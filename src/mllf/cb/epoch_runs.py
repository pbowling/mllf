"""One online-training epoch's simulations: propose biases, run, record.

``EpochRun`` holds everything one ``train_epoch`` call shares between its
phases (policy, config, per-pair state, the run's accumulators). Phase 1
(``prepare_all``) builds each combo's graph, samples actions (shifted by
per-pair offsets in hindsight mode) and writes ``variables.py``; phase 2
(``simulate``) submits the runs to SLURM, polls them, retries or replaces
failures and records each finished run's ``epoch_results.pt``. The updates
that read those records are in ``mllf.cb.epoch_updates``.
"""
import concurrent.futures
import logging
import shutil
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from mllf.cb.bandit_config import BanditConfig
from mllf.cb.cost_ledger import simulated_ns
from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.group_correction import apply_group_values, correction_config
from mllf.cb.hindsight import (
    apply_pair_offsets,
    cap_quadratic,
    combo_node_labels,
    linear_edge_variance,
)
from mllf.cb.msld_reward import compute_msld_reward_improved
from mllf.cb.slurm import submit_epoch_simulation
from mllf.cb.workflow_utils import (
    check_simulation_success,
    format_coverage_line,
    format_crossing_line,
    parse_simulation_metrics,
    record_combo_failure,
    set_msld_steps,
    template_steps,
)
from mllf.file_handling.combo_sets import is_group_combo
from mllf.file_handling.manifests import build_combo_dir_lookup, ensure_combo_dir_exists
from mllf.file_handling.write_bias_coeff import write_variables_from_actions
from mllf.slurm import refresh_exclude


log = logging.getLogger(__name__)


# Policy forward passes are not thread-safe on CPU; phase-1 workers share one policy.
policy_inference_lock = threading.Lock()

# reward: keys -> defaults for compute_msld_reward_improved; a cached run is
# re-scored when any of them changed since it was recorded.
REWARD_DEFAULTS = {
    'w_P': 0.5, 'w_T': 0.75, 'w_U': 0.3, 'gamma': 4.0, 'P_baseline': 500.0, 'T_baseline': 50.0,
    'min_transitions_per_site': 10, 'min_coverage_ratio': 0.5, 'entropy_bonus': 8.0,
    'concentration_penalty_threshold': 0.8,
}
# parse_simulation_metrics fields kept in epoch_results.pt, with their defaults.
METRIC_FIELDS = (
    ('populations', {}), ('transitions', {}), ('ddg_pairs', {}), ('fraction_physical', None),
    ('barrier_pairs', {}), ('skew_pairs', {}), ('end_pairs', {}), ('transition_pairs', {}),
    ('endpoint_pairs', {}), ('site_holdoff', []), ('endpoint_nodes', None),
)


@dataclass
class EpochOptions:
    """How one ``train_epoch`` call trains (all optional).

    update_policy: False runs and records without any update (validation).
    bc_mode / bc_source_epoch / bc_auxiliary_weight: behavioral cloning toward
        saved actions (all of them, or an auxiliary loss next to REINFORCE).
    replay_from_epoch: reuse that epoch's results instead of simulating.
    deterministic_actions: posterior / policy means, no exploration.
    ema_baseline: REINFORCE's running reward baseline (updated in place).
    reserve_pool: combos that replace permanently failed ones.
    bandit_state: per-pair offsets and per-combo failures (updated in place);
        None runs the shared model alone.
    baseline_net / baseline_optimizer / system_to_idx: REINFORCE's
        context-dependent baseline (``ContextBaseline``).
    history_replay_cfg: REINFORCE replay-AWR on past epochs' runs.
    update_shared: False keeps the shared model fixed while per-pair state
        still learns (``mode: deploy``).
    stage_of_system / stage_cfg_of_system: each system's curriculum stage
        name and settings (per-pair stage counts, short-run success test).
    run_steps_fn(system, combo name) -> (equil, prod) or None: each run's
        length (None: the system template's full length).
    explicit_combos: combo path -> info for combos in no manifest
        (``combo_sets``), so their directories can be created.
    replay_cache: replay sets for the hindsight regression, by cache path;
        pass the same dict every epoch of a run to build each set once.
    """
    update_policy: bool = True
    bc_mode: bool = False
    bc_source_epoch: Optional[int] = None
    replay_from_epoch: Optional[int] = None
    deterministic_actions: bool = False
    bc_auxiliary_weight: float = 0.0
    ema_baseline: Optional[dict] = None
    reserve_pool: Optional[List[str]] = None
    verbose: bool = False
    bandit_state: Optional[dict] = None
    baseline_net: Optional[torch.nn.Module] = None
    baseline_optimizer: Optional[torch.optim.Optimizer] = None
    system_to_idx: Optional[Dict[str, int]] = None
    history_replay_cfg: Optional[Dict] = None
    update_shared: bool = True
    stage_of_system: Optional[Dict[str, str]] = None
    run_steps_fn: Optional[Callable[[str, str], Optional[Tuple[int, int]]]] = None
    stage_cfg_of_system: Optional[Dict[str, Dict]] = None
    explicit_combos: Optional[Dict[str, Dict]] = None
    replay_cache: Dict[str, list] = field(default_factory=dict)


# small helpers

def sample_actions(policy, x, edge_index, edge_type, deterministic: bool = False,
                   exploration_scale: float = 1.0):
    """Sample actions from *policy*, dispatching on its bandit mode.

    Bayesian-head policies (``use_bayesian_heads=True``) use
    ``get_actions_thompson`` (``exploration_scale`` widens or narrows the
    posterior spread); others use REINFORCE's ``get_actions``. Callers hold
    ``policy_inference_lock`` and ``torch.no_grad()``.

    Returns (actions, logp); ``logp`` is None in Bayesian mode.
    """
    if getattr(policy, 'use_bayesian_heads', False):
        actions, _mean, _var = policy.get_actions_thompson(
            x, edge_index, edge_type, deterministic=deterministic,
            exploration_scale=exploration_scale)
        return actions, None
    actions, logp, _mean, _log_std = policy.get_actions(
        x, edge_index, edge_type, deterministic=deterministic)
    return actions, logp


def detach_cpu_or_none(t):
    """``t.detach().cpu()`` if *t* is a tensor, else ``None`` (for Bayesian-mode logp)."""
    return t.detach().cpu() if t is not None else None


class CachedGraphData:
    """The graph tensors the update step needs (x, edge_index, edge_type),
    reloaded from the small per-run cache (``save_graph_data_cache``)."""
    __slots__ = ('x', 'edge_index', 'edge_type')

    def __init__(self, x, edge_index, edge_type):
        self.x = x
        self.edge_index = edge_index
        self.edge_type = edge_type


def save_graph_data_cache(epoch_dir: Path, data) -> None:
    """Save CPU copies of graph tensors for reuse after simulation completes."""
    try:
        torch.save({
            'x': data.x.detach().cpu(),
            'edge_index': data.edge_index.detach().cpu(),
            'edge_type': (data.edge_type.detach().cpu()
                          if data.edge_type is not None else None),
        }, epoch_dir / 'graph_data.pt')
    except Exception:
        pass


def load_graph_data_cache(epoch_dir: Path, device: str) -> Optional[CachedGraphData]:
    """Load graph tensors saved by save_graph_data_cache, or None if absent."""
    cache_path = epoch_dir / 'graph_data.pt'
    if not cache_path.exists():
        return None
    try:
        saved = torch.load(cache_path, weights_only=False)
        edge_type = saved.get('edge_type')
        return CachedGraphData(
            saved['x'].to(device),
            saved['edge_index'].to(device),
            edge_type.to(device) if edge_type is not None else None,
        )
    except Exception:
        return None


def msld_reward(epoch_dir, reward_cfg: Dict, verbose: bool) -> float:
    """``compute_msld_reward_improved`` for a run directory with the config's reward weights."""
    return compute_msld_reward_improved(
        str(epoch_dir), **{k: reward_cfg.get(k, d) for k, d in REWARD_DEFAULTS.items()},
        suppress_warnings=not verbose)


def results_record(reward: float, actions: torch.Tensor, logp, reward_cfg: Dict,
                   metrics: Dict, run_id: str) -> Dict:
    """The ``epoch_results.pt`` record of one finished run."""
    return {'reward': reward, 'actions': actions.cpu(), 'logp': logp, 'reward_config': reward_cfg,
            **{k: metrics.get(k, d) for k, d in METRIC_FIELDS}, 'run_id': run_id}


@dataclass
class _Job:
    combo_path: Path
    epoch_dir: Path
    job_id: str
    actions: torch.Tensor
    logp: Optional[torch.Tensor]
    retries: int = 0


# the epoch

class EpochRun:
    """State shared by the phases of one ``train_epoch`` call."""

    def __init__(self, policy, combos: List[str], combo_system_map: Dict[str, Dict], epoch: int,
                 config: Dict, device: str, opts: EpochOptions):
        self.policy = policy
        self.combos = combos
        self.combo_system_map = combo_system_map
        self.epoch = epoch
        self.config = config
        self.device = device
        self.opts = opts
        self.bandit = BanditConfig.from_config(config)
        self.reward_cfg = config.get('reward', {})
        self.bandit_state = opts.bandit_state
        # Per-pair offsets need bandit_state, so validation passes
        # (bandit_state=None) always see the shared model alone.
        self.use_offsets = (self.bandit.use_hindsight and self.bandit_state is not None
                            and bool(self.bandit.pair_offsets))
        self.hs_dims = self.bandit.offset_dims
        # bandit.group_offset_dims: x / s offsets are tuned to a pair alone
        # and stack over a substituent's partners in a group
        self.group_dims = self.bandit.group_dims
        self.offset_sampling = self.use_offsets and bool(self.bandit.offset_sampling)
        # bandit.group_correction: a group run whose end terms hold an
        # endpoint (or a whole site) off shrinks those pairs' x / s for that
        # group; its later runs use the corrected values.
        self.gc_cfg = correction_config(self.bandit.raw) if self.use_offsets else None
        # Tags every epoch_results.pt with the training run (checkpoint dir)
        # that produced it: combo trees are shared between runs of the same
        # systems, and a run must not reuse another run's results.
        self.run_id = str(config.get('output', {}).get('base_dir', 'unknown'))
        self.bias_clip = config.get('training', {}).get('bias_clip', 1000.0)
        self._templates: Dict[str, Optional[Dict[str, float]]] = {}
        self._map_lock = threading.Lock()
        self.dir_lookup: Dict[str, Dict] = {}
        # accumulated over the epoch
        self.epoch_rewards: List[float] = []
        self.deferred_updates: List[Tuple[Path, Path, float]] = []
        self.permanently_failed: List[str] = []
        self.failed_run_dirs: List[Tuple[str, str, bool]] = []   # cost ledger: failed runs
        # Lines for epoch_summary.log (written after the epoch's header) and
        # structured regression / hindsight numbers.
        self.detail_lines: List[str] = []
        self.epoch_info: Dict[str, Dict] = {}
        self.epoch_loss = 0.0

    # -- per-system lookups -------------------------------------------------
    def sys_cfg(self, combo_path) -> Dict:
        return self.combo_system_map.get(str(combo_path), {})

    def sys_name(self, combo_path, default: str = 'unknown') -> str:
        return self.sys_cfg(combo_path).get('name', default)

    def template(self, sys_cfg: Dict) -> Optional[Dict[str, float]]:
        idir = str(sys_cfg.get('input_dir', ''))
        if idir not in self._templates:
            self._templates[idir] = template_steps(idir) if idir else None
        return self._templates[idir]

    def apply_run_length(self, combo_path, sys_cfg: Dict) -> None:
        """Write this run's length into the combo's msld_flat.py (no-op
        without run_steps_fn or a readable template)."""
        if self.opts.run_steps_fn is None:
            return
        tpl = self.template(sys_cfg)
        if tpl is None:
            return
        steps = self.opts.run_steps_fn(sys_cfg.get('name', 'unknown'), Path(combo_path).name)
        equil, prod = steps if steps else (tpl['equil'], tpl['prod'])
        set_msld_steps(Path(combo_path), equil, prod)

    def stage_cfg(self, sys_name: str) -> Dict:
        return (self.opts.stage_cfg_of_system or {}).get(sys_name) or {}

    def short_test(self, sys_name: str) -> Tuple[int, float]:
        """(min transitions, min minority fraction) of the system's short-run
        success test (stage keys explore_min_transitions / explore_min_fraction)."""
        st = self.stage_cfg(sys_name)
        return (int(st.get('explore_min_transitions', 2)),
                float(st.get('explore_min_fraction', 0.2)))

    def is_short_run(self, combo_path, epoch_dir) -> bool:
        """Simulated well under (< half) the template's full length."""
        if self.opts.run_steps_fn is None:
            return False
        tpl = self.template(self.sys_cfg(combo_path))
        ns = simulated_ns(Path(epoch_dir) / 'output.out')
        if tpl is None or ns is None:
            return False
        return ns < 0.5 * (tpl['equil'] + tpl['prod']) * tpl['timestep'] / 1000.0

    # -- proposals ----------------------------------------------------------
    def with_pair_offsets(self, actions, data, combo_path, sys_cfg):
        """Shift a Thompson-sampled proposal by the pair's offsets, if any,
        then set the group's corrected x / s (bandit.group_correction).

        Returns (actions, linear_var): linear_var is each edge's b-difference
        variance after its pair's offset (``linear_edge_variance``), weighting
        the per-site b fit in write_variables_from_actions; None without
        offsets."""
        policy = self.policy
        if (not self.use_offsets or data.edge_type is not None
                or not getattr(policy, 'use_bayesian_heads', False)):
            return actions, None
        with torch.no_grad(), policy_inference_lock:
            mean, _, _ = policy.get_actions_thompson(
                data.x, data.edge_index, data.edge_type, deterministic=True)
        labels = combo_node_labels(combo_path)
        sys_name = sys_cfg.get('name', 'unknown')
        b = self.bandit
        dims = self.group_dims if is_group_combo(combo_path) else self.hs_dims
        shifted, _n = apply_pair_offsets(
            actions, mean.detach(), self.bandit_state, sys_name,
            labels, data.edge_index,
            prior_sd=b.offset_prior_sd,
            sample=self.offset_sampling and not self.opts.deterministic_actions,
            base_sd=b.offset_base_sd,
            dims=dims)
        shifted, _ = cap_quadratic(shifted, mean.detach(), b.c_cap)
        if self.gc_cfg is not None:
            shifted, _ = apply_group_values(shifted, self.bandit_state, sys_name,
                                            Path(combo_path).name,
                                            labels, data.edge_index)
        # without b offsets the b fit weights every edge by the prior (model alone)
        lin_var = (linear_edge_variance(self.bandit_state if 'linear' in dims else None,
                                        sys_name, labels, data.edge_index,
                                        prior_sd=b.offset_prior_sd)
                   if 'linear' in self.hs_dims else None)
        return shifted, lin_var

    def propose(self, data, combo_path, sys_cfg):
        """(actions, logp, linear_var) for one combo's run this epoch."""
        policy = self.policy
        lin_var = None
        with torch.no_grad():
            if getattr(policy, 'use_bayesian_heads', False):
                # Thompson sampling from each bias head's posterior (or, for
                # eval combos, the posterior mean).
                with policy_inference_lock:
                    actions, logp = sample_actions(
                        policy, data.x, data.edge_index, data.edge_type,
                        deterministic=self.opts.deterministic_actions,
                        exploration_scale=self.bandit.exploration_scale)
                actions, lin_var = self.with_pair_offsets(actions.detach(), data, combo_path,
                                                          sys_cfg)
            elif self.opts.deterministic_actions:
                with policy_inference_lock:
                    mean, _ = policy._forward_edges(data.x, data.edge_index, data.edge_type)
                actions = mean.detach()
                with policy_inference_lock:
                    logp = policy.evaluate_logp(data.x, data.edge_index, data.edge_type,
                                                actions)[0].detach()
            else:
                with policy_inference_lock:
                    actions, logp, _, _ = policy.get_actions(data.x, data.edge_index,
                                                             data.edge_type)
                actions = actions.detach()
                logp = logp.detach()
        return actions, logp, lin_var

    # -- phase 1 ------------------------------------------------------------
    def _recover(self, combo_path: Path, sys_cfg: Dict, epoch_dir: Path):
        """A previous run's simulation finished but its epoch_results.pt was
        never saved: score it now. Returns the reward, or None."""
        prev_output = epoch_dir / 'output.out'
        try:
            reward = msld_reward(epoch_dir, self.reward_cfg, self.opts.verbose)
            data, _ = build_graph_and_data(str(combo_path), sys_cfg, self.config, self.device)
            save_graph_data_cache(epoch_dir, data)
            with torch.no_grad():
                with policy_inference_lock:
                    actions, logp = sample_actions(
                        self.policy, data.x, data.edge_index, data.edge_type,
                        exploration_scale=self.bandit.exploration_scale)
            torch.save(results_record(reward, actions.detach(), detach_cpu_or_none(logp),
                                      self.reward_cfg,
                                      parse_simulation_metrics(prev_output), self.run_id),
                       epoch_dir / 'epoch_results.pt')
            if self.opts.verbose:
                log.debug(f"  Recovered from existing output.out: reward={reward:.4f}")
            return reward
        except Exception as e:
            warnings.warn(f"Recovery from output.out failed for {combo_path.name}: {e}")
            return None

    def _cached_reward(self, combo_path: Path, epoch_results_file: Path):
        """This run's own recorded reward for the combo (resuming a crashed
        run), re-scored if the reward settings changed; None if absent or
        another run's."""
        verbose = self.opts.verbose
        try:
            cached = torch.load(epoch_results_file, weights_only=False)
            if cached.get('run_id') != self.run_id:
                raise LookupError(f"results belong to run {cached.get('run_id')!r}")
            cached_reward_config = cached.get('reward_config', {})
            if any(cached_reward_config.get(k) != self.reward_cfg.get(k) for k in REWARD_DEFAULTS):
                if verbose:
                    log.debug("  Reward config changed - recomputing reward")
                try:
                    reward = msld_reward(epoch_results_file.parent, self.reward_cfg, verbose)
                    if verbose:
                        log.debug(f"  Old reward: {cached.get('reward', 'N/A')} -> "
                                  f"New reward: {reward:.4f}")
                except Exception as _e:
                    warnings.warn(f"Could not recompute reward for {combo_path.name}: {_e}")
                    reward = cached['reward']
            else:
                reward = cached['reward']
            if verbose:
                log.debug(f"  Using cached reward: {reward:.4f}")
            return reward
        except LookupError as e:
            if verbose:
                log.debug(f"  Not reusing {epoch_results_file}: {e}")
        except Exception as e:
            warnings.warn(f"Failed to load cached results for {combo_path.name}: {e}")
        return None

    def prepare_combo(self, args):
        """Prepare one combo. Returns one of:

          ('cached',  idx, reward, combo_path, epoch_dir)
          ('pending', idx, combo_path, epoch_dir, actions, logp, sys_cfg)
          ('skip',    idx)   # run_sims=False
          ('error',   idx)   # graph build or dir-creation failure
        """
        combo_idx, combo_dir = args
        combo_path = Path(combo_dir)
        sys_cfg = self.sys_cfg(combo_path)
        original = str(combo_path)
        try:
            combo_path = ensure_combo_dir_exists(combo_path, sys_cfg, self.config,
                                                 verbose=self.opts.verbose,
                                                 _lookup=self.dir_lookup)
        except Exception as e:
            log.error(f"  ERROR: Could not create combo dir {combo_path.name}: {e}")
            return ('error', combo_idx)
        if str(combo_path) != original and sys_cfg:
            with self._map_lock:
                self.combo_system_map[str(combo_path)] = sys_cfg
        self.apply_run_length(combo_path, sys_cfg)

        epoch_dir = combo_path / f"run_{self.epoch:03d}"
        epoch_dir.mkdir(exist_ok=True, parents=True)
        if self.opts.replay_from_epoch is not None:
            source_pt = combo_path / f"run_{self.opts.replay_from_epoch:03d}" / 'epoch_results.pt'
            dest_pt = epoch_dir / 'epoch_results.pt'
            if not dest_pt.exists() and source_pt.exists():
                shutil.copy2(str(source_pt), str(dest_pt))

        epoch_results_file = epoch_dir / 'epoch_results.pt'
        prev_output = epoch_dir / 'output.out'
        if (not epoch_results_file.exists() and prev_output.exists()
                and check_simulation_success(prev_output)):
            reward = self._recover(combo_path, sys_cfg, epoch_dir)
            if reward is not None:
                return ('cached', combo_idx, reward, combo_path, epoch_dir)
        if epoch_results_file.exists():
            reward = self._cached_reward(combo_path, epoch_results_file)
            if reward is not None:
                return ('cached', combo_idx, reward, combo_path, epoch_dir)

        try:
            data, extras = build_graph_and_data(str(combo_path), sys_cfg, self.config, self.device)
        except Exception as e:
            log.error(f"  ERROR: Failed to build graph for {combo_path.name}: {e}")
            return ('error', combo_idx)
        save_graph_data_cache(epoch_dir, data)
        actions, logp, lin_var = self.propose(data, combo_path, sys_cfg)
        write_variables_from_actions(str(epoch_dir), data, extras, actions,
                                     bias_clip=self.bias_clip,
                                     linear_var=lin_var)
        if not self.config.get('run_sims', True):
            if self.opts.verbose:
                log.debug("  Skipping simulation (run_sims=False)")
            return ('skip', combo_idx)
        if self.opts.verbose:
            log.debug(f"  Prepared {combo_idx + 1}/{len(self.combos)}: "
                      f"{sys_cfg.get('name', 'unknown')}/{combo_path.name}")
        return ('pending', combo_idx, combo_path, epoch_dir, actions, logp, sys_cfg)

    def prepare_all(self):
        """Phase 1: prepare every combo with a small thread pool. Results are
        handled in combo order, so the epoch's records are deterministic.
        Returns (pending submissions, counts)."""
        # Pre-scan combo_metadata.json once for combos whose dirs don't exist
        # yet (memory O(missing combos), not O(all metadata)). Explicit combos
        # stay findable for reserve-pool replacements too.
        self.dir_lookup = {**(self.opts.explicit_combos or {}),
                           **build_combo_dir_lookup(self.combos, self.combo_system_map,
                                                    explicit=self.opts.explicit_combos)}
        n_workers = self.config.get('phase1_workers', 2)
        log.info(f"\nPhase 1: preparing {len(self.combos)} combos ({n_workers} workers)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(self.prepare_combo, enumerate(self.combos)))
        pending = []
        counts = {'cached': 0, 'pending': 0, 'skip': 0, 'error': 0}
        for result in results:
            kind = result[0]
            counts[kind] += 1
            if kind == 'cached':
                _, _, reward, cp, ed = result
                self.epoch_rewards.append(reward)
                self.deferred_updates.append((cp, ed, reward))
            elif kind == 'pending':
                _, _, cp, ed, act, lp, sc = result
                pending.append((cp, ed, act, lp, sc))
        log.info(f"  Prep done: {counts['cached']} cached, {counts['pending']} to submit, "
                 f"{counts['skip']} skipped, {counts['error']} errors")
        return pending, counts

    # -- phase 2 ------------------------------------------------------------
    def _abort_if_too_many_failures(self) -> None:
        max_failed = self.config.get('max_failed_combos_per_epoch', 50)
        if len(self.permanently_failed) >= max_failed:
            log.error(f"\nFATAL: {len(self.permanently_failed)} combos permanently failed "
                      f"(>= max_failed_combos_per_epoch={max_failed}). Aborting.")
            log.error("  Failed combos:")
            for fc in self.permanently_failed:
                log.error(f"    {fc}")
            sys.exit(1)

    def _submit_replacement(self, reserve: List[str], job_queue: List[_Job]) -> None:
        """Run the next reserve combo in place of a permanently failed one."""
        while reserve:
            rep_str = reserve.pop(0)
            rep_path = Path(rep_str)
            rep_sys = self.combo_system_map.get(rep_str, {})
            try:
                rep_path = ensure_combo_dir_exists(rep_path, rep_sys, self.config,
                                                   verbose=self.opts.verbose,
                                                   _lookup=self.dir_lookup)
                self.apply_run_length(rep_path, rep_sys)
                rep_epoch_dir = rep_path / f"run_{self.epoch:03d}"
                rep_epoch_dir.mkdir(exist_ok=True, parents=True)
                if (rep_epoch_dir / 'epoch_results.pt').exists():
                    break  # already cached; counts as done
                rep_data, rep_extras = build_graph_and_data(str(rep_path), rep_sys, self.config,
                                                            self.device)
                save_graph_data_cache(rep_epoch_dir, rep_data)
                with torch.no_grad():
                    rep_act, rep_lp = sample_actions(
                        self.policy, rep_data.x, rep_data.edge_index, rep_data.edge_type,
                        exploration_scale=self.bandit.exploration_scale)
                rep_act, rep_lv = self.with_pair_offsets(rep_act.detach(), rep_data, rep_path,
                                                         rep_sys)
                rep_lp = detach_cpu_or_none(rep_lp)
                write_variables_from_actions(str(rep_epoch_dir), rep_data, rep_extras, rep_act,
                                             bias_clip=self.bias_clip, linear_var=rep_lv)
                rep_job_id = submit_epoch_simulation(
                    str(rep_path), str(rep_epoch_dir), config=self.config,
                    solvent_state=rep_sys.get('solvent_state', 'solv'))
                job_queue.append(_Job(rep_path, rep_epoch_dir, rep_job_id, rep_act, rep_lp))
                log.debug(f"  Replacement: {rep_sys.get('name', 'unknown')}/{rep_path.name}"
                          f"  ->  job {rep_job_id}")
                break
            except Exception as rep_e:
                log.debug(f"  Replacement {rep_path.name} failed: {rep_e}; trying next")

    def _finish_job(self, job: _Job, job_queue: List[_Job], reserve: List[str]) -> None:
        """A submitted job left the queue: retry it, give it up (and replace
        it from the reserve pool) or record its results."""
        max_retries = self.config.get('max_retries', 2)
        output_file = job.epoch_dir / 'output.out'
        success = check_simulation_success(output_file)
        if not success and job.retries < max_retries:
            log.debug(f"  Simulation failed for {job.combo_path.name}, retrying "
                      f"(attempt {job.retries + 2}/{max_retries + 1})...")
            try:
                # nodes added to slurm.exclude_file since the first attempt apply
                refresh_exclude(job.epoch_dir / 'run_epoch.sh', self.config.get('slurm'))
                result = subprocess.run(['sbatch', str(job.epoch_dir / 'run_epoch.sh')],
                                        cwd=str(job.epoch_dir), capture_output=True, text=True,
                                        check=True)
                job_queue.append(_Job(job.combo_path, job.epoch_dir,
                                      result.stdout.strip().split()[-1],
                                      job.actions, job.logp, job.retries + 1))
            except Exception as e:
                log.error(f"  Error resubmitting: {e}; skipping {job.combo_path.name}")
            return
        if not success:
            failed_sys = self.sys_name(job.combo_path)
            log.warning(f"  SKIP: {failed_sys}/{job.combo_path.name} permanently failed "
                        f"after {max_retries + 1} attempt(s); continuing without it")
            self.permanently_failed.append(f"{failed_sys}/{job.combo_path.name}")
            self.failed_run_dirs.append((failed_sys, str(job.epoch_dir), False))
            if self.bandit_state is not None:
                # An infrastructure/software failure, not a poor-sampling
                # result: uncertainty selection stops re-selecting a combo
                # that cannot be simulated (record_combo_failure).
                record_combo_failure(self.bandit_state, str(job.combo_path),
                                     cap=self.bandit.failure_retry_cap)
            self._abort_if_too_many_failures()
            self._submit_replacement(reserve, job_queue)
            return

        log.debug(f"  {self.sys_name(job.combo_path)}/{job.combo_path.name}:")
        try:
            reward = msld_reward(job.epoch_dir, self.reward_cfg, self.opts.verbose)
        except Exception as e:
            warnings.warn(f"Failed to compute reward for {job.combo_path.name}: {e}")
            reward = -100.0
        self.epoch_rewards.append(reward)
        raw_metrics = parse_simulation_metrics(output_file)
        log.debug(f"  {format_coverage_line(raw_metrics)}")
        crossing = format_crossing_line(raw_metrics)
        if crossing:
            log.debug(f"  {crossing}")
        torch.save(results_record(reward, job.actions, detach_cpu_or_none(job.logp),
                                  self.reward_cfg,
                                  raw_metrics, self.run_id), job.epoch_dir / 'epoch_results.pt')
        self.deferred_updates.append((job.combo_path, job.epoch_dir, reward))

    def simulate(self, pending: list, counts: Dict[str, int]) -> int:
        """Phase 2: submit to SLURM and poll until every job has finished.
        Returns the number submitted."""
        max_concurrent = self.config.get('max_concurrent_jobs', 10)
        poll_interval = self.config.get('poll_interval', 10)
        reserve = list(self.opts.reserve_pool) if self.opts.reserve_pool else []
        job_queue: List[_Job] = []
        n_submitted = 0
        live_status = sys.stdout.isatty() and log.isEnabledFor(logging.INFO)
        if pending:
            log.info(f"\nPhase 2: submitting {counts['pending']} simulations "
                     f"(max {max_concurrent} concurrent SLURM jobs)")
        while job_queue or pending:
            sq = subprocess.run(['squeue', '--me', '--noheader', '--format=%i'],
                                capture_output=True, text=True)
            active_ids = {ln.strip() for ln in sq.stdout.splitlines() if ln.strip()}
            done = [i for i, j in enumerate(job_queue) if str(j.job_id) not in active_ids]
            for i in reversed(done):
                self._finish_job(job_queue.pop(i), job_queue, reserve)

            slots = max_concurrent - len(job_queue)
            while slots > 0 and pending:
                p_combo, p_edir, p_act, p_lp, p_sys = pending.pop(0)
                p_sys_name = p_sys.get('name', 'unknown')
                try:
                    job_id = submit_epoch_simulation(str(p_combo), str(p_edir), config=self.config,
                                                     solvent_state=p_sys.get('solvent_state',
                                                                             'solv'))
                    job_queue.append(_Job(p_combo, p_edir, job_id, p_act, p_lp))
                    n_submitted += 1
                    log.debug(f"  Submitted {n_submitted}/{counts['pending']}: "
                              f"{p_sys_name}/{p_combo.name}  ->  job {job_id}")
                    slots -= 1
                except Exception as e:
                    # Usually a cluster/config problem (e.g. missing slurm:
                    # partition/gres), not the combo: no record_combo_failure,
                    # but counted so a broken setup aborts.
                    stderr = getattr(e, 'stderr', None)
                    log.error(f"  ERROR: Failed to submit {p_combo.name}: {e}"
                              + (f"\n    sbatch: {stderr.strip()}" if stderr else ""))
                    self.permanently_failed.append(f"{p_sys_name}/{p_combo.name} (submission)")
                    self._abort_if_too_many_failures()

            if job_queue or pending:
                n_active = len([j for j in job_queue if str(j.job_id) in active_ids])
                if live_status:   # rewritten in place; a log file would keep every copy
                    print(f"  [running={n_active} | finishing={len(job_queue) - n_active} | "
                          f"queued={len(pending)}]", end='\r', flush=True)
                time.sleep(poll_interval)
        if live_status:
            print()   # end the status line
        log.info(f"  Complete: {n_submitted} submitted, {counts['cached']} cached, "
                 f"{counts['skip']} skipped, {counts['error']} errors")
        return n_submitted

    def prepare_and_simulate(self) -> None:
        """Phases 1 and 2. A training epoch with no successful run aborts
        (updating on no data would record an empty epoch); a validation pass
        only warns."""
        pending, counts = self.prepare_all()
        n_submitted = self.simulate(pending, counts)
        if self.deferred_updates:
            return
        summary = (f"{n_submitted} submitted, {counts['cached']} cached, "
                   f"{len(self.permanently_failed)} failed, {counts['error']} prep errors")
        if self.opts.update_policy:
            log.error(f"\nFATAL: epoch {self.epoch} produced no successful simulations "
                      f"({summary}). Aborting.")
            for fc in self.permanently_failed:
                log.info(f"    {fc}")
            sys.exit(1)
        log.warning(f"  WARNING: no successful simulations this pass ({summary})")
