"""Per-epoch online training step for the UniMol CB policy.

The core of the online-training driver: for a batch of combo directories,
this builds (or reuses cached) Uni-Mol graph data, samples policy actions,
submits/polls SLURM simulations, computes per-edge rewards, and updates the
policy — either via REINFORCE (``bandit.algorithm: reinforce``, the default)
or a closed-form NeuralLinear + Thompson Sampling posterior update
(``bandit.algorithm: neurallinear_ts``). Also supports a behavioral-cloning
mode and replay-style AWR (see ``train_epoch``'s docstring).

Moved here from ``training/workflow.py`` as part of making the online CB
training driver importable as a library (see ``mllf.cli.train_workflow``).
The helper functions below (leading underscore) are implementation details
of ``train_epoch`` and not a public API of this module.
"""
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.graph_utils import build_directed_pairs
from mllf.cb.hindsight import (
    DEFAULT_TEMPERATURE,
    apply_pair_offsets,
    combo_node_labels,
    hindsight_targets,
    preact_weights,
    read_applied_biases,
    record_pair_targets,
)
from mllf.cb.policy import ContextBaseline, UnimolPolicy
from mllf.cb.pretrain_policy import compute_pairwise_confidence_weights
from mllf.cb.train_improved import compute_msld_reward_improved
from mllf.cb.workflow_utils import (
    BARRIER_TARGET,
    BARRIER_WIDTH,
    END_WIDTH,
    SKEW_WIDTH,
    build_edge_weights,
    check_simulation_success,
    compute_failure_reward,
    compute_pair_reward,
    compute_pair_success_stats,
    fix_msld_flat_for_single_site,
    parse_simulation_metrics,
    record_combo_failure,
)
from mllf.cli.slurm import submit_epoch_simulation
from mllf.cli.workflow import write_variables_from_actions
from mllf.file_handling.manifests import build_combo_dir_lookup, ensure_combo_dir_exists


# Thread-safety lock for policy inference (policy forward passes are not thread-safe on CPU)
_policy_inference_lock = threading.Lock()


def _policy_sample_actions(policy, x, edge_index, edge_type, deterministic: bool = False,
                            exploration_scale: float = 1.0):
    """Sample actions from *policy*, dispatching on its bandit mode.

    NeuralLinear + Thompson Sampling policies (``use_bayesian_heads=True``) use
    ``get_actions_thompson``; everything else uses the original REINFORCE
    ``get_actions``. Callers should hold ``_policy_inference_lock`` and be
    inside ``torch.no_grad()`` around this call, same as the previous
    direct ``policy.get_actions(...)`` call sites this replaces.

    Args:
        exploration_scale: Forwarded to ``get_actions_thompson`` (see its
            docstring) — a runtime dial to widen/narrow Thompson-sampling
            spread beyond whatever the posterior's own calibrated/narrowed
            covariance currently gives, without needing to regenerate the
            pretrained Bayesian checkpoint. Ignored in REINFORCE mode.

    Returns:
        (actions, logp) — ``logp`` is ``None`` in Bayesian mode (a
        REINFORCE-only concept; nothing downstream reads it for that mode).
    """
    if getattr(policy, 'use_bayesian_heads', False):
        actions, _mean, _var = policy.get_actions_thompson(
            x, edge_index, edge_type, deterministic=deterministic,
            exploration_scale=exploration_scale)
        return actions, None
    actions, logp, _mean, _log_std = policy.get_actions(
        x, edge_index, edge_type, deterministic=deterministic)
    return actions, logp


def _detach_cpu_or_none(t):
    """``t.detach().cpu()`` if *t* is a tensor, else ``None`` (for Bayesian-mode logp)."""
    return t.detach().cpu() if t is not None else None


# No atom deletions needed for these pretraining systems.
SITE_ATOMS_MAPPING: Dict[int, str] = {}


class _CachedGraphData:
    """Lightweight holder for the graph tensors needed by the REINFORCE
    update step (x, edge_index, edge_type). Reloaded from the small on-disk
    cache written by _save_graph_data_cache; duck-types the subset of
    UnimolData's attributes actually used downstream.
    """
    __slots__ = ('x', 'edge_index', 'edge_type')

    def __init__(self, x, edge_index, edge_type):
        self.x = x
        self.edge_index = edge_index
        self.edge_type = edge_type


def _save_graph_data_cache(epoch_dir: Path, data) -> None:
    """Persist the graph tensors needed later for the REINFORCE update to a
    small file in *epoch_dir*.

    This lets the REINFORCE prep loop reuse the same Uni-Mol embeddings
    computed during Phase 1 instead of recomputing them, WITHOUT keeping
    every combo's data resident in RAM across the (potentially long) SLURM
    wait in Phase 2 — an earlier in-memory-dict version of this cache caused
    an OOM because it held all epoch combos' embeddings live for the entire
    epoch. Persisting the (tiny — a few tens of KB) tensors to disk and
    reloading them one at a time in Phase 3 keeps the memory profile the
    same as before caching was introduced.
    """
    try:
        torch.save({
            'x': data.x.detach().cpu(),
            'edge_index': data.edge_index.detach().cpu(),
            'edge_type': (data.edge_type.detach().cpu()
                          if data.edge_type is not None else None),
        }, epoch_dir / 'graph_data.pt')
    except Exception:
        pass


def _load_graph_data_cache(epoch_dir: Path, device: str) -> Optional['_CachedGraphData']:
    """Load graph tensors saved by _save_graph_data_cache, or None if absent."""
    cache_path = epoch_dir / 'graph_data.pt'
    if not cache_path.exists():
        return None
    try:
        saved = torch.load(cache_path, weights_only=False)
        edge_type = saved.get('edge_type')
        return _CachedGraphData(
            saved['x'].to(device),
            saved['edge_index'].to(device),
            edge_type.to(device) if edge_type is not None else None,
        )
    except Exception:
        return None


def _prepare_success_combo_reward(
    combo_path_u: Path,
    epoch_dir_u: Path,
    combo_system_map: Dict[str, Dict],
    config: Dict,
    device: str,
    no_transition_weight: float,
):
    """Load cached graph/result data for one crash-free combo and compute its
    per-edge reward + confidence weights.

    Shared by the REINFORCE and NeuralLinear+TS epoch-batch update paths in
    ``train_epoch`` — reward computation (``compute_pair_reward``) is identical
    between the two algorithms; only what happens with the result differs
    (gradient step vs. closed-form posterior update).

    Returns:
        (vdata_u, saved_data, saved_actions, r_pair, edge_weights_u, trans_vals)

    Raises on I/O/graph-build failure — callers catch and skip the combo, same
    as the inline code this replaces.
    """
    sys_cfg_u = combo_system_map.get(str(combo_path_u), {})
    # Reuse the graph/Uni-Mol data built earlier this epoch (Phase 1 prep,
    # replacement, or recovery path), loaded from the small on-disk cache
    # written by _save_graph_data_cache, instead of recomputing embeddings for
    # every combo a second time. Only combos whose reward came from a
    # pre-existing cached epoch_results.pt (no build this epoch) fall back to
    # a fresh build here.
    vdata_u = _load_graph_data_cache(Path(epoch_dir_u), device)
    if vdata_u is None:
        vdata_u, _ = build_graph_and_data(str(combo_path_u), sys_cfg_u, config, device)
    # graph_data.pt is only needed to bridge Phase 1 -> here within the same
    # epoch; remove it now so per-epoch combo dirs don't accumulate this
    # artifact indefinitely across the run.
    _gd_cache_path = Path(epoch_dir_u) / 'graph_data.pt'
    if _gd_cache_path.exists():
        try:
            _gd_cache_path.unlink()
        except Exception:
            pass

    results_pt = Path(epoch_dir_u) / 'epoch_results.pt'
    saved_data = torch.load(results_pt, weights_only=False)

    site_trans = saved_data.get('transitions', [])
    trans_vals = list(site_trans.values()) if isinstance(site_trans, dict) else list(site_trans)

    saved_actions = saved_data['actions'].to(device)
    ddg_pairs = saved_data.get('ddg_pairs', {})
    populations = saved_data.get('populations', [])
    _trans = saved_data.get('transitions', [])
    _trans_vals = list(_trans.values()) if isinstance(_trans, dict) else list(_trans)
    _total_trans = sum(_trans_vals)
    _t_base = config.get('reward', {}).get('T_baseline', 50.0)
    _rew_cfg = config.get('reward', {})
    r_pair = compute_pair_reward(
        vdata_u.edge_index, ddg_pairs, populations,
        total_transitions=_total_trans,
        t_baseline=_t_base,
        fraction_physical=saved_data.get('fraction_physical'),
        barrier_pairs=saved_data.get('barrier_pairs'),
        barrier_target=_rew_cfg.get('barrier_target', BARRIER_TARGET),
        barrier_width=_rew_cfg.get('barrier_width', BARRIER_WIDTH),
        skew_pairs=saved_data.get('skew_pairs'),
        end_pairs=saved_data.get('end_pairs'),
        skew_width=_rew_cfg.get('skew_width', SKEW_WIDTH),
        end_width=_rew_cfg.get('end_width', END_WIDTH),
    ).to(device)
    edge_weights_u = build_edge_weights(
        vdata_u.edge_index, ddg_pairs, no_transition_weight, device,
    )

    return vdata_u, saved_data, saved_actions, r_pair, edge_weights_u, trans_vals


def _sample_replay_items(
    combos: List[str],
    current_epoch: int,
    k_total: int,
    max_lookback: Optional[int],
    rng: np.random.RandomState,
) -> List[tuple]:
    """Sample up to *k_total* (combo_path, epoch_dir) pairs from combos' own
    simulation history, for replay-style AWR (see ``train_epoch``'s
    ``replay_cfg``).

    This is the "buffer" for replay-style AWR: every combo already persists
    ``run_{epoch:03d}/epoch_results.pt`` each epoch it's simulated (actions,
    reward, ddg_pairs, populations) -- the same records the existing
    ``bc_mode``/``bc_source_epoch`` path reads for one fixed epoch. This
    generalizes that to a growing, randomly-sampled set of past epochs, so
    one gradient step draws on many more (context, action, reward) triples
    than the current epoch's fresh batch alone, without any new simulations.

    Args:
        combos: Combo path strings to look for history in (typically this
            epoch's active pool -- scoring history is cheap since embeddings
            are cache-hits, but still bounded to combos already in play).
        current_epoch: Exclude this epoch's own (not-yet-simulated-when-
            sampled) directory.
        k_total: Cap on how many replay items to return.
        max_lookback: If given, only consider epochs within this many epochs
            of ``current_epoch`` (bounds cost as history grows very long;
            None = unbounded).
        rng: Seeded RNG for reproducible sampling.
    """
    candidates = []
    for combo_str in combos:
        combo_path = Path(combo_str)
        if not combo_path.is_dir():
            continue
        for run_dir in combo_path.glob('run_*'):
            m = re.match(r'run_(\d+)$', run_dir.name)
            if not m:
                continue
            run_epoch = int(m.group(1))
            if run_epoch >= current_epoch:
                continue
            if max_lookback is not None and run_epoch < current_epoch - max_lookback:
                continue
            if (run_dir / 'epoch_results.pt').exists():
                candidates.append((combo_path, run_dir))
    if not candidates or k_total <= 0:
        return []
    if len(candidates) <= k_total:
        return candidates
    idx = rng.choice(len(candidates), size=k_total, replace=False)
    return [candidates[i] for i in idx]


def _compute_replay_awr_loss(
    policy: UnimolPolicy,
    combo_path: Path,
    epoch_dir: Path,
    combo_system_map: Dict[str, Dict],
    config: Dict,
    device: str,
    no_transition_weight: float,
    awr_temperature: float,
) -> Optional[torch.Tensor]:
    """AWR-style loss for one replayed (combo, past-epoch) observation.

    Generalizes the confidence-weighted Gaussian-NLL loss the existing
    ``bc_mode`` path already computes toward one fixed ``bc_source_epoch``,
    by additionally reward-weighting it with the same ``exp(reward /
    temperature)``, clamp<=20 convention ``pretrain_policy.py``'s AWR
    pretraining and the NeuralLinear+TS posterior update both use -- so a
    replayed observation with a good reward pulls the policy toward its
    saved action much harder than a mediocre one, instead of every replayed
    action getting equal weight regardless of how good it actually was.

    Returns None if this replay item couldn't be loaded/scored (the caller
    just skips it -- a missing/corrupt old run directory shouldn't abort the
    epoch).
    """
    try:
        vdata, saved_data, saved_actions, r_pair, edge_weights, _ = (
            _prepare_success_combo_reward(
                combo_path, epoch_dir, combo_system_map, config, device,
                no_transition_weight,
            )
        )
        mean, log_std = policy._forward_edges(vdata.x, vdata.edge_index, vdata.edge_type)
        std = torch.exp(log_std)
        logp = torch.distributions.Normal(mean, std).log_prob(saved_actions)  # [E, D]
        active_mask = saved_actions.abs() > 1e-8

        # Confidence weight from population balance / DDG reliability -- same
        # mechanism as pretrain_policy.py's AWR pretraining and bc_mode above.
        ddg_pairs = saved_data.get('ddg_pairs', {})
        populations = saved_data.get('populations', [])
        pairwise_weight = None
        if ddg_pairs and populations:
            try:
                nsubs_per_site = []
                unique_sites = vdata.site_index.unique() if hasattr(vdata, 'site_index') else []
                for site_id in sorted(unique_sites.tolist()):
                    nsubs_per_site.append(int((vdata.site_index == site_id).sum().item()))
                pairs = build_directed_pairs(nsubs_per_site)
                _w = compute_pairwise_confidence_weights(
                    {'ddg_pairs': ddg_pairs, 'populations': populations},
                    nsubs_per_site=nsubs_per_site, pairs=pairs,
                )
                pairwise_weight = torch.tensor(_w, dtype=torch.float32, device=device).unsqueeze(-1)  # [E,1]
            except Exception:
                pairwise_weight = None
        if pairwise_weight is None:
            # Fall back to the cheaper DDG-transition weight already computed
            # by _prepare_success_combo_reward.
            pairwise_weight = edge_weights.unsqueeze(-1)

        awr_w = torch.exp(r_pair / awr_temperature).clamp(max=20.0)  # [E, D]
        combined_weight = pairwise_weight * awr_w  # [E, D]

        weighted_logp = logp[active_mask] * combined_weight[active_mask]
        if weighted_logp.numel() == 0:
            return None
        return -weighted_logp.mean()
    except Exception:
        return None


def append_summary(summary_path: Optional[Path], text: str) -> None:
    """Append a line (or block) of text to the lightweight epoch summary log.

    The main training_status_*.out log captures every print statement
    (including per-combo Uni-Mol computation chatter) and can grow to
    hundreds of MB over a long run. This writes just the high-signal
    per-epoch lines (batch reward stats, failure gradient, epoch summary)
    to a small separate file that's easy to skim or diff across runs.
    Silently no-ops if summary_path is None or the write fails.
    """
    if summary_path is None:
        return
    try:
        with open(summary_path, 'a') as f:
            f.write(text + '\n')
    except Exception:
        pass


def format_pair_success(ps: Dict[str, int], label: str) -> Optional[str]:
    """Format a compute_pair_success_stats() dict as one summary line, or
    None if there are no pairs to report (e.g. run_sims=False)."""
    total = ps.get('total', 0)
    if total == 0:
        return None
    pct = lambda n: 100.0 * n / total
    return (
        f"  Pair success ({label}): crossed {ps['crossed']}/{total} ({pct(ps['crossed']):.1f}%), "
        f"balanced {ps['balanced']}/{total} ({pct(ps['balanced']):.1f}%), "
        f"both {ps['both']}/{total} ({pct(ps['both']):.1f}%)"
    )


def train_epoch(
    policy: UnimolPolicy,
    rl_optimizer: torch.optim.Optimizer,
    combos: List[str],
    combo_system_map: Dict[str, Dict],  # combo_path -> system_config
    epoch: int,
    config: Dict,
    device: str = 'cpu',
    update_policy: bool = True,
    bc_mode: bool = False,
    bc_source_epoch: Optional[int] = None,
    replay_from_epoch: Optional[int] = None,
    deterministic_actions: bool = False,
    bc_auxiliary_weight: float = 0.0,
    ema_baseline: Optional[dict] = None,
    reserve_pool: Optional[List[str]] = None,
    verbose: bool = False,
    summary_log_path: Optional[Path] = None,
    bandit_state: Optional[dict] = None,
    baseline_net: Optional['ContextBaseline'] = None,
    baseline_optimizer: Optional[torch.optim.Optimizer] = None,
    system_to_idx: Optional[Dict[str, int]] = None,
    history_replay_cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    """Run one training epoch across all combos from all systems.

    Parameters match run_workflow_sitepool.py train_epoch exactly, with the
    addition of combo_system_map which maps each combo path string to its
    per-system config dict (needed for solvent_state and ensure_combo_dir_exists).

    Bandit algorithm: controlled by config['bandit']['algorithm']
    ('reinforce', the default, or 'neurallinear_ts'). In 'neurallinear_ts'
    mode, action sampling uses policy.get_actions_thompson instead of
    policy.get_actions, and the epoch-batch update is a closed-form
    Bayesian posterior update (policy.update_bayesian_posteriors) instead of
    a REINFORCE backward()/optimizer.step() — see the branch on `algorithm`
    below. bandit_state (mutated in place, same pattern as ema_baseline) is
    the per-combo infra-failure/defer tracker used by uncertainty-driven
    combo selection; harmless to omit in 'reinforce' mode.

    REINFORCE-only additions (both opt-in, default off, no effect on
    'neurallinear_ts' mode):

    baseline_net / baseline_optimizer / system_to_idx: when baseline_net is
    given, REINFORCE's advantage uses ``ContextBaseline`` (features +
    system identity) in place of the pooled scalar-per-bias-type EMA
    baseline. baseline_optimizer trains it (a separate step from
    rl_optimizer, so its gradient never touches the policy). system_to_idx
    maps a combo's system name (via combo_system_map) to the embedding index
    ContextBaseline expects; required together with baseline_net.

    history_replay_cfg: when given (``{'k_per_epoch': int, 'max_lookback':
    Optional[int], 'awr_temperature': float}``), each epoch additionally
    samples up to k_per_epoch past (combo, epoch) observations from combos'
    own simulation history (see ``_sample_replay_items``) and folds an
    AWR-weighted loss for them (see ``_compute_replay_awr_loss``) into the
    same gradient step as the fresh on-policy batch -- more (context,
    action, reward) signal per expensive simulation, no new simulations.
    """
    policy.train()
    # Uni-Mol representation is loaded on-demand (no explicit encoder model)
    # Runtime Thompson-sampling exploration dial (see _policy_sample_actions);
    # 1.0 reproduces the posterior's own calibrated/narrowed spread exactly.
    _exploration_scale = config.get('bandit', {}).get('exploration_scale', 1.0)
    # NeuralLinear+TS update target: 'awr' regresses toward the sampled action
    # weighted by exp(r/beta); 'hindsight' regresses the linear and quadratic
    # heads toward the biases each run measured it should have used (see
    # mllf.cb.hindsight), and by default keeps per-pair offsets for what the
    # shared model can't fit. Offsets need bandit_state, so validation passes
    # (bandit_state=None) always see the shared model alone.
    _bandit_cfg = config.get('bandit', {})
    _use_hindsight = _bandit_cfg.get('update_target', 'awr') == 'hindsight'
    _use_offsets = (_use_hindsight and bandit_state is not None
                    and bool(_bandit_cfg.get('pair_offsets', True)))

    def _with_pair_offsets(actions, data, combo_path, sys_cfg):
        """Shift a Thompson-sampled proposal by the pair's offsets, if any."""
        if (not _use_offsets or data.edge_type is not None
                or not getattr(policy, 'use_bayesian_heads', False)):
            return actions
        with torch.no_grad(), _policy_inference_lock:
            mean, _, _ = policy.get_actions_thompson(
                data.x, data.edge_index, data.edge_type, deterministic=True)
        shifted, _n = apply_pair_offsets(
            actions, mean.detach(), bandit_state, sys_cfg.get('name', 'unknown'),
            combo_node_labels(combo_path), data.edge_index,
            prior_sd=_bandit_cfg.get('offset_prior_sd'))
        return shifted
    # Tags every epoch_results.pt written this call with which training run
    # (checkpoint dir) produced it -- out_dir/combo trees are shared across
    # runs reusing the same systems config, so this lets analysis tooling
    # (training/analysis/) tell them apart instead of silently reading
    # whichever run's data last overwrote a given (combo, epoch).
    run_id = str(config.get('output', {}).get('base_dir', 'unknown'))

    epoch_loss = 0.0
    epoch_rewards = []
    deferred_updates = []
    # Diagnostic-only per-pair success breakdown (see compute_pair_success_stats);
    # accumulated across every crash-free combo regardless of algorithm/floor gate.
    pair_stats_total = {'total': 0, 'crossed': 0, 'balanced': 0, 'both': 0}
    # Lines destined for epoch_summary.log, buffered rather than appended
    # immediately -- main() writes the "Epoch N Summary" header first, then
    # this whole batch of detail lines right after it, so the file reads
    # top-down per epoch instead of details-then-summary. Console/verbose-log
    # prints are untouched and stay in their original chronological order.
    _detail_lines: List[str] = []

    target_norm: Optional[torch.Tensor] = None
    if (bc_mode or bc_auxiliary_weight > 0.0) and bc_source_epoch is not None:
        _norm_list = []
        for _cd in combos:
            _cp = Path(_cd)
            _rpt = _cp / f"run_{bc_source_epoch:03d}" / 'epoch_results.pt'
            if _rpt.exists():
                try:
                    _d = torch.load(_rpt, weights_only=False)
                    _norm_list.append(_d['actions'].float())
                except Exception:
                    pass
        if _norm_list:
            _all_targets = torch.cat(_norm_list, dim=0)
            target_norm = _all_targets.std(dim=0).clamp(min=1.0).to(device)
        else:
            target_norm = policy.scale_factors.to(device)
        print(f"  BC target_norm (per-dim std, min=1): "
              f"{[f'{v:.3f}' for v in target_norm.tolist()]}")
    if target_norm is None:
        target_norm = policy.scale_factors.to(device)

    # ── Behavioral Cloning mode ──
    if bc_mode and update_policy:
        rl_optimizer.zero_grad()
        n_bc = 0
        for combo_dir in combos:
            combo_path = Path(combo_dir)
            sys_cfg = combo_system_map.get(str(combo_path), {})
            combo_path = ensure_combo_dir_exists(combo_path, sys_cfg, config, verbose=verbose)
            if bc_source_epoch is not None:
                target_run = combo_path / f"run_{bc_source_epoch:03d}"
                candidate = target_run / 'epoch_results.pt'
                results_pt = candidate if candidate.exists() else None
            else:
                run_dirs = sorted(
                    combo_path.glob('run_*'),
                    key=lambda p: int(p.name.split('_')[1]) if p.name.split('_')[1].isdigit() else -1
                )
                results_pt = None
                for rd in reversed(run_dirs):
                    ept = rd / 'epoch_results.pt'
                    if ept.exists():
                        results_pt = ept
                        break
            if results_pt is None:
                continue
            try:
                saved_data = torch.load(results_pt, weights_only=False)
                saved_actions = saved_data['actions'].to(device)
                reward_bc = saved_data['reward']
                epoch_rewards.append(reward_bc)

                vdata_bc, _ = build_graph_and_data(
                    str(combo_path), sys_cfg, config, device
                )
                mean_bc, log_std_bc = policy._forward_edges(
                    vdata_bc.x, vdata_bc.edge_index, vdata_bc.edge_type
                )
                if saved_actions.dim() == 1:
                    saved_actions = saved_actions.unsqueeze(-1).expand_as(mean_bc)
                _et_bc = vdata_bc.edge_type // 2
                _bc_mask = torch.zeros(mean_bc.shape, dtype=torch.bool, device=device)
                for _d in range(policy.mlp_out_dim):
                    _bc_mask[_et_bc == _d, _d] = True
                
                # Phase 1.2: Compute pairwise confidence weights from simulation results
                ddg_pairs_bc = saved_data.get('ddg_pairs', {})
                populations_bc = saved_data.get('populations', [])
                pairwise_weights_bc = None
                
                if ddg_pairs_bc and populations_bc:
                    try:
                        # Build pairs for this combo's graph
                        nsubs_per_site = []
                        n_nodes = vdata_bc.x.size(0)
                        unique_sites = vdata_bc.site_index.unique() if hasattr(vdata_bc, 'site_index') else []
                        if len(unique_sites) == 0:
                            # Fallback: infer from edge_type distribution
                            for d in range(policy.mlp_out_dim):
                                count = (vdata_bc.edge_type // 2 == d).sum().item()
                                if count > 0:
                                    nsubs_per_site.append(int((1 + (1 + 8*count)**0.5) / 2))
                        else:
                            for site_id in sorted(unique_sites.tolist()):
                                nsubs_per_site.append((vdata_bc.site_index == site_id).sum().item())
                        
                        pairs = build_directed_pairs(nsubs_per_site)
                        pairwise_weights_bc = compute_pairwise_confidence_weights(
                            {'ddg_pairs': ddg_pairs_bc, 'populations': populations_bc},
                            nsubs_per_site=nsubs_per_site,
                            pairs=pairs,
                        )
                    except Exception as e:
                        if verbose:
                            print(f"    Warning: Could not compute pairwise weights: {e}")
                
                # Phase 1.1: Use log-likelihood loss instead of MSE (trains both mean and log_std)
                std_bc = torch.exp(log_std_bc)
                logp_bc = torch.distributions.Normal(mean_bc, std_bc).log_prob(saved_actions)
                
                # Phase 1.4: Apply active mask (only train on non-zero target edges)
                active_mask_bc = saved_actions.abs() > 1e-8
                combined_mask = _bc_mask & active_mask_bc
                
                # Phase 1.2: Apply pairwise weighting if available
                if pairwise_weights_bc is not None:
                    try:
                        weights_for_loss = torch.tensor(
                            pairwise_weights_bc, dtype=torch.float32, device=device
                        )
                        weights_for_loss = weights_for_loss.unsqueeze(-1)  # [E, 1]
                        weighted_logp = logp_bc[combined_mask] * weights_for_loss[combined_mask]
                        if weighted_logp.numel() > 0:
                            bc_loss = -weighted_logp.mean()
                        else:
                            bc_loss = -logp_bc[combined_mask].mean() if combined_mask.sum() > 0 else torch.tensor(0.0, device=device)
                    except Exception as e:
                        if verbose:
                            print(f"    Warning: Could not apply pairwise weights: {e}")
                        bc_loss = -logp_bc[combined_mask].mean() if combined_mask.sum() > 0 else torch.tensor(0.0, device=device)
                else:
                    # Fallback: use log-likelihood without pairwise weighting
                    bc_loss = -logp_bc[combined_mask].mean() if combined_mask.sum() > 0 else torch.tensor(0.0, device=device)
                
                if not (torch.isnan(bc_loss) or torch.isinf(bc_loss)):
                    bc_loss.backward()
                    epoch_loss += bc_loss.item()
                    n_bc += 1
                    print(f"  BC update {combo_path.name}: loss={bc_loss.item():.4f}  "
                          f"reward={reward_bc:.2f}")
            except Exception as e:
                print(f"  Warning: BC update failed for {combo_path.name}: {e}")
        if n_bc > 0:
            # Phase 1.3: Expand gradient clipping to all parameters (not just edge_mlp)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            rl_optimizer.step()
            epoch_loss /= n_bc
        print(f"  BC: {n_bc}/{len(combos)} combos updated")
        avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        reward_std = float(np.std(epoch_rewards)) if epoch_rewards else 0.0
        return {
            'loss': epoch_loss,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'num_combos': len(combos),
            'update_policy': True,
            'bc_mode': True,
            'n_success': 0,
        }

    job_queue = []
    max_concurrent = config.get('max_concurrent_jobs', 10)
    max_retries = config.get('max_retries', 2)
    _reserve = list(reserve_pool) if reserve_pool else []
    _permanently_failed: List[str] = []
    _max_failed = config.get('max_failed_combos_per_epoch', 50)

    def _abort_if_too_many_failures() -> None:
        if len(_permanently_failed) >= _max_failed:
            print(f"\nFATAL: {len(_permanently_failed)} combos permanently failed "
                  f"(>= max_failed_combos_per_epoch={_max_failed}). Aborting.")
            print("  Failed combos:")
            for _fc in _permanently_failed:
                print(f"    {_fc}")
            sys.exit(1)

    def _process_completed_job(idx: int) -> None:
        """Process a completed SLURM job at ``job_queue[idx]``.

        Pops the entry, checks simulation success, handles retries,
        permanent failures with reserve-pool replacement, and on success
        computes the reward and records it in *epoch_rewards* /
        *deferred_updates*.
        """
        jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, retry_count = job_queue.pop(idx)

        output_file = jepoch_dir / 'output.out'
        simulation_success = check_simulation_success(output_file)

        if not simulation_success and retry_count < max_retries:
            print(f"  Simulation failed for {jcombo_path.name}, retrying "
                  f"(attempt {retry_count + 2}/{max_retries + 1})...")
            epoch_run_script = jepoch_dir / 'run_epoch.sh'
            try:
                result = subprocess.run(
                    ['sbatch', str(epoch_run_script)],
                    cwd=str(jepoch_dir),
                    capture_output=True, text=True, check=True
                )
                new_job_id = result.stdout.strip().split()[-1]
                job_queue.append((jcombo_path, jepoch_dir, new_job_id,
                                  jactions, jlogp, retry_count + 1))
            except Exception as e:
                print(f"  Error resubmitting: {e} — skipping {jcombo_path.name}")
            return

        if not simulation_success:
            _failed_sys = combo_system_map.get(str(jcombo_path), {}).get('name', 'unknown')
            print(f"  SKIP: {_failed_sys}/{jcombo_path.name} permanently failed "
                  f"after {max_retries + 1} attempt(s) — continuing without it")
            _permanently_failed.append(f"{_failed_sys}/{jcombo_path.name}")
            if bandit_state is not None:
                # Infrastructure/software failure, not a poor-sampling result —
                # record it so uncertainty-driven selection (NeuralLinear+TS)
                # stops repeatedly re-selecting a combo that cannot be
                # simulated, without discarding it as "bad" (see
                # record_combo_failure).
                _cap = config.get('bandit', {}).get('failure_retry_cap', 3)
                record_combo_failure(bandit_state, str(jcombo_path), cap=_cap)
            _abort_if_too_many_failures()
            # Try to submit a replacement from the reserve pool.
            while _reserve:
                _rep_str = _reserve.pop(0)
                _rep_path = Path(_rep_str)
                _rep_sys = combo_system_map.get(_rep_str, {})
                try:
                    _rep_path = ensure_combo_dir_exists(
                        _rep_path, _rep_sys, config, verbose=verbose,
                        _lookup=_combo_dir_lookup)
                    fix_msld_flat_for_single_site(_rep_path, SITE_ATOMS_MAPPING)
                    _rep_epoch_dir = _rep_path / f"run_{epoch:03d}"
                    _rep_epoch_dir.mkdir(exist_ok=True, parents=True)
                    if (_rep_epoch_dir / 'epoch_results.pt').exists():
                        break  # already cached — counts as done
                    _rep_data, _rep_extras = build_graph_and_data(
                        str(_rep_path), _rep_sys, config, device)
                    _save_graph_data_cache(_rep_epoch_dir, _rep_data)
                    with torch.no_grad():
                        _rep_act, _rep_lp = _policy_sample_actions(
                            policy, _rep_data.x, _rep_data.edge_index, _rep_data.edge_type,
                            exploration_scale=_exploration_scale)
                    _rep_act = _with_pair_offsets(_rep_act.detach(), _rep_data, _rep_path, _rep_sys)
                    _rep_lp = _detach_cpu_or_none(_rep_lp)
                    write_variables_from_actions(
                        str(_rep_epoch_dir), _rep_data, _rep_extras, _rep_act,
                        bias_clip=config.get('training', {}).get('bias_clip', 1000.0))
                    _rep_job_id = submit_epoch_simulation(
                        str(_rep_path), str(_rep_epoch_dir), config=config,
                        solvent_state=_rep_sys.get('solvent_state', 'solv'))
                    _rep_sys_name = _rep_sys.get('name', 'unknown')
                    job_queue.append((_rep_path, _rep_epoch_dir, _rep_job_id,
                                      _rep_act, _rep_lp, 0))
                    print(f"  Replacement: {_rep_sys_name}/{_rep_path.name}"
                          f"  →  job {_rep_job_id}")
                    break
                except Exception as _rep_e:
                    print(f"  Replacement {_rep_path.name} failed: {_rep_e} — trying next")
            return

        reward_config = config.get('reward', {})
        _jsys = combo_system_map.get(str(jcombo_path), {}).get('name', 'unknown')
        print(f"  {_jsys}/{jcombo_path.name}:")
        try:
            reward = compute_msld_reward_improved(
                str(jepoch_dir),
                w_P=reward_config.get('w_P', 0.5),
                w_T=reward_config.get('w_T', 0.75),
                w_U=reward_config.get('w_U', 0.3),
                gamma=reward_config.get('gamma', 4.0),
                P_baseline=reward_config.get('P_baseline', 500.0),
                T_baseline=reward_config.get('T_baseline', 50.0),
                min_transitions_per_site=reward_config.get('min_transitions_per_site', 10),
                min_coverage_ratio=reward_config.get('min_coverage_ratio', 0.5),
                entropy_bonus=reward_config.get('entropy_bonus', 8.0),
                concentration_penalty_threshold=reward_config.get('concentration_penalty_threshold', 0.8),
                suppress_warnings=not verbose,
            )
        except Exception as e:
            warnings.warn(f"Failed to compute reward for {jcombo_path.name}: {e}")
            reward = -100.0

        epoch_rewards.append(reward)
        raw_metrics = parse_simulation_metrics(output_file)
        torch.save({
            'reward': reward,
            'actions': jactions.cpu(),
            'logp': _detach_cpu_or_none(jlogp),
            'reward_config': reward_config,
            'populations': raw_metrics.get('populations', {}),
            'transitions': raw_metrics.get('transitions', {}),
            'ddg_pairs': raw_metrics.get('ddg_pairs', {}),
            'fraction_physical': raw_metrics.get('fraction_physical'),
            'barrier_pairs': raw_metrics.get('barrier_pairs', {}),
            'skew_pairs': raw_metrics.get('skew_pairs', {}),
            'end_pairs': raw_metrics.get('end_pairs', {}),
            'transition_pairs': raw_metrics.get('transition_pairs', {}),
            'endpoint_pairs': raw_metrics.get('endpoint_pairs', {}),
            'run_id': run_id,
        }, jepoch_dir / 'epoch_results.pt')

        deferred_updates.append((jcombo_path, jepoch_dir, reward))

    # Phase 1: Prepare ALL combos in parallel.
    # Each worker creates the combo dir (if needed), checks the cache,
    # builds the PyG graph, samples actions, and writes variables.py.
    # Results are returned in a list and processed in original combo order
    # after the pool finishes so that epoch_rewards / deferred_updates /
    # pending_submissions are filled deterministically.
    _prep_map_lock = threading.Lock()
    n_prep_workers = config.get('phase1_workers', 2)
    run_config = config.get('run_sims', True)
    bias_clip = config.get('training', {}).get('bias_clip', 1000.0)
    poll_interval = config.get('poll_interval', 10)

    # Pre-scan combo_metadata.json for combos whose dirs don't exist yet.
    # Loads each system's file once, keeps only the needed entries, then
    # explicitly frees the full list.  Memory cost is O(missing_combos),
    # not O(total_metadata) — critical for systems like 12fuzed (4.18 M rows).
    _combo_dir_lookup = build_combo_dir_lookup(combos, combo_system_map)

    def _prepare_combo_worker(args):
        """Prepare one combo and return a result tuple.

        Returns one of:
          ('cached',  combo_idx, reward, combo_path, epoch_dir)
          ('pending', combo_idx, combo_path, epoch_dir, actions, logp, sys_cfg)
          ('skip',    combo_idx)   # run_sims=False
          ('error',   combo_idx)   # graph build or dir-creation failure
        """
        combo_idx, combo_dir = args
        combo_path = Path(combo_dir)
        sys_cfg = combo_system_map.get(str(combo_path), {})
        _original_combo_str = str(combo_path)

        try:
            combo_path = ensure_combo_dir_exists(
                combo_path, sys_cfg, config, verbose=verbose,
                _lookup=_combo_dir_lookup,
            )
        except Exception as e:
            print(f"  ERROR: Could not create combo dir {combo_path.name}: {e}")
            return ('error', combo_idx)

        if str(combo_path) != _original_combo_str and sys_cfg:
            with _prep_map_lock:
                combo_system_map[str(combo_path)] = sys_cfg

        fix_msld_flat_for_single_site(combo_path, SITE_ATOMS_MAPPING)

        epoch_dir = combo_path / f"run_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True, parents=True)

        if replay_from_epoch is not None:
            source_pt = combo_path / f"run_{replay_from_epoch:03d}" / 'epoch_results.pt'
            dest_pt = epoch_dir / 'epoch_results.pt'
            if not dest_pt.exists() and source_pt.exists():
                shutil.copy2(str(source_pt), str(dest_pt))

        epoch_results_file = epoch_dir / 'epoch_results.pt'
        _prev_output = epoch_dir / 'output.out'

        # Recover from a previous run whose simulation completed but whose
        # epoch_results.pt was never saved.
        if not epoch_results_file.exists() and _prev_output.exists() and check_simulation_success(_prev_output):
            try:
                _rc = config.get('reward', {})
                _rec_reward = compute_msld_reward_improved(
                    str(epoch_dir),
                    w_P=_rc.get('w_P', 0.5), w_T=_rc.get('w_T', 0.75),
                    w_U=_rc.get('w_U', 0.3), gamma=_rc.get('gamma', 4.0),
                    P_baseline=_rc.get('P_baseline', 500.0),
                    T_baseline=_rc.get('T_baseline', 50.0),
                    min_transitions_per_site=_rc.get('min_transitions_per_site', 10),
                    min_coverage_ratio=_rc.get('min_coverage_ratio', 0.5),
                    entropy_bonus=_rc.get('entropy_bonus', 8.0),
                    concentration_penalty_threshold=_rc.get('concentration_penalty_threshold', 0.8),
                    suppress_warnings=not verbose,
                )
                _rec_data, _rec_extras = build_graph_and_data(
                    str(combo_path), sys_cfg, config, device)
                _save_graph_data_cache(epoch_dir, _rec_data)
                with torch.no_grad():
                    # Thread-safety lock: policy forward passes are not thread-safe
                    with _policy_inference_lock:
                        _rec_act, _rec_lp = _policy_sample_actions(
                            policy, _rec_data.x, _rec_data.edge_index, _rec_data.edge_type,
                            exploration_scale=_exploration_scale)
                _rec_act = _rec_act.detach()
                _rec_lp = _detach_cpu_or_none(_rec_lp)
                _rec_metrics = parse_simulation_metrics(_prev_output)
                torch.save({
                    'reward': _rec_reward, 'actions': _rec_act.cpu(), 'logp': _rec_lp,
                    'reward_config': _rc,
                    'populations': _rec_metrics.get('populations', {}),
                    'transitions': _rec_metrics.get('transitions', {}),
                    'ddg_pairs': _rec_metrics.get('ddg_pairs', {}),
                    'fraction_physical': _rec_metrics.get('fraction_physical'),
                    'barrier_pairs': _rec_metrics.get('barrier_pairs', {}),
                    'skew_pairs': _rec_metrics.get('skew_pairs', {}),
                    'end_pairs': _rec_metrics.get('end_pairs', {}),
                    'transition_pairs': _rec_metrics.get('transition_pairs', {}),
                    'endpoint_pairs': _rec_metrics.get('endpoint_pairs', {}),
                    'run_id': run_id,
                }, epoch_results_file)
                if verbose:
                    print(f"  Recovered from existing output.out: reward={_rec_reward:.4f}")
                return ('cached', combo_idx, _rec_reward, combo_path, epoch_dir)
            except Exception as _rec_e:
                warnings.warn(f"Recovery from output.out failed for {combo_path.name}: {_rec_e}")

        # Check cache: only this run's own results (resuming a crashed run);
        # another run's results for the same (combo, epoch) are re-simulated.
        if epoch_results_file.exists():
            try:
                cached = torch.load(epoch_results_file, weights_only=False)
                if cached.get('run_id') != run_id:
                    raise LookupError(f"results belong to run {cached.get('run_id')!r}")
                reward_config = config.get('reward', {})
                cached_reward_config = cached.get('reward_config', {})
                reward_changed = any(
                    cached_reward_config.get(k) != reward_config.get(k)
                    for k in ('w_P', 'w_T', 'w_U', 'gamma', 'P_baseline', 'T_baseline',
                              'min_transitions_per_site', 'min_coverage_ratio',
                              'entropy_bonus', 'concentration_penalty_threshold')
                )
                if reward_changed:
                    if verbose:
                        print(f"  Reward config changed - recomputing reward")
                    try:
                        reward = compute_msld_reward_improved(
                            str(epoch_dir),
                            w_P=reward_config.get('w_P', 0.5),
                            w_T=reward_config.get('w_T', 0.75),
                            w_U=reward_config.get('w_U', 0.3),
                            gamma=reward_config.get('gamma', 4.0),
                            P_baseline=reward_config.get('P_baseline', 500.0),
                            T_baseline=reward_config.get('T_baseline', 50.0),
                            min_transitions_per_site=reward_config.get('min_transitions_per_site', 10),
                            min_coverage_ratio=reward_config.get('min_coverage_ratio', 0.5),
                            entropy_bonus=reward_config.get('entropy_bonus', 8.0),
                            concentration_penalty_threshold=reward_config.get('concentration_penalty_threshold', 0.8),
                            suppress_warnings=not verbose,
                        )
                        if verbose:
                            print(f"  Old reward: {cached.get('reward', 'N/A')} -> New reward: {reward:.4f}")
                    except Exception as _e:
                        warnings.warn(f"Could not recompute reward for {combo_path.name}: {_e}")
                        reward = cached['reward']
                else:
                    reward = cached['reward']
                if verbose:
                    print(f"  Using cached reward: {reward:.4f}")
                return ('cached', combo_idx, reward, combo_path, epoch_dir)
            except LookupError as e:
                if verbose:
                    print(f"  Not reusing {epoch_results_file}: {e}")
            except Exception as e:
                warnings.warn(f"Failed to load cached results for {combo_path.name}: {e}")

        # Build graph and sample actions
        try:
            data, extras = build_graph_and_data(
                str(combo_path), sys_cfg, config, device
            )
        except Exception as e:
            print(f"  ERROR: Failed to build graph for {combo_path.name}: {e}")
            return ('error', combo_idx)
        _save_graph_data_cache(epoch_dir, data)

        with torch.no_grad():
            if getattr(policy, 'use_bayesian_heads', False):
                # NeuralLinear + Thompson Sampling: sample (or, for eval combos,
                # use the posterior mean) directly from each bias head's
                # posterior — no REINFORCE (mean, log_std) distribution involved.
                with _policy_inference_lock:
                    actions, logp = _policy_sample_actions(
                        policy, data.x, data.edge_index, data.edge_type,
                        deterministic=deterministic_actions, exploration_scale=_exploration_scale)
                actions = _with_pair_offsets(actions.detach(), data, combo_path, sys_cfg)
            elif deterministic_actions:
                # Thread-safety lock: policy forward passes are not thread-safe
                with _policy_inference_lock:
                    mean, _ = policy._forward_edges(
                        data.x, data.edge_index, data.edge_type
                    )
                actions = mean.detach()
                # Thread-safety lock: policy forward passes are not thread-safe
                with _policy_inference_lock:
                    logp = policy.evaluate_logp(
                        data.x, data.edge_index, data.edge_type, actions
                    )[0].detach()
            else:
                # Thread-safety lock: policy forward passes are not thread-safe
                with _policy_inference_lock:
                    actions, logp, _, _ = policy.get_actions(
                        data.x, data.edge_index, data.edge_type
                    )
                actions = actions.detach()
                logp = logp.detach()

        write_variables_from_actions(str(epoch_dir), data, extras, actions, bias_clip=bias_clip)

        if not run_config:
            if verbose:
                print(f"  Skipping simulation (run_sims=False)")
            return ('skip', combo_idx)

        _sys_display = sys_cfg.get('name', 'unknown')
        if verbose:
            print(f"  Prepared {combo_idx+1}/{len(combos)}: {_sys_display}/{combo_path.name}")
        return ('pending', combo_idx, combo_path, epoch_dir, actions, logp, sys_cfg)

    # ── Phase 1: Prepare all combos with a small thread pool ───────────────
    print(f"\n=== Phase 1: preparing {len(combos)} combos "
          f"({n_prep_workers} workers) ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_prep_workers) as _prep_exec:
        _prep_results = list(_prep_exec.map(_prepare_combo_worker, enumerate(combos)))

    _pending_to_submit = []
    _n_cached = _n_skipped = _n_errors = 0
    _n_prep_pending = 0
    for _result in _prep_results:
        _kind = _result[0]
        if _kind == 'cached':
            _, _ci, _reward, _cp, _ed = _result
            epoch_rewards.append(_reward)
            deferred_updates.append((_cp, _ed, _reward))
            _n_cached += 1
        elif _kind == 'pending':
            _, _ci, _cp, _ed, _act, _lp, _sc = _result
            _pending_to_submit.append((_cp, _ed, _act, _lp, _sc))
            _n_prep_pending += 1
        elif _kind == 'skip':
            _n_skipped += 1
        else:
            _n_errors += 1

    print(f"  Prep done: {_n_cached} cached, {_n_prep_pending} to submit, "
          f"{_n_skipped} skipped, {_n_errors} errors")

    # ── Phase 2: Submit to SLURM and monitor until all jobs complete ────────
    _n_submitted = 0
    if _pending_to_submit:
        print(f"\n=== Phase 2: submitting {_n_prep_pending} simulations "
              f"(max {max_concurrent} concurrent SLURM jobs) ===")

    while job_queue or _pending_to_submit:
        # ── Poll SLURM ────────────────────────────────────────────────────
        _sq = subprocess.run(
            ['squeue', '--me', '--noheader', '--format=%i'],
            capture_output=True, text=True,
        )
        _active_ids = {ln.strip() for ln in _sq.stdout.splitlines() if ln.strip()}

        _completed_indices = [
            i for i, _j in enumerate(job_queue) if str(_j[2]) not in _active_ids
        ]
        for i in reversed(_completed_indices):
            _process_completed_job(i)

        # ── Fill available SLURM slots ────────────────────────────────────
        slots = max_concurrent - len(job_queue)
        _submitted_this_round = 0
        while slots > 0 and _pending_to_submit:
            _p_combo, _p_edir, _p_act, _p_lp, _p_sys = _pending_to_submit.pop(0)
            _p_sys_name = _p_sys.get('name', 'unknown')
            try:
                job_id = submit_epoch_simulation(
                    str(_p_combo), str(_p_edir), config=config,
                    solvent_state=_p_sys.get('solvent_state', 'solv'),
                )
                job_queue.append((_p_combo, _p_edir, job_id, _p_act, _p_lp, 0))
                _n_submitted += 1
                print(f"  Submitted {_n_submitted}/{_n_prep_pending}: "
                      f"{_p_sys_name}/{_p_combo.name}  \u2192  job {job_id}")
                slots -= 1
                _submitted_this_round += 1
            except Exception as e:
                # Usually a cluster/config problem (e.g. missing slurm:
                # partition/gres), not the combo, so no record_combo_failure.
                # Counted toward max_failed_combos_per_epoch so a broken
                # setup aborts instead of "training" on no data.
                _stderr = getattr(e, 'stderr', None)
                print(f"  ERROR: Failed to submit {_p_combo.name}: {e}"
                      + (f"\n    sbatch: {_stderr.strip()}" if _stderr else ""))
                _permanently_failed.append(f"{_p_sys_name}/{_p_combo.name} (submission)")
                _abort_if_too_many_failures()

        # ── Status line ───────────────────────────────────────────────────
        if job_queue or _pending_to_submit:
            n_active = len([j for j in job_queue if str(j[2]) in _active_ids])
            print(f"  [running={n_active} | finishing={len(job_queue) - n_active} | "
                  f"queued={len(_pending_to_submit)}]",
                  end='\r', flush=True)
            time.sleep(poll_interval)

    print()  # newline after the \r status line
    print(f"  Complete: {_n_submitted} submitted, {_n_cached} cached, "
          f"{_n_skipped} skipped, {_n_errors} errors")

    if not deferred_updates:
        # Nothing simulated successfully this epoch. For a training epoch,
        # updating the policy, bandit state or checkpoints would record an
        # epoch with no data, so abort; a (small) validation set only warns.
        _counts = (f"{_n_submitted} submitted, {_n_cached} cached, "
                   f"{len(_permanently_failed)} failed, {_n_errors} prep errors")
        if update_policy:
            print(f"\nFATAL: epoch {epoch} produced no successful simulations "
                  f"({_counts}). Aborting.")
            for _fc in _permanently_failed:
                print(f"    {_fc}")
            sys.exit(1)
        print(f"  WARNING: no successful simulations this pass ({_counts})")

    bandit_cfg = config.get('bandit', {})
    algorithm = bandit_cfg.get('algorithm', 'reinforce')

    if algorithm == 'neurallinear_ts':
        # === Epoch-batch NeuralLinear + TS closed-form posterior update ===
        # Replaces the REINFORCE backward()/optimizer.step() below with a
        # closed-form Bayesian posterior update per bias head — no gradient
        # involved. compute_pair_reward's -1.0-for-"not visited" convention
        # already encodes "valid but poor sampling" as a real (low-weight)
        # observation, so — unlike REINFORCE — there is no separate EMA
        # baseline, success floor, or failure-batch needed here: every combo
        # in deferred_updates is already crash/infra-failure-free (retries and
        # permanent-failure handling happen earlier in this function and never
        # populate deferred_updates; see the record_combo_failure call above).
        reward_config = config.get('reward', {})
        no_transition_weight = float(reward_config.get('no_transition_weight', 0.2))
        awr_temperature = float(bandit_cfg.get('awr_temperature', 0.5))
        pair_min_trans = int(reward_config.get(
            'floor_mask_min_transitions', reward_config.get('min_transitions_per_site', 3)))
        pair_balance_threshold = float(reward_config.get('pair_balance_threshold', 0.1))

        # n_updated counts every combo fed into the posterior (all crash-free
        # combos, including poorly-sampled ones — see above). n_success uses
        # the same gate as the REINFORCE path (>=1 site meets the transition
        # floor) so "Success batch" and the plateau LR scheduler, which both
        # consume n_success, mean the same thing for either algorithm.
        n_updated = 0
        n_success = 0
        _r_relevant_all = []
        _hs_barrier_target = float(reward_config.get('barrier_target', BARRIER_TARGET))
        _hs_temperature = float(bandit_cfg.get('temperature', DEFAULT_TEMPERATURE))
        _hs_linear_cap = float(bandit_cfg.get('hindsight_linear_step_cap', 3.0))
        _hs_quadratic_cap = float(bandit_cfg.get('hindsight_quadratic_step_cap', 12.0))
        _hs_weight_scale = float(bandit_cfg.get('hindsight_weight_scale', 1.0))
        _hs_steps = ([], [])      # |target - applied| per head, for the summary
        _hs_n_edges = 0
        _hs_n_offsets = 0
        for (combo_path_u, epoch_dir_u, reward_u) in deferred_updates:
            try:
                (vdata_u, saved_data, saved_actions, r_pair,
                 edge_weights_u, _trans_vals) = _prepare_success_combo_reward(
                    combo_path_u, epoch_dir_u, combo_system_map, config, device,
                    no_transition_weight,
                )
                _ps = compute_pair_success_stats(
                    vdata_u.edge_index, saved_data.get('ddg_pairs', {}),
                    saved_data.get('populations', []),
                    total_transitions=sum(_trans_vals), min_transitions=pair_min_trans,
                    balance_threshold=pair_balance_threshold,
                )
                for _k in pair_stats_total:
                    pair_stats_total[_k] += _ps[_k]
                if vdata_u.edge_type is None:
                    # Online-training graph structure (the only one
                    # build_graph_and_data produces): one edge per pair,
                    # every bias type independently meaningful for every
                    # edge. Use compute_pair_reward's full [E, 4] output and
                    # a genuinely per-dimension AWR weight — each bias type
                    # gets its own reward-derived pull, not one shared value.
                    r_relevant = r_pair  # [E, 4]
                    edge_weights_bcast = edge_weights_u.unsqueeze(-1)  # [E, 1] -> broadcasts
                else:
                    # Routed graph structure (pretraining's fully-connected
                    # 4x-edges-per-pair layout): only the edge's own routed
                    # dimension is meaningful.
                    bias_type_u = vdata_u.edge_type // 2
                    r_relevant = r_pair.gather(1, bias_type_u.unsqueeze(-1)).squeeze(-1)  # [E]
                    edge_weights_bcast = edge_weights_u
                # AWR-style reward weight (same exp(r/beta), clamp<=20
                # convention pretrain_policy.py's AWR pretraining already
                # uses): a good outcome pulls the posterior toward the
                # observed action much harder than a poor one — ridge
                # regression has no symmetric "push away from a bad action"
                # mechanism, so avoidance comes from good actions dominating
                # the reward-weighted fit, not from a negative update.
                awr_w = torch.exp(r_relevant / awr_temperature).clamp(max=20.0)
                combined_weight = edge_weights_bcast * awr_w
                update_targets = saved_actions

                if _use_hindsight and vdata_u.edge_type is None:
                    # Linear/quadratic heads: regress toward the biases this
                    # run measured it should have used, each weighted by its
                    # own measurement noise (no reward weighting -- a trapped
                    # or over-flattened run is as informative as a good one).
                    # Skew/end keep the AWR target above.
                    _applied = read_applied_biases(epoch_dir_u)
                    if _applied is not None:
                        hs_t, hs_v = hindsight_targets(
                            vdata_u.edge_index, _applied, saved_data,
                            barrier_target=_hs_barrier_target,
                            temperature=_hs_temperature,
                            linear_step_cap=_hs_linear_cap,
                            quadratic_step_cap=_hs_quadratic_cap,
                        )
                        hs_t = hs_t.to(saved_actions.dtype).to(device)
                        finite = torch.isfinite(hs_v).to(device)
                        update_targets = saved_actions.clone()
                        update_targets[:, :2] = torch.where(finite, hs_t, saved_actions[:, :2])
                        combined_weight = combined_weight.expand_as(saved_actions).clone()
                        combined_weight[:, :2] = preact_weights(
                            policy, hs_t.cpu(), hs_v, dims=(0, 1), scale=_hs_weight_scale,
                        ).to(device)
                        _applied_t = saved_actions[:, :2].detach().cpu().double()
                        _step = (hs_t.cpu().double() - _applied_t)
                        for _col in range(2):
                            _m = torch.isfinite(hs_v[:, _col])
                            _hs_steps[_col].extend(_step[_m, _col].abs().tolist())
                        _hs_n_edges += int(vdata_u.edge_index.size(1))
                        if update_policy and _use_offsets:
                            _labels = combo_node_labels(combo_path_u)
                            if _labels:
                                _hs_n_offsets += record_pair_targets(
                                    bandit_state,
                                    combo_system_map.get(str(combo_path_u), {}).get('name', 'unknown'),
                                    _labels, vdata_u.edge_index, hs_t.cpu(), hs_v,
                                    history=int(bandit_cfg.get('offset_history', 5)),
                                )

                if update_policy:
                    policy.update_bayesian_posteriors(
                        vdata_u.x, vdata_u.edge_index, vdata_u.edge_type,
                        update_targets, weights=combined_weight,
                    )
                n_updated += 1
                if _trans_vals and max(_trans_vals) >= pair_min_trans:
                    n_success += 1
                _r_relevant_all.append(r_relevant)
            except Exception as e:
                print(f"  Warning: NeuralLinear+TS update failed for {combo_path_u.name}: {e}")

        if _r_relevant_all:
            _all_r = torch.cat(_r_relevant_all)
            dim_names = ['linear', 'quadratic', 'skew', 'end']
            _n_obs_str = ', '.join(
                f"{name}:{int(policy.edge_mlp.mlps[d].bayesian_head.n_obs.item())}"
                for d, name in enumerate(dim_names)
            )
            _stats_msg = (
                f"  NeuralLinear+TS posterior update: {n_updated}/{len(deferred_updates)} "
                f"combos ({n_success} meeting transition floor), "
                f"{_all_r.shape[0]} edges, reward mean={_all_r.mean().item():.3f} "
                f"std={_all_r.std().item():.3f}, n_obs per head=[{_n_obs_str}]"
            )
            print(_stats_msg)
            _detail_lines.append(_stats_msg)
        if _hs_n_edges:
            def _med(v):
                return f"{float(np.median(v)):.2f} (n={len(v)})" if v else "n/a"
            _hs_msg = (
                f"  Hindsight targets: median |step| linear {_med(_hs_steps[0])} kcal/mol, "
                f"quadratic {_med(_hs_steps[1])}; pair-offset observations recorded: {_hs_n_offsets}"
            )
            print(_hs_msg)
            _detail_lines.append(_hs_msg)

    # === Epoch-batch REINFORCE update ===
    elif deferred_updates:
        reward_config = config.get('reward', {})
        lambda_e = reward_config.get('lambda_entropy', 0.0)
        min_trans = reward_config.get(
            'floor_mask_min_transitions',
            reward_config.get('min_transitions_per_site', 3)
        )
        failure_advantage = reward_config.get('failure_advantage', None)
        pair_balance_threshold = float(reward_config.get('pair_balance_threshold', 0.1))
        # Per-edge down-weight applied to pairs with no observed DDG (no
        # lambda-space crossings between that specific pair). Rather than
        # relying on the advantage floor to silently zero these out (see
        # success_floor below), every pair is included in training with a
        # small, tunable, non-zero weight — so pairs that were simply
        # out-competed by another substituent at the same site (rather than
        # having a genuinely bad bias themselves) still get a fair, muted
        # signal instead of either full-strength blame or total exclusion.
        no_transition_weight = float(reward_config.get('no_transition_weight', 0.2))

        raw_batch = []
        failure_batch = []
        _n_partial = 0   # combos where ≥1 site passed but not all
        for (combo_path_u, epoch_dir_u, reward_u) in deferred_updates:
            sys_cfg_u = combo_system_map.get(str(combo_path_u), {})
            try:
                # Reuse the graph/Uni-Mol data built earlier this epoch
                # (Phase 1 prep, replacement, or recovery path), loaded from
                # the small on-disk cache written by _save_graph_data_cache,
                # instead of recomputing embeddings for every combo a second
                # time. Only combos whose reward came from a pre-existing
                # cached epoch_results.pt (no build this epoch) fall back to
                # a fresh build here.
                vdata_u = _load_graph_data_cache(Path(epoch_dir_u), device)
                if vdata_u is None:
                    vdata_u, _ = build_graph_and_data(
                        str(combo_path_u), sys_cfg_u, config, device
                    )
                # graph_data.pt is only needed to bridge Phase 1 -> here within
                # the same epoch; remove it now so per-epoch combo dirs don't
                # accumulate this artifact indefinitely across the run.
                _gd_cache_path = Path(epoch_dir_u) / 'graph_data.pt'
                if _gd_cache_path.exists():
                    try:
                        _gd_cache_path.unlink()
                    except Exception:
                        pass
                results_pt = Path(epoch_dir_u) / 'epoch_results.pt'
                saved_data = torch.load(results_pt, weights_only=False)

                site_trans = saved_data.get('transitions', [])
                trans_vals = list(site_trans.values()) if isinstance(site_trans, dict) else list(site_trans)
                _ps = compute_pair_success_stats(
                    vdata_u.edge_index, saved_data.get('ddg_pairs', {}),
                    saved_data.get('populations', []),
                    total_transitions=sum(trans_vals), min_transitions=min_trans,
                    balance_threshold=pair_balance_threshold,
                )
                for _k in pair_stats_total:
                    pair_stats_total[_k] += _ps[_k]
                # Gate: include if ANY site meets the floor.  Failing-site pairs
                # have DDG=None and receive -1.0 from compute_pair_reward, which
                # naturally suppresses the actions that produced no sampling there.
                max_site_trans = max(trans_vals) if trans_vals else 0
                if max_site_trans < min_trans:
                    if failure_advantage is not None:
                        saved_actions = saved_data['actions'].to(device)
                        fail_pops = saved_data.get('populations', [])
                        failure_batch.append((combo_path_u, vdata_u, saved_actions, fail_pops))
                    continue
                if trans_vals and min(trans_vals) < min_trans:
                    _n_partial += 1

                saved_actions = saved_data['actions'].to(device)
                ddg_pairs = saved_data.get('ddg_pairs', {})
                populations = saved_data.get('populations', [])
                _trans = saved_data.get('transitions', [])
                _trans_vals = list(_trans.values()) if isinstance(_trans, dict) else list(_trans)
                _total_trans = sum(_trans_vals)
                _t_base = config.get('reward', {}).get('T_baseline', 50.0)
                _rew_cfg = config.get('reward', {})
                r_pair = compute_pair_reward(
                    vdata_u.edge_index, ddg_pairs, populations,
                    total_transitions=_total_trans,
                    t_baseline=_t_base,
                    fraction_physical=saved_data.get('fraction_physical'),
                    barrier_pairs=saved_data.get('barrier_pairs'),
                    barrier_target=_rew_cfg.get('barrier_target', BARRIER_TARGET),
                    barrier_width=_rew_cfg.get('barrier_width', BARRIER_WIDTH),
                    skew_pairs=saved_data.get('skew_pairs'),
                    end_pairs=saved_data.get('end_pairs'),
                    skew_width=_rew_cfg.get('skew_width', SKEW_WIDTH),
                    end_width=_rew_cfg.get('end_width', END_WIDTH),
                ).to(device)
                edge_weights_u = build_edge_weights(
                    vdata_u.edge_index, ddg_pairs, no_transition_weight, device,
                )

                logp_old = saved_data['logp'].to(device).detach()
                raw_batch.append((combo_path_u, vdata_u, saved_actions, r_pair, logp_old, edge_weights_u))

            except Exception as e:
                print(f"  Warning: Could not prepare update for {combo_path_u.name}: {e}")

        # Baseline selection
        ema_alpha = reward_config.get('ema_alpha', 0.2)
        ema_min_batch = int(reward_config.get('ema_min_batch', 5))
        success_floor = reward_config.get('success_advantage_floor', 0.0)
        # reward.safety_lock_n is no longer read (see the note further down,
        # where the fixed-constant fallback it used to trigger was removed).
        _gb_cfg = reward_config.get('global_baseline', [0.0, -1.0, -1.0, -1.0])
        global_baseline_t = torch.tensor(_gb_cfg, dtype=torch.float32, device=device)
        dim_names = ['linear', 'quadratic', 'skew', 'end']

        n_success = len(raw_batch)
        if raw_batch:
            all_r = torch.cat([item[3] for item in raw_batch])
            all_w = torch.cat([item[5] for item in raw_batch])
            # Weighted baseline: down-weighted (no-crossing) edges contribute
            # proportionally less to the mean/std so they don't distort the
            # normalisation used for the well-sampled (crossed) edges.
            _w_sum = all_w.sum().clamp(min=1e-6)
            r_mean_batch = (all_r * all_w.unsqueeze(-1)).sum(dim=0) / _w_sum
            r_var_batch = (all_w.unsqueeze(-1) * (all_r - r_mean_batch) ** 2).sum(dim=0) / _w_sum
            r_std_batch = r_var_batch.clamp(min=1e-12).sqrt().clamp(min=1e-6)
            n_downweighted = int((all_w < 1.0 - 1e-6).sum().item())
            stats_str = ', '.join(
                f'{n}: {r_mean_batch[i]:.3f}±{r_std_batch[i]:.3f}'
                for i, n in enumerate(dim_names)
            )
            # No separate "safety lock" on top of this: ema_min_batch already
            # gates whether a small batch gets folded INTO the running
            # estimate (below), and seeding a brand-new EMA from a small-but-
            # real batch is still strictly more informative than an arbitrary
            # config constant unrelated to these systems. A prior version
            # additionally forced r_mean/r_std to a fixed global constant
            # whenever n_success < safety_lock_n (default 10) -- redundant
            # with ema_min_batch where it mattered, and actively harmful for
            # systems whose n_success chronically sits below that threshold:
            # they never got to use their own accumulated EMA at all, every
            # epoch reverting to a number with no relationship to what these
            # systems actually look like.
            if ema_baseline is not None:
                if len(ema_baseline) == 0:
                    ema_baseline['mean'] = r_mean_batch.detach().clone().cpu()
                    ema_baseline['std'] = r_std_batch.detach().clone().cpu()
                    ema_baseline['n_updates'] = 1
                elif n_success >= ema_min_batch:
                    ema_baseline['mean'] = (
                        ema_alpha * r_mean_batch.detach().cpu()
                        + (1.0 - ema_alpha) * ema_baseline['mean']
                    )
                    ema_baseline['std'] = (
                        ema_alpha * r_std_batch.detach().cpu()
                        + (1.0 - ema_alpha) * ema_baseline['std']
                    )
                    ema_baseline['n_updates'] += 1
            if ema_baseline and 'mean' in ema_baseline:
                r_mean = ema_baseline['mean'].to(device)
                r_std = ema_baseline['std'].to(device)
                baseline_src = f"EMA(n_upd={ema_baseline['n_updates']})"
            else:
                r_mean = r_mean_batch
                r_std = r_std_batch
                baseline_src = "batch"

            _batch_stats_msg = (
                f"  Batch reward stats (n={n_success} combos"
                f"{f', {_n_partial} partial' if _n_partial else ''}, "
                f"{all_r.shape[0]} edges [{n_downweighted} down-weighted "
                f"x{no_transition_weight}], baseline={baseline_src}): {stats_str}"
            )
            print(_batch_stats_msg)
            _detail_lines.append(_batch_stats_msg)
        else:
            r_mean = global_baseline_t
            r_std = torch.ones(4, device=device)

        # Combined policy gradient: success batch + (down-weighted) failure
        # batch now share a single zero_grad/backward-accumulate/step cycle.
        # Previously each batch got its own independent Adam step; since Adam's
        # per-parameter step size is normalised by gradient RMS (not raw loss
        # magnitude), two separate full-strength steps per epoch moved the
        # policy roughly twice as far as intended, with the failure step
        # (driven mostly by simulator/sampling difficulty, not policy quality)
        # contributing equally to that movement regardless of how small its
        # loss was. Merging into one step means the failure contribution is
        # additive to (and bounded relative to) the success gradient instead
        # of an independent full-sized nudge.
        _use_context_baseline = baseline_net is not None and baseline_optimizer is not None
        if _use_context_baseline:
            # Diagnostics only (not used in the loss itself): per-dimension
            # weighted squared error of the context baseline's own
            # predictions, plus the same for the naive pooled-batch-mean it
            # replaced (r_mean_batch, already computed above) -- so the
            # epoch report can show whether it's actually beating the thing
            # it replaced, not just that it's running.
            _baseline_sq_err_sum = torch.zeros(4, device=device)
            _naive_sq_err_sum = torch.zeros(4, device=device)
            _baseline_w_sum = torch.zeros(4, device=device)
        if (raw_batch or failure_batch) and update_policy:
            rl_optimizer.zero_grad()
            if _use_context_baseline:
                baseline_optimizer.zero_grad()
            for (combo_path_u, vdata_u, saved_actions, r_pair, logp_old, edge_weights_u) in raw_batch:
                try:
                    if _use_context_baseline:
                        # Context-dependent baseline: "typical reward for a
                        # context like this edge/system", replacing the
                        # pooled-across-every-system r_mean above. z is
                        # detached from the policy trunk before the baseline
                        # sees it -- its regression loss (below) must never
                        # reach the policy's own parameters. r_std stays the
                        # existing batch/EMA scalar; only the mean reference
                        # becomes context-dependent.
                        _sys_name = combo_system_map.get(str(combo_path_u), {}).get('name', '__unknown__')
                        _sys_idx = (system_to_idx or {}).get(_sys_name, 0)
                        _sys_ids = torch.full((r_pair.size(0),), _sys_idx, dtype=torch.long, device=device)
                        _z_ctx = policy.forward_features(
                            vdata_u.x, vdata_u.edge_index, vdata_u.edge_type
                        ).detach()
                        _baseline_pred = baseline_net(_z_ctx, _sys_ids)  # [E, D], grad flows to baseline only
                        a_pair = (r_pair - _baseline_pred.detach()) / r_std
                    else:
                        a_pair = (r_pair - r_mean) / r_std
                    if success_floor is not None:
                        # Clamp every edge's advantage at the same floor,
                        # regardless of confidence weight. Down-weighted
                        # (no-crossing) edges get a small effective loss
                        # contribution via edge_weights_u below, but their raw
                        # (pre-weight) advantage can be several std widths more
                        # negative than a full-weight edge's floor-clamped
                        # advantage — weighting alone doesn't keep that bounded,
                        # so the floor is now applied universally.
                        a_pair = a_pair.clamp(min=float(success_floor))

                    logp_per_dim, log_std_saved = policy.evaluate_logp(
                        vdata_u.x, vdata_u.edge_index, vdata_u.edge_type,
                        saved_actions,
                    )
                    
                    # Log importance ratio to detect off-policy staleness:
                    # If ratio >> 1, old logp is stale and gradients are wrong-signed.
                    # This diagnostic confirms whether PPO clipping would help.
                    if logp_old is not None and logp_old.numel() == logp_per_dim.numel():
                        try:
                            importance_ratio = torch.exp(logp_per_dim - logp_old)
                            ratio_mean = importance_ratio.mean().item()
                            ratio_std = importance_ratio.std().item()
                            ratio_max = importance_ratio.max().item()
                        except Exception:
                            pass
                    
                    # Correct multi-output REINFORCE: multiply each dimension's log prob
                    # by its corresponding advantage before summing over dimensions.
                    # This ensures each bias type gradient is weighted by its own action's
                    # log probability, not by the sum of all log probabilities.
                    # Per-edge weighting: every pair contributes (weighted mean
                    # over edges, using edge_weights_u), rather than a uniform
                    # per-edge mean — down-weighted no-crossing edges still
                    # contribute, just proportionally less.
                    _per_edge_term = (logp_per_dim * a_pair).sum(dim=-1)  # [E]
                    _w_sum_combo = edge_weights_u.sum().clamp(min=1e-6)
                    policy_loss = -(_per_edge_term * edge_weights_u).sum() / _w_sum_combo / len(raw_batch)
                    if lambda_e > 0:
                        policy_loss = policy_loss - lambda_e * log_std_saved.mean() / len(raw_batch)

                    if bc_auxiliary_weight > 0.0 and bc_source_epoch is not None:
                        _bc_pt = combo_path_u / f"run_{bc_source_epoch:03d}" / 'epoch_results.pt'
                        if _bc_pt.exists():
                            try:
                                _bc_d = torch.load(_bc_pt, weights_only=False)
                                _bc_target = _bc_d['actions'].to(device)
                                _bc_mean, _ = policy._forward_edges(
                                    vdata_u.x, vdata_u.edge_index, vdata_u.edge_type
                                )
                                if _bc_target.dim() == 1:
                                    _bc_target = _bc_target.unsqueeze(-1).expand_as(_bc_mean)
                                _et_aux = vdata_u.edge_type // 2
                                _aux_mask = torch.zeros(_bc_mean.shape, dtype=torch.bool, device=device)
                                for _d in range(policy.mlp_out_dim):
                                    _aux_mask[_et_aux == _d, _d] = True
                                _bc_aux = F.mse_loss(
                                    (_bc_mean / target_norm)[_aux_mask],
                                    (_bc_target.detach() / target_norm)[_aux_mask],
                                )
                                policy_loss = (policy_loss
                                               + bc_auxiliary_weight * _bc_aux / len(raw_batch))
                            except Exception:
                                pass

                    if not (torch.isnan(policy_loss) or torch.isinf(policy_loss)):
                        policy_loss.backward()
                        epoch_loss += policy_loss.item() * len(raw_batch)

                    # Train the baseline toward this combo's *observed* reward
                    # (not detached this time) -- a plain weighted-MSE
                    # regression, disjoint from the policy graph above since
                    # _z_ctx was detached before baseline_net saw it.
                    if _use_context_baseline:
                        _b_err = (_baseline_pred - r_pair.detach()) ** 2  # [E, D]
                        _b_w = edge_weights_u.unsqueeze(-1).expand_as(_b_err)
                        baseline_loss = (_b_err * _b_w).sum() / _b_w.sum().clamp(min=1e-6) / len(raw_batch)
                        if not (torch.isnan(baseline_loss) or torch.isinf(baseline_loss)):
                            baseline_loss.backward()
                        with torch.no_grad():
                            _baseline_sq_err_sum += (_b_err.detach() * _b_w).sum(dim=0)
                            _naive_err = (r_mean_batch.unsqueeze(0) - r_pair) ** 2  # [E, D]
                            _naive_sq_err_sum += (_naive_err * _b_w).sum(dim=0)
                            _baseline_w_sum += _b_w.sum(dim=0)

                except Exception as e:
                    print(f"  Warning: REINFORCE update failed for {combo_path_u.name}: {e}")

            # Failure-batch gradient: accumulated onto the SAME gradient as the
            # success batch above (no intermediate zero_grad/step). The
            # contribution is heavily down-weighted (failure_loss_weight,
            # default 0.05) and its per-dimension advantage is clamped to the
            # same success_advantage_floor, so a below-floor combo can never
            # exert more per-edge influence than the strongest correction
            # allowed on the success side.
            if failure_batch and failure_advantage is not None:
                beta_fail = float(reward_config.get('failure_loss_weight', 0.05))
                n_fail = len(failure_batch)
                n_ref = max(len(raw_batch), 1)
                for (combo_path_f, vdata_f, saved_actions_f, fail_pops) in failure_batch:
                    try:
                        a_fail = compute_failure_reward(
                            vdata_f.edge_index, fail_pops,
                            failure_advantage=float(failure_advantage),
                        ).to(device)
                        if success_floor is not None:
                            a_fail = a_fail.clamp(min=float(success_floor))
                        logp_fail, _ = policy.evaluate_logp(
                            vdata_f.x, vdata_f.edge_index, vdata_f.edge_type,
                            saved_actions_f,
                        )
                        fail_loss = beta_fail * -(logp_fail * a_fail).sum(dim=-1).mean() / n_ref
                        if not (torch.isnan(fail_loss) or torch.isinf(fail_loss)):
                            fail_loss.backward()
                    except Exception as e:
                        print(f"  Warning: failure gradient failed for {combo_path_f.name}: {e}")
                _failure_grad_msg = (
                    f"  Failure gradient: {n_fail} below-floor combos, "
                    f"A={failure_advantage:.1f} (loss_weight={beta_fail:.3f}, "
                    f"floor-clamped, merged into success step)"
                )
                print(_failure_grad_msg)
                _detail_lines.append(_failure_grad_msg)

            # Replay-style AWR: fold in additional (context, action, reward)
            # observations sampled from combos' own simulation history, onto
            # the SAME gradient as the fresh batch above -- more signal per
            # expensive simulation, no new simulations. Opt-in, default off.
            if history_replay_cfg and history_replay_cfg.get('enabled', False):
                _k_replay = int(history_replay_cfg.get('k_per_epoch', 0))
                _max_lookback = history_replay_cfg.get('max_lookback', None)
                _awr_temp = float(history_replay_cfg.get('awr_temperature', 1.0))
                _replay_rng = np.random.RandomState(
                    history_replay_cfg.get('seed', 0) + epoch
                )
                _replay_items = _sample_replay_items(
                    combos, epoch, _k_replay, _max_lookback, _replay_rng,
                )
                _n_replay_used = 0
                if _replay_items:
                    for (_r_combo_path, _r_epoch_dir) in _replay_items:
                        _r_loss = _compute_replay_awr_loss(
                            policy, _r_combo_path, _r_epoch_dir, combo_system_map,
                            config, device, no_transition_weight, _awr_temp,
                        )
                        if _r_loss is not None:
                            (_r_loss / len(_replay_items)).backward()
                            _n_replay_used += 1
                if _replay_items or _k_replay > 0:
                    _replay_msg = (
                        f"  History replay (AWR): {_n_replay_used}/{len(_replay_items)} "
                        f"sampled observations used (k_per_epoch={_k_replay}, "
                        f"awr_temperature={_awr_temp})"
                    )
                    print(_replay_msg)
                    _detail_lines.append(_replay_msg)

            # Phase 1.3: Expand gradient clipping to all parameters (consistent with pretraining)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            rl_optimizer.step()
            if _use_context_baseline:
                torch.nn.utils.clip_grad_norm_(baseline_net.parameters(), max_norm=1.0)
                baseline_optimizer.step()
                # Report whether the context baseline is actually beating the
                # naive pooled-batch-mean it replaced, per bias type -- not
                # just that it ran. Skipped if raw_batch was empty this epoch
                # (nothing to compare; _baseline_w_sum stays all-zero then).
                if float(_baseline_w_sum.sum()) > 0:
                    _bl_mse = _baseline_sq_err_sum / _baseline_w_sum.clamp(min=1e-6)
                    _naive_mse = _naive_sq_err_sum / _baseline_w_sum.clamp(min=1e-6)
                    _bl_msg = (
                        "  Context baseline MSE vs naive pooled-mean MSE (lower is better): "
                        + ", ".join(
                            f"{n}: {_bl_mse[i]:.4f} vs {_naive_mse[i]:.4f}"
                            for i, n in enumerate(dim_names)
                        )
                    )
                    print(_bl_msg)
                    _detail_lines.append(_bl_msg)
    else:
        n_success = 0

    avg_loss = epoch_loss / len(epoch_rewards) if epoch_rewards else 0.0
    avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
    reward_std = float(np.std(epoch_rewards)) if epoch_rewards else 0.0
    return {
        'loss': avg_loss,
        'avg_reward': avg_reward,
        'reward_std': reward_std,
        'num_combos': len(combos),
        'update_policy': update_policy,
        'bc_mode': False,
        'n_success': n_success,
        'permanently_failed': _permanently_failed,
        'pair_success': pair_stats_total,
        'detail_lines': _detail_lines,
    }
