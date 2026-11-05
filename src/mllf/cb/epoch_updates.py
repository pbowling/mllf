"""Policy updates from one epoch's finished runs (``mllf.cb.epoch_runs``).

* ``behavior_cloning_epoch``: regress toward saved actions instead of running.
* ``hindsight_update``: NeuralLinear + Thompson sampling -- per-pair offsets,
  exploration and outcomes, then the shared model's closed-form posterior
  update or hindsight regression (``bandit.shared_update``).
* ``reinforce_update``: one REINFORCE step on the epoch's batch (plus the
  optional failure batch, context baseline and history replay).
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mllf.cb.epoch_runs import EpochRun, load_graph_data_cache
from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.pairs import build_directed_pairs
from mllf.cb.group_correction import correction_targets, update_group
from mllf.cb.hindsight import (
    BIAS_DIMS,
    OFFSET_DIMS,
    SUCCESS_MIN_FRACTION,
    SUCCESS_MIN_TRANSITIONS,
    barrier_trapped_c_targets,
    combo_node_labels,
    grow_directional_b_steps,
    held_off_counts,
    hindsight_targets,
    lambda_states,
    node_transition_counts,
    pair_succeeded,
    preact_weights,
    read_applied_biases,
    record_pair_outcomes,
    record_pair_targets,
    site_nodes,
    uncrossed_pairs,
    update_exploration,
)
from mllf.cb.hindsight_regression import (
    build_replay_set,
    online_item,
    regression_config,
    regression_step,
)
from mllf.cb.pair_weights import compute_pairwise_confidence_weights
from mllf.cb.workflow_utils import (
    BARRIER_TARGET,
    BARRIER_WIDTH,
    END_WIDTH,
    SKEW_WIDTH,
    build_edge_weights,
    compute_failure_reward,
    compute_pair_reward,
    compute_pair_success_stats,
)
from mllf.file_handling.manifests import ensure_combo_dir_exists

PAIR_STAT_KEYS = ('total', 'crossed', 'balanced', 'both')


def empty_pair_stats() -> Dict[str, int]:
    return dict.fromkeys(PAIR_STAT_KEYS, 0)


# ── reading a finished run ──────────────────────────────────────────────────

def _trans_values(saved_data: Dict) -> List[float]:
    trans = saved_data.get('transitions', [])
    return list(trans.values()) if isinstance(trans, dict) else list(trans)


def load_run(combo_path: Path, epoch_dir: Path, combo_system_map: Dict[str, Dict], config: Dict,
             device: str):
    """(graph data, saved epoch_results) of a finished run. The graph comes
    from the run's small cache (built in phase 1) when present, else is
    rebuilt; the cache is removed either way so run dirs don't keep it."""
    vdata = load_graph_data_cache(Path(epoch_dir), device)
    if vdata is None:
        vdata, _ = build_graph_and_data(str(combo_path), combo_system_map.get(str(combo_path), {}),
                                        config, device)
    gd_cache = Path(epoch_dir) / 'graph_data.pt'
    if gd_cache.exists():
        try:
            gd_cache.unlink()
        except Exception:
            pass
    saved_data = torch.load(Path(epoch_dir) / 'epoch_results.pt', weights_only=False)
    return vdata, saved_data


def pair_reward(edge_index: torch.Tensor, saved_data: Dict, reward_cfg: Dict,
                device: str) -> torch.Tensor:
    """[E, 4] per-pair reward of one run (``compute_pair_reward``)."""
    return compute_pair_reward(
        edge_index, saved_data.get('ddg_pairs', {}), saved_data.get('populations', []),
        total_transitions=sum(_trans_values(saved_data)),
        t_baseline=reward_cfg.get('T_baseline', 50.0),
        fraction_physical=saved_data.get('fraction_physical'),
        barrier_pairs=saved_data.get('barrier_pairs'),
        barrier_target=reward_cfg.get('barrier_target', BARRIER_TARGET),
        barrier_width=reward_cfg.get('barrier_width', BARRIER_WIDTH),
        skew_pairs=saved_data.get('skew_pairs'),
        end_pairs=saved_data.get('end_pairs'),
        skew_width=reward_cfg.get('skew_width', SKEW_WIDTH),
        end_width=reward_cfg.get('end_width', END_WIDTH),
    ).to(device)


def prepare_success_combo_reward(combo_path: Path, epoch_dir: Path,
                                 combo_system_map: Dict[str, Dict],
                                 config: Dict, device: str, no_transition_weight: float):
    """A finished run's graph and results with its per-edge reward and
    confidence weights: (vdata, saved_data, saved_actions, r_pair,
    edge_weights, transition values). Raises on I/O or graph-build failure."""
    vdata, saved_data = load_run(combo_path, epoch_dir, combo_system_map, config, device)
    saved_actions = saved_data['actions'].to(device)
    r_pair = pair_reward(vdata.edge_index, saved_data, config.get('reward', {}), device)
    edge_weights = build_edge_weights(vdata.edge_index, saved_data.get('ddg_pairs', {}),
                                      no_transition_weight, device)
    return vdata, saved_data, saved_actions, r_pair, edge_weights, _trans_values(saved_data)


def _nsubs_from_site_index(vdata) -> List[int]:
    unique_sites = vdata.site_index.unique() if hasattr(vdata, 'site_index') else []
    return [int((vdata.site_index == s).sum().item()) for s in sorted(unique_sites.tolist())]


def pair_test_counts(edge_index, saved_data, min_trans: int, min_frac: float) -> Dict[str, int]:
    """Per undirected pair: crossed (>= min_trans transitions), balanced
    (both sampled, minority >= min_frac) and both (the success test,
    hindsight.pair_succeeded)."""
    out = empty_pair_stats()
    pops = saved_data.get('populations') or []
    sites = site_nodes(edge_index)
    for k in range(edge_index.size(1)):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i >= j:
            continue
        site = sites.get(i, [i, j])
        if len(site) > 2:
            nt = node_transition_counts(saved_data, site) or {}
            t = min(nt.get(i, 0), nt.get(j, 0))
        else:
            t = (saved_data.get('transition_pairs') or {}).get(f"{i + 2}_{j + 2}", 0) or 0
        p_i = pops[i] if i < len(pops) else 0
        p_j = pops[j] if j < len(pops) else 0
        out['total'] += 1
        out['crossed'] += int(t >= min_trans)
        out['balanced'] += int(p_i > 0 and p_j > 0 and min(p_i, p_j) / (p_i + p_j) >= min_frac)
        out['both'] += int(pair_succeeded(saved_data, i, j, site=site,
                                          min_transitions=min_trans, min_fraction=min_frac))
    return out


def _add(total: Dict[str, int], part: Dict[str, int]) -> None:
    for k in total:
        total[k] += part[k]


# ── behavioral cloning ──────────────────────────────────────────────────────

def bc_target_norm(run: EpochRun) -> torch.Tensor:
    """Per-dimension scale of the BC targets: the std of bc_source_epoch's
    saved actions (at least 1) when cloning toward them, else the policy's
    scale factors."""
    opts, policy, device = run.opts, run.policy, run.device
    if (opts.bc_mode or opts.bc_auxiliary_weight > 0.0) and opts.bc_source_epoch is not None:
        norm_list = []
        for cd in run.combos:
            rpt = Path(cd) / f"run_{opts.bc_source_epoch:03d}" / 'epoch_results.pt'
            if rpt.exists():
                try:
                    norm_list.append(torch.load(rpt, weights_only=False)['actions'].float())
                except Exception:
                    pass
        if norm_list:
            target_norm = torch.cat(norm_list, dim=0).std(dim=0).clamp(min=1.0).to(device)
        else:
            target_norm = policy.scale_factors.to(device)
        print("  BC target_norm (per-dim std, min=1): "
              f"{[f'{v:.3f}' for v in target_norm.tolist()]}")
        return target_norm
    return policy.scale_factors.to(device)


def _routed_mask(policy, edge_type, shape, device) -> torch.Tensor:
    """True at each edge's own bias type (routed graphs: edge_type // 2)."""
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    et = edge_type // 2
    for d in range(policy.mlp_out_dim):
        mask[et == d, d] = True
    return mask


def _latest_results(combo_path: Path, source_epoch: Optional[int]) -> Optional[Path]:
    if source_epoch is not None:
        candidate = combo_path / f"run_{source_epoch:03d}" / 'epoch_results.pt'
        return candidate if candidate.exists() else None
    run_dirs = sorted(combo_path.glob('run_*'),
                      key=lambda p: (int(p.name.split('_')[1]) if p.name.split('_')[1].isdigit()
                                     else -1))
    for rd in reversed(run_dirs):
        if (rd / 'epoch_results.pt').exists():
            return rd / 'epoch_results.pt'
    return None


def _bc_loss(run: EpochRun, combo_path: Path, sys_cfg: Dict, saved_data: Dict) -> torch.Tensor:
    """Confidence-weighted Gaussian NLL of one combo's saved actions."""
    policy, device, verbose = run.policy, run.device, run.opts.verbose
    saved_actions = saved_data['actions'].to(device)
    vdata, _ = build_graph_and_data(str(combo_path), sys_cfg, run.config, device)
    mean, log_std = policy._forward_edges(vdata.x, vdata.edge_index, vdata.edge_type)
    if saved_actions.dim() == 1:
        saved_actions = saved_actions.unsqueeze(-1).expand_as(mean)
    routed = _routed_mask(policy, vdata.edge_type, mean.shape, device)

    # Pairwise confidence weights from the run's populations / DDGs.
    ddg_pairs = saved_data.get('ddg_pairs', {})
    populations = saved_data.get('populations', [])
    weights = None
    if ddg_pairs and populations:
        try:
            unique_sites = vdata.site_index.unique() if hasattr(vdata, 'site_index') else []
            nsubs_per_site = []
            if len(unique_sites) == 0:
                # Fallback: infer from edge_type distribution
                for d in range(policy.mlp_out_dim):
                    count = (vdata.edge_type // 2 == d).sum().item()
                    if count > 0:
                        nsubs_per_site.append(int((1 + (1 + 8 * count) ** 0.5) / 2))
            else:
                for site_id in sorted(unique_sites.tolist()):
                    nsubs_per_site.append((vdata.site_index == site_id).sum().item())
            weights = compute_pairwise_confidence_weights(
                {'ddg_pairs': ddg_pairs, 'populations': populations},
                nsubs_per_site=nsubs_per_site, pairs=build_directed_pairs(nsubs_per_site))
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not compute pairwise weights: {e}")

    logp = torch.distributions.Normal(mean, torch.exp(log_std)).log_prob(saved_actions)
    mask = routed & (saved_actions.abs() > 1e-8)       # only non-zero target edges

    def unweighted():
        return -logp[mask].mean() if mask.sum() > 0 else torch.tensor(0.0, device=device)
    if weights is None:
        return unweighted()
    try:
        w = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)  # [E, 1]
        weighted = logp[mask] * w[mask]
        return -weighted.mean() if weighted.numel() > 0 else unweighted()
    except Exception as e:
        if verbose:
            print(f"    Warning: Could not apply pairwise weights: {e}")
        return unweighted()


def behavior_cloning_epoch(run: EpochRun, rl_optimizer) -> Dict:
    """One BC gradient step toward each combo's saved actions (those of
    bc_source_epoch, else its latest run)."""
    policy = run.policy
    rl_optimizer.zero_grad()
    n_bc = 0
    epoch_loss = 0.0
    for combo_dir in run.combos:
        combo_path = Path(combo_dir)
        sys_cfg = run.sys_cfg(combo_path)
        combo_path = ensure_combo_dir_exists(combo_path, sys_cfg, run.config,
                                             verbose=run.opts.verbose,
                                             _lookup=run.opts.explicit_combos)
        results_pt = _latest_results(combo_path, run.opts.bc_source_epoch)
        if results_pt is None:
            continue
        try:
            saved_data = torch.load(results_pt, weights_only=False)
            reward_bc = saved_data['reward']
            run.epoch_rewards.append(reward_bc)
            bc_loss = _bc_loss(run, combo_path, sys_cfg, saved_data)
            if not (torch.isnan(bc_loss) or torch.isinf(bc_loss)):
                bc_loss.backward()
                epoch_loss += bc_loss.item()
                n_bc += 1
                print(f"  BC update {combo_path.name}: loss={bc_loss.item():.4f}  "
                      f"reward={reward_bc:.2f}")
        except Exception as e:
            print(f"  Warning: BC update failed for {combo_path.name}: {e}")
    if n_bc > 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        rl_optimizer.step()
        epoch_loss /= n_bc
    print(f"  BC: {n_bc}/{len(run.combos)} combos updated")
    rewards = run.epoch_rewards
    return {
        'loss': epoch_loss,
        'avg_reward': float(np.mean(rewards)) if rewards else 0.0,
        'reward_std': float(np.std(rewards)) if rewards else 0.0,
        'num_combos': len(run.combos),
        'update_policy': True,
        'bc_mode': True,
        'n_success': 0,
    }


# ── hindsight / NeuralLinear + Thompson sampling ───────────────────────────

class _HindsightCounts:
    """What the hindsight update did this epoch (for the summary line)."""

    def __init__(self, dims):
        self.steps = {d: [] for d in dims}   # |target - proposed| per head
        self.unlocks: Dict[str, int] = {}    # inverted b / c brackets unlocked
        for k in ('grown', 'bracketed', 'c_trapped', 'c_down', 'pinned', 'stranded', 'uncrossed',
                  'edges', 'offsets', 'held_off', 'gc_runs', 'gc_stepped', 'gc_guarded',
                  'widened', 'narrowed'):
            setattr(self, k, 0)


def _hindsight_settings(run: EpochRun) -> Dict:
    """The hindsight update's settings, read once per epoch."""
    b, rc = run.bandit, run.reward_cfg
    caps = b.step_caps
    b_base = float(b.hindsight_linear_step_cap)
    return {
        'barrier_target': float(rc.get('barrier_target', BARRIER_TARGET)),
        'temperature': float(b.temperature),
        'caps': caps,
        'weight_scale': b.weight_scale(run.hs_dims),
        'b_base': b_base,
        'b_max': float(b.linear_step_max if b.linear_step_max is not None else b_base),
        'b_needs_crossing': bool(b.b_direction_requires_crossing),
        # c_up_requires_b_bracket: a run pinned at its start is treated like
        # any one-sided run (a b step toward the other substituent); c steps
        # up only through the barrier trap (b bracket closed, no crossing).
        'c_needs_bracket': bool(b.c_up_requires_b_bracket),
        'c_trap': dict(base_step=caps['quadratic_step_cap'], max_step=float(b.quadratic_step_max),
                       trap_width=float(b.barrier_trap_width)),
        'factor_max': float(b.explore_factor_max),
        'persistent': int(b.explore_after_failures),
        'cols': [OFFSET_DIMS[d] for d in run.hs_dims],
    }


def _hindsight_targets_for_run(run: EpochRun, s: Dict, n: _HindsightCounts, combo_path_u: Path,
                               epoch_dir_u: Path, vdata_u, saved_data: Dict, saved_actions,
                               applied,
                               combined_weight):
    """Per-pair state updates for one run, and the shared model's targets and
    weights. Returns (update_targets, combined_weight, hs_t, hs_v)."""
    bandit_state, opts, device, hs_dims = run.bandit_state, run.opts, run.device, run.hs_dims
    hs_t, hs_v, hs_dir = hindsight_targets(
        vdata_u.edge_index, applied, saved_data,
        barrier_target=s['barrier_target'],
        temperature=s['temperature'], return_directional=True, **s['caps'],
    )
    labels = combo_node_labels(combo_path_u) if run.use_offsets else None
    sys_name = run.sys_name(combo_path_u)
    # A run pinned at its start says the barrier, not b, held lambda: no b
    # target (shared or per-pair) and its b streak/bracket are left alone; it
    # steps c up. A stranded run (neither endpoint) steps c down.
    pinned, stranded = lambda_states(vdata_u.edge_index, saved_data)
    fwd = vdata_u.edge_index[0] < vdata_u.edge_index[1]
    n.pinned += int((pinned & fwd).sum())
    n.stranded += int((stranded & fwd).sum())
    # b_direction_requires_crossing: any run that never crossed (pinned or
    # not) gives no b direction and steps c up, like a pinned run.
    if s['b_needs_crossing']:
        uncrossed = uncrossed_pairs(vdata_u.edge_index, saved_data)
        n.uncrossed += int((uncrossed & ~pinned & fwd).sum())
        pinned = pinned | uncrossed
    if s['c_needs_bracket']:
        pinned = torch.zeros_like(pinned)
    if pinned.any():
        hs_v[pinned, 0] = float('inf')
        hs_dir[pinned, 0] = False
    # bandit.group_correction: a group run whose end terms hold an endpoint
    # (or a whole site) off steps those pairs' x / s down for the group (or a
    # corrected site that overshot is stepped back). The run then says
    # nothing per pair about that site: its edges skip the per-pair rules
    # below, and the shared model learns the group's corrected x / s.
    gc_t = gc_v = gc_mask = None
    if opts.update_policy and labels and run.gc_cfg is not None:
        gc_res = update_group(bandit_state, sys_name, combo_path_u.name, saved_data,
                              labels, applied, run.gc_cfg, run.epoch, str(epoch_dir_u))
        if gc_res['flagged'] or gc_res['guarded']:
            gc_t, gc_v, gc_mask = correction_targets(vdata_u.edge_index, gc_res, run.gc_cfg)
            n.gc_runs += 1
            n.gc_stepped += int(gc_res['stepped'])
            n.gc_guarded += len(gc_res['guarded'])
    sel = None if gc_mask is None or not gc_mask.any() else (~gc_mask).nonzero(as_tuple=True)[0]

    def _p(t):
        return t if sel is None else t[sel]
    if opts.update_policy and labels:
        # Per-pair offsets get grown directional b and (barrier-trapped) c
        # steps; the shared model below keeps the base steps.
        ei_p = vdata_u.edge_index if sel is None else vdata_u.edge_index[:, sel]
        pin_p, str_p, dir_p = _p(pinned), _p(stranded), _p(hs_dir)
        hs_t_pair, hs_v_pair = _p(hs_t).clone(), _p(hs_v).clone()
        hs_dir_pair = dir_p.clone()
        if 'linear' in hs_dims and s['b_max'] >= s['b_base']:
            g, br = grow_directional_b_steps(
                bandit_state, sys_name, labels, ei_p, hs_t_pair, dir_p, applied,
                base_step=s['b_base'], max_step=s['b_max'], var=hs_v_pair,
                skip=pin_p | str_p, counts=n.unlocks)
            n.grown += g
            n.bracketed += br
            if sel is not None:
                hs_dir[sel] = dir_p
        if 'quadratic' in hs_dims:
            up, down = barrier_trapped_c_targets(
                bandit_state, sys_name, labels, ei_p, hs_t_pair, hs_v_pair, hs_dir_pair, applied,
                pinned=pin_p, stranded=str_p, counts=n.unlocks, **s['c_trap'])
            n.c_trapped += up
            n.c_down += down
        n.offsets += record_pair_targets(
            bandit_state, sys_name, labels, ei_p, hs_t_pair, hs_v_pair,
            history=int(run.bandit.offset_history), dims=hs_dims, directional=hs_dir_pair)
        short = run.is_short_run(combo_path_u, epoch_dir_u)
        short_min, short_frac = run.short_test(sys_name)
        stage_cfg = run.stage_cfg(sys_name)
        record_pair_outcomes(bandit_state, sys_name, labels, ei_p, saved_data,
                             combo=str(combo_path_u),
                             stage=(opts.stage_of_system or {}).get(sys_name),
                             short=short, short_min_transitions=short_min,
                             short_min_fraction=short_frac,
                             confirm=bool(stage_cfg.get('confirm_full_length', True)),
                             epoch=run.epoch,
                             partial_min_transitions=int(
                                 stage_cfg.get('partial_min_transitions', 3)),
                             partial_runs=int(stage_cfg.get('partial_runs_to_resolve', 0)))
        if run.offset_sampling:
            wd, nr = update_exploration(
                bandit_state, sys_name, labels, ei_p, hs_v_pair, saved_data,
                factor_max=s['factor_max'], dims=hs_dims, persistent_failures=s['persistent'],
                min_transitions=short_min if short else SUCCESS_MIN_TRANSITIONS,
                min_fraction=short_frac if short else SUCCESS_MIN_FRACTION)
            n.widened += wd
            n.narrowed += nr
    if labels and 'quadratic' in hs_dims:
        barrier_trapped_c_targets(
            bandit_state, sys_name, labels, vdata_u.edge_index, hs_t, hs_v, hs_dir.clone(),
            applied,
            pinned=pinned, stranded=stranded, **{**s['c_trap'], 'grow': False})
    if sel is not None:
        # flagged / guarded site: only the group's corrected x / s
        hs_v[gc_mask] = float('inf')
        fin = torch.isfinite(gc_v)
        hs_t[fin] = gc_t[fin]
        hs_v[fin] = gc_v[fin]
    hs_t_dev = hs_t.to(saved_actions.dtype).to(device)
    finite = torch.isfinite(hs_v).to(device)
    update_targets = saved_actions.clone()
    combined_weight = combined_weight.expand_as(saved_actions).clone()
    cols = s['cols']
    w = preact_weights(run.policy, hs_t[:, cols], hs_v[:, cols], dims=cols).to(device)
    w = w * torch.tensor([s['weight_scale'][d] for d in hs_dims], dtype=w.dtype, device=w.device)
    step = hs_t - saved_actions.detach().cpu().double()
    for k, (d, col) in enumerate(zip(hs_dims, cols)):
        update_targets[:, col] = torch.where(finite[:, col], hs_t_dev[:, col],
                                             saved_actions[:, col])
        combined_weight[:, col] = w[:, k]
        m = torch.isfinite(hs_v[:, col])
        n.steps[d].extend(step[m, col].abs().tolist())
    n.edges += int(vdata_u.edge_index.size(1))
    n.held_off += held_off_counts(vdata_u.edge_index, saved_data)
    return update_targets, combined_weight, hs_t, hs_v


def _regression_update(run: EpochRun, items: list) -> None:
    """bandit.shared_update: regression -- gradient steps on the full network
    toward the epoch's (and recent epochs') hindsight targets, with
    pretraining replay."""
    config = run.config
    reg_cfg = regression_config(run.bandit.raw)
    reg_cfg['env_input'] = config.get('unimol', {}).get('env_input', 'dual')
    reg_cfg['ligand_other_sites'] = bool(config.get('unimol', {}).get('ligand_other_sites', False))
    reg_cfg['site_size_input'] = bool(config.get('unimol', {}).get('site_size_input', False))
    ck = Path(config.get('output', {}).get('base_dir', 'checkpoints'))
    buf_path = ck / 'regression_online_buffer.pt'
    buf = torch.load(buf_path, weights_only=False) if buf_path.exists() else []
    buf.append(items)
    buf = buf[-int(reg_cfg['online_history']):]
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(buf, buf_path)
    if reg_cfg.get('replay_cache') is None:
        reg_cfg['replay_cache'] = str(ck / 'regression_replay.pt')
    cache = run.opts.replay_cache
    replay = cache.get(reg_cfg['replay_cache'])
    if replay is None and float(reg_cfg['replay_weight']) > 0:
        replay = build_replay_set(reg_cfg, seed=0)
        cache[reg_cfg['replay_cache']] = replay
    online_all = [it for ep in buf for it in ep]
    rs = regression_step(run.policy, online_all, replay or [], reg_cfg, device=run.device,
                         seed=run.epoch)
    msg = (f"  Hindsight regression (full network): {rs.get('steps', 0)} steps on "
           f"{len(online_all)} online graphs ({len(buf)} epoch(s)) + "
           f"{len(replay or [])} replay; loss online "
           f"{rs.get('online_before', float('nan')):.3f} -> "
           f"{rs.get('online_after', float('nan')):.3f}, replay "
           f"{rs.get('replay_before', float('nan')):.3f} -> "
           f"{rs.get('replay_after', float('nan')):.3f}")
    print(msg)
    run.detail_lines.append(msg)
    run.epoch_info['regression'] = dict(rs)


def hindsight_update(run: EpochRun, pair_stats: Dict[str, Dict[str, int]]) -> int:
    """NeuralLinear + TS update from every crash-free run of the epoch.
    Unlike REINFORCE there is no baseline, success floor or failure batch:
    compute_pair_reward's -1 for an unvisited pair already makes a poorly
    sampled run a real, low-weight observation. Returns n_success (runs with
    >= 1 site meeting the transition floor, as for REINFORCE)."""
    policy, opts, b, rc = run.policy, run.opts, run.bandit, run.reward_cfg
    no_transition_weight = float(rc.get('no_transition_weight', 0.2))
    awr_temperature = float(b.awr_temperature)
    pair_min_trans = int(rc.get('floor_mask_min_transitions',
                                rc.get('min_transitions_per_site', 3)))
    pair_balance_threshold = float(rc.get('pair_balance_threshold', 0.1))
    s = _hindsight_settings(run)
    n = _HindsightCounts(run.hs_dims)
    n_updated = n_success = 0
    r_relevant_all = []
    reg_items = []
    for (combo_path_u, epoch_dir_u, _reward) in run.deferred_updates:
        try:
            (vdata_u, saved_data, saved_actions, r_pair, edge_weights_u,
             trans_vals) = prepare_success_combo_reward(
                combo_path_u, epoch_dir_u, run.combo_system_map, run.config, run.device,
                no_transition_weight)
            _add(pair_stats['total'], compute_pair_success_stats(
                vdata_u.edge_index, saved_data.get('ddg_pairs', {}),
                saved_data.get('populations', []),
                total_transitions=sum(trans_vals), min_transitions=pair_min_trans,
                balance_threshold=pair_balance_threshold))
            sys_u = run.sys_name(combo_path_u)
            if run.is_short_run(combo_path_u, epoch_dir_u):
                _add(pair_stats['short'], pair_test_counts(vdata_u.edge_index, saved_data,
                                                           *run.short_test(sys_u)))
            else:
                _add(pair_stats['full'], pair_test_counts(vdata_u.edge_index, saved_data,
                                                          SUCCESS_MIN_TRANSITIONS,
                                                          SUCCESS_MIN_FRACTION))
            if vdata_u.edge_type is None:
                # Online graphs (one edge per pair): every bias type is
                # meaningful for every edge, each with its own AWR weight.
                r_relevant = r_pair                                  # [E, 4]
                edge_weights_bcast = edge_weights_u.unsqueeze(-1)    # [E, 1]
            else:
                # Routed graphs (pretraining's 4 edges per pair): only the
                # edge's own bias type is meaningful.
                r_relevant = r_pair.gather(1, (vdata_u.edge_type // 2).unsqueeze(-1)).squeeze(-1)
                edge_weights_bcast = edge_weights_u
            # AWR weight exp(r / beta), clamp <= 20: good outcomes pull the
            # posterior toward the observed action much harder than poor ones.
            combined_weight = (edge_weights_bcast
                               * torch.exp(r_relevant / awr_temperature).clamp(max=20.0))
            update_targets = saved_actions
            applied = None
            if b.use_hindsight and vdata_u.edge_type is None:
                # Heads in hindsight_dims regress toward the biases this run
                # measured it should have used, weighted by their measurement
                # noise; other heads keep the AWR target.
                applied = read_applied_biases(epoch_dir_u)
                if applied is not None:
                    update_targets, combined_weight, hs_t, hs_v = _hindsight_targets_for_run(
                        run, s, n, combo_path_u, epoch_dir_u, vdata_u, saved_data, saved_actions,
                        applied, combined_weight)
            if opts.update_policy and opts.update_shared and b.shared_regression:
                if vdata_u.edge_type is None and applied is not None:
                    item = online_item(vdata_u.x, vdata_u.edge_index, hs_t, hs_v)
                    if item is not None:
                        reg_items.append(item)
            elif opts.update_policy and opts.update_shared:
                policy.update_bayesian_posteriors(vdata_u.x, vdata_u.edge_index, vdata_u.edge_type,
                                                  update_targets, weights=combined_weight)
            n_updated += 1
            if trans_vals and max(trans_vals) >= pair_min_trans:
                n_success += 1
            r_relevant_all.append(r_relevant)
        except Exception as e:
            print(f"  Warning: NeuralLinear+TS update failed for {combo_path_u.name}: {e}")

    if r_relevant_all:
        all_r = torch.cat(r_relevant_all)
        n_obs = ', '.join(f"{name}:{int(policy.edge_mlp.mlps[d].bayesian_head.n_obs.item())}"
                          for d, name in enumerate(BIAS_DIMS))
        msg = (f"  NeuralLinear+TS posterior update: {n_updated}/{len(run.deferred_updates)} "
               f"combos ({n_success} meeting transition floor), "
               f"{all_r.shape[0]} edges, reward mean={all_r.mean().item():.3f} "
               f"std={all_r.std().item():.3f}, n_obs per head=[{n_obs}]")
        print(msg)
        run.detail_lines.append(msg)
    if opts.update_policy and opts.update_shared and b.shared_regression:
        _regression_update(run, reg_items)
    if n.edges:
        def _med(v):
            return f"{float(np.median(v)):.2f} (n={len(v)})" if v else "n/a"
        msg = ("  Hindsight targets: median |step| "
               + ', '.join(f"{d} {_med(v)}" for d, v in n.steps.items())
               + f"; held-off endpoints {n.held_off}; grown b steps {n.grown}"
               + f"; bracketed b {n.bracketed}"
               + f"; barrier-trapped c {n.c_trapped}"
               + f"; c stepped down {n.c_down}"
               + f"; pinned runs {n.pinned}, stranded runs {n.stranded}"
               + f", other uncrossed runs {n.uncrossed}"
               + f"; pair-offset observations {n.offsets}"
               + (f"; exploration widened {n.widened}, narrowed {n.narrowed}"
                  if run.offset_sampling else "")
               + (f"; group corrections {n.gc_runs} run(s), stepped {n.gc_stepped}, "
                  f"guarded sites {n.gc_guarded}" if run.gc_cfg is not None else ""))
        print(msg)
        run.detail_lines.append(msg)
        run.epoch_info['hindsight'] = {
            'median_step': {d: (float(np.median(v)) if v else None) for d, v in n.steps.items()},
            'held_off': n.held_off, 'b_grown': n.grown, 'b_bisected': n.bracketed,
            'b_unlocked': n.unlocks.get('b_unlocked', 0), 'c_up': n.c_trapped,
            'c_down': n.c_down, 'c_unlocked': n.unlocks.get('c_unlocked', 0),
            'pinned': n.pinned, 'stranded': n.stranded,
            'group_corrected': n.gc_runs}
    return n_success


# ── REINFORCE ───────────────────────────────────────────────────────────────

def sample_replay_items(combos: List[str], current_epoch: int, k_total: int,
                        max_lookback: Optional[int], rng: np.random.RandomState) -> List[tuple]:
    """Up to *k_total* (combo_path, run_dir) pairs from the combos' own
    earlier epochs (``run_{epoch:03d}/epoch_results.pt``, before
    ``current_epoch`` and within ``max_lookback`` epochs of it), for replay
    AWR: more (context, action, reward) signal per simulation."""
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


def compute_replay_awr_loss(policy, combo_path: Path, epoch_dir: Path,
                            combo_system_map: Dict[str, Dict],
                            config: Dict, device: str, no_transition_weight: float,
                            awr_temperature: float) -> Optional[torch.Tensor]:
    """AWR loss for one replayed (combo, past epoch) run: the confidence-
    weighted Gaussian NLL of its saved action, weighted by
    exp(reward / temperature) clamped at 20 (as in AWR pretraining). None if
    the run cannot be loaded or scored."""
    try:
        vdata, saved_data, saved_actions, r_pair, edge_weights, _ = prepare_success_combo_reward(
            combo_path, epoch_dir, combo_system_map, config, device, no_transition_weight)
        mean, log_std = policy._forward_edges(vdata.x, vdata.edge_index, vdata.edge_type)
        logp = torch.distributions.Normal(mean, torch.exp(log_std)).log_prob(
            saved_actions)  # [E, D]
        active_mask = saved_actions.abs() > 1e-8
        ddg_pairs = saved_data.get('ddg_pairs', {})
        populations = saved_data.get('populations', [])
        pairwise_weight = None
        if ddg_pairs and populations:
            try:
                nsubs_per_site = _nsubs_from_site_index(vdata)
                w = compute_pairwise_confidence_weights(
                    {'ddg_pairs': ddg_pairs, 'populations': populations},
                    nsubs_per_site=nsubs_per_site, pairs=build_directed_pairs(nsubs_per_site))
                pairwise_weight = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(-1)
            except Exception:
                pairwise_weight = None
        if pairwise_weight is None:
            pairwise_weight = edge_weights.unsqueeze(-1)
        combined_weight = pairwise_weight * torch.exp(r_pair / awr_temperature).clamp(max=20.0)
        weighted_logp = logp[active_mask] * combined_weight[active_mask]
        if weighted_logp.numel() == 0:
            return None
        return -weighted_logp.mean()
    except Exception:
        return None


def _reinforce_batches(run: EpochRun, pair_stats: Dict[str, int], min_trans, failure_advantage,
                       pair_balance_threshold: float, no_transition_weight: float):
    """(success batch, failure batch, partial count) from the epoch's runs.
    A run enters the success batch if any site meets the transition floor;
    failing sites' pairs get -1 from compute_pair_reward."""
    raw_batch, failure_batch, n_partial = [], [], 0
    for (combo_path_u, epoch_dir_u, _reward) in run.deferred_updates:
        try:
            vdata_u, saved_data = load_run(combo_path_u, epoch_dir_u, run.combo_system_map,
                                           run.config,
                                           run.device)
            trans_vals = _trans_values(saved_data)
            _add(pair_stats, compute_pair_success_stats(
                vdata_u.edge_index, saved_data.get('ddg_pairs', {}),
                saved_data.get('populations', []),
                total_transitions=sum(trans_vals), min_transitions=min_trans,
                balance_threshold=pair_balance_threshold))
            if (max(trans_vals) if trans_vals else 0) < min_trans:
                if failure_advantage is not None:
                    failure_batch.append((combo_path_u, vdata_u,
                                          saved_data['actions'].to(run.device),
                                          saved_data.get('populations', [])))
                continue
            if trans_vals and min(trans_vals) < min_trans:
                n_partial += 1
            saved_actions = saved_data['actions'].to(run.device)
            r_pair = pair_reward(vdata_u.edge_index, saved_data, run.config.get('reward', {}),
                                 run.device)
            edge_weights_u = build_edge_weights(vdata_u.edge_index,
                                                saved_data.get('ddg_pairs', {}),
                                                no_transition_weight, run.device)
            logp_old = saved_data['logp'].to(run.device).detach()
            raw_batch.append((combo_path_u, vdata_u, saved_actions, r_pair, logp_old,
                              edge_weights_u))
        except Exception as e:
            print(f"  Warning: Could not prepare update for {combo_path_u.name}: {e}")
    return raw_batch, failure_batch, n_partial


def _reward_baseline(run: EpochRun, raw_batch, n_partial: int, no_transition_weight: float):
    """(r_mean, r_std, r_mean_batch): the REINFORCE baseline -- the EMA of
    the edge-weighted batch reward (once it has enough runs), else the batch
    itself; the configured global baseline with no successful run."""
    rc, device, ema_baseline = run.reward_cfg, run.device, run.opts.ema_baseline
    if not raw_batch:
        return (torch.tensor(rc.get('global_baseline', [0.0, -1.0, -1.0, -1.0]),
                             dtype=torch.float32,
                             device=device), torch.ones(4, device=device), None)
    ema_alpha = rc.get('ema_alpha', 0.2)
    ema_min_batch = int(rc.get('ema_min_batch', 5))
    n_success = len(raw_batch)
    all_r = torch.cat([item[3] for item in raw_batch])
    all_w = torch.cat([item[5] for item in raw_batch])
    # Down-weighted (no-crossing) edges count proportionally less.
    w_sum = all_w.sum().clamp(min=1e-6)
    r_mean_batch = (all_r * all_w.unsqueeze(-1)).sum(dim=0) / w_sum
    r_var_batch = (all_w.unsqueeze(-1) * (all_r - r_mean_batch) ** 2).sum(dim=0) / w_sum
    r_std_batch = r_var_batch.clamp(min=1e-12).sqrt().clamp(min=1e-6)
    n_downweighted = int((all_w < 1.0 - 1e-6).sum().item())
    stats_str = ', '.join(f'{name}: {r_mean_batch[i]:.3f}±{r_std_batch[i]:.3f}'
                          for i, name in enumerate(BIAS_DIMS))
    if ema_baseline is not None:
        if len(ema_baseline) == 0:
            ema_baseline['mean'] = r_mean_batch.detach().clone().cpu()
            ema_baseline['std'] = r_std_batch.detach().clone().cpu()
            ema_baseline['n_updates'] = 1
        elif n_success >= ema_min_batch:
            ema_baseline['mean'] = (ema_alpha * r_mean_batch.detach().cpu()
                                    + (1.0 - ema_alpha) * ema_baseline['mean'])
            ema_baseline['std'] = (ema_alpha * r_std_batch.detach().cpu()
                                   + (1.0 - ema_alpha) * ema_baseline['std'])
            ema_baseline['n_updates'] += 1
    if ema_baseline and 'mean' in ema_baseline:
        r_mean, r_std = ema_baseline['mean'].to(device), ema_baseline['std'].to(device)
        baseline_src = f"EMA(n_upd={ema_baseline['n_updates']})"
    else:
        r_mean, r_std = r_mean_batch, r_std_batch
        baseline_src = "batch"
    msg = (f"  Batch reward stats (n={n_success} combos"
           f"{f', {n_partial} partial' if n_partial else ''}, "
           f"{all_r.shape[0]} edges [{n_downweighted} down-weighted "
           f"x{no_transition_weight}], baseline={baseline_src}): {stats_str}")
    print(msg)
    run.detail_lines.append(msg)
    return r_mean, r_std, r_mean_batch


def _bc_auxiliary_loss(run: EpochRun, combo_path_u: Path, vdata_u,
                       target_norm) -> Optional[torch.Tensor]:
    """MSE toward bc_source_epoch's saved actions (bc_auxiliary_weight)."""
    bc_pt = combo_path_u / f"run_{run.opts.bc_source_epoch:03d}" / 'epoch_results.pt'
    if not bc_pt.exists():
        return None
    try:
        bc_target = torch.load(bc_pt, weights_only=False)['actions'].to(run.device)
        bc_mean, _ = run.policy._forward_edges(vdata_u.x, vdata_u.edge_index, vdata_u.edge_type)
        if bc_target.dim() == 1:
            bc_target = bc_target.unsqueeze(-1).expand_as(bc_mean)
        mask = _routed_mask(run.policy, vdata_u.edge_type, bc_mean.shape, run.device)
        return F.mse_loss((bc_mean / target_norm)[mask], (bc_target.detach() / target_norm)[mask])
    except Exception:
        return None


def reinforce_update(run: EpochRun, rl_optimizer, target_norm, pair_stats: Dict[str, int]) -> int:
    """One REINFORCE step: the success batch, the down-weighted failure batch
    and history replay share a single zero_grad / backward / step (two Adam
    steps would move the policy about twice as far). Returns n_success."""
    policy, opts, rc, device = run.policy, run.opts, run.reward_cfg, run.device
    lambda_e = rc.get('lambda_entropy', 0.0)
    min_trans = rc.get('floor_mask_min_transitions', rc.get('min_transitions_per_site', 3))
    failure_advantage = rc.get('failure_advantage', None)
    pair_balance_threshold = float(rc.get('pair_balance_threshold', 0.1))
    # Pairs with no observed DDG train with this small, non-zero weight: a
    # pair out-competed at its site gets a muted signal, not full blame.
    no_transition_weight = float(rc.get('no_transition_weight', 0.2))
    raw_batch, failure_batch, n_partial = _reinforce_batches(
        run, pair_stats, min_trans, failure_advantage, pair_balance_threshold,
        no_transition_weight)
    success_floor = rc.get('success_advantage_floor', 0.0)
    r_mean, r_std, r_mean_batch = _reward_baseline(run, raw_batch, n_partial, no_transition_weight)
    n_success = len(raw_batch)

    baseline_net, baseline_optimizer = opts.baseline_net, opts.baseline_optimizer
    use_context_baseline = baseline_net is not None and baseline_optimizer is not None
    if use_context_baseline:
        # Diagnostics only: the context baseline's weighted squared error
        # next to that of the pooled batch mean it replaces.
        baseline_sq_err = torch.zeros(4, device=device)
        naive_sq_err = torch.zeros(4, device=device)
        baseline_w = torch.zeros(4, device=device)
    if not ((raw_batch or failure_batch) and opts.update_policy):
        return n_success

    rl_optimizer.zero_grad()
    if use_context_baseline:
        baseline_optimizer.zero_grad()
    for (combo_path_u, vdata_u, saved_actions, r_pair, logp_old, edge_weights_u) in raw_batch:
        try:
            if use_context_baseline:
                # "Typical reward for a context like this edge/system": the
                # trunk features are detached so the baseline's loss never
                # reaches the policy; r_std stays the batch/EMA scale.
                sys_idx = (opts.system_to_idx or {}).get(
                    run.sys_name(combo_path_u, '__unknown__'), 0)
                sys_ids = torch.full((r_pair.size(0),), sys_idx, dtype=torch.long, device=device)
                z_ctx = policy.forward_features(vdata_u.x, vdata_u.edge_index,
                                                vdata_u.edge_type).detach()
                baseline_pred = baseline_net(z_ctx, sys_ids)          # [E, D]
                a_pair = (r_pair - baseline_pred.detach()) / r_std
            else:
                a_pair = (r_pair - r_mean) / r_std
            if success_floor is not None:
                a_pair = a_pair.clamp(min=float(success_floor))
            logp_per_dim, log_std_saved = policy.evaluate_logp(
                vdata_u.x, vdata_u.edge_index, vdata_u.edge_type, saved_actions)
            # Each dimension's log prob times its own advantage; edges
            # weighted by their confidence (edge_weights_u).
            per_edge = (logp_per_dim * a_pair).sum(dim=-1)                # [E]
            w_sum = edge_weights_u.sum().clamp(min=1e-6)
            policy_loss = -(per_edge * edge_weights_u).sum() / w_sum / len(raw_batch)
            if lambda_e > 0:
                policy_loss = policy_loss - lambda_e * log_std_saved.mean() / len(raw_batch)
            if opts.bc_auxiliary_weight > 0.0 and opts.bc_source_epoch is not None:
                bc_aux = _bc_auxiliary_loss(run, combo_path_u, vdata_u, target_norm)
                if bc_aux is not None:
                    policy_loss = policy_loss + opts.bc_auxiliary_weight * bc_aux / len(raw_batch)
            if not (torch.isnan(policy_loss) or torch.isinf(policy_loss)):
                policy_loss.backward()
                run.epoch_loss += policy_loss.item() * len(raw_batch)
            if use_context_baseline:
                # Train the baseline toward the observed reward (weighted MSE).
                b_err = (baseline_pred - r_pair.detach()) ** 2              # [E, D]
                b_w = edge_weights_u.unsqueeze(-1).expand_as(b_err)
                baseline_loss = (b_err * b_w).sum() / b_w.sum().clamp(min=1e-6) / len(raw_batch)
                if not (torch.isnan(baseline_loss) or torch.isinf(baseline_loss)):
                    baseline_loss.backward()
                with torch.no_grad():
                    baseline_sq_err += (b_err.detach() * b_w).sum(dim=0)
                    naive_sq_err += (((r_mean_batch.unsqueeze(0) - r_pair) ** 2) * b_w).sum(dim=0)
                    baseline_w += b_w.sum(dim=0)
        except Exception as e:
            print(f"  Warning: REINFORCE update failed for {combo_path_u.name}: {e}")

    # Failure batch: heavily down-weighted (failure_loss_weight) and
    # floor-clamped, so a below-floor run never outweighs the strongest
    # success-side correction.
    if failure_batch and failure_advantage is not None:
        beta_fail = float(rc.get('failure_loss_weight', 0.05))
        n_ref = max(len(raw_batch), 1)
        for (combo_path_f, vdata_f, saved_actions_f, fail_pops) in failure_batch:
            try:
                a_fail = compute_failure_reward(vdata_f.edge_index, fail_pops,
                                                failure_advantage=float(failure_advantage)
                                                ).to(device)
                if success_floor is not None:
                    a_fail = a_fail.clamp(min=float(success_floor))
                logp_fail, _ = policy.evaluate_logp(vdata_f.x, vdata_f.edge_index,
                                                    vdata_f.edge_type,
                                                    saved_actions_f)
                fail_loss = beta_fail * -(logp_fail * a_fail).sum(dim=-1).mean() / n_ref
                if not (torch.isnan(fail_loss) or torch.isinf(fail_loss)):
                    fail_loss.backward()
            except Exception as e:
                print(f"  Warning: failure gradient failed for {combo_path_f.name}: {e}")
        msg = (f"  Failure gradient: {len(failure_batch)} below-floor combos, "
               f"A={failure_advantage:.1f} (loss_weight={beta_fail:.3f}, "
               "floor-clamped, merged into success step)")
        print(msg)
        run.detail_lines.append(msg)

    replay_cfg = opts.history_replay_cfg
    if replay_cfg and replay_cfg.get('enabled', False):
        k_replay = int(replay_cfg.get('k_per_epoch', 0))
        awr_temp = float(replay_cfg.get('awr_temperature', 1.0))
        items = sample_replay_items(run.combos, run.epoch, k_replay,
                                    replay_cfg.get('max_lookback', None),
                                    np.random.RandomState(replay_cfg.get('seed', 0) + run.epoch))
        n_used = 0
        for (r_combo_path, r_epoch_dir) in items:
            r_loss = compute_replay_awr_loss(policy, r_combo_path, r_epoch_dir,
                                             run.combo_system_map,
                                             run.config, device, no_transition_weight, awr_temp)
            if r_loss is not None:
                (r_loss / len(items)).backward()
                n_used += 1
        if items or k_replay > 0:
            msg = (f"  History replay (AWR): {n_used}/{len(items)} "
                   f"sampled observations used (k_per_epoch={k_replay}, "
                   f"awr_temperature={awr_temp})")
            print(msg)
            run.detail_lines.append(msg)

    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    rl_optimizer.step()
    if use_context_baseline:
        torch.nn.utils.clip_grad_norm_(baseline_net.parameters(), max_norm=1.0)
        baseline_optimizer.step()
        if float(baseline_w.sum()) > 0:
            bl_mse = baseline_sq_err / baseline_w.clamp(min=1e-6)
            naive_mse = naive_sq_err / baseline_w.clamp(min=1e-6)
            msg = ("  Context baseline MSE vs naive pooled-mean MSE (lower is better): "
                   + ", ".join(f"{name}: {bl_mse[i]:.4f} vs {naive_mse[i]:.4f}"
                               for i, name in enumerate(BIAS_DIMS)))
            print(msg)
            run.detail_lines.append(msg)
    return n_success
