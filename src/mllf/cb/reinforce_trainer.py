"""Run one online training epoch for the Uni-Mol bias policy.

``train_epoch`` proposes biases for each combo, simulates them
(``mllf.cb.epoch_runs``), then updates the policy from the finished runs
(``mllf.cb.epoch_updates``): REINFORCE, or Bayesian heads with Thompson sampling
with per-pair offsets and a posterior or hindsight-regression update of the
shared model (``bandit.algorithm`` / ``bandit.shared_update``). Behavioral
cloning toward saved actions replaces the simulations in ``bc_mode``.
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mllf.cb.epoch_runs import EpochOptions, EpochRun
from mllf.cb.epoch_updates import (
    bc_target_norm,
    behavior_cloning_epoch,
    empty_pair_stats,
    hindsight_update,
    reinforce_update,
)


def train_epoch(policy, rl_optimizer, combos: List[str], combo_system_map: Dict[str, Dict],
                epoch: int, config: Dict, device: str = 'cpu',
                options: Optional[EpochOptions] = None, **kwargs) -> Dict:
    """Run one training (or, with ``update_policy=False``, validation) epoch
    over ``combos`` from every system.

    ``combo_system_map`` maps each combo path to its ``systems:`` entry (it
    gains entries for combo dirs created under another path). How the epoch
    trains is given by ``options`` or the same fields as keyword arguments
    (see ``EpochOptions``); ``bandit_state`` and ``ema_baseline`` there are
    updated in place.

    Returns the epoch's stats: loss, reward mean/std, n_success, the
    per-pair success counts (all runs, short and full-length runs), the
    summary-log detail lines, permanently failed combos and ``run_dirs``
    ((system, run dir, ok) for every simulation that finished, for the cost
    ledger).
    """
    if options is not None and kwargs:
        raise TypeError("pass either options or keyword arguments, not both")
    opts = options if options is not None else EpochOptions(**kwargs)
    policy.train()
    run = EpochRun(policy, combos, combo_system_map, epoch, config, device, opts)
    target_norm = bc_target_norm(run)
    if opts.bc_mode and opts.update_policy:
        return behavior_cloning_epoch(run, rl_optimizer)

    run.prepare_and_simulate()
    pair_stats = {
        'total': empty_pair_stats(),
        'short': empty_pair_stats(),
        'full': empty_pair_stats()}
    if run.bandit.algorithm == 'hindsight':
        n_success = hindsight_update(run, pair_stats)
    elif run.deferred_updates:
        n_success = reinforce_update(run, rl_optimizer, target_norm, pair_stats['total'])
    else:
        n_success = 0

    rewards = run.epoch_rewards
    return {
        'loss': run.epoch_loss / len(rewards) if rewards else 0.0,
        'avg_reward': float(np.mean(rewards)) if rewards else 0.0,
        'reward_std': float(np.std(rewards)) if rewards else 0.0,
        'num_combos': len(combos),
        'update_policy': opts.update_policy,
        'bc_mode': False,
        'n_success': n_success,
        'permanently_failed': run.permanently_failed,
        'pair_success': pair_stats['total'],
        'pair_success_short': pair_stats['short'],
        'pair_success_full': pair_stats['full'],
        'detail_lines': run.detail_lines,
        **run.epoch_info,
        'run_dirs': [(run.sys_name(cp), str(ed), True) for cp, ed, _ in run.deferred_updates]
        + run.failed_run_dirs,
    }


def append_summary(summary_path: Optional[Path], text: str) -> None:
    """Append a line (or block) to the small per-epoch summary log
    (``epoch_summary.log``), which stays easy to skim where the job's stdout
    grows to hundreds of MB. No-op if summary_path is None or the write fails.
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

    def pct(n):
        return 100.0 * n / total
    return (
        f"  Pair success ({label}): crossed {ps['crossed']}/{total} ({pct(ps['crossed']):.1f}%), "
        f"balanced {ps['balanced']}/{total} ({pct(ps['balanced']):.1f}%), "
        f"both {ps['both']}/{total} ({pct(ps['both']):.1f}%)"
    )
