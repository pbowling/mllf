"""Run- and pair-level rewards for pretraining's advantage weighting."""
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mllf.cb.hindsight import (
    SUCCESS_MIN_FRACTION,
    SUCCESS_MIN_TRANSITIONS,
    pair_succeeded,
    site_nodes,
)
from mllf.cb.workflow_utils import (
    BARRIER_TARGET,
    BARRIER_WIDTH,
    END_WIDTH,
    SKEW_WIDTH,
    compute_pair_reward,
)


log = logging.getLogger(__name__)


# ── Run length ───────────────────────────────────────────────────────────────
# Pretraining mixes ALF's short runs (~0.1 ns) with 1 ns production runs. A
# short run has had a tenth of the time to cross and to balance, so it is
# judged the way online training judges its short exploratory runs: by the
# looser success test of the curriculum stage keys explore_min_transitions /
# explore_min_fraction (see training/online_A_well_solv_vac.yaml), with
# count-based reward terms scaled by its length. Set in the config's
# ``reward`` section (``full_length_ns``, ``explore_min_transitions``,
# ``explore_min_fraction``); runs collected without 'simulated_ns' count as
# full length (``collect_pretraining_data --backfill-barrier`` adds it).
FULL_LENGTH_NS = 1.0


EXPLORE_MIN_TRANSITIONS = 3


EXPLORE_MIN_FRACTION = 0.0


# Keyword arguments of compute_reward_from_sim_results a config's reward section may set.
REWARD_KWARGS = {
    'w_P', 'w_T', 'w_U', 'gamma', 'P_baseline', 'T_baseline',
    'min_transitions_per_site', 'min_coverage_ratio', 'entropy_bonus',
    'concentration_penalty_threshold', 'full_length_ns', 'explore_min_transitions',
}


def reward_kwargs(reward_config: Optional[Dict]) -> Dict:
    return {k: v for k, v in (reward_config or {}).items() if k in REWARD_KWARGS}


def run_length(sim_results: Dict, full_length_ns: float = FULL_LENGTH_NS) -> Tuple[float, bool]:
    """(length / full length, short) of one run from its 'simulated_ns'.

    Short means under half the full length, online's test for an exploratory
    run (``train_epoch``'s ``_is_short_run``). Without a usable length the
    run counts as full: (1.0, False).
    """
    try:
        ns = float(sim_results.get('simulated_ns'))
    except (TypeError, ValueError):
        return 1.0, False
    if not math.isfinite(ns) or ns <= 0 or full_length_ns <= 0:
        return 1.0, False
    return ns / full_length_ns, ns < 0.5 * full_length_ns


def compute_reward_from_sim_results(
    sim_results: Dict,
    num_sites: int,
    nsubs_per_site: List[int],
    w_P: float = 0.5,
    w_T: float = 0.75,
    w_U: float = 0.3,
    gamma: float = 4.0,
    P_baseline: float = 500.0,
    T_baseline: float = 50.0,
    min_transitions_per_site: int = 10,
    min_coverage_ratio: float = 0.5,
    entropy_bonus: float = 8.0,
    concentration_penalty_threshold: float = 0.8,
    full_length_ns: float = FULL_LENGTH_NS,
    explore_min_transitions: int = EXPLORE_MIN_TRANSITIONS,
) -> float:
    """Compute reward from simulation results dict using improved reward logic.

    This implements the same logic as train_improved.py but works with cached
    simulation results instead of reading from output files.

    **Confidence Factor (C_F)**: Scales population reward by data reliability

    Uses a **graduated transition penalty system** to provide continuous feedback:
    - **0 transitions**: -40.0 penalty (death floor - zero activity)
    - **1 transition**: -32.0 penalty (very low activity)
    - **2 transitions**: -24.0 penalty (very low activity)
    - **3-9 transitions**: -2.0 - (2.0 × deficit) penalty (climbing ramp, ranges -16 to -4)
    - **≥10 transitions**: 0.0 penalty, unlocks R_T reward (success zone)

    Default parameters match the updated reward configuration which achieved
    improved gradient signal for low-transition runs (graduated penalties help
    1-2 transition runs compared to uniform -40 penalty).

    Args:
        sim_results: Dict with 'populations' and 'transitions' keys
        num_sites: Number of sites in the system
        nsubs_per_site: List of number of substituents per site
        w_P, w_T: Reward weights for populations and transitions
        w_U: Accepted for API compatibility but unused in current formula;
             coverage is handled by the quadratic coverage_factor multiplier.
        gamma: Scaling factor for rewards (used in coverage/concentration penalties)
        P_baseline: Normalization baseline for populations
        T_baseline: Normalization baseline for transitions
        min_transitions_per_site: Minimum transitions required per site (default: 10)
        min_coverage_ratio: Minimum ratio of substituents that must be visited
        entropy_bonus: Bonus for uniform distributions
        concentration_penalty_threshold: Threshold for concentration penalty (e.g., 0.8 = 80%)
        full_length_ns, explore_min_transitions: run-length handling (see
            ``run_length``). Populations and transitions are counts, so
            P_baseline and T_baseline scale with the run's length; a short run
            needs only explore_min_transitions per site (online's short-run
            test) where a full one needs min_transitions_per_site.

    Returns:
        Scalar reward value
    """
    length_ratio, short = run_length(sim_results, full_length_ns)
    P_baseline *= length_ratio
    T_baseline *= length_ratio
    if short:
        min_transitions_per_site = min(min_transitions_per_site, explore_min_transitions)

    populations = sim_results.get("populations", {})
    transitions = sim_results.get("transitions", {})

    # Extract population counts (use only HIGHEST lambda value)
    pop_list = []
    for block_id in sorted([int(k) for k in populations.keys()]):
        block_data = populations[str(block_id)]
        counts = block_data.get("counts", {})
        if counts:
            # Use only the highest lambda value
            max_lambda = max(counts.keys(), key=lambda x: float(x))
            pop_list.append(counts[max_lambda])

    if not pop_list:
        return -50.0  # No population data (capped at penalty limit)

    pop_array = np.array(pop_list, dtype=float)
    total_pop = pop_array.sum()

    if total_pop == 0:
        return -50.0  # No sampling occurred (capped at penalty limit)

    # Extract transition counts per site (use only HIGHEST lambda value)
    trans_per_site = []
    for site_id in sorted([int(k) for k in transitions.keys()]):
        site_data = transitions[str(site_id)]
        if site_data and isinstance(site_data, dict):
            # Use only the highest lambda value
            max_lambda = max(site_data.keys(), key=lambda x: float(x))
            total_trans = site_data[max_lambda]
            trans_per_site.append(total_trans)
        else:
            trans_per_site.append(0)

    # Ensure we have transition data for all sites
    while len(trans_per_site) < num_sites:
        trans_per_site.append(0)

    trans_array = np.array(trans_per_site[:num_sites], dtype=float)
    total_trans = trans_array.sum()

    # Track minimum transitions across all sites for Confidence Factor
    min_transitions_across_sites = int(trans_array.min())

    # === STRICT REQUIREMENTS (penalties) ===
    # Match train_improved.py penalty calculation
    penalties = 0.0

    # 1. Multi-site aware transition penalty system
    # Apply base penalty once based on worst site, then add scaling for additional bad sites
    # This prevents unfair accumulation when multiple sites are degenerate
    sites_below_threshold = 0

    # Count sites below threshold
    for trans_count in trans_array:
        if trans_count < min_transitions_per_site:
            sites_below_threshold += 1

    # Determine base penalty based on worst site (minimum transitions)
    if min_transitions_across_sites == 0:
        base_penalty = 40.0
    elif min_transitions_across_sites == 1:
        base_penalty = 32.0
    elif min_transitions_across_sites == 2:
        base_penalty = 24.0
    elif min_transitions_across_sites < min_transitions_per_site:
        # Tier 2: "Climbing Ramp" - softened gradient from ~-16 to -4
        # Formula: -2.0 - (2.0 * deficit)
        deficit = min_transitions_per_site - min_transitions_across_sites
        base_penalty = 2.0 + 2.0 * deficit
    else:
        base_penalty = 0.0

    # Add multi-site degradation penalty if multiple sites are bad
    # Each additional bad site beyond the first adds a smaller incremental penalty
    if sites_below_threshold > 1:
        multisite_penalty = (sites_below_threshold - 1) * 4.0
        total_transition_penalty = base_penalty + multisite_penalty
        penalties -= total_transition_penalty
    elif sites_below_threshold == 1:
        penalties -= base_penalty

    # 2. Coverage requirement
    # NOTE: adaptive coverage penalty removed — replaced by coverage_factor below
    num_populated = np.count_nonzero(pop_array)
    total_subs = sum(nsubs_per_site)
    coverage_ratio = num_populated / total_subs if total_subs > 0 else 0.0

    # 3. Concentration penalty (per-site check)
    pop_idx = 0
    for site_idx, nsubs in enumerate(nsubs_per_site):
        site_pops = pop_array[pop_idx:pop_idx + nsubs]
        site_total = site_pops.sum()

        if site_total > 0:
            concentration_ratio = site_pops.max() / site_total
            if concentration_ratio > concentration_penalty_threshold:
                # Reduced coefficient (2.0 instead of 5.0) to prevent excessive
                # accumulation in multi-site systems
                penalties -= gamma * 2.0 * (concentration_ratio - concentration_penalty_threshold)

        pop_idx += nsubs

    # === REWARD COMPONENTS ===

    # Confidence Factor (C_F): Scale population reward by data reliability
    # Low-transition runs have unreliable population distributions
    confidence_factor = min(1.0, min_transitions_across_sites / (2.0 * min_transitions_per_site))

    # R_P: Population balance
    pop_probs = pop_array / total_pop  # needed for entropy below
    nonzero_pops = pop_array[pop_array > 0]

    R_P = 0.0
    if len(nonzero_pops) > 1:
        min_meaningful_coverage = max(2, num_sites * 1.5)

        if len(nonzero_pops) >= min_meaningful_coverage:
            # balance_factor removed: R_entropy captures within-visited uniformity
            total_pop_normalized = sum(p / P_baseline for p in nonzero_pops)
            R_P = w_P * total_pop_normalized * confidence_factor
        else:
            # Insufficient coverage: minimal reward proportional to coverage
            R_P = w_P * 0.01 * coverage_ratio * confidence_factor

    # R_T: Transitions (Tier 3: "Success Zone" - only if all sites >= min_transitions_per_site)
    if sites_below_threshold == 0:
        R_T = w_T * (total_trans / T_baseline)
        avg_trans_per_site = total_trans / num_sites if num_sites > 0 else 0
        if avg_trans_per_site > min_transitions_per_site * 2:
            R_T *= 1.5  # 50% bonus for very active sampling
    else:
        R_T = 0.0  # No transition reward if some sites below threshold (Tier 1 or 2)

    # R_entropy: Shannon entropy bonus for uniform distributions
    entropy = -np.sum(pop_probs * np.log(pop_probs + 1e-10))
    max_entropy = np.log(len(pop_probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    R_entropy = entropy_bonus * normalized_entropy

    # ========== PENALTY CLAMPING ==========
    max_penalty = 60.0
    if penalties < -max_penalty:
        penalties = -max_penalty

    # coverage_factor: smooth quadratic multiplier replacing hard completeness gate
    coverage_factor = (num_populated / total_subs) ** 2 if total_subs > 0 else 0.0
    reward = coverage_factor * (R_P + R_T + R_entropy) + penalties

    return reward


def _extract_highest_lambda_counts(populations) -> list:
    """Convert simulation_results.json populations dict to a flat list of raw counts.

    ``simulation_results.json`` stores populations as a dict::

        {"2": {"counts": {"0.95": N, "0.99": M}, "site": S}, ...}

    Each block's counts sub-dict holds MSLD population counts sampled at
    different lambda values (e.g. 0.95 and 0.99).  These are **not** averaged
    or normalised — only the single highest-lambda count is used (e.g. 0.99
    when both 0.95 and 0.99 are present), since it is the most physically
    meaningful value and the values at different lambdas are incommensurable.

    Returns a list indexed by node_idx where::

        result[node_idx] = highest-lambda population count for block (node_idx + 2)

    (MSLD block_offset=2 convention: block 1 is the reference.)

    If *populations* is already a list it is returned unchanged.
    """
    if not populations:
        return []
    if isinstance(populations, list):
        return populations
    block_offset = 2
    try:
        max_node = max(int(k) - block_offset for k in populations.keys())
    except (ValueError, TypeError):
        return []
    result = [0] * (max_node + 1)
    for block_id_str, block_info in populations.items():
        try:
            node_idx = int(block_id_str) - block_offset
        except ValueError:
            continue
        if node_idx < 0:
            continue
        counts = block_info.get('counts', {}) if isinstance(block_info, dict) else {}
        if counts:
            # Take only the highest-lambda value — do not average across lambdas.
            max_lambda = max(counts.keys(), key=lambda x: float(x))
            result[node_idx] = counts[max_lambda]
    return result


def _extract_total_transitions(transitions) -> int:
    """Sum per-site highest-lambda transition counts from a
    ``simulation_results.json`` 'transitions' dict::

        {"1": {"0.95": a, "0.99": b}, "2": {...}, ...}

    Mirrors the per-site highest-lambda selection used elsewhere in this
    module (``compute_reward_from_sim_results``) and in online training
    (``parse_transitions_and_rates`` / ``parse_simulation_metrics``).

    Returns 0 if *transitions* is empty or malformed.
    """
    if not transitions:
        return 0
    total = 0
    for site_data in transitions.values():
        if isinstance(site_data, dict) and site_data:
            max_lambda = max(site_data.keys(), key=lambda x: float(x))
            try:
                total += int(site_data[max_lambda])
            except (TypeError, ValueError):
                pass
    return total


def run_pair_reward(
    run: Dict,
    edge_index: torch.Tensor,
    reward_config: Dict,
) -> Optional[torch.Tensor]:
    """Compute the per-edge, per-dimension reward tensor for one pretraining run.

    Uses the SAME ``compute_pair_reward()`` that online REINFORCE training
    uses (``training/workflow.py`` / ``mllf.cb.workflow_utils``), so BC/AWR
    pretraining and online training score individual substituent pairs on
    identical criteria: per-pair population balance (linear), per-pair
    midpoint barrier from the lambda trajectory (quadratic; falls back to DDG
    existence + transition quality when ``barrier_pairs`` is absent from
    ``simulation_results.json`` -- see ``--backfill-barrier`` in
    collect_pretraining_data), the pair's skew and end asymmetries of that
    profile (skew / end; ``skew_pairs`` / ``end_pairs``, same backfill), else
    the FRACTION PHYSICAL LIGAND
    proxy scaled by pair count (skew), and the population/quality combined
    proxy (end). Without this, every pair within an otherwise-successful run
    would share one whole-run scalar reward, even though some pairs are
    typically much better resolved (more transitions, cleaner DDG) than
    others in the same run.

    The legacy transition-quality proxy compares the run's transitions with
    T_baseline scaled by the run's length (``run_length``), so a short run is
    not scored against what a full run has time for.

    Returns None if the run's ``sim_results`` lacks the population or
    per-pair DDG data needed (e.g. older collected runs, or a
    ``simulation_results.json`` that predates per-pair DDG collection) --
    callers should fall back to the whole-run scalar reward in that case.
    """
    sim_results = run.get("sim_results", {})
    populations_raw = sim_results.get("populations", {})
    if not populations_raw:
        return None
    ddg_pairs = sim_results.get("ddg_pairs", {})
    if not ddg_pairs:
        return None
    try:
        populations = _extract_highest_lambda_counts(populations_raw)
        total_transitions = _extract_total_transitions(sim_results.get("transitions", {}))
        length_ratio, _ = run_length(
            sim_results, reward_config.get(
                "full_length_ns", FULL_LENGTH_NS))
        t_baseline = reward_config.get("T_baseline", 50.0) * length_ratio
        fraction_physical = sim_results.get("fraction_physical")
        return compute_pair_reward(
            edge_index.detach().cpu(), ddg_pairs, populations,
            total_transitions=total_transitions,
            t_baseline=t_baseline,
            fraction_physical=fraction_physical,
            barrier_pairs=sim_results.get("barrier_pairs"),
            barrier_target=reward_config.get("barrier_target", BARRIER_TARGET),
            barrier_width=reward_config.get("barrier_width", BARRIER_WIDTH),
            skew_pairs=sim_results.get("skew_pairs"),
            end_pairs=sim_results.get("end_pairs"),
            skew_width=reward_config.get("skew_width", SKEW_WIDTH),
            end_width=reward_config.get("end_width", END_WIDTH),
        )
    except Exception:
        return None


def run_pair_success(run: Dict, edge_index: torch.Tensor,
                     reward_config: Dict) -> Optional[torch.Tensor]:
    """Per edge: did the run pass online training's success test for that pair?

    The test online training uses to call a pair's run good
    (``hindsight.pair_succeeded``): both substituents sampled with minority
    fraction >= min_fraction, and >= min_transitions crossings (on a site of
    3+ substituents, each of the two entered or left that often). A full run
    needs SUCCESS_MIN_TRANSITIONS / SUCCESS_MIN_FRACTION; a short one
    (``run_length``) the reward config's explore_min_transitions /
    explore_min_fraction, the curriculum stage keys of the same name.

    Returns a [E] bool tensor, or None when the run lacks per-pair transitions
    ('transition_pairs'; ``collect_pretraining_data --backfill-barrier``).
    """
    sim_results = run.get("sim_results", {})
    pair_trans = sim_results.get("transition_pairs")
    if not pair_trans:
        return None
    _, short = run_length(sim_results, reward_config.get("full_length_ns", FULL_LENGTH_NS))
    if short:
        min_t = int(reward_config.get("explore_min_transitions", EXPLORE_MIN_TRANSITIONS))
        min_f = float(reward_config.get("explore_min_fraction", EXPLORE_MIN_FRACTION))
    else:
        min_t, min_f = SUCCESS_MIN_TRANSITIONS, SUCCESS_MIN_FRACTION
    saved = {"populations": _extract_highest_lambda_counts(sim_results.get("populations", {})),
             "transition_pairs": pair_trans}
    ei = edge_index.detach().cpu()
    sites = site_nodes(ei)
    return torch.tensor([
        pair_succeeded(saved, int(ei[0, k]), int(ei[1, k]), min_fraction=min_f,
                       min_transitions=min_t, site=sites.get(int(ei[0, k])))
        for k in range(ei.size(1))
    ], dtype=torch.bool)


def pair_advantages(
    pair_rewards: List[Optional[torch.Tensor]],
    pair_success: Optional[List[Optional[torch.Tensor]]] = None,
    success_floor: Optional[float] = None,
) -> Tuple[List[Optional[torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Per-pair AWR advantages for every run.

    * Rewards are normalised per bias type by the mean / std over all runs.
    * ``success_floor`` (``reward.success_advantage_floor``) clamps from below,
      so one badly scored pair cannot dominate its run's loss.
    * Success gate (``pair_success``, from ``run_pair_success``): the online
      success test for a run of that length decides whether a pair's biases
      are imitated more or less than average -- a pair that passed gets
      advantage >= 0, one that failed <= 0 -- and the reward ranks pairs
      within each side. As online, a short run's success counts as much as a
      full run's: the length sets which test a pair must pass, not how much a
      pass is worth. Runs without a success entry are not gated.

    Returns (advantages aligned with ``pair_rewards``, baseline mean [4], std [4]).
    """
    valid = [r.reshape(-1, 4) for r in pair_rewards if r is not None]
    if not valid:
        return [None] * len(pair_rewards), torch.zeros(4), torch.ones(4)
    stacked = torch.cat(valid)
    mean, std = stacked.mean(dim=0), stacked.std(dim=0).clamp(min=1e-6)

    out: List[Optional[torch.Tensor]] = []
    for idx, r in enumerate(pair_rewards):
        if r is None:
            out.append(None)
            continue
        adv = (r - mean) / std
        if success_floor is not None:
            adv = adv.clamp(min=float(success_floor))
        ok = pair_success[idx] if pair_success is not None else None
        if ok is not None and ok.shape[0] == adv.shape[0]:
            adv = torch.where(ok.unsqueeze(-1), adv.clamp(min=0.0), adv.clamp(max=0.0))
        out.append(adv)
    return out, mean, std


def build_pair_advantages(runs: List[Dict], graph_cache: List,
                          reward_config: Dict) -> Optional[List]:
    """Per-edge, per-bias-type advantages for each run (``pretrain_epoch``'s
    per-pair AWR), from the online pair reward; with the success gate (on by
    default) edges whose run failed the online success test get at most
    ``success_advantage_floor``. None if no run has pair data."""
    full_length_ns = reward_config.get("full_length_ns", FULL_LENGTH_NS)
    short_runs = [run_length(run.get("sim_results", {}), full_length_ns)[1] for run in runs]
    success_gate = bool(reward_config.get("success_gate", True))
    pair_reward_cache = [None] * len(runs)
    pair_success_cache = [None] * len(runs)

    log.info(f"\n{'=' * 60}")
    log.info("Building per-pair reward cache for per-edge, per-dimension AWR...")
    log.info(f"{'=' * 60}")
    n_pair_ok = 0
    for run_idx, run in enumerate(runs):
        try:
            if graph_cache[run_idx] is None:
                continue
            edge_idx = graph_cache[run_idx][1]
            r_pair = run_pair_reward(run, edge_idx, reward_config)
            if r_pair is not None:
                pair_reward_cache[run_idx] = r_pair
                if success_gate:
                    pair_success_cache[run_idx] = run_pair_success(run, edge_idx, reward_config)
                n_pair_ok += 1
        except Exception:
            pass   # the run falls back to its scalar reward
        if (run_idx + 1) % 100 == 0 or (run_idx + 1) == len(runs):
            log.info(f"  {run_idx + 1}/{len(runs)} runs ({n_pair_ok} with pair rewards)...")

    if not n_pair_ok:
        return None
    success_floor = reward_config.get("success_advantage_floor", -2.0)
    advantages, baseline_mean, baseline_std = pair_advantages(
        pair_reward_cache, pair_success=pair_success_cache if success_gate else None,
        success_floor=success_floor)
    log.info("\nPer-pair reward baseline computed:")
    log.info(f"  Runs with per-pair rewards: {n_pair_ok}/{len(runs)}")
    log.info(f"  Baseline mean [lin, quad, skew, end]: {baseline_mean.numpy()}")
    log.info(f"  Baseline  std [lin, quad, skew, end]: {baseline_std.numpy()}")
    log.info(f"  Success floor: {success_floor}")
    if success_gate:
        for name, short in (('full', False), ('short', True)):
            flags = [ok for ok, sh in zip(pair_success_cache, short_runs)
                     if ok is not None and sh == short]
            n_edges = sum(int(f.numel()) for f in flags)
            n_passed = sum(int(f.sum()) for f in flags)
            log.info(f"  Success gate, {name} runs (short = under "
                     f"{0.5 * full_length_ns:g} ns): {n_passed}/{n_edges} edges passed "
                     f"the online {'short-run' if short else 'full-run'} test")
    return advantages


def assign_run_rewards(runs: List[Dict], reward_config: Dict) -> int:
    """Set run["_bc_reward"] = max(0, reward) on every run (the scalar AWR
    weight is exp(r / beta)); returns the number of positive-reward runs."""
    filtered_rc = reward_kwargs(reward_config)
    log.info("\nPre-computing per-run rewards for AWR loss weighting...")
    n_pos = 0
    total_pos = 0.0
    for run in runs:
        try:
            with open(Path(run["run_dir"]) / "graph_info.json") as f:
                gi = json.load(f)
            site_cnts: Dict[int, int] = {}
            for bd in gi.get("sites", {}).values():
                site_cnts[bd["site"]] = site_cnts.get(bd["site"], 0) + 1
            nsubs_per_site = [site_cnts[s] for s in sorted(site_cnts)]
            r = compute_reward_from_sim_results(
                run["sim_results"], len(site_cnts), nsubs_per_site, **filtered_rc)
            run["_bc_reward"] = max(0.0, r)
            if r > 0:
                n_pos += 1
                total_pos += r
        except Exception:
            run["_bc_reward"] = 0.0
    log.info(f"  Positive-reward runs: {n_pos}/{len(runs)}  "
             f"(total positive reward mass: {total_pos:.2f})")
    return n_pos
