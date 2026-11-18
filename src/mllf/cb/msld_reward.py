"""Improved reward function for MSLD simulations that prevents degenerate solutions.

This module provides an enhanced reward function that explicitly penalizes
concentrated populations and low transition counts, encouraging the policy
to explore the full alchemical space rather than converging to single-substituent
solutions.
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from mllf.file_handling.read_output import (
    parse_single_population,
    parse_transitions_and_rates,
    terminated_normally,
)


def compute_msld_reward_improved(
    combo_dir: str,
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
    suppress_warnings: bool = False,
) -> float:
    """Compute improved scalarized reward that prevents degenerate solutions.

    This reward function uses a Confidence Factor to prevent degenerate behavior:

    **Confidence Factor (C_F)**
    Scales population rewards based on data reliability:
        C_F = min(1.0, min_transitions / (2 * N_req))
        R_P is multiplied by C_F, reducing false rewards from low-transition runs

    **Tiered Transition Penalty System:**
    - **Tier 1: "Death Floor" (0-2 transitions)**: Fixed penalty of -40.0
    - **Tier 2: "Climbing Ramp" (3-9 transitions)**: Linear gradient from ~-16 to -4
      Formula: -2.0 - (2.0 x deficit)
    - **Tier 3: "Success Zone" (>=10 transitions)**: Penalty = 0.0, unlocks R_T

    **Reward Components:**
        R = coverage_factor * (R_P + R_T + R_entropy) + R_penalties

    where:
        R_P: Population balance reward x C_F (confidence-scaled)
        R_T: Transition reward (gated: only if all sites >= min_transitions)
        coverage_factor: (nonzero_count/total_subs)^2; smooth partial-coverage multiplier
        R_entropy: Bonus for high-entropy (uniform) distributions
        R_penalties: Tiered transition penalties + coverage/concentration penalties

    Args:
        combo_dir: Path to combination directory with simulation outputs.
        w_P: Weight for population term (default: 0.5).
        w_T: Weight for transition term (default: 0.75).
        w_U: Weight for uniformity term (default: 0.3).
        gamma: Base penalty coefficient (default: 4.0).
        P_baseline: Normalization baseline for populations (default: 500.0).
        T_baseline: Normalization baseline for transitions (default: 50.0).
        min_transitions_per_site: Minimum transitions required per site (default: 10).
        min_coverage_ratio: Minimum fraction of substituents that must be visited (default: 0.5).
        entropy_bonus: Bonus coefficient for high-entropy distributions (default: 8.0).
        concentration_penalty_threshold: Threshold for concentration penalty (default: 0.8).

    Returns:
        Scalar reward value (higher is better). Returns large negative value
        if simulation failed or if degenerate behavior is detected.
    """

    combo_path = Path(combo_dir)

    # Find output file
    output_file = None
    possible_outputs = [
        combo_path / 'output.out',
        combo_path / f'{combo_path.name}.out',
    ]
    for out_file in combo_path.glob('*.out'):
        possible_outputs.append(out_file)

    for candidate in possible_outputs:
        if candidate.exists():
            output_file = candidate
            break

    if output_file is None:
        warnings.warn(f"No output file found in {combo_dir}")
        return -50.0  # Capped penalty for failed simulation

    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
    except Exception as e:
        warnings.warn(f"Could not read {output_file}: {e}")
        return -50.0

    if not terminated_normally(output_text):
        warnings.warn(f"Simulation did not terminate normally in {combo_dir}")
        return -50.0

    # Parse outputs
    population_data = parse_single_population(output_text)
    transitions_data, _ = parse_transitions_and_rates(output_text)

    if not population_data or not transitions_data:
        warnings.warn(f"No population or transition data in {output_file}")
        return -50.0

    # Extract populations (per block/substituent) - use only HIGHEST lambda value
    populations = []
    for block_id, block_info in population_data.items():
        counts_dict = block_info.get('counts', {})
        if counts_dict:
            # Use only the highest lambda value
            max_lambda = max(counts_dict.keys(), key=lambda x: float(x))
            populations.append(counts_dict[max_lambda])
        else:
            populations.append(0)

    # Extract transitions (per site) - use only HIGHEST lambda value
    site_transitions = {}
    for site_id, trans_dict in transitions_data.items():
        if trans_dict:
            # Use only the highest lambda value
            max_lambda = max(trans_dict.keys(), key=lambda x: float(x))
            site_transitions[site_id] = trans_dict[max_lambda]
        else:
            site_transitions[site_id] = 0

    total_subs = len(populations)
    total_sites = len(site_transitions)

    if total_subs == 0 or total_sites == 0:
        return -50.0

    # ========== STRICT DEGENERATE BEHAVIOR CHECKS ==========

    # Check 1: Multi-site aware transition penalty system
    # Apply base penalty once based on worst site, then add scaling for additional bad sites
    # This prevents unfair accumulation when multiple sites are degenerate
    penalties = 0.0
    sites_below_threshold = 0
    min_transitions_across_sites = float('inf')

    # First pass: find minimum transitions and count sites below threshold
    for site_id, trans_count in site_transitions.items():
        min_transitions_across_sites = min(min_transitions_across_sites, trans_count)
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
        # At trans=3, deficit=7: -2.0 - 14.0 = -16.0
        # At trans=9, deficit=1: -2.0 - 2.0 = -4.0
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
        if not suppress_warnings:
            warnings.warn(f"{sites_below_threshold} sites below threshold: -{base_penalty:.1f} "
                          f"(base) + -{multisite_penalty:.1f} (multisite) = "
                          f"-{total_transition_penalty:.1f}")
    elif sites_below_threshold == 1:
        penalties -= base_penalty
        if not suppress_warnings:
            warnings.warn(f"1 site below threshold: -{base_penalty:.1f}")

    # Check 2: Minimum coverage (fraction of substituents visited)
    pop_array = np.array(populations)
    nonzero_count = np.sum(pop_array > 0)
    coverage_ratio = nonzero_count / total_subs if total_subs > 0 else 0.0

    # Coverage is handled by coverage_factor (quadratic penalty) rather than a hard threshold.
    # This gives smooth partial credit instead of a cliff at 100% coverage.

    # Check 3: Detect single-dominant-population per site
    # Extract actual substituents per site from graph_info.json
    nsubs_per_site = None
    graph_info_path = combo_path / 'graph_info.json'
    if graph_info_path.exists():
        try:
            with open(graph_info_path, 'r') as f:
                graph_info = json.load(f)
            if 'sites' in graph_info:
                site_counts = defaultdict(int)
                for site_key in graph_info['sites']:
                    site_num = int(site_key.split('_')[0].replace('site', ''))
                    site_counts[site_num] += 1
                nsubs_per_site = [site_counts[i] for i in sorted(site_counts.keys())]
        except Exception:
            pass

    # Fallback: uniform distribution (asymmetric-safe)
    if nsubs_per_site is None:
        subs_per_site = total_subs // total_sites if total_sites > 0 else total_subs
        nsubs_per_site = [subs_per_site] * total_sites
        for i in range(total_subs % total_sites):
            nsubs_per_site[i] += 1

    pop_idx = 0
    for site_idx, nsubs in enumerate(nsubs_per_site):
        site_pops = pop_array[pop_idx:pop_idx + nsubs]

        if len(site_pops) > 0 and np.sum(site_pops) > 0:
            max_pop = np.max(site_pops)
            total_pop = np.sum(site_pops)
            concentration_ratio = max_pop / total_pop

            # If concentration exceeds threshold: penalty
            # Reduced coefficient (2.0 instead of 5.0) to prevent excessive
            # accumulation in multi-site systems
            if concentration_ratio > concentration_penalty_threshold:
                penalties -= gamma * 2.0 * (concentration_ratio - concentration_penalty_threshold)
                if not suppress_warnings:
                    warnings.warn(f"Site {site_idx} has {concentration_ratio:.2%} concentration")

        pop_idx += nsubs

    # ========== POSITIVE REWARD COMPONENTS ==========

    # Confidence Factor: Scale population reward based on minimum transitions
    # C_F = min(1.0, min_transitions / (2 * N_req))
    # This prevents rewarding low-transition runs with misleading population data
    confidence_factor = min(1.0, min_transitions_across_sites / (2.0 * min_transitions_per_site))

    # R_P: Population balance reward (scaled by confidence factor)
    R_P = 0.0
    if len(populations) > 1:
        nonzero_pops = pop_array[pop_array > 0]

        # Require meaningful coverage: at least 2 subs per site on average
        # If we have 8 subs total and only 2 are visited (one per site), that's degenerate
        min_meaningful_coverage = max(2, total_sites * 1.5)  # At least 1.5 subs per site

        if len(nonzero_pops) > 1 and len(nonzero_pops) >= min_meaningful_coverage:
            # Use coefficient of variation (std/mean) for balance
            # Normalized population reward (balance_factor removed: R_entropy captures uniformity)
            total_pop_normalized = sum(p / P_baseline for p in nonzero_pops)
            R_P = w_P * total_pop_normalized * confidence_factor
        else:
            # Insufficient coverage: minimal reward proportional to coverage
            R_P = w_P * 0.01 * coverage_ratio * confidence_factor

    # R_T: Transition reward (Tier 3: "Success Zone" - only if all sites >=
    # min_transitions_per_site)
    R_T = 0.0
    if sites_below_threshold == 0:  # All sites in Success Zone
        total_trans = sum(site_transitions.values())
        R_T = w_T * (total_trans / T_baseline)

        # Bonus for exceeding minimum significantly
        avg_trans_per_site = total_trans / total_sites if total_sites > 0 else 0
        if avg_trans_per_site > min_transitions_per_site * 2:
            R_T *= 1.5  # 50% bonus for high transition counts

    # R_entropy: Shannon entropy bonus for uniform distributions
    R_entropy = 0.0
    if np.sum(pop_array) > 0:
        # Normalize to probability distribution
        prob_dist = pop_array / np.sum(pop_array)
        prob_dist = prob_dist[prob_dist > 0]  # Remove zeros

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        max_entropy = np.log(len(prob_dist))  # Maximum possible entropy

        # Normalized entropy score [0, 1]
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0
        R_entropy = entropy_bonus * entropy_score

    # ========== PENALTY CLAMPING ==========
    # Prevent gradient explosion by capping maximum negative penalty
    # Increased from 50 to 60 to preserve gradient information with multi-site systems
    max_penalty = 60.0
    if penalties < -max_penalty:
        penalties = -max_penalty
        if not suppress_warnings:
            warnings.warn(f"Penalties clamped to -{max_penalty}")

    # ========== FINAL REWARD ==========

    # coverage_factor: smooth quadratic multiplier; full credit at 100% coverage,
    # proportional partial credit below (e.g. 2/3 -> 0.44x, 1/3 -> 0.11x).
    coverage_factor = (nonzero_count / total_subs) ** 2 if total_subs > 0 else 0.0
    R = coverage_factor * (R_P + R_T + R_entropy) + penalties
    return R
