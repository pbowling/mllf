"""Improved reward function for MSLD simulations that prevents degenerate solutions.

This module provides an enhanced reward function that explicitly penalizes
concentrated populations and low transition counts, encouraging the policy
to explore the full alchemical space rather than converging to single-substituent
solutions.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from mllf.file_handling.read_output import (
    parse_single_population,
    parse_transitions_and_rates,
    terminated_normally
)


def compute_msld_reward_improved(
    combo_dir: str,
    w_P: float = 0.5,
    w_T: float = 0.5,
    w_U: float = 0.3,
    gamma: float = 4.0,
    P_baseline: float = 500.0,
    T_baseline: float = 50.0,
    min_transitions_per_site: int = 10,
    min_coverage_ratio: float = 0.5,
    entropy_bonus: float = 8.0,
    concentration_penalty_threshold: float = 0.8
) -> float:
    """Compute improved scalarized reward that prevents degenerate solutions.
    
    This reward function explicitly addresses the issue of the policy converging
    to single-substituent solutions using a **tiered transition penalty system**:
    
    **Tiered Transition Penalty System:**
    - **Tier 1: "Death Floor" (0 transitions)**: Fixed penalty of -40.0
      Worst possible state, signaling total inactivity is unacceptable
    
    - **Tier 2: "Climbing Ramp" (1-9 transitions)**: Linear gradient from ~-30 to -8
      Formula: -5.0 - (2.8 × deficit)
      Provides continuous feedback where each additional transition reduces penalty
    
    - **Tier 3: "Success Zone" (≥10 transitions)**: Penalty = 0.0
      Site is "unlocked" - earns positive R_T reward and potential 1.5× bonus
    
    **Additional Protections:**
    1. **Minimum coverage requirement**: Penalty if fewer than `min_coverage_ratio`
       of substituents are visited (have non-zero population)
    
    2. **Concentration penalty**: Per-site penalty if any single substituent
       exceeds `concentration_penalty_threshold` of that site's population
    
    3. **Entropy-based uniformity bonus**: Rewards more uniform population
       distributions using Shannon entropy
    
    The reward function is:
        R = R_P + R_T + R_U + R_entropy + R_penalties
    
    where:
        R_P: Population balance reward (weighted)
        R_T: Transition reward (Tier 3 only, weighted, with bonus for high counts)
        R_U: Coverage uniformity reward (weighted)
        R_entropy: Bonus for high-entropy (uniform) distributions
        R_penalties: Tiered transition penalties + coverage/concentration penalties
    
    Args:
        combo_dir: Path to combination directory with simulation outputs.
        w_P: Weight for population term (default: 0.5).
        w_T: Weight for transition term (default: 0.5).
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
        print(f"  Warning: No output file found in {combo_dir}")
        return -100.0 * gamma  # Large penalty for failed simulation
    
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {output_file}: {e}")
        return -100.0 * gamma
    
    if not terminated_normally(output_text):
        print(f"  Warning: Simulation did not terminate normally in {combo_dir}")
        return -100.0 * gamma
    
    # Parse outputs
    population_data = parse_single_population(output_text)
    transitions_data, _ = parse_transitions_and_rates(output_text)
    
    if not population_data or not transitions_data:
        print(f"  Warning: No population or transition data in {output_file}")
        return -100.0 * gamma
    
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
        return -100.0 * gamma
    
    # ========== STRICT DEGENERATE BEHAVIOR CHECKS ==========
    
    # Check 1: Tiered transition penalty system
    # Replaces binary threshold with continuous gradient feedback
    penalties = 0.0
    sites_below_threshold = 0
    
    for site_id, trans_count in site_transitions.items():
        if trans_count < min_transitions_per_site:
            sites_below_threshold += 1
            
            if trans_count == 0:
                # Tier 1: "Death Floor" - worst possible state
                penalties -= 40.0
            elif trans_count < min_transitions_per_site:
                # Tier 2: "Climbing Ramp" - linear gradient from ~-30 to -8
                # Formula: -5.0 - (2.8 * deficit)
                # At trans=1, deficit=9: -5.0 - 25.2 = -30.2
                # At trans=9, deficit=1: -5.0 - 2.8 = -7.8
                deficit = min_transitions_per_site - trans_count
                penalties -= (5.0 + 2.8 * deficit)
    
    # Note: sites_below_threshold tracked but no additional penalty
    # (tiered penalties already applied above)
    if sites_below_threshold > 0:
        print(f"  Warning: {sites_below_threshold} site(s) below {min_transitions_per_site} transitions")
    
    # Check 2: Minimum coverage (fraction of substituents visited)
    pop_array = np.array(populations)
    nonzero_count = np.sum(pop_array > 0)
    coverage_ratio = nonzero_count / total_subs if total_subs > 0 else 0.0
    
    if coverage_ratio < min_coverage_ratio:
        # Heavy penalty for low coverage
        deficit = min_coverage_ratio - coverage_ratio
        penalties -= gamma * 20.0 * deficit
        print(f"  Warning: Coverage {coverage_ratio:.2f} below minimum {min_coverage_ratio}")
    
    # Check 3: Detect single-dominant-population per site
    # Extract actual substituents per site from graph_info.json
    nsubs_per_site = None
    graph_info_path = combo_path / 'graph_info.json'
    if graph_info_path.exists():
        try:
            import json
            with open(graph_info_path, 'r') as f:
                graph_info = json.load(f)
            if 'sites' in graph_info:
                from collections import defaultdict
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
            if concentration_ratio > concentration_penalty_threshold:
                penalties -= gamma * 5.0 * (concentration_ratio - concentration_penalty_threshold)
                print(f"  Warning: Site {site_idx} has {concentration_ratio:.2%} concentration")
        
        pop_idx += nsubs
    
    # ========== POSITIVE REWARD COMPONENTS ==========
    
    # R_P: Population balance reward
    R_P = 0.0
    if len(populations) > 1:
        nonzero_pops = pop_array[pop_array > 0]
        
        # Require meaningful coverage: at least 2 subs per site on average
        # If we have 8 subs total and only 2 are visited (one per site), that's degenerate
        min_meaningful_coverage = max(2, total_sites * 1.5)  # At least 1.5 subs per site
        
        if len(nonzero_pops) > 1 and len(nonzero_pops) >= min_meaningful_coverage:
            # Use coefficient of variation (std/mean) for balance
            pop_mean = np.mean(nonzero_pops)
            pop_std = np.std(nonzero_pops)
            cv = pop_std / pop_mean if pop_mean > 0 else 1.0
            
            # Balance factor: exp(-cv) ranges from ~0.37 (CV=1) to 1.0 (CV=0)
            balance_factor = np.exp(-cv)
            
            # Normalized population reward (only count non-zero populations)
            total_pop_normalized = sum(p / P_baseline for p in nonzero_pops)
            R_P = w_P * total_pop_normalized * balance_factor
        else:
            # Insufficient coverage: minimal reward proportional to coverage
            R_P = w_P * 0.01 * coverage_ratio
    
    # R_T: Transition reward (Tier 3: "Success Zone" - only if all sites >= min_transitions_per_site)
    R_T = 0.0
    if sites_below_threshold == 0:  # All sites in Success Zone
        total_trans = sum(site_transitions.values())
        R_T = w_T * (total_trans / T_baseline)
        
        # Bonus for exceeding minimum significantly
        avg_trans_per_site = total_trans / total_sites if total_sites > 0 else 0
        if avg_trans_per_site > min_transitions_per_site * 2:
            R_T *= 1.5  # 50% bonus for high transition counts
    
    # R_U: Coverage uniformity reward
    R_U = w_U * coverage_ratio
    
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
    max_penalty = 50.0
    if penalties < -max_penalty:
        penalties = -max_penalty
        print(f"  Warning: Penalties clamped to -{max_penalty}")
    
    # ========== FINAL REWARD ==========
    
    R = R_P + R_T + R_U + R_entropy + penalties
    
    # Debug output
    print(f"  Reward breakdown: R_P={R_P:.2f}, R_T={R_T:.2f}, R_U={R_U:.2f}, "
          f"R_entropy={R_entropy:.2f}, penalties={penalties:.2f}, total={R:.2f}")
    print(f"  Coverage: {nonzero_count}/{total_subs} ({coverage_ratio:.2%}), "
          f"Transitions: {list(site_transitions.values())}")
    
    return R


def compute_msld_reward_per_site(
    combo_dir: str,
    **kwargs
) -> Tuple[float, Dict]:
    """Compute reward with detailed per-site breakdown for analysis.
    
    Returns:
        reward: Total scalar reward
        metrics: Dict with detailed per-site metrics for logging/analysis
    """
    # Parse outputs (same as above)
    combo_path = Path(combo_dir)
    
    output_file = None
    for candidate in [combo_path / 'output.out', combo_path / f'{combo_path.name}.out']:
        if candidate.exists():
            output_file = candidate
            break
    
    if output_file is None:
        return -100.0, {'error': 'No output file'}
    
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
    except Exception:
        return -100.0, {'error': 'Read failed'}
    
    if not terminated_normally(output_text):
        return -100.0, {'error': 'Abnormal termination'}
    
    population_data = parse_single_population(output_text)
    transitions_data, _ = parse_transitions_and_rates(output_text)
    
    # Compute reward
    reward = compute_msld_reward_improved(combo_dir, **kwargs)
    
    # Extract detailed metrics
    populations = [sum(block['counts'].values()) for block in population_data.values()]
    pop_array = np.array(populations)
    
    metrics = {
        'reward': reward,
        'total_subs': len(populations),
        'nonzero_subs': np.sum(pop_array > 0),
        'coverage': np.sum(pop_array > 0) / len(populations) if len(populations) > 0 else 0,
        'site_transitions': {site: sum(trans.values()) for site, trans in transitions_data.items()},
        'populations': populations,
        'max_pop_ratio': np.max(pop_array) / np.sum(pop_array) if np.sum(pop_array) > 0 else 0,
    }
    
    return reward, metrics
