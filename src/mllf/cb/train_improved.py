"""Improved reward function for MSLD simulations that prevents degenerate solutions.

This module provides an enhanced reward function that explicitly penalizes
concentrated populations and low transition counts, encouraging the policy
to explore the full alchemical space rather than converging to single-substituent
solutions.
"""
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
import numpy as np

from mllf.file_handling.read_output import (
    parse_single_population,
    parse_transitions_and_rates,
    terminated_normally
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
    concentration_penalty_threshold: float = 0.8
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
      Formula: -2.0 - (2.0 × deficit)
    - **Tier 3: "Success Zone" (≥10 transitions)**: Penalty = 0.0, unlocks R_T
    
    **Reward Components:**
        R = R_P + R_T + R_U + R_entropy + R_penalties
    
    where:
        R_P: Population balance reward × C_F (confidence-scaled)
        R_T: Transition reward (gated: only if all sites ≥ min_transitions)
        R_U: Coverage uniformity reward
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
        print(f"  Warning: No output file found in {combo_dir}")
        return -50.0  # Capped penalty for failed simulation
    
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {output_file}: {e}")
        return -50.0
    
    if not terminated_normally(output_text):
        print(f"  Warning: Simulation did not terminate normally in {combo_dir}")
        return -50.0
    
    # Parse outputs
    population_data = parse_single_population(output_text)
    transitions_data, _ = parse_transitions_and_rates(output_text)
    
    if not population_data or not transitions_data:
        print(f"  Warning: No population or transition data in {output_file}")
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
        print(f"  Warning: {sites_below_threshold} sites below threshold: -{base_penalty:.1f} (base) + -{multisite_penalty:.1f} (multisite) = -{total_transition_penalty:.1f}")
    elif sites_below_threshold == 1:
        penalties -= base_penalty
        print(f"  Warning: 1 site below threshold: -{base_penalty:.1f}")
    
    # Check 2: Minimum coverage (fraction of substituents visited)
    pop_array = np.array(populations)
    nonzero_count = np.sum(pop_array > 0)
    coverage_ratio = nonzero_count / total_subs if total_subs > 0 else 0.0
    
    # Adaptive coverage requirement: scales with system size to encourage visiting multiple subs
    # Formula: min_subs = 1 + 0.5*(total-1)
    # Examples: 2 subs→1.5 (75%), 3 subs→2.0 (67%), 4 subs→2.5 (62.5%), 6 subs→3.5 (58%)
    min_subs_required = 1.0 + 0.5 * (total_subs - 1) if total_subs > 1 else 0.5
    adaptive_min_coverage = min_subs_required / total_subs if total_subs > 0 else 0.0
    
    # NO DOUBLE JEOPARDY: Don't penalize coverage if transitions are too low for reliable statistics
    # Coverage is only meaningful when there are enough transitions to have statistical confidence
    # Only apply coverage penalty if transitions are at or above the success threshold
    total_transitions = sum(site_transitions.values())
    
    # Coverage penalties only apply when the system has enough transitions for reliable sampling
    # Below min_transitions_per_site, the transition penalty already captures the problem
    if coverage_ratio < adaptive_min_coverage and min_transitions_across_sites >= min_transitions_per_site:
        # System has sufficient transitions but poor coverage - penalize the sampling inefficiency
        deficit = adaptive_min_coverage - coverage_ratio
        penalty_scale = np.sqrt(total_subs) if total_subs > 1 else 1.0
        penalties -= gamma * 20.0 * deficit / penalty_scale
        print(f"  Warning: Coverage {coverage_ratio:.2f} below adaptive minimum {adaptive_min_coverage:.2f} (need {min_subs_required:.1f}/{total_subs} subs)")
    elif coverage_ratio < adaptive_min_coverage and min_transitions_across_sites < min_transitions_per_site:
        print(f"  Info: Coverage {coverage_ratio:.2f} below minimum, but transitions too low for reliable statistics ({min_transitions_across_sites}/{min_transitions_per_site}), no double penalty")
    
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
            # Reduced coefficient (2.0 instead of 5.0) to prevent excessive accumulation in multi-site systems
            if concentration_ratio > concentration_penalty_threshold:
                penalties -= gamma * 2.0 * (concentration_ratio - concentration_penalty_threshold)
                print(f"  Warning: Site {site_idx} has {concentration_ratio:.2%} concentration")
        
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
            pop_mean = np.mean(nonzero_pops)
            pop_std = np.std(nonzero_pops)
            cv = pop_std / pop_mean if pop_mean > 0 else 1.0
            
            # Balance factor: exp(-cv) ranges from ~0.37 (CV=1) to 1.0 (CV=0)
            balance_factor = np.exp(-cv)
            
            # Normalized population reward (only count non-zero populations)
            total_pop_normalized = sum(p / P_baseline for p in nonzero_pops)
            
            # Apply confidence factor to prevent rewarding low-transition runs
            R_P = w_P * total_pop_normalized * balance_factor * confidence_factor
        else:
            # Insufficient coverage: minimal reward proportional to coverage
            R_P = w_P * 0.01 * coverage_ratio * confidence_factor
    
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
    # Increased from 50 to 60 to preserve gradient information with multi-site systems
    max_penalty = 60.0
    if penalties < -max_penalty:
        penalties = -max_penalty
        print(f"  Warning: Penalties clamped to -{max_penalty}")
    
    # ========== FINAL REWARD ==========

    # Completeness gate: if any substituent was never visited, replace the positive
    # reward components with -0.01 so the total is always negative. Penalties are
    # still added to preserve gradient signal (worse behaviour = more negative).
    if nonzero_count < total_subs:
        R = -0.01 + penalties
    else:
        R = R_P + R_T + R_U + R_entropy + penalties

    # Debug output
    print(f"  Reward breakdown: R_P={R_P:.2f}, R_T={R_T:.2f}, R_U={R_U:.2f}, "
          f"R_entropy={R_entropy:.2f}, penalties={penalties:.2f}, total={R:.2f}")
    if nonzero_count < total_subs:
        print(f"  Completeness gate applied ({nonzero_count}/{total_subs} subs visited) → positive components replaced with -0.01, penalties retained")
    print(f"  Coverage: {nonzero_count}/{total_subs} ({coverage_ratio:.2%}), "
          f"Transitions: {list(site_transitions.values())}, "
          f"Confidence: {confidence_factor:.2f}")

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
        return -50.0, {'error': 'No output file'}
    
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
    except Exception:
        return -50.0, {'error': 'Read failed'}
    
    if not terminated_normally(output_text):
        return -50.0, {'error': 'Abnormal termination'}
    
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


def compute_reward_from_raw_metrics(
    populations: list,
    transitions: list,
    w_P: float = 0.5,
    w_T: float = 0.75,
    w_U: float = 0.3,
    gamma: float = 4.0,
    P_baseline: float = 500.0,
    T_baseline: float = 50.0,
    min_transitions_per_site: int = 10,
    min_coverage_ratio: float = 0.5,
    entropy_bonus: float = 8.0,
    concentration_penalty_threshold: float = 0.8
) -> float:
    """Compute scalarized reward from raw simulation metrics using improved reward logic.
    
    This function allows recomputing rewards with different hyperparameters
    without re-running simulations. It uses the same reward logic as
    `compute_msld_reward_improved` but operates on pre-parsed metrics.
    
    This enables:
    - Testing different reward configurations on existing simulation data
    - Using simulations as "pretraining data" with flexible reward functions
    - Hyperparameter tuning without expensive re-simulation
    
    Args:
        populations: List of population counts per block/substituent.
        transitions: List of transition counts per site.
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
        Scalar reward value (higher is better). Returns -50.0 if inputs are empty.
    """
    if not populations and not transitions:
        return -50.0
    
    total_subs = len(populations)
    
    # Assume uniform distribution across sites if not provided
    # For transitions list, each element is a site's transition count
    total_sites = len(transitions) if transitions else 1
    
    if total_subs == 0 or total_sites == 0:
        return -50.0
    
    # ========== TRANSITION PENALTIES ==========
    penalties = 0.0
    sites_below_threshold = 0
    min_transitions_across_sites = float('inf')
    
    for trans_count in transitions:
        min_transitions_across_sites = min(min_transitions_across_sites, trans_count)
        
        if trans_count < min_transitions_per_site:
            sites_below_threshold += 1
            
            if trans_count == 0:
                penalties -= 40.0
            elif trans_count == 1:
                penalties -= 32.0
            elif trans_count == 2:
                penalties -= 24.0
            elif trans_count < min_transitions_per_site:
                deficit = min_transitions_per_site - trans_count
                penalties -= (2.0 + 2.0 * deficit)
    
    # ========== COVERAGE PENALTIES ==========
    pop_array = np.array(populations)
    nonzero_count = np.sum(pop_array > 0)
    coverage_ratio = nonzero_count / total_subs if total_subs > 0 else 0.0
    
    if coverage_ratio < min_coverage_ratio:
        deficit = min_coverage_ratio - coverage_ratio
        penalties -= gamma * 20.0 * deficit
    
    # ========== CONCENTRATION PENALTIES ==========
    # Assume uniform subs per site distribution
    subs_per_site = total_subs // total_sites if total_sites > 0 else total_subs
    nsubs_per_site = [subs_per_site] * total_sites
    for i in range(total_subs % total_sites):
        nsubs_per_site[i] += 1
    
    pop_idx = 0
    for nsubs in nsubs_per_site:
        site_pops = pop_array[pop_idx:pop_idx + nsubs]
        
        if len(site_pops) > 0 and np.sum(site_pops) > 0:
            max_pop = np.max(site_pops)
            total_pop = np.sum(site_pops)
            concentration_ratio = max_pop / total_pop
            
            if concentration_ratio > concentration_penalty_threshold:
                penalties -= gamma * 5.0 * (concentration_ratio - concentration_penalty_threshold)
        
        pop_idx += nsubs
    
    # ========== POSITIVE REWARDS ==========
    
    # Confidence Factor
    confidence_factor = min(1.0, min_transitions_across_sites / (2.0 * min_transitions_per_site))
    
    # R_P: Population balance reward
    R_P = 0.0
    if len(populations) > 1:
        nonzero_pops = pop_array[pop_array > 0]
        min_meaningful_coverage = max(2, total_sites * 1.5)
        
        if len(nonzero_pops) > 1 and len(nonzero_pops) >= min_meaningful_coverage:
            pop_mean = np.mean(nonzero_pops)
            pop_std = np.std(nonzero_pops)
            cv = pop_std / pop_mean if pop_mean > 0 else 1.0
            balance_factor = np.exp(-cv)
            total_pop_normalized = sum(p / P_baseline for p in nonzero_pops)
            R_P = w_P * total_pop_normalized * balance_factor * confidence_factor
        else:
            R_P = w_P * 0.01 * coverage_ratio * confidence_factor
    
    # R_T: Transition reward
    R_T = 0.0
    if sites_below_threshold == 0:
        total_trans = sum(transitions)
        R_T = w_T * (total_trans / T_baseline)
        avg_trans_per_site = total_trans / total_sites if total_sites > 0 else 0
        if avg_trans_per_site > min_transitions_per_site * 2:
            R_T *= 1.5
    
    # R_U: Coverage uniformity reward
    R_U = w_U * coverage_ratio
    
    # R_entropy: Shannon entropy bonus
    R_entropy = 0.0
    if np.sum(pop_array) > 0:
        prob_dist = pop_array / np.sum(pop_array)
        prob_dist = prob_dist[prob_dist > 0]
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        max_entropy = np.log(len(prob_dist))
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0
        R_entropy = entropy_bonus * entropy_score
    
    # Clamp penalties
    max_penalty = 50.0
    if penalties < -max_penalty:
        penalties = -max_penalty
    
    # Completeness gate: replace positive components with -0.01, keep penalties
    if nonzero_count < total_subs:
        R = -0.01 + penalties
    else:
        R = R_P + R_T + R_U + R_entropy + penalties

    return R


def reinforce_train_step(
    policy,
    optimizer: torch.optim.Optimizer,
    data,
    env_reward_fn: Callable,
    baseline: float = 0.0,
    lambda_entropy: float = 0.0
):
    """Perform one REINFORCE update with optional entropy regularization.

    Args:
        policy: instance of EdgePolicy
        optimizer: optimizer for policy parameters
        data: PyG data object with x, edge_index, edge_type, edge_attr
        env_reward_fn: callable(actions: torch.Tensor) -> float (or tensor)
        baseline: scalar baseline to reduce variance (default: 0.0)
        lambda_entropy: entropy regularization coefficient (default: 0.0).
            Positive values encourage exploration by maximizing policy entropy.
    
    Returns:
        (loss_value, reward): Tuple of scalar loss and scalar reward
    """
    policy.train()
    optimizer.zero_grad()

    x = data.x
    edge_index = data.edge_index
    edge_type = data.edge_type
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

    actions, logp, mean, std = policy.get_actions(x, edge_index, edge_type, edge_attr)

    # Evaluate reward from environment (user-provided function)
    # allow env_reward_fn to accept torch tensor or convert to numpy
    with torch.no_grad():
        r = env_reward_fn(actions)
        if isinstance(r, torch.Tensor):
            reward = float(r.item())
        else:
            reward = float(r)

    # policy loss: -E[logp * (reward - baseline)] ; sum over edges
    # here we treat reward as a scalar applied to all edge actions
    adv = reward - baseline
    policy_loss = -(logp.sum() * adv)
    
    # Entropy regularization: encourages exploration
    # H(π) = 0.5 * log(2πe * σ²) for Gaussian policy
    # We want to maximize entropy, so subtract it from loss (or add negative entropy)
    entropy_loss = 0.0
    if lambda_entropy > 0:
        # std is already returned from get_actions as log_std in position 3
        log_std = policy.log_std if hasattr(policy, 'log_std') else torch.zeros_like(mean)
        entropy = 0.5 * torch.log(2 * 3.14159 * 2.71828 * torch.exp(2 * log_std)).sum()
        entropy_loss = -lambda_entropy * entropy
    
    # Total loss
    loss = policy_loss + entropy_loss
    loss.backward()
    optimizer.step()

    return float(loss.item()), reward
