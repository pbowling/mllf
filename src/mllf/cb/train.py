"""Simple REINFORCE training loop helper for the edge-policy.

This file provides a helper `reinforce_train_step` that performs one
policy-gradient update given a callable environment evaluator that maps
actions -> reward. The environment evaluator must accept a NumPy or torch
tensor of edge actions and return a scalar reward.

Also includes `compute_msld_reward`, a default reward function for MSLD
simulations that computes scalarized rewards based on populations and
transitions from CHARMM output files.
"""
from pathlib import Path
from typing import Callable

import torch
import numpy as np

from mllf.file_handling.read_output import (
    parse_single_population,
    parse_transitions_and_rates,
    terminated_normally
) 


def compute_msld_reward(
    combo_dir: str,
    w_P: float = 0.5,
    w_T: float = 0.5,
    gamma: float = 10.0,
    P_baseline: float = 1000.0,
    T_baseline: float = 100.0
) -> float:
    """Compute scalarized reward from MSLD simulation outputs.
    
    This is the default reward function for MSLD simulations. It implements
    a scalarized reward that encourages:
      1. Higher total populations (normalized)
      2. Higher total transitions (normalized)
      3. Non-zero populations across all perturbations (bonus term)
    
    The reward function is:
        R = R_normalized + R_non_zero
    where:
        R_normalized = w_P * sum(P_i / P_baseline) + w_T * sum(T_j / T_baseline)
        R_non_zero = gamma * count(P_i > 0)
    
    Args:
        combo_dir: Path to combination directory with simulation outputs.
        w_P: Weight for population term (default: 0.5). Should be in [0, 1].
        w_T: Weight for transition term (default: 0.5). Should be in [0, 1].
        gamma: Bonus coefficient for non-zero populations (default: 10.0).
            Larger values more strongly enforce the non-zero constraint.
        P_baseline: Normalization baseline for populations (default: 1000.0).
            Should be set based on expected population magnitudes.
        T_baseline: Normalization baseline for transitions (default: 100.0).
            Should be set based on expected transition magnitudes.
    
    Returns:
        Scalar reward value (higher is better). Returns 0.0 if simulation
        did not complete successfully or if outputs cannot be parsed.
    
    Example:
        >>> reward = compute_msld_reward(
        ...     '/path/to/combo_001',
        ...     w_P=0.6, w_T=0.4,
        ...     gamma=15.0,
        ...     P_baseline=1500.0,
        ...     T_baseline=200.0
        ... )
    """
    
    combo_path = Path(combo_dir)
    
    # Look for simulation output file (typically .out file from SLURM or stdout redirect)
    output_file = None
    
    # Search for output files in common locations
    possible_outputs = [
        combo_path / 'output.out',  # From run.sh redirect
        combo_path / f'{combo_path.name}.out',  # SLURM default pattern
    ]
    
    # Also check for any .out files in the directory
    for out_file in combo_path.glob('*.out'):
        possible_outputs.append(out_file)
    
    # Find the first existing output file
    for candidate in possible_outputs:
        if candidate.exists():
            output_file = candidate
            break
    
    if output_file is None:
        print(f"  Warning: No output file found in {combo_dir}")
        return 0.0
    
    # Read output file
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
    except Exception as e:
        print(f"  Warning: Could not read {output_file}: {e}")
        return 0.0
    
    # Check if simulation terminated normally
    if not terminated_normally(output_text):
        print(f"  Warning: Simulation did not terminate normally in {combo_dir}")
        return 0.0
    
    # Parse populations: block -> {"counts": {lambda: count}, "site": site_idx}
    population_data = parse_single_population(output_text)
    
    # Parse transitions: site -> {lambda: count}
    transitions_data, _ = parse_transitions_and_rates(output_text)
    
    if not population_data and not transitions_data:
        print(f"  Warning: No population or transition data found in {output_file}")
        return 0.0
    
    # Extract P_i values: total counts across all lambdas for each block
    populations = []
    for block_id, block_info in population_data.items():
        counts_dict = block_info.get('counts', {})
        total_count = sum(counts_dict.values())
        populations.append(total_count)
    
    # Extract T_j values: total transitions across all lambdas for each site
    transitions = []
    for site_id, trans_dict in transitions_data.items():
        total_trans = sum(trans_dict.values())
        transitions.append(total_trans)
    
    # Calculate normalized reward components
    # Encourage balanced populations (not concentrated in one substituent)
    R_P = 0.0
    if populations:
        # Penalize if most population is concentrated in a single substituent
        # Use coefficient of variation (std/mean) as a measure of imbalance
        pop_array = np.array(populations)
        nonzero_pops = pop_array[pop_array > 0]
        
        if len(nonzero_pops) > 1:
            # Reward distributed populations (lower CV is better)
            pop_mean = np.mean(nonzero_pops)
            pop_std = np.std(nonzero_pops)
            cv = pop_std / pop_mean if pop_mean > 0 else 1.0
            
            # Balance factor: ranges from 0 (highly imbalanced) to 1 (perfectly balanced)
            # Use exp(-cv) to map CV to balance score
            balance_factor = np.exp(-cv)
            
            # Reward based on total population magnitude AND balance
            total_pop_normalized = sum(P_i / P_baseline for P_i in populations)
            R_P = w_P * total_pop_normalized * balance_factor
        else:
            # Only one non-zero population is bad, give minimal reward
            R_P = w_P * 0.1 * (populations[0] / P_baseline)
    
    R_T = 0.0
    if transitions:
        # Reward total transitions (more transitions = better sampling)
        total_trans = sum(transitions)
        R_T = w_T * (total_trans / T_baseline)
        
        # Penalize if transitions are too low (< 10)
        if total_trans < 10:
            R_T *= 0.1  # Severely penalize low transition counts
    
    R_normalized = R_P + R_T
    
    # Calculate non-zero constraint bonus (encourage all substituents to be sampled)
    nonzero_count = sum(1 for P_i in populations if P_i > 0)
    R_non_zero = gamma * nonzero_count
    
    # Penalty for worst case: single population dominant, very few transitions
    penalty = 0.0
    if populations and transitions:
        total_trans = sum(transitions)
        max_pop = max(populations)
        total_pop = sum(populations)
        
        # If one population has >90% of total AND transitions < 10: heavy penalty
        if total_pop > 0 and max_pop / total_pop > 0.9 and total_trans < 10:
            penalty = -gamma * 2.0  # Double the gamma penalty
    
    # Final reward
    R = R_normalized + R_non_zero + penalty
    
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
