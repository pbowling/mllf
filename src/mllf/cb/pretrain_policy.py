"""Policy pretraining on collected MSLD simulation data via Behavior Cloning.

This script trains the policy to directly predict bias coefficients from successful
simulations using supervised learning (behavior cloning). This is fundamentally
different from REINFORCE training:

**Behavior Cloning Approach:**
- Extract bias coefficients from successful runs as training targets
- Train policy to predict these coefficients using MSE loss
- Filter data to use only the best runs (highest rewards per system)
- Requires 50-100 epochs for convergence

**Key Differences from run_workflow.py:**
- No REINFORCE: Uses supervised MSE loss instead of policy gradients
- No new simulations run: Learns from historical bias coefficients
- Uses only best runs: Filters for highest-reward runs per system
- Multiple epochs needed: Not deterministic - gradient descent on MSE

**Data Requirements:**
- Must have bias coefficient matrices (c, x, s, b) in variables.py
- Must have simulation results to compute rewards for filtering
- RTF files used to build graph structure

This allows the policy to learn good bias coefficient predictions from
successful simulations before running expensive RL episodes.

Usage:
    python -m mllf.cb.pretrain_policy \\
        --pretraining-dir pretraining/14benz_solv \\
        --pretraining-dir pretraining/indole_solv \\
        --output-dir models/pretrained_policy \\
        --config examples/workflow_pretrain.yaml \\
        --epochs 1
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data

from mllf.cli.workflow import build_data_and_targets_from_combo
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy


def build_graph_from_saved_data(run_dir: Path, toppar_dir=None, toppar_files=None, warn_missing_types=True):
    """Build PyG graph from saved variables.py.
    
    Args:
        run_dir: Directory containing variables.py
        toppar_dir: Path to toppar directory (None uses package default)
        toppar_files: List of specific toppar filenames to include
        warn_missing_types: If True, warn when sub RTF files contain atom types not in vocabulary
    
    Returns:
        Tuple of (data, targets, extras)
    """
    # Use the existing workflow function to build graph from variables.py
    return build_data_and_targets_from_combo(
        str(run_dir), 
        toppar_dir=toppar_dir,
        toppar_files=toppar_files,
        warn_missing_types=warn_missing_types
    )


def filter_best_runs_per_system(runs: List[Dict]) -> List[Dict]:
    """Filter runs to keep only the best run per unique system.
    
    Groups runs by their dataset (pretraining parent directory) and keeps only the
    run with the highest reward for each dataset. This ensures we get one representative
    run from each system (e.g., one from 14benz_solv, one from indole_prot, etc.)
    rather than training on all runs.
    
    For 14benz_combos_best, which contains different combinations, we group by the
    combination name (e.g., comb_0001_site1_1__site1_2) to get one run per combination.
    
    Args:
        runs: List of run dicts with 'run_dir', 'source_dir', 'metadata', 'sim_results'
    
    Returns:
        Filtered list containing only best run per system/combination
    """
    from collections import defaultdict
    
    # Group runs by system identifier
    systems = defaultdict(list)
    for run in runs:
        run_dir = run.get('run_dir')
        
        # Extract system identifier from the run directory path
        # For directories like pretraining/14benz_solv/run1, we want "14benz_solv"
        # For pretraining/14benz_combos_best/comb_0001_..._run_001, we want the combo name
        if run_dir:
            parent_dir = run_dir.parent.name  # e.g., "14benz_solv" or "14benz_combos_best"
            
            # Special handling for 14benz_combos_best: group by combination, not parent
            if 'combo' in parent_dir.lower():
                # Extract combination name (everything before _run_NNN)
                run_name = run_dir.name
                if '_run_' in run_name:
                    combo_name = run_name.rsplit('_run_', 1)[0]
                    system_id = f"{parent_dir}/{combo_name}"
                else:
                    system_id = f"{parent_dir}/{run_name}"
            else:
                # For other datasets, use parent directory as system ID
                system_id = parent_dir
        else:
            # Fallback to source_dir if run_dir not available
            source = run.get('source_dir', 'unknown')
            system_id = source
        
        systems[system_id].append(run)
    
    # Compute rewards and keep best per system
    best_runs = []
    for system_name, system_runs in sorted(systems.items()):
        if not system_runs:
            continue
        
        # Compute reward for each run
        run_rewards = []
        for run in system_runs:
            metadata = run.get('metadata', {})
            num_sites = metadata.get('num_sites', 2)
            num_substituents = metadata.get('num_substituents', 0)
            
            # Extract actual nsubs_per_site from graph_info.json if available
            run_dir = run.get('run_dir')
            nsubs_per_site = None
            if run_dir:
                graph_info_path = run_dir / 'graph_info.json'
                if graph_info_path.exists():
                    try:
                        import json
                        with open(graph_info_path, 'r') as f:
                            graph_info = json.load(f)
                        if 'sites' in graph_info:
                            # Count substituents per site from graph_info
                            from collections import defaultdict
                            site_counts = defaultdict(int)
                            for site_key in graph_info['sites']:
                                # Parse "site1_sub2" -> site number
                                site_num = int(site_key.split('_')[0].replace('site', ''))
                                site_counts[site_num] += 1
                            # Convert to ordered list
                            nsubs_per_site = [site_counts[i] for i in sorted(site_counts.keys())]
                    except Exception as e:
                        print(f"  Warning: Could not parse graph_info.json: {e}")
            
            # Fallback: estimate from total (asymmetric-safe distribution)
            if nsubs_per_site is None:
                if num_substituents > 0 and num_sites > 0:
                    nsubs_per_site = [num_substituents // num_sites] * num_sites
                    for i in range(num_substituents % num_sites):
                        nsubs_per_site[i] += 1
                else:
                    nsubs_per_site = [3] * num_sites
            
            # Compute reward (using defaults for filtering)
            reward = compute_reward_from_sim_results(
                run['sim_results'],
                num_sites=num_sites,
                nsubs_per_site=nsubs_per_site
            )
            run_rewards.append((run, reward))
        
        # Keep only the best run
        if run_rewards:
            best_run, best_reward = max(run_rewards, key=lambda x: x[1])
            best_runs.append(best_run)
            print(f"  {system_name}: selected run with reward {best_reward:.2f}")
    
    return best_runs


def load_pretraining_runs(pretraining_dir: Path) -> List[Dict]:
    """Load all collected pretraining runs.
    
    Args:
        pretraining_dir: Directory with collected run data
    
    Returns:
        List of dicts with run_dir, metadata, sim_results for each run
    """
    runs = []
    
    for run_dir in sorted(pretraining_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Load metadata
        metadata_file = run_dir / "metadata.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Skip if simulation didn't terminate normally
        if not metadata.get("terminated_normally", False):
            print(f"  Skipping {run_dir.name}: did not terminate normally")
            continue
        
        # Load simulation results
        results_file = run_dir / "simulation_results.json"
        if not results_file.exists():
            print(f"  Skipping {run_dir.name}: no simulation_results.json")
            continue
        
        with open(results_file, 'r') as f:
            sim_results = json.load(f)
        
        runs.append({
            "run_dir": run_dir,
            "source_dir": metadata.get("source_run_dir"),
            "metadata": metadata,
            "sim_results": sim_results,
        })
    
    return runs


def compute_reward_from_sim_results(
    sim_results: Dict,
    num_sites: int,
    nsubs_per_site: List[int],
    w_P: float = 0.5,
    w_T: float = 0.5,
    w_U: float = 0.3,
    gamma: float = 4.0,
    P_baseline: float = 500.0,
    T_baseline: float = 50.0,
    min_transitions_per_site: int = 10,
    min_coverage_ratio: float = 0.5,
    entropy_bonus: float = 8.0,
    concentration_penalty_threshold: float = 0.8,
) -> float:
    """Compute reward from simulation results dict using improved reward logic.
    
    This implements the same logic as train_improved.py but works with cached
    simulation results instead of reading from output files.
    
    Args:
        sim_results: Dict with 'populations' and 'transitions' keys
        num_sites: Number of sites in the system
        nsubs_per_site: List of number of substituents per site
        w_P, w_T, w_U: Reward weights for populations, transitions, and uniformity
        gamma: Scaling factor for rewards
        P_baseline: Normalization baseline for populations
        T_baseline: Normalization baseline for transitions
        min_transitions_per_site: Minimum transitions required per site
        min_coverage_ratio: Minimum ratio of substituents that must be visited
        entropy_bonus: Bonus for uniform distributions
        concentration_penalty_threshold: Threshold for concentration penalty (e.g., 0.8 = 80%)
    
    Returns:
        Scalar reward value
    """
    populations = sim_results.get("populations", {})
    transitions = sim_results.get("transitions", {})
    
    # Extract population counts (use highest lambda value)
    pop_list = []
    for block_id in sorted([int(k) for k in populations.keys()]):
        block_data = populations[str(block_id)]
        counts = block_data.get("counts", {})
        if counts:
            max_lambda = max(counts.keys(), key=lambda x: float(x))
            pop_list.append(counts[max_lambda])
    
    if not pop_list:
        return -100.0 * gamma  # No population data
    
    pop_array = np.array(pop_list, dtype=float)
    total_pop = pop_array.sum()
    
    if total_pop == 0:
        return -100.0 * gamma  # No sampling occurred
    
    # Extract transition counts per site
    trans_per_site = []
    for site_id in sorted([int(k) for k in transitions.keys()]):
        site_data = transitions[str(site_id)]
        if site_data:
            max_lambda = max(site_data.keys(), key=lambda x: float(x))
            trans_per_site.append(site_data[max_lambda])
        else:
            trans_per_site.append(0)
    
    # Ensure we have transition data for all sites
    while len(trans_per_site) < num_sites:
        trans_per_site.append(0)
    
    trans_array = np.array(trans_per_site[:num_sites], dtype=float)
    total_trans = trans_array.sum()
    
    # === STRICT REQUIREMENTS (penalties) ===
    penalty = 0.0
    
    # 1. Per-site minimum transitions requirement
    for site_idx, trans_count in enumerate(trans_array):
        if trans_count < min_transitions_per_site:
            shortage = min_transitions_per_site - trans_count
            penalty += gamma * shortage  # Linear penalty for shortage
    
    # 2. Coverage requirement (minimum % of substituents visited)
    num_populated = np.count_nonzero(pop_array)
    total_subs = sum(nsubs_per_site)
    coverage_ratio = num_populated / total_subs if total_subs > 0 else 0.0
    
    if coverage_ratio < min_coverage_ratio:
        shortage = min_coverage_ratio - coverage_ratio
        penalty += gamma * 10.0 * shortage  # Heavy penalty for poor coverage
    
    # 3. Concentration penalty (per-site check)
    pop_idx = 0
    for site_idx, nsubs in enumerate(nsubs_per_site):
        site_pops = pop_array[pop_idx:pop_idx + nsubs]
        site_total = site_pops.sum()
        
        if site_total > 0:
            site_max_ratio = site_pops.max() / site_total
            if site_max_ratio > concentration_penalty_threshold:
                excess = site_max_ratio - concentration_penalty_threshold
                penalty += gamma * 5.0 * excess  # Penalty for concentration
        
        pop_idx += nsubs
    
    # Clamp penalties to prevent gradient explosion
    # Cap at -50.0 to ensure single bad run doesn't destroy model weights
    max_penalty = 50.0
    if penalty > max_penalty:
        penalty = max_penalty
    
    # If penalties are severe, return negative reward immediately
    if penalty > gamma * 20:
        return -penalty
    
    # === REWARD COMPONENTS ===
    
    # R_P: Population balance (coefficient of variation - lower is better/more uniform)
    pop_probs = pop_array / total_pop  # Needed for entropy calculation below
    nonzero_pops = pop_array[pop_array > 0]
    
    if len(nonzero_pops) > 1:
        pop_mean = np.mean(nonzero_pops)
        pop_std = np.std(nonzero_pops)
        cv = pop_std / pop_mean if pop_mean > 0 else 10.0
        balance_factor = np.exp(-cv)  # Use exp(-cv) like train_improved.py
        
        # Sum per-substituent normalized populations (matching train_improved.py)
        total_pop_normalized = np.sum(pop_array / P_baseline)
        R_P = w_P * balance_factor * total_pop_normalized
    else:
        # Only one substituent visited: minimal reward
        R_P = w_P * 0.01
    
    # R_T: Transitions (normalized by baseline)
    R_T = w_T * (total_trans / T_baseline)
    
    # R_U: Coverage uniformity reward (matching train_improved.py)
    R_U = w_U * coverage_ratio
    
    # R_entropy: Shannon entropy bonus for uniform distributions
    entropy = -np.sum(pop_probs * np.log(pop_probs + 1e-10))
    max_entropy = np.log(len(pop_probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    R_entropy = entropy_bonus * normalized_entropy
    
    # Total reward
    reward = R_P + R_T + R_U + R_entropy - penalty
    
    return reward


def pretrain_epoch(
    encoder: nn.Module,
    policy: nn.Module,
    optimizer: optim.Optimizer,
    runs: List[Dict],
    reward_config: Dict,
    device: torch.device,
    toppar_dir=None,
    toppar_files=None,
    warn_missing_types=True,
) -> Dict[str, float]:
    """Run one behavior cloning epoch with MSE loss.
    
    Args:
        encoder: GNN encoder
        policy: Edge policy
        optimizer: Optimizer
        runs: List of pretraining run dicts (should be best runs only)
        reward_config: Reward function configuration (unused in BC)
        device: Device for computation
        toppar_dir: Path to toppar directory (None uses package default)
        toppar_files: List of specific toppar filenames to include
        warn_missing_types: If True, warn when sub RTF files contain atom types not in vocabulary
    
    Returns:
        Dict with epoch statistics
    """
    policy.train()
    encoder.train()
    
    epoch_loss = 0.0
    num_updates = 0
    
    for run_idx, run in enumerate(runs):
        run_dir = run["run_dir"]
        
        # Build graph from saved data AND get target coefficients
        try:
            data, targets, extras = build_graph_from_saved_data(
                run_dir, 
                toppar_dir=toppar_dir,
                toppar_files=toppar_files,
                warn_missing_types=warn_missing_types
            )
            data = data.to(device)
            
            # Targets contain the actual bias coefficients from successful run
            if targets is None or len(targets) == 0:
                print(f"  Warning: No target coefficients for {run_dir.name}, skipping")
                continue
            
            # Convert targets list to tensor
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            
        except Exception as e:
            print(f"  Error building graph for {run_dir.name}: {e}")
            continue
        
        # Get predicted coefficients from policy (deterministic mean)
        _, _, predicted_means, _ = policy.get_actions(
            data.x, data.edge_index, data.edge_type, data.edge_attr,
            deterministic=True  # Use mean predictions, not sampled
        )
        
        # Behavior Cloning: MSE loss between predicted and target coefficients
        mse_loss = nn.functional.mse_loss(predicted_means, targets)
        
        # Check for NaN/inf
        if torch.isnan(mse_loss) or torch.isinf(mse_loss):
            print(f"  Warning: NaN/inf loss for {run['run_dir'].name}, skipping")
            continue
        
        # Update
        optimizer.zero_grad()
        mse_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += mse_loss.item()
        num_updates += 1
        
        if (run_idx + 1) % 10 == 0:
            avg_loss = epoch_loss / num_updates
            print(f"  Run {run_idx+1}/{len(runs)}: mse_loss={mse_loss.item():.4f}, avg_loss={avg_loss:.4f}")
    
    avg_loss = epoch_loss / num_updates if num_updates > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'num_runs': num_updates,
    }


def pretrain(
    pretraining_dir: Path,
    output_dir: Path,
    config: Dict,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
):
    """Run policy pretraining (legacy single-directory interface).
    
    Args:
        pretraining_dir: Directory with collected pretraining data
        output_dir: Directory to save pretrained policy
        config: Configuration dict (same format as workflow_sample.yaml)
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device ('cuda' or 'cpu'). If None, auto-detect.
    """
    # Load runs from single directory
    print(f"\nLoading pretraining data from {pretraining_dir}...")
    runs = load_pretraining_runs(pretraining_dir)
    print(f"Loaded {len(runs)} runs with successful simulations")
    
    # Call the main pretraining function
    pretrain_with_runs(runs, output_dir, config, epochs, learning_rate, device)


def pretrain_with_runs(
    runs: List[Dict],
    output_dir: Path,
    config: Dict,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
):
    """Run policy pretraining with provided runs.
    
    Args:
        runs: List of run dicts (from load_pretraining_runs)
        output_dir: Directory to save pretrained policy
        config: Configuration dict (same format as workflow_sample.yaml)
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device ('cuda' or 'cpu'). If None, auto-detect.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    if len(runs) == 0:
        print("Error: No valid runs found")
        return
    
    # Filter to keep only best run per system for behavior cloning
    print(f"\nFiltering {len(runs)} runs to keep only best per system...")
    best_runs = filter_best_runs_per_system(runs)
    print(f"Filtered to {len(best_runs)} best runs for training\n")
    
    if len(best_runs) == 0:
        print("Error: No valid runs after filtering")
        return
    
    # Update runs to use filtered best runs
    runs = best_runs
    
    # Extract toppar configuration
    vocab_config = config.get('vocabulary', {})
    toppar_dir = vocab_config.get('toppar_dir')
    toppar_files = vocab_config.get('toppar_files')
    warn_missing_types = vocab_config.get('warn_missing_types', True)
    
    # Get a sample run to build model architecture (find one with edges)
    sample_data = None
    sample_extras = None
    for run in runs:
        try:
            data, _, extras = build_data_and_targets_from_combo(
                str(run["run_dir"]),
                toppar_dir=toppar_dir,
                toppar_files=toppar_files,
                warn_missing_types=warn_missing_types
            )
            if data.edge_index.size(1) > 0:  # Has edges
                sample_data = data
                sample_extras = extras
                break
        except Exception as e:
            continue
    
    if sample_data is None:
        print("Error: Could not find a valid graph with edges")
        return
    
    # Create model using config (same as workflow)
    train_config = config.get('training', {})
    encoder_config = train_config.get('encoder', {})
    policy_config = train_config.get('policy', {})
    
    encoder = RGCNEncoder(
        in_dim=sample_data.x.size(1),
        hidden_dims=encoder_config.get('hidden_dims', [64, 64]),
        out_dim=encoder_config.get('out_dim', 32),
        num_relations=sample_data.edge_type.max().item() + 1
    ).to(device)
    
    policy = EdgePolicy.from_pyg_data(
        encoder=encoder,
        emb_dim=encoder_config.get('out_dim', 32),
        data=sample_data,
        mlp_hidden=policy_config.get('mlp_hidden', 64),
        mlp_out_dim=len(sample_extras['relation_names']) // 2
    ).to(device)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(policy.parameters()),
        lr=learning_rate
    )
    
    print(f"\nModel architecture:")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())} params")
    print(f"  Policy: {sum(p.numel() for p in policy.parameters())} params")
    
    # Training loop
    reward_config = config.get('reward', {})
    best_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Starting behavior cloning for {epochs} epochs")
    print(f"Training on {len(runs)} best runs per system")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        stats = pretrain_epoch(
            encoder, policy, optimizer, runs, reward_config, device,
            toppar_dir=toppar_dir,
            toppar_files=toppar_files,
            warn_missing_types=warn_missing_types
        )
        
        print(f"  MSE Loss: {stats['loss']:.4f}")
        print(f"  Runs processed: {stats['num_runs']}")
        
        # Save best model (lowest loss)
        if stats['loss'] < best_loss:
            best_loss = stats['loss']
            
            best_path = output_dir / "best_policy.pt"
            torch.save({
                'encoder_state': encoder.state_dict(),
                'policy_state': policy.state_dict(),
                'epoch': epoch + 1,
                'loss': stats['loss'],
            }, best_path)
            print(f"  Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        torch.save({
            'encoder_state': encoder.state_dict(),
            'policy_state': policy.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch + 1,
            'stats': stats,
        }, checkpoint_path)
    
    # Save final model
    final_path = output_dir / "final_policy.pt"
    torch.save({
        'encoder_state': encoder.state_dict(),
        'policy_state': policy.state_dict(),
        'epoch': epochs,
    }, final_path)
    
    # Save metadata
    metadata = {
        'node_feat_dim': sample_data.x.size(1),
        'num_relations': sample_data.edge_type.max().item() + 1,
        'encoder_config': encoder_config,
        'policy_config': policy_config,
        'num_pretraining_runs': len(runs),
        'epochs': epochs,
        'best_loss': best_loss,
        'training_method': 'behavior_cloning',
    }
    
    with open(output_dir / "pretrain_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Behavior cloning complete!")
    print(f"Best MSE loss: {best_loss:.4f}")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain MSLD policy on collected simulation data"
    )
    parser.add_argument(
        "--pretraining-dir",
        type=str,
        action='append',
        required=True,
        help="Directory containing collected pretraining data (can specify multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save pretrained policy",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/workflow_sample.yaml",
        help="Config file (same format as workflow_sample.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50 for behavior cloning convergence)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Combine runs from all pretraining directories
    all_runs = []
    for pretrain_dir_str in args.pretraining_dir:
        pretrain_dir = Path(pretrain_dir_str)
        print(f"\nLoading from {pretrain_dir}...")
        runs = load_pretraining_runs(pretrain_dir)
        print(f"  Loaded {len(runs)} runs")
        all_runs.extend(runs)
    
    print(f"\nTotal runs from all directories: {len(all_runs)}")
    
    # Run pretraining with combined runs
    pretrain_with_runs(
        runs=all_runs,
        output_dir=Path(args.output_dir),
        config=config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
