"""Pairwise MLP policy pretraining on collected MSLD simulation data.

This script trains the pairwise MLP policy to directly predict bias coefficients
from successful simulations using supervised learning (behavior cloning).

**Key Differences from Graph-Based Pretraining:**
- No graph construction: Uses substituent features directly
- Pairwise predictions: Predicts coefficients for each directed pair
- Linear bias conversion: Converts relative-to-first-sub format to absolute pairwise

**Linear Bias Handling:**
In variables.py files, linear biases (b matrix) are stored relative to the first
substituent at each site. For example, if site 1 has subs [s0, s1, s2]:
  - b[0,1] = linear bias of s1 relative to s0
  - b[0,2] = linear bias of s2 relative to s0
  - b[1,2] = b[0,2] - b[0,1] (linear bias of s2 relative to s1)

For pairwise MLP training, we convert these to absolute directional biases:
  - linear(s0→s1) = b[0,1]
  - linear(s1→s0) = -b[0,1] (antisymmetric)
  - linear(s1→s2) = b[1,2] = b[0,2] - b[0,1]
  
Linear biases only exist within sites. Cross-site pairs have linear=0.

**Data Requirements:**
- Must have bias coefficient matrices (c, x, s, b) in variables.py
- Must have simulation results to compute rewards for filtering
- RTF files used to extract substituent features

Usage:
    python -m mllf.cb.pretrain_pairwise_policy \\
        --pretraining-dir pretraining/14benz_solv \\
        --pretraining-dir pretraining/indole_solv \\
        --output-dir models/pretrained_pairwise_policy \\
        --config examples/workflow_pretrain.yaml \\
        --epochs 50
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

from mllf.cb.pairwise_mlp_policy import PairwiseMLPPolicy
from mllf.cb.pairwise_utils import (
    load_substituent_features_from_combo,
    load_substituent_features_from_graph_info,
)
from mllf.cb.pretrain_policy import (
    load_pretraining_runs,
    filter_best_runs_per_system,
    compute_reward_from_sim_results,
)


def extract_pairwise_targets_from_variables(
    variables_path: Path,
    nsubs_per_site: List[int]
) -> torch.Tensor:
    """Extract pairwise bias targets from variables.py file.
    
    Converts stored bias matrices to pairwise format for training.
    Extracts BOTH directions (i→j and j→i) to allow independent predictions
    for skew and end biases.
    
    Linear biases are converted from relative-to-first-sub to absolute values.
    For pair (i,j), linear_ij = b[j] - b[i] where b is the stored vector.
    
    Nonlinear biases (c, x, s) are extracted for all directed pairs.
    
    Args:
        variables_path: Path to variables.py file
        nsubs_per_site: Number of substituents per site [n1, n2, ...]
    
    Returns:
        Tensor of shape [num_pairs, 4] with [linear, quadratic, skew, end]
        for each directed pair (i,j) where i≠j within same site
    """
    # Load variables.py by parsing YAML (safer than exec)
    import yaml
    with open(variables_path, 'r') as f:
        content = f.read()
    
    # Extract YAML string between bias_string = """...""" or bias_string="""..."""
    yaml_start = content.find('bias_string = """') + len('bias_string = """')
    if yaml_start < len('bias_string = """'):
        # Try alternative format without space
        yaml_start = content.find('bias_string="""') + len('bias_string="""')
    yaml_end = content.find('"""', yaml_start)
    yaml_str = content[yaml_start:yaml_end]
    
    data = yaml.safe_load(yaml_str)
    
    # Extract bias coefficients
    # b is stored as [[val1, val2, ...]] - 2D list with single row
    # These are linear biases relative to first sub at each site
    b_list = data['b']
    if isinstance(b_list[0], list):
        # Flatten if it's [[...]]
        b_vector = np.array(b_list[0], dtype=float)
    else:
        b_vector = np.array(b_list, dtype=float)
    
    # c, x, s are stored as full NxN matrices in YAML
    c_matrix = np.array(data['c'], dtype=float)
    x_matrix = np.array(data['x'], dtype=float)
    s_matrix = np.array(data['s'], dtype=float)
    
    total_subs = sum(nsubs_per_site)
    if c_matrix.shape != (total_subs, total_subs):
        raise ValueError(f"c matrix shape {c_matrix.shape} doesn't match expected ({total_subs}, {total_subs})")
    num_sites = len(nsubs_per_site)
    
    # Build directed pairs (both directions) using same logic as build_directed_pairs
    from mllf.cb.pairwise_utils import build_directed_pairs
    pairs = build_directed_pairs(nsubs_per_site)
    
    # Extract coefficients for each directed pair
    targets = []
    for sub_i, sub_j in pairs:
        # Linear bias: convert from relative-to-first-sub to absolute pairwise
        # For pair (i,j): linear_ij = b[j] - b[i]
        # This naturally gives antisymmetric values: linear_ji = b[i] - b[j] = -linear_ij
        linear = b_vector[sub_j] - b_vector[sub_i]
        
        # Nonlinear coefficients: extract from stored matrices
        # For quadratic (antisymmetric): Use stored value which should satisfy c[j,i] = -c[i,j]
        # For skew/end (independent): Use stored value directly
        quadratic = c_matrix[sub_i, sub_j]
        skew = x_matrix[sub_i, sub_j]
        end = s_matrix[sub_i, sub_j]
        
        targets.append([linear, quadratic, skew, end])
    
    return torch.tensor(targets, dtype=torch.float32)


def pretrain_epoch(
    policy: PairwiseMLPPolicy,
    optimizer: optim.Optimizer,
    runs: List[Dict],
    device: torch.device,
) -> Dict[str, float]:
    """Run one behavior cloning epoch with MSE loss.
    
    Args:
        policy: Pairwise MLP policy
        optimizer: Optimizer
        runs: List of pretraining run dicts (should be best runs only)
        device: Device for computation
    
    Returns:
        Dict with epoch statistics
    """
    policy.train()
    
    epoch_loss = 0.0
    num_updates = 0
    
    for run_idx, run in enumerate(runs):
        run_dir = run["run_dir"]
        
        # Load substituent features from run directory
        # Prefer graph_info.json if available (for pretraining data)
        graph_info_path = run_dir / "graph_info.json"
        
        try:
            if graph_info_path.exists():
                # Load from graph_info.json (pretraining data)
                features, pairs, metadata = load_substituent_features_from_graph_info(
                    str(graph_info_path)
                )
            else:
                # Fallback to RTF parsing (if running on fresh combo dirs)
                try:
                    features, pairs, metadata = load_substituent_features_from_combo(
                        str(run_dir),
                        solvent_override=None
                    )
                except ValueError as e:
                    # Skip runs without graph_info.json or RTF files (incomplete data)
                    continue
            
            features = features.to(device)
            
            # Load target coefficients from variables.py
            variables_path = run_dir / "variables.py"
            if not variables_path.exists():
                print(f"  Warning: No variables.py for {run_dir.name}, skipping")
                continue
            
            nsubs_per_site = metadata['nsubs_per_site']
            targets = extract_pairwise_targets_from_variables(
                variables_path, nsubs_per_site
            ).to(device)
            
            if len(targets) == 0:
                print(f"  Warning: No target coefficients for {run_dir.name}, skipping")
                continue
            
        except Exception as e:
            print(f"  Error loading data for {run_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Convert pairs list to tensor for policy
        pairs_tensor = torch.tensor(pairs, dtype=torch.long, device=device)
        
        # Get predicted coefficients from policy (deterministic mean)
        _, _, predicted_means, _ = policy.get_actions(
            features, pairs_tensor, deterministic=True
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


def pretrain_pairwise(
    runs: List[Dict],
    output_dir: Path,
    config: Dict,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
):
    """Run pairwise MLP policy pretraining.
    
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
    reward_config = config.get('reward', {})
    print(f"\nFiltering {len(runs)} runs to keep only best per system...")
    print(f"Using reward config: {reward_config}")
    best_runs = filter_best_runs_per_system(runs, reward_config=reward_config)
    print(f"Filtered to {len(best_runs)} best runs for training\n")
    
    if len(best_runs) == 0:
        print("Error: No valid runs after filtering")
        return
    
    runs = best_runs
    
    # Get a sample run to determine feature dimension
    sample_features = None
    sample_metadata = None
    for run in runs:
        run_dir = run["run_dir"]
        graph_info_path = run_dir / "graph_info.json"
        
        try:
            if graph_info_path.exists():
                # Load from graph_info.json (pretraining data)
                features, pairs, metadata = load_substituent_features_from_graph_info(
                    str(graph_info_path)
                )
            else:
                # Fallback to RTF parsing
                features, pairs, metadata = load_substituent_features_from_combo(
                    str(run_dir),
                    solvent_override=None
                )
            
            sample_features = features
            sample_metadata = metadata
            break
        except Exception as e:
            print(f"  Warning: Could not load {run['run_dir'].name}: {e}")
            continue
    
    if sample_features is None:
        print("Error: Could not load any valid run for model initialization")
        return
    
    feature_dim = sample_metadata['feature_dim']
    
    # Create pairwise MLP policy using config
    train_config = config.get('training', {})
    policy_config = train_config.get('policy', {})
    
    # Support both pairwise_mlp and policy config keys
    pairwise_config = config.get('pairwise_mlp', policy_config)
    
    policy = PairwiseMLPPolicy(
        feature_dim=feature_dim,
        hidden_dims=pairwise_config.get('hidden_dims', [256, 128]),
        num_bias_types=4,
        bias_embed_dim=pairwise_config.get('bias_embed_dim', 16),
        dropout=pairwise_config.get('dropout', 0.1),
        feature_mode=pairwise_config.get('feature_mode', 'difference')  # Default: difference features
    ).to(device)
    
    optimizer = optim.Adam(
        policy.parameters(),
        lr=learning_rate
    )
    
    print(f"\nModel architecture:")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Feature mode: {policy.feature_mode}")
    print(f"  Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Training loop
    best_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Starting pairwise MLP behavior cloning for {epochs} epochs")
    print(f"Training on {len(runs)} best runs per system")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        stats = pretrain_epoch(
            policy, optimizer, runs, device
        )
        
        print(f"  MSE Loss: {stats['loss']:.4f}")
        print(f"  Runs processed: {stats['num_runs']}")
        
        # Save best model (lowest loss)
        if stats['loss'] < best_loss:
            best_loss = stats['loss']
            
            best_path = output_dir / "best_pairwise_policy.pt"
            torch.save({
                'policy_state': policy.state_dict(),
                'epoch': epoch + 1,
                'loss': stats['loss'],
            }, best_path)
            print(f"  Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        torch.save({
            'policy_state': policy.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch + 1,
            'stats': stats,
        }, checkpoint_path)
    
    # Save final model
    final_path = output_dir / "final_pairwise_policy.pt"
    torch.save({
        'policy_state': policy.state_dict(),
        'epoch': epochs,
    }, final_path)
    
    # Save metadata
    metadata = {
        'feature_dim': feature_dim,
        'policy_config': pairwise_config,
        'num_pretraining_runs': len(runs),
        'epochs': epochs,
        'best_loss': best_loss,
        'training_method': 'behavior_cloning_pairwise',
        'model_type': 'pairwise_mlp',
    }
    
    with open(output_dir / "pretrain_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Pairwise MLP behavior cloning complete!")
    print(f"Best MSE loss: {best_loss:.4f}")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain pairwise MLP policy on collected simulation data"
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
    pretrain_pairwise(
        runs=all_runs,
        output_dir=Path(args.output_dir),
        config=config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
