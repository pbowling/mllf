#!/usr/bin/env python3
"""Test script for pairwise MLP with 14benz data.

This script validates that the pairwise MLP can:
1. Load the pretrained model
2. Process 14benz RTF files  
3. Generate predictions for real combinations
4. Convert predictions to valid bias coefficients

Usage:
    python test_14benz_mlp.py [--pretrain-path PATH]
"""
import argparse
from pathlib import Path
import torch
import numpy as np

from mllf.cb.pairwise_mlp_policy import PairwiseMLPPolicy
from mllf.cb.pairwise_utils import (
    load_substituent_features_from_combo,
    build_directed_pairs,
    predictions_to_bias_dict,
)


def test_14benz_combinations(pretrain_path: Path):
    """Test pairwise MLP on 14benz combinations.
    
    Args:
        pretrain_path: Path to pretrained model
    """
    print("=" * 70)
    print("Pairwise MLP Test with 14benz Data")
    print("=" * 70)
    print()
    
    # Find 14benz combinations
    combo_base = Path("/home/pbowling/mllf/examples/14benz/generated_combos")
    
    if not combo_base.exists():
        print(f"Error: 14benz combinations not found at {combo_base}")
        print("Please generate combinations first:")
        print("  cd /home/pbowling/mllf/examples/mlp")
        print("  python -m mllf.file_handling.generate_combinations workflow_14benz.yaml")
        return 1
    
    # Find all combination directories
    combo_dirs = sorted(combo_base.glob("comb_*"))
    
    if not combo_dirs:
        print(f"Error: No combination directories found in {combo_base}")
        return 1
    
    print(f"Found {len(combo_dirs)} combinations")
    print()
    
    # Test on first few combinations
    test_combos = combo_dirs[:5]
    
    # Load pretrained model
    print("Loading pretrained model...")
    if not pretrain_path.exists():
        print(f"Error: Pretrained model not found at {pretrain_path}")
        print("Please run pretraining first:")
        print("  cd /home/pbowling/mllf/examples/mlp")
        print("  ./run_pretraining.sh")
        return 1
    
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    
    # Load metadata
    metadata_path = pretrain_path.parent / "pretrain_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return 1
    
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Initialize policy with same architecture as pretraining
    policy = PairwiseMLPPolicy(
        feature_dim=metadata['feature_dim'],
        hidden_dims=metadata['policy_config']['hidden_dims'],
        num_bias_types=metadata['policy_config']['num_bias_types'],
        bias_embed_dim=metadata['policy_config']['bias_embed_dim'],
        dropout=metadata['policy_config']['dropout']
    )
    
    policy.load_state_dict(checkpoint['policy_state'])
    policy.eval()
    
    print(f"  Model loaded: {sum(p.numel() for p in policy.parameters()):,} parameters")
    print(f"  Feature dimension: {metadata['feature_dim']}")
    print(f"  Pretrained on {metadata['num_pretraining_runs']} runs")
    print(f"  Best pretraining loss: {metadata['best_loss']:.2f}")
    print()
    
    # Test each combination
    print("Testing combinations...")
    print()
    
    successful = 0
    failed = 0
    
    for combo_dir in test_combos:
        combo_name = combo_dir.name
        print(f"Testing {combo_name}...")
        
        try:
            # Load features from RTF files
            features, pairs, metadata = load_substituent_features_from_combo(
                str(combo_dir),
                solvent_override=None
            )
            
            nsubs_per_site = metadata['nsubs_per_site']
            print(f"  Sites: {nsubs_per_site}")
            print(f"  Features: {features.shape}")
            print(f"  Pairs: {len(pairs)}")
            
            # Generate predictions
            pairs_tensor = torch.tensor(pairs, dtype=torch.long)
            
            with torch.no_grad():
                actions, logp, means, log_stds = policy.get_actions(
                    features, pairs_tensor, deterministic=True
                )
            
            print(f"  Predictions shape: {actions.shape}")
            
            # Check prediction ranges
            linear = actions[:, 0]
            quadratic = actions[:, 1]
            skew = actions[:, 2]
            end = actions[:, 3]
            
            print(f"  Linear range: [{linear.min():.2f}, {linear.max():.2f}]")
            print(f"  Quadratic range: [{quadratic.min():.2f}, {quadratic.max():.2f}]")
            print(f"  Skew range: [{skew.min():.2f}, {skew.max():.2f}]")
            print(f"  End range: [{end.min():.2f}, {end.max():.2f}]")
            
            # Convert to bias matrices
            bias_dict = predictions_to_bias_dict(actions, pairs, nsubs_per_site)
            
            # Verify shapes
            total_subs = sum(nsubs_per_site)
            b = np.array(bias_dict['b'])
            c = np.array(bias_dict['c'])
            
            assert b.shape == (total_subs, total_subs), f"Invalid b shape: {b.shape}"
            assert c.shape == (total_subs, total_subs), f"Invalid c shape: {c.shape}"
            
            # Verify antisymmetry
            b_antisym_error = np.abs(b + b.T).max()
            c_antisym_error = np.abs(c + c.T).max()
            
            print(f"  Antisymmetry: b={b_antisym_error:.6f}, c={c_antisym_error:.6f}")
            
            if b_antisym_error < 1e-5 and c_antisym_error < 1e-5:
                print(f"  ✓ Success")
                successful += 1
            else:
                print(f"  ✗ Antisymmetry violation")
                failed += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        
        print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"  Successful: {successful}/{len(test_combos)}")
    print(f"  Failed: {failed}/{len(test_combos)}")
    print()
    
    if successful == len(test_combos):
        print("✓ All tests passed! Pairwise MLP is ready for 14benz training.")
        print()
        print("Next steps:")
        print("  1. Review configuration: workflow_14benz_mlp.yaml")
        print("  2. Start training (requires SLURM setup):")
        print("     python run_pairwise_training.py workflow_14benz_mlp.yaml")
        return 0
    else:
        print("✗ Some tests failed. Please review errors above.")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test pairwise MLP with 14benz data"
    )
    parser.add_argument(
        '--pretrain-path',
        type=str,
        default='models/pretrained_pairwise/best_pairwise_policy.pt',
        help='Path to pretrained model'
    )
    
    args = parser.parse_args()
    
    pretrain_path = Path(args.pretrain_path)
    return test_14benz_combinations(pretrain_path)


if __name__ == '__main__':
    exit(main())
