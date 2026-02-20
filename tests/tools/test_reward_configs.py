#!/usr/bin/env python3
"""Test different reward configurations on existing simulation data.

This script loads cached epoch results (with raw simulation metrics) and
recomputes rewards using different hyperparameter configurations. This allows
rapid experimentation with reward functions without re-running expensive
simulations.

Usage:
    python test_reward_configs.py <results_dir> [--configs configs.yaml]

Example:
    # Test multiple configurations on epoch 5 results
    python test_reward_configs.py training_output/epoch_005 --configs reward_configs.yaml
    
    # Or test a single configuration interactively
    python test_reward_configs.py training_output/epoch_005 \\
        --w_P 0.7 --w_T 0.3 --gamma 15.0
"""
import argparse
from pathlib import Path
import torch
import yaml
import numpy as np
from typing import Dict, List

from mllf.cb.train_improved import compute_reward_from_raw_metrics


def load_epoch_results(epoch_dir: Path) -> List[Dict]:
    """Load all epoch_results.pt files from an epoch directory.
    
    Args:
        epoch_dir: Directory containing combination subdirectories with epoch_results.pt
    
    Returns:
        List of dicts containing checkpoint data (actions, logp, populations, transitions, etc.)
    """
    results = []
    
    for combo_dir in epoch_dir.iterdir():
        if not combo_dir.is_dir():
            continue
        
        results_file = combo_dir / 'epoch_results.pt'
        if not results_file.exists():
            continue
        
        try:
            checkpoint = torch.load(results_file, map_location='cpu', weights_only=False)
            
            # Skip if raw metrics not available
            if 'populations' not in checkpoint or 'transitions' not in checkpoint:
                print(f"  Warning: {combo_dir.name} missing raw metrics, skipping")
                continue
            
            checkpoint['combo_dir'] = combo_dir.name
            results.append(checkpoint)
        except Exception as e:
            print(f"  Error loading {results_file}: {e}")
    
    return results


def evaluate_reward_config(
    results: List[Dict],
    w_P: float = 0.5,
    w_T: float = 0.5,
    gamma: float = 10.0,
    P_baseline: float = 1000.0,
    T_baseline: float = 100.0,
    config_name: str = "default"
) -> Dict:
    """Evaluate a reward configuration on cached results.
    
    Args:
        results: List of checkpoint dicts with raw metrics
        w_P: Weight for population term
        w_T: Weight for transition term
        gamma: Non-zero population bonus
        P_baseline: Population normalization baseline
        T_baseline: Transition normalization baseline
        config_name: Name for this configuration
    
    Returns:
        Dict with statistics about this configuration
    """
    rewards = []
    
    for result in results:
        reward = compute_reward_from_raw_metrics(
            populations=result['populations'],
            transitions=result['transitions'],
            w_P=w_P,
            w_T=w_T,
            gamma=gamma,
            P_baseline=P_baseline,
            T_baseline=T_baseline
        )
        rewards.append(reward)
    
    return {
        'config_name': config_name,
        'w_P': w_P,
        'w_T': w_T,
        'gamma': gamma,
        'P_baseline': P_baseline,
        'T_baseline': T_baseline,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'median_reward': np.median(rewards),
        'num_samples': len(rewards)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test different reward configurations on existing simulation data'
    )
    parser.add_argument(
        'results_dir',
        type=Path,
        help='Directory containing epoch results (e.g., training_output/epoch_005)'
    )
    parser.add_argument(
        '--configs',
        type=Path,
        help='YAML file with reward configurations to test'
    )
    parser.add_argument('--w_P', type=float, help='Population weight')
    parser.add_argument('--w_T', type=float, help='Transition weight')
    parser.add_argument('--gamma', type=float, help='Non-zero population bonus')
    parser.add_argument('--P_baseline', type=float, help='Population baseline')
    parser.add_argument('--T_baseline', type=float, help='Transition baseline')
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")
    
    print(f"Loading epoch results from {args.results_dir}...")
    results = load_epoch_results(args.results_dir)
    
    if not results:
        print("No valid epoch results found!")
        return
    
    print(f"Loaded {len(results)} combination results")
    print(f"  Each has populations: {len(results[0]['populations'])}")
    print(f"  Each has transitions: {len(results[0]['transitions'])}")
    
    # Test configurations
    test_configs = []
    
    if args.configs:
        # Load configs from YAML
        print(f"\nLoading configurations from {args.configs}...")
        with open(args.configs) as f:
            configs_data = yaml.safe_load(f)
        
        for name, config in configs_data.items():
            test_configs.append({
                'config_name': name,
                **config
            })
    elif any([args.w_P, args.w_T, args.gamma, args.P_baseline, args.T_baseline]):
        # Single config from command line
        test_configs.append({
            'config_name': 'custom',
            'w_P': args.w_P or 0.5,
            'w_T': args.w_T or 0.5,
            'gamma': args.gamma or 10.0,
            'P_baseline': args.P_baseline or 1000.0,
            'T_baseline': args.T_baseline or 100.0
        })
    else:
        # Default: test a few interesting configurations
        print("\nNo configurations specified, testing defaults...")
        test_configs = [
            {
                'config_name': 'baseline',
                'w_P': 0.5, 'w_T': 0.5, 'gamma': 10.0,
                'P_baseline': 1000.0, 'T_baseline': 100.0
            },
            {
                'config_name': 'population_focused',
                'w_P': 0.7, 'w_T': 0.3, 'gamma': 10.0,
                'P_baseline': 1000.0, 'T_baseline': 100.0
            },
            {
                'config_name': 'transition_focused',
                'w_P': 0.3, 'w_T': 0.7, 'gamma': 10.0,
                'P_baseline': 1000.0, 'T_baseline': 100.0
            },
            {
                'config_name': 'strong_nonzero',
                'w_P': 0.5, 'w_T': 0.5, 'gamma': 20.0,
                'P_baseline': 1000.0, 'T_baseline': 100.0
            },
            {
                'config_name': 'equal_weight',
                'w_P': 0.5, 'w_T': 0.5, 'gamma': 15.0,
                'P_baseline': 1000.0, 'T_baseline': 100.0
            }
        ]
    
    # Test each configuration
    print(f"\nTesting {len(test_configs)} reward configurations...")
    print("=" * 80)
    
    stats = []
    for config in test_configs:
        result_stats = evaluate_reward_config(results, **config)
        stats.append(result_stats)
        
        print(f"\nConfiguration: {result_stats['config_name']}")
        print(f"  Parameters: w_P={config['w_P']}, w_T={config['w_T']}, "
              f"gamma={config['gamma']}")
        print(f"  Baselines: P={config['P_baseline']}, T={config['T_baseline']}")
        print(f"  Mean reward: {result_stats['mean_reward']:.4f} ± {result_stats['std_reward']:.4f}")
        print(f"  Range: [{result_stats['min_reward']:.4f}, {result_stats['max_reward']:.4f}]")
        print(f"  Median: {result_stats['median_reward']:.4f}")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("Summary Comparison (sorted by mean reward):")
    print("=" * 80)
    
    stats.sort(key=lambda x: x['mean_reward'], reverse=True)
    
    print(f"{'Config':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)
    for s in stats:
        print(f"{s['config_name']:<20} {s['mean_reward']:>10.4f} "
              f"{s['std_reward']:>10.4f} {s['min_reward']:>10.4f} {s['max_reward']:>10.4f}")
    
    print("\n" + "=" * 80)
    print("Best configuration:", stats[0]['config_name'])
    print(f"  w_P={stats[0]['w_P']}, w_T={stats[0]['w_T']}, gamma={stats[0]['gamma']}")


if __name__ == '__main__':
    main()
