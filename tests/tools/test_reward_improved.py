#!/usr/bin/env python3
"""Test improved reward function on pretraining data to compare good vs bad results.

This script loads pretraining data and computes the improved reward function
to verify it correctly distinguishes between good sampling (high transitions,
broad coverage) and bad sampling (degenerate single-substituent concentration).

Usage:
    python test_reward_improved.py <pretraining_dir> [--good-threshold <N>] [--bad-threshold <N>]

Example:
    # Test on indolizine_solv: run180+ is good, run1-19 is bad
    python test_reward_improved.py pretraining/indolizine_solv --good-threshold 180 --bad-threshold 20
"""
import argparse
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple


def compute_improved_reward_from_json(
    run_dir: Path,
    w_P: float = 0.4,
    w_T: float = 0.4,
    w_U: float = 0.2,
    gamma: float = 5.0,
    P_baseline: float = 1000.0,
    T_baseline: float = 100.0,
    min_transitions_per_site: int = 10,
    min_coverage_ratio: float = 0.5,
    entropy_bonus: float = 5.0,
    concentration_penalty_threshold: float = 0.8,
    verbose: bool = False
) -> Tuple[float, Dict]:
    """Compute improved reward from pretraining JSON data.
    
    Args:
        run_dir: Path to run directory with JSON files
        w_P: Weight for population term (default: 0.4)
        w_T: Weight for transition term (default: 0.4)
        w_U: Weight for uniformity term (default: 0.2)
        gamma: Base penalty coefficient (default: 5.0)
        P_baseline: Population normalization baseline (default: 1000.0)
        T_baseline: Transition normalization baseline (default: 100.0)
        min_transitions_per_site: Minimum transitions per site (default: 10)
        min_coverage_ratio: Minimum coverage fraction (default: 0.5)
        entropy_bonus: Entropy bonus coefficient (default: 5.0)
        concentration_penalty_threshold: Threshold for concentration penalty (default: 0.8)
        min_coverage_ratio: Minimum coverage fraction (default: 0.5)
        entropy_bonus: Entropy bonus coefficient (default: 5.0)
        verbose: Print detailed breakdown
    
    Returns:
        (reward, metrics_dict)
    """
    
    # Load simulation results
    sim_results_file = run_dir / 'simulation_results.json'
    if not sim_results_file.exists():
        return -500.0, {'error': 'No simulation_results.json'}
    
    try:
        with open(sim_results_file) as f:
            sim_data = json.load(f)
    except Exception as e:
        return -500.0, {'error': f'Failed to load JSON: {e}'}
    
    if not sim_data.get('terminated_normally', False):
        return -500.0, {'error': 'Abnormal termination'}
    
    # Extract populations
    population_data = sim_data.get('populations', {})
    transitions_data = sim_data.get('transitions', {})
    
    if not population_data or not transitions_data:
        return -500.0, {'error': 'Missing population or transition data'}
    
    # Parse populations per block
    # Parse populations per block - use only HIGHEST lambda value
    populations = []
    for block_id, block_info in population_data.items():
        if isinstance(block_info, dict) and 'counts' in block_info:
            counts_dict = block_info['counts']
            if counts_dict:
                # Use only the highest lambda value
                max_lambda = max(counts_dict.keys(), key=lambda x: float(x))
                populations.append(counts_dict[max_lambda])
            else:
                populations.append(0)
        else:
            populations.append(0)
    
    # Parse transitions per site - use only HIGHEST lambda value
    site_transitions = {}
    for site_id, trans_dict in transitions_data.items():
        if isinstance(trans_dict, dict):
            if trans_dict:
                # Use only the highest lambda value
                max_lambda = max(trans_dict.keys(), key=lambda x: float(x))
                total_trans = trans_dict[max_lambda]
            else:
                total_trans = 0
        else:
            total_trans = trans_dict if isinstance(trans_dict, (int, float)) else 0
        site_transitions[site_id] = total_trans
    
    total_subs = len(populations)
    total_sites = len(site_transitions)
    
    if total_subs == 0 or total_sites == 0:
        return -500.0, {'error': 'No substituents or sites'}
    
    pop_array = np.array(populations)
    
    # ========== STRICT DEGENERATE BEHAVIOR CHECKS ==========
    
    penalties = 0.0
    penalty_messages = []
    
    # Check 1: Minimum transitions per site
    sites_below_threshold = 0
    min_site_trans = min(site_transitions.values()) if site_transitions else 0
    
    for site_id, trans_count in site_transitions.items():
        if trans_count < min_transitions_per_site:
            sites_below_threshold += 1
            deficit = min_transitions_per_site - trans_count
            penalties -= gamma * (1 + deficit)
    
    if sites_below_threshold > 0:
        penalties -= gamma * 10.0 * sites_below_threshold
        penalty_messages.append(f"{sites_below_threshold} site(s) below {min_transitions_per_site} transitions")
    
    # Check 2: Minimum coverage
    nonzero_count = np.sum(pop_array > 0)
    coverage_ratio = nonzero_count / total_subs if total_subs > 0 else 0.0
    
    if coverage_ratio < min_coverage_ratio:
        deficit = min_coverage_ratio - coverage_ratio
        penalties -= gamma * 20.0 * deficit
        penalty_messages.append(f"Coverage {coverage_ratio:.2%} below minimum {min_coverage_ratio:.0%}")
    
    # Check 3: Detect single-dominant-population per site
    # Extract actual substituents per site from graph_info.json
    nsubs_per_site = None
    graph_info_path = run_dir / 'graph_info.json'
    if graph_info_path.exists():
        try:
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
    
    max_concentration = 0.0
    pop_idx = 0
    for site_idx, nsubs in enumerate(nsubs_per_site):
        site_pops = pop_array[pop_idx:pop_idx + nsubs]
        
        if len(site_pops) > 0 and np.sum(site_pops) > 0:
            max_pop = np.max(site_pops)
            total_pop = np.sum(site_pops)
            concentration_ratio = max_pop / total_pop
            max_concentration = max(max_concentration, concentration_ratio)
            
            if concentration_ratio > concentration_penalty_threshold:
                penalties -= gamma * 5.0 * (concentration_ratio - concentration_penalty_threshold)
                penalty_messages.append(f"Site {site_idx} has {concentration_ratio:.0%} concentration")
        
        pop_idx += nsubs
    
    # ========== POSITIVE REWARD COMPONENTS ==========
    
    # R_P: Population balance reward
    R_P = 0.0
    balance_factor = 0.0
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
            balance_factor = 0.01
    
    # R_T: Transition reward
    R_T = 0.0
    if sites_below_threshold == 0:
        total_trans = sum(site_transitions.values())
        R_T = w_T * (total_trans / T_baseline)
        
        avg_trans_per_site = total_trans / total_sites if total_sites > 0 else 0
        if avg_trans_per_site > min_transitions_per_site * 2:
            R_T *= 1.5
    
    # R_U: Coverage uniformity reward
    R_U = w_U * coverage_ratio
    
    # R_entropy: Shannon entropy bonus
    R_entropy = 0.0
    entropy_score = 0.0
    if np.sum(pop_array) > 0:
        prob_dist = pop_array / np.sum(pop_array)
        prob_dist = prob_dist[prob_dist > 0]
        
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        max_entropy = np.log(len(prob_dist)) if len(prob_dist) > 0 else 1.0
        entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0
        R_entropy = entropy_bonus * entropy_score
    
    # ========== PENALTY CLAMPING ==========
    # Prevent gradient explosion by capping maximum negative penalty
    max_penalty = 50.0
    if penalties < -max_penalty:
        penalties = -max_penalty
    
    # Final reward
    R = R_P + R_T + R_U + R_entropy + penalties
    
    # Metrics for analysis
    metrics = {
        'reward': R,
        'R_P': R_P,
        'R_T': R_T,
        'R_U': R_U,
        'R_entropy': R_entropy,
        'penalties': penalties,
        'total_subs': total_subs,
        'nonzero_subs': int(nonzero_count),
        'coverage': coverage_ratio,
        'total_transitions': sum(site_transitions.values()),
        'min_site_transitions': min_site_trans,
        'max_concentration': max_concentration,
        'balance_factor': balance_factor,
        'entropy_score': entropy_score,
        'sites_below_threshold': sites_below_threshold,
        'penalty_messages': penalty_messages
    }
    
    if verbose:
        print(f"  R_P={R_P:.2f} R_T={R_T:.2f} R_U={R_U:.2f} R_entropy={R_entropy:.2f} penalties={penalties:.2f}")
        print(f"  Coverage: {nonzero_count}/{total_subs} ({coverage_ratio:.0%}), "
              f"Transitions: {list(site_transitions.values())}, Max conc: {max_concentration:.0%}")
        if penalty_messages:
            for msg in penalty_messages:
                print(f"  ⚠️  {msg}")
    
    return R, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Test improved reward function on pretraining data'
    )
    parser.add_argument(
        'pretraining_dir',
        type=Path,
        help='Pretraining directory (e.g., pretraining/indolizine_solv)'
    )
    parser.add_argument(
        '--good-threshold',
        type=int,
        default=180,
        help='Run numbers >= this are considered "good" (default: 180)'
    )
    parser.add_argument(
        '--bad-threshold',
        type=int,
        default=20,
        help='Run numbers < this are considered "bad" (default: 20)'
    )
    parser.add_argument(
        '--configs',
        type=Path,
        help='YAML file with reward configurations to test'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed breakdown for each run'
    )
    parser.add_argument(
        '--balance-runs',
        action='store_true',
        help='Use equal number of good and bad runs'
    )
    
    args = parser.parse_args()
    
    if not args.pretraining_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.pretraining_dir}")
    
    print(f"Testing improved reward function on: {args.pretraining_dir}")
    print(f"Good runs: >= run{args.good_threshold}")
    print(f"Bad runs: < run{args.bad_threshold}")
    print("=" * 80)
    
    # Collect runs
    good_runs = []
    bad_runs = []
    
    for run_dir in sorted(args.pretraining_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Extract run number
        run_name = run_dir.name
        if not run_name.startswith('run'):
            continue
        
        try:
            run_num = int(run_name[3:])
        except ValueError:
            continue
        
        if run_num >= args.good_threshold:
            good_runs.append((run_num, run_dir))
        elif run_num < args.bad_threshold:
            bad_runs.append((run_num, run_dir))
    
    print(f"\nFound {len(good_runs)} good runs, {len(bad_runs)} bad runs")
    
    # Balance runs if requested
    if args.balance_runs:
        min_runs = min(len(good_runs), len(bad_runs))
        good_runs = good_runs[:min_runs]
        bad_runs = bad_runs[:min_runs]
        print(f"Balanced to {min_runs} runs each")
    
    # Load configurations
    if args.configs:
        print(f"\nLoading configurations from {args.configs}...")
        import yaml
        with open(args.configs) as f:
            configs_data = yaml.safe_load(f)
        print(f"Loaded {len(configs_data)} configurations")
    else:
        configs_data = {'default': {}}
    
    # Test each configuration
    all_results = []
    
    for config_name, config_params in configs_data.items():
        print("\n" + "=" * 80)
        print(f"TESTING CONFIGURATION: {config_name}")
        print("=" * 80)
        if config_params:
            print(f"Parameters: {config_params}")
        
        # Test good runs
        print("\nGood Runs:")
        
        good_rewards = []
        good_metrics = []
        
        for run_num, run_dir in sorted(good_runs):
            if args.verbose:
                print(f"\n{run_dir.name} (run {run_num}):")
            
            reward, metrics = compute_improved_reward_from_json(run_dir, verbose=args.verbose, **config_params)
            good_rewards.append(reward)
            good_metrics.append(metrics)
            
            if not args.verbose and len(good_runs) <= 20:
                print(f"  {run_dir.name}: reward={reward:7.2f}, coverage={metrics['coverage']:.0%}, "
                      f"trans={metrics['total_transitions']:3d}, max_conc={metrics['max_concentration']:.0%}")
        
        # Test bad runs
        print("\nBad Runs:")
        
        bad_rewards = []
        bad_metrics = []
        
        for run_num, run_dir in sorted(bad_runs):
            if args.verbose:
                print(f"\n{run_dir.name} (run {run_num}):")
            
            reward, metrics = compute_improved_reward_from_json(run_dir, verbose=args.verbose, **config_params)
            bad_rewards.append(reward)
            bad_metrics.append(metrics)
            
            if not args.verbose and len(bad_runs) <= 20:
                print(f"  {run_dir.name}: reward={reward:7.2f}, coverage={metrics['coverage']:.0%}, "
                      f"trans={metrics['total_transitions']:3d}, max_conc={metrics['max_concentration']:.0%}")
    
        # Summary statistics for this configuration
        print("\nSummary:")
        
        if good_rewards:
            print(f"  Good Runs (n={len(good_rewards)}):")
            print(f"    Mean: {np.mean(good_rewards):7.2f} ± {np.std(good_rewards):6.2f}")
            print(f"    Range: [{np.min(good_rewards):7.2f}, {np.max(good_rewards):7.2f}]")
        
        if bad_rewards:
            print(f"  Bad Runs (n={len(bad_rewards)}):")
            print(f"    Mean: {np.mean(bad_rewards):7.2f} ± {np.std(bad_rewards):6.2f}")
            print(f"    Range: [{np.min(bad_rewards):7.2f}, {np.max(bad_rewards):7.2f}]")
        
        # Compare
        if good_rewards and bad_rewards:
            diff = np.mean(good_rewards) - np.mean(bad_rewards)
            print(f"  Difference (good - bad): {diff:7.2f}")
            
            result = {
                'config_name': config_name,
                'config_params': config_params,
                'good_mean': np.mean(good_rewards),
                'good_std': np.std(good_rewards),
                'good_min': np.min(good_rewards),
                'good_max': np.max(good_rewards),
                'bad_mean': np.mean(bad_rewards),
                'bad_std': np.std(bad_rewards),
                'bad_min': np.min(bad_rewards),
                'bad_max': np.max(bad_rewards),
                'difference': diff,
                'correctly_distinguishes': np.mean(good_rewards) > np.mean(bad_rewards)
            }
            all_results.append(result)
            
            if np.mean(good_rewards) > np.mean(bad_rewards):
                print(f"  ✅ Correctly distinguishes good from bad")
            else:
                print(f"  ❌ Bad runs have higher reward")
    
    # Final comparison across all configs
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("CONFIGURATION COMPARISON")
        print("=" * 80)
        
        # Sort by good run mean reward (higher is better)
        all_results.sort(key=lambda x: x['good_mean'], reverse=True)
        
        print(f"\n{'Config':<25} {'Good Mean':>10} {'Bad Mean':>10} {'Difference':>12} {'Status':>8}")
        print("-" * 80)
        for r in all_results:
            status = "✅" if r['correctly_distinguishes'] else "❌"
            print(f"{r['config_name']:<25} {r['good_mean']:>10.2f} {r['bad_mean']:>10.2f} "
                  f"{r['difference']:>12.2f} {status:>8}")
        
        print("\n" + "=" * 80)
        print("BEST CONFIGURATIONS")
        print("=" * 80)
        
        # Filter to only those that correctly distinguish
        valid_results = [r for r in all_results if r['correctly_distinguishes']]
        
        if valid_results:
            # Best for highest good rewards
            best = valid_results[0]
            print(f"\nHighest Good Reward: {best['config_name']}")
            print(f"  Good mean: {best['good_mean']:.2f}")
            print(f"  Bad mean: {best['bad_mean']:.2f}")
            print(f"  Difference: {best['difference']:.2f}")
            
            # Best for largest separation
            best_sep = max(valid_results, key=lambda x: x['difference'])
            if best_sep != best:
                print(f"\nLargest Separation: {best_sep['config_name']}")
                print(f"  Good mean: {best_sep['good_mean']:.2f}")
                print(f"  Bad mean: {best_sep['bad_mean']:.2f}")
                print(f"  Difference: {best_sep['difference']:.2f}")
        else:
            print("\n⚠️  No configurations correctly distinguished good from bad!")
    elif all_results:
        # Single configuration summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        r = all_results[0]
        print(f"\nConfiguration: {r['config_name']}")
        print(f"  Good runs: {r['good_mean']:.2f} ± {r['good_std']:.2f}")
        print(f"  Bad runs: {r['bad_mean']:.2f} ± {r['bad_std']:.2f}")
        print(f"  Difference: {r['difference']:.2f}")
        print(f"  Status: {'✅ Correctly distinguishes' if r['correctly_distinguishes'] else '❌ Does not distinguish'}")


if __name__ == '__main__':
    main()
