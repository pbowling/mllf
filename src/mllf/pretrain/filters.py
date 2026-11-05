"""Run selection before behavior cloning: best run per system, reward and
transition thresholds, stratified sampling of negative-reward runs and
coefficient outliers.
"""
import logging
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from mllf.file_handling.generate_combinations import is_combo_dir_name

from mllf.pretrain.rewards import (
    compute_reward_from_sim_results,
    reward_kwargs,
)

log = logging.getLogger(__name__)


def filter_best_runs_per_system(
        runs: List[Dict], reward_config: Optional[Dict] = None) -> List[Dict]:
    """Filter runs to keep only the best run per unique system.

    Groups runs by their dataset (pretraining parent directory) and keeps only the
    run with the highest reward for each dataset. This ensures we get one representative
    run from each system (e.g., one from 14benz_solv, one from indole_prot, etc.)
    rather than training on all runs.

    For 14benz_combos_best, which contains different combinations, we group by the
    combination name (e.g., comb_0001_site1_1__site1_2) to get one run per combination.

    Args:
        runs: List of run dicts with 'run_dir', 'source_dir', 'metadata', 'sim_results'
        reward_config: Optional reward configuration dict from config file

    Returns:
        Filtered list containing only best run per system/combination
    """
    if reward_config is None:
        reward_config = {}

    # Group runs by system identifier
    systems = defaultdict(list)
    for run in runs:
        run_dir = run.get('run_dir')

        # Extract system identifier from the run directory path
        # Supports both flat and nested structures:
        # - Flat: pretraining/14benz_solv/run_001 → "14benz_solv"
        # - Nested: pretraining/14benz_good_runs/comb_XXXX/run_001 → "14benz_good_runs/comb_XXXX"
        if run_dir:
            parent_dir = run_dir.parent.name  # e.g., "14benz_solv" or "comb_0063_site2_5__site2_3"
            # e.g., "pretraining" or "14benz_good_runs"
            grandparent_dir = run_dir.parent.parent.name

            # Check if parent looks like a combo directory (starts with "comb_")
            if is_combo_dir_name(parent_dir):
                # Nested structure: use grandparent/combo as system ID
                system_id = f"{grandparent_dir}/{parent_dir}"
            # Check if dataset name contains 'combo' or 'best' (legacy flat structure)
            elif 'combo' in grandparent_dir.lower() or 'best' in grandparent_dir.lower():
                # Legacy flat structure with combo in dataset name
                run_name = run_dir.name
                if '_run_' in run_name:
                    combo_name = run_name.rsplit('_run_', 1)[0]
                    system_id = f"{parent_dir}/{combo_name}"
                else:
                    system_id = f"{parent_dir}/{run_name}"
            else:
                # Standard flat structure: use parent directory as system ID
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
                        with open(graph_info_path, 'r') as f:
                            graph_info = json.load(f)
                        if 'sites' in graph_info:
                            # Count substituents per site from graph_info
                            site_counts = defaultdict(int)
                            for site_key in graph_info['sites']:
                                # Parse "site1_sub2" -> site number
                                site_num = int(site_key.split('_')[0].replace('site', ''))
                                site_counts[site_num] += 1
                            # Convert to ordered list
                            nsubs_per_site = [site_counts[i] for i in sorted(site_counts.keys())]
                    except Exception as e:
                        log.warning(f"  Warning: Could not parse graph_info.json: {e}")

            # Fallback: estimate from total (asymmetric-safe distribution)
            if nsubs_per_site is None:
                if num_substituents > 0 and num_sites > 0:
                    nsubs_per_site = [num_substituents // num_sites] * num_sites
                    for i in range(num_substituents % num_sites):
                        nsubs_per_site[i] += 1
                else:
                    nsubs_per_site = [3] * num_sites

            # Compute reward using config parameters (filter out legacy/unknown parameters)
            filtered_config = reward_kwargs(reward_config)

            reward = compute_reward_from_sim_results(
                run['sim_results'],
                num_sites=num_sites,
                nsubs_per_site=nsubs_per_site,
                **filtered_config  # Use filtered reward config from workflow yaml
            )
            run_rewards.append((run, reward))

        # Keep only the best run
        if run_rewards:
            best_run, best_reward = max(run_rewards, key=lambda x: x[1])
            best_runs.append(best_run)
            log.info(f"  {system_name}: selected run with reward {best_reward:.2f}")

    return best_runs


def compute_coefficient_statistics(runs: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std dev for each bias coefficient type across all runs.

    Args:
        runs: List of run dictionaries with run_dir paths

    Returns:
        Dict with statistics per bias type:
        {
            'linear': {'mean': float, 'std': float, 'values': [...]},
            'quadratic': {'mean': float, 'std': float, 'values': [...]},
            'skew': {'mean': float, 'std': float, 'values': [...]},
            'end': {'mean': float, 'std': float, 'values': [...]}
        }
    """

    # Collect all coefficient values by type
    coeff_values = {
        'linear': [],
        'quadratic': [],
        'skew': [],
        'end': []
    }

    log.info(f"Computing coefficient statistics across {len(runs)} runs...")

    for run in runs:
        run_dir = run["run_dir"]
        variables_path = run_dir / "variables.py"

        if not variables_path.exists():
            continue

        try:
            # Load bias coefficients
            with open(variables_path, 'r') as f:
                content = f.read()

            # Extract YAML
            yaml_start = content.find('bias_string = """') + len('bias_string = """')
            if yaml_start < len('bias_string = """'):
                yaml_start = content.find('bias_string="""') + len('bias_string="""')
            yaml_end = content.find('"""', yaml_start)
            if yaml_end == -1:
                continue

            yaml_content = content[yaml_start:yaml_end]
            bias_data = yaml.safe_load(yaml_content)

            # Extract linear biases (b vector)
            b = bias_data.get('b', [])
            if isinstance(b, list):
                for row in b:
                    if isinstance(row, list):
                        coeff_values['linear'].extend([abs(float(x)) for x in row if x != 0.0])
                    else:
                        val = float(row)
                        if val != 0.0:
                            coeff_values['linear'].append(abs(val))

            # Extract matrix coefficients
            c_matrix = bias_data.get('c', [])
            x_matrix = bias_data.get('x', [])
            s_matrix = bias_data.get('s', [])

            for matrix, key in [(c_matrix, 'quadratic'), (x_matrix, 'skew'), (s_matrix, 'end')]:
                if isinstance(matrix, list):
                    for row in matrix:
                        if isinstance(row, list):
                            coeff_values[key].extend([abs(float(x)) for x in row if x != 0.0])

        except Exception:
            continue

    # Compute statistics
    stats = {}
    for bias_type, values in coeff_values.items():
        if len(values) > 0:
            values_array = np.array(values)
            stats[bias_type] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'median': float(np.median(values_array)),
                'p95': float(np.percentile(values_array, 95)),
                'max': float(np.max(values_array)),
                'count': len(values)
            }
        else:
            stats[bias_type] = {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'p95': 0.0,
                'max': 0.0,
                'count': 0
            }

    # Print statistics
    log.info('%s', "\n" + "=" * 80)
    log.info("Coefficient Statistics Across All Runs")
    log.info('%s', "=" * 80)
    for bias_type, stat in stats.items():
        log.info(f"{bias_type:12s}: mean={stat['mean']:7.2f}, std={stat['std']:7.2f}, "
                 f"p95={stat['p95']:7.2f}, max={stat['max']:7.2f}, n={stat['count']}")
    log.info('%s', "=" * 80 + "\n")

    return stats


def filter_runs_by_coefficient_range(
    runs: List[Dict],
    n_std: float = 3.0,
    min_runs_to_compute_stats: int = 10
) -> List[Dict]:
    """Filter runs to exclude those with coefficients outside statistical range.

    Removes runs where ANY coefficient exceeds mean ± n_std for its type.
    This filters out outlier runs that would dominate the loss during training.

    Args:
        runs: List of run dictionaries
        n_std: Number of standard deviations for threshold (default: 3.0)
        min_runs_to_compute_stats: Minimum runs needed to compute statistics

    Returns:
        Filtered list of runs
    """
    if len(runs) < min_runs_to_compute_stats:
        log.warning(f"Warning: Only {len(runs)} runs available, skipping statistical filtering")
        return runs

    # Compute statistics across all runs
    stats = compute_coefficient_statistics(runs)

    # Compute thresholds
    thresholds = {}
    log.info(f"Filtering thresholds (mean ± {n_std}σ):")
    for bias_type, stat in stats.items():
        threshold = stat['mean'] + n_std * stat['std']
        thresholds[bias_type] = threshold
        log.info(f"  {bias_type:12s}: ≤ {threshold:.2f}")
    log.info('')

    # Filter runs
    filtered_runs = []
    excluded_runs = []

    for run in runs:
        run_dir = run["run_dir"]
        variables_path = run_dir / "variables.py"

        if not variables_path.exists():
            continue

        try:
            # Load and parse coefficients
            with open(variables_path, 'r') as f:
                content = f.read()

            yaml_start = content.find('bias_string = """') + len('bias_string = """')
            if yaml_start < len('bias_string = """'):
                yaml_start = content.find('bias_string="""') + len('bias_string="""')
            yaml_end = content.find('"""', yaml_start)
            if yaml_end == -1:
                continue

            yaml_content = content[yaml_start:yaml_end]
            bias_data = yaml.safe_load(yaml_content)

            # Check all coefficients against thresholds
            exceeds_threshold = False
            violated_type = None
            max_violation = 0.0

            # Check linear (b vector)
            b = bias_data.get('b', [])
            for row in (b if isinstance(b, list) else [b]):
                vals = row if isinstance(row, list) else [row]
                for val in vals:
                    if val != 0.0 and abs(float(val)) > thresholds['linear']:
                        exceeds_threshold = True
                        violated_type = 'linear'
                        max_violation = max(max_violation, abs(float(val)))
                        break
                if exceeds_threshold:
                    break

            # Check matrices
            if not exceeds_threshold:
                for matrix_key, bias_type in [('c', 'quadratic'), ('x', 'skew'), ('s', 'end')]:
                    matrix = bias_data.get(matrix_key, [])
                    if isinstance(matrix, list):
                        for row in matrix:
                            if isinstance(row, list):
                                for val in row:
                                    if val != 0.0 and abs(float(val)) > thresholds[bias_type]:
                                        exceeds_threshold = True
                                        violated_type = bias_type
                                        max_violation = max(max_violation, abs(float(val)))
                                        break
                            if exceeds_threshold:
                                break
                        if exceeds_threshold:
                            break

            if exceeds_threshold:
                excluded_runs.append(
                    (run_dir.parent.name, run_dir.name, violated_type, max_violation))
            else:
                filtered_runs.append(run)

        except Exception:
            # On error, include the run (conservative approach)
            filtered_runs.append(run)

    # Report results
    log.info(f"\n{'=' * 80}")
    log.info(f"Statistical Filtering Results (±{n_std}σ threshold)")
    log.info(f"{'=' * 80}")
    log.info(f"  Total runs: {len(runs)}")
    log.info(f"  Kept: {len(filtered_runs)} ({100 * len(filtered_runs) / len(runs):.1f}%)")
    log.info(f"  Excluded: {len(excluded_runs)} ({100 * len(excluded_runs) / len(runs):.1f}%)")

    if excluded_runs:
        log.info("\n  Top 10 excluded runs:")
        excluded_runs.sort(key=lambda x: x[3], reverse=True)
        for system, run, bias_type, max_val in excluded_runs[:10]:
            log.info(f"    {system:30s} {run:15s} ({bias_type:10s}: {max_val:.2f})")

    log.info(f"{'=' * 80}\n")

    return filtered_runs


def filter_runs_by_reward(
    runs: List[Dict],
    min_reward: float = 0.0,
    reward_config: Optional[Dict] = None,
) -> List[Dict]:
    """Filter runs by minimum reward threshold.

    Computes reward for each run using compute_reward_from_sim_results() and
    excludes runs below the threshold.

    Args:
        runs: List of run dicts from load_pretraining_runs
        min_reward: Minimum reward threshold (default: 0.0)
        reward_config: Optional config ``reward`` section (reward weights and
            run-length handling; defaults otherwise)

    Returns:
        Filtered list of runs meeting reward threshold
    """

    log.info(f"\n{'=' * 80}")
    log.info(f"Reward Filtering (threshold: >= {min_reward})")
    log.info(f"{'=' * 80}")

    filtered_runs = []
    excluded_runs = []
    error_counts = {}

    for run in runs:
        try:
            run_dir = Path(run['run_dir'])

            # Load simulation_results.json (note: not sim_results.json)
            sim_results_path = run_dir / "simulation_results.json"
            if not sim_results_path.exists():
                # If no sim results, exclude the run
                error_reason = "No simulation_results.json"
                error_counts[error_reason] = error_counts.get(error_reason, 0) + 1
                excluded_runs.append((run_dir.parent.name, run_dir.name, None, error_reason))
                continue

            with open(sim_results_path, 'r') as f:
                sim_results = json.load(f)

            # Load graph_info.json to get nsubs_per_site
            graph_info_path = run_dir / "graph_info.json"
            if not graph_info_path.exists():
                error_reason = "No graph_info.json"
                error_counts[error_reason] = error_counts.get(error_reason, 0) + 1
                excluded_runs.append((run_dir.parent.name, run_dir.name, None, error_reason))
                continue

            with open(graph_info_path, 'r') as f:
                graph_info = json.load(f)

            # Extract nsubs_per_site from sites data (sites is a dict with block names as keys)
            sites = graph_info.get('sites', {})
            if not sites:
                error_reason = "No sites in graph_info"
                error_counts[error_reason] = error_counts.get(error_reason, 0) + 1
                excluded_runs.append((run_dir.parent.name, run_dir.name, None, error_reason))
                continue

            # Count substitutions per site
            site_counts = {}
            for block_name, block_data in sites.items():
                site_num = extract_site_number(block_name, block_data)
                site_counts[site_num] = site_counts.get(site_num, 0) + 1

            num_sites = len(site_counts)
            nsubs_per_site = [site_counts[site] for site in sorted(site_counts.keys())]

            # Compute reward
            reward = compute_reward_from_sim_results(sim_results, num_sites, nsubs_per_site,
                                                     **reward_kwargs(reward_config))

            # Filter by threshold
            if reward >= min_reward:
                filtered_runs.append(run)
            else:
                excluded_runs.append(
                    (run_dir.parent.name, run_dir.name, reward, "Below threshold"))

        except Exception as e:
            # On error, exclude the run (conservative approach for reward filtering)
            error_reason = f"Error: {str(e)}"
            error_counts[error_reason] = error_counts.get(error_reason, 0) + 1
            excluded_runs.append((run_dir.parent.name, run_dir.name, None, error_reason))

    # Report results
    log.info(f"  Total runs: {len(runs)}")
    log.info(f"  Kept: {len(filtered_runs)} ({100 * len(filtered_runs) / len(runs):.1f}%)")
    log.info(f"  Excluded: {len(excluded_runs)} ({100 * len(excluded_runs) / len(runs):.1f}%)")

    # Show error breakdown
    if error_counts:
        log.error("\n  Error breakdown:")
        for error_reason, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            log.info(f"    {error_reason}: {count} runs")

    if excluded_runs:
        # Show distribution of excluded runs
        reward_excluded = [x for x in excluded_runs if isinstance(x[2], (int, float))]
        if reward_excluded:
            log.info("\n  Reward statistics for excluded runs:")
            rewards = [x[2] for x in reward_excluded]
            log.info(f"    Mean: {sum(rewards) / len(rewards):.2f}")
            log.info(f"    Min: {min(rewards):.2f}")
            log.info(f"    Max: {max(rewards):.2f}")

        log.info("\n  Top 10 excluded runs (by reward):")
        # Sort by reward (descending) - show the "best" of the excluded runs
        reward_excluded_sorted = sorted(reward_excluded, key=lambda x: x[2], reverse=True)
        for system, run, reward, reason in reward_excluded_sorted[:10]:
            log.info(f"    {system:30s} {run:15s} reward={reward:7.2f}")

    log.info(f"{'=' * 80}\n")

    return filtered_runs


def filter_runs_by_min_transitions(
    runs: List[Dict],
    min_transitions: int = 3,
) -> List[Dict]:
    """Filter runs by minimum transitions on the worst site.

    Reads the ``transitions`` dict from each run's ``simulation_results.json``
    and keeps only runs where *every* site has at least ``min_transitions``
    transitions at the highest lambda value recorded.

    This is a direct data-quality criterion that avoids the reward formula
    entirely — a run with 3 transitions on every site is always kept regardless
    of what any penalty coefficients compute to.

    Args:
        runs: List of run dicts from load_pretraining_runs
        min_transitions: Minimum transitions required on every site (default: 3)

    Returns:
        Filtered list of runs where all sites meet the threshold
    """

    log.info(f"\n{'=' * 80}")
    log.info(f"Transition Filtering (min transitions per site: >= {min_transitions})")
    log.info(f"{'=' * 80}")

    filtered_runs = []
    excluded_runs = []   # (system, run, min_t, reason)
    n_errors = 0

    for run in runs:
        try:
            run_dir = Path(run['run_dir'])
            sim_path = run_dir / "simulation_results.json"
            if not sim_path.exists():
                n_errors += 1
                excluded_runs.append((run_dir.parent.name, run_dir.name, None, "no sim_results"))
                continue

            sim = json.loads(sim_path.read_text())
            trans_dict = sim.get("transitions", {})
            if not trans_dict:
                # No transition data at all — treat as 0 transitions
                excluded_runs.append((run_dir.parent.name, run_dir.name, 0, "no transitions"))
                continue

            # Per-site count at the highest lambda value
            per_site = []
            for site_id in sorted(trans_dict.keys(), key=int):
                site_data = trans_dict[site_id]
                if isinstance(site_data, dict) and site_data:
                    max_lam = max(site_data.keys(), key=float)
                    per_site.append(int(site_data[max_lam]))
                else:
                    per_site.append(0)

            worst = min(per_site) if per_site else 0

            if worst >= min_transitions:
                filtered_runs.append(run)
            else:
                excluded_runs.append((run_dir.parent.name, run_dir.name, worst, "below threshold"))

        except Exception as exc:
            n_errors += 1
            excluded_runs.append((run_dir.parent.name, run_dir.name, None, f"error: {exc}"))

    log.info(f"  Total runs : {len(runs)}")
    log.info(f"  Kept       : {len(filtered_runs)} "
             f"({100 * len(filtered_runs) / max(len(runs), 1):.1f}%)")
    log.info(f"  Excluded   : {len(excluded_runs)} "
             f"({100 * len(excluded_runs) / max(len(runs), 1):.1f}%)")
    if n_errors:
        log.error(f"  Errors     : {n_errors}")

    # Histogram of worst-site transition counts in excluded runs
    excl_counts = [t for _, _, t, _ in excluded_runs if isinstance(t, int)]
    if excl_counts:
        hist = Counter(excl_counts)
        log.info("  Excluded by worst-site transition count:")
        for k in sorted(hist):
            log.info(f"    {k:3d} transitions: {hist[k]} runs")

    log.info(f"{'=' * 80}\n")
    return filtered_runs


def extract_site_number(block_id: str, block_data: Dict) -> int:
    """Extract site number from block data, handling missing 'site' key.

    Some multi-site systems have graph_info.json blocks without the 'site' key.
    This function handles both cases:
    1. If 'site' key exists, return it
    2. If missing, extract from block_id format (e.g., "site1_sub2" → 1)

    Args:
        block_id: Block identifier (e.g., "site1_sub1", "site2_sub3")
        block_data: Block data dict (may or may not have 'site' key)

    Returns:
        Site number as integer

    Raises:
        KeyError: If 'site' key exists and is required but missing
        ValueError: If extraction from block_id fails
    """
    # Try to use 'site' key if it exists
    if 'site' in block_data:
        return block_data['site']

    # Otherwise, extract from block_id format: "siteN_subM" → N
    try:
        # Split on underscore: ["siteN", "subM"]
        parts = block_id.split('_')
        if len(parts) >= 1 and parts[0].startswith('site'):
            # Extract number from "siteN"
            site_str = parts[0][4:]  # Remove "site" prefix
            return int(site_str)
    except (IndexError, ValueError):
        pass

    # If we get here, we couldn't extract the site number
    raise ValueError(
        f"Cannot extract site number from block_id '{block_id}' and block_data has no 'site' key")


def sample_runs_stratified_negative(
    runs: List[Dict],
    fraction_per_bucket: float = 0.55,
    seed: int = 42,
    reward_config: Optional[Dict] = None,
) -> List[Dict]:
    """Keep all positive-reward runs and sample negative-reward buckets with a
    quadratic ramp: the worst bucket ((-inf, -50]) keeps 0% and the best
    negative bucket ((-10, 0)) keeps ``fraction_per_bucket``.  Intermediate
    buckets follow a squared schedule (fraction = max × (i / (N-1))²), which
    concentrates sampling on near-zero runs whose coefficients were almost
    correct.

    Buckets (left-exclusive, right-inclusive except the last):
        (-inf, -50], (-50, -40], (-40, -30], (-30, -20], (-20, -10], (-10, 0)

    All runs with reward >= 0 are always kept in full.
    Runs whose reward cannot be computed are excluded.

    Args:
        runs: List of run dicts from load_pretraining_runs.
        fraction_per_bucket: Maximum sampling fraction applied to the best negative
            bucket ((-10, 0)).  Worst bucket gets 0%.  Default: 0.55.
        seed: Random seed for reproducibility (default: 42).
        reward_config: Optional config ``reward`` section (reward weights and
            run-length handling; defaults otherwise).

    Returns:
        Filtered + sampled list of runs.
    """
    # Upper bounds for negative buckets; the implicit lower bound of the first
    # bucket is -inf.  For a reward r < 0 we assign it to the first bucket
    # whose upper bound satisfies r <= upper_bound.
    BUCKET_UPPER_BOUNDS = [-50, -40, -30, -20, -10, 0]

    log.info(f"\n{'=' * 80}")
    log.info(f"Stratified Negative Sampling  (max_fraction={fraction_per_bucket:.0%}, quadratic "
             "ramp)")
    log.info(f"{'=' * 80}")

    # Compute reward for every run -----------------------------------------
    scored: List[tuple] = []  # (reward, run_dict)
    n_error = 0
    error_log = []  # Track errors for detailed reporting

    for run in runs:
        try:
            run_dir = Path(run['run_dir'])

            sim_results_path = run_dir / "simulation_results.json"
            if not sim_results_path.exists():
                n_error += 1
                error_log.append((str(run_dir), "missing_simulation_results.json"))
                continue
            with open(sim_results_path) as f:
                sim_results = json.load(f)

            graph_info_path = run_dir / "graph_info.json"
            if not graph_info_path.exists():
                n_error += 1
                error_log.append((str(run_dir), "missing_graph_info.json"))
                continue
            with open(graph_info_path) as f:
                graph_info = json.load(f)

            sites = graph_info.get('sites', {})
            if not sites:
                n_error += 1
                error_log.append((str(run_dir), "empty_sites_dict"))
                continue

            site_counts: Dict = {}
            for block_id, block_data in sites.items():
                s = extract_site_number(block_id, block_data)
                site_counts[s] = site_counts.get(s, 0) + 1
            num_sites = len(site_counts)
            nsubs_per_site = [site_counts[s] for s in sorted(site_counts)]

            reward = compute_reward_from_sim_results(sim_results, num_sites, nsubs_per_site,
                                                     **reward_kwargs(reward_config))
            scored.append((reward, run))
        except Exception as e:
            n_error += 1
            error_log.append((str(run_dir), f"{type(e).__name__}: {str(e)[:80]}"))

    log.info(f"  Total runs scored: {len(scored):,}")
    if n_error:
        log.info(f"  Skipped (scoring error): {n_error:,}")
        # Print top error types
        error_types = {}
        for run_path, error_msg in error_log:
            if ":" in error_msg:
                error_type = error_msg.split(":")[0]
            else:
                error_type = error_msg
            error_types[error_type] = error_types.get(error_type, 0) + 1

        log.error("\n  Error Type Breakdown:")
        for error_type in sorted(error_types.keys(), key=lambda x: error_types[x], reverse=True):
            count = error_types[error_type]
            log.info(f"    {error_type:<40} {count:>6,}")

        # Log to file for detailed analysis
        error_log_path = Path("scoring_errors.log")
        with open(error_log_path, 'w') as f:
            f.write("Scoring Errors Log\n")
            f.write(f"Total errors: {n_error}\n\n")
            for run_path, error_msg in error_log[:20]:  # Log first 20
                f.write(f"{run_path}\n  {error_msg}\n")
        log.info(f"\n  Full error log saved to: {error_log_path}")

    # Separate positive runs (kept in full) and bucket negative ones ---------
    positive_runs = [r for rew, r in scored if rew >= 0]
    buckets: Dict[int, List[Dict]] = {i: [] for i in range(len(BUCKET_UPPER_BOUNDS))}
    for rew, r in scored:
        if rew >= 0:
            continue
        for i, hi in enumerate(BUCKET_UPPER_BOUNDS):
            if rew <= hi:
                buckets[i].append(r)
                break

    # Build bucket labels for display
    bucket_labels = []
    prev = "-inf"
    for hi in BUCKET_UPPER_BOUNDS:
        bucket_labels.append(f"({prev}, {hi}]" if hi < 0 else f"({prev}, {hi})")
        prev = str(hi)

    # Sample from each bucket with quadratic ramp --------------------------------
    # bucket 0 (worst, (-inf,-50]) → fraction 0.0
    # bucket N-1 (best, (-10,0))   → fraction fraction_per_bucket
    # Intermediate buckets: fraction = max × (i / (N-1))²
    n_buckets = len(BUCKET_UPPER_BOUNDS)
    rng = random.Random(seed)
    sampled_negative: List[Dict] = []
    log.info(f"\n  {'Bucket':<18} {'Fraction':>8} {'Available':>10} {'Sampled':>9}")
    log.info(f"  {'-' * 18} {'-' * 8} {'-' * 10} {'-' * 9}")
    for i, label in enumerate(bucket_labels):
        available = buckets[i]
        n_avail = len(available)
        # Quadratic ramp: bucket 0 → 0%, bucket (n_buckets-1) → fraction_per_bucket
        bucket_frac = fraction_per_bucket * (i / max(n_buckets - 1, 1)) ** 2
        n_sample = max(1, int(math.ceil(n_avail * bucket_frac))
                       ) if (n_avail > 0 and bucket_frac > 0) else 0
        if n_avail <= n_sample:
            selected = available
        else:
            selected = rng.sample(available, n_sample) if n_sample > 0 else []
        sampled_negative.extend(selected)
        log.info(f"  {label:<18} {bucket_frac:>7.0%} {n_avail:>10,} {len(selected):>9,}")

    result = positive_runs + sampled_negative

    # Positive bucket breakdown for visibility
    POS_BUCKET_UPPER_BOUNDS = [10, 20, 30, 40, 50, math.inf]
    pos_buckets: Dict[int, int] = {i: 0 for i in range(len(POS_BUCKET_UPPER_BOUNDS))}
    for rew, _ in scored:
        if rew < 0:
            continue
        for i, hi in enumerate(POS_BUCKET_UPPER_BOUNDS):
            if rew <= hi:
                pos_buckets[i] += 1
                break
    pos_labels = []
    prev_p = "0"
    for hi in POS_BUCKET_UPPER_BOUNDS:
        if hi == math.inf:
            pos_labels.append(f"({prev_p}, +inf)")
        else:
            pos_labels.append(f"[{prev_p}, {hi}]")
        prev_p = str(int(hi)) if hi != math.inf else str(int(POS_BUCKET_UPPER_BOUNDS[-2]))

    log.info(f"\n  Positive (kept all):   {len(positive_runs):>7,}")
    log.info(f"  {'Bucket':<18} {'Count':>10}")
    log.info(f"  {'-' * 18} {'-' * 10}")
    for i, label in enumerate(pos_labels):
        log.info(f"  {label:<18} {pos_buckets[i]:>10,}")

    log.info(f"\n  Negative (sampled):    {len(sampled_negative):>7,}")
    log.info(f"  Total after sampling:  {len(result):>7,}")
    log.info(f"{'=' * 80}\n")
    return result


def select_runs(runs: List[Dict], opts, reward_config: Dict) -> Optional[List[Dict]]:
    """Apply the run filters of ``PretrainOptions`` in order; None (after
    printing why) when no run is left."""
    if opts.use_best_only:
        log.info(f"\nFiltering {len(runs)} runs to keep only best per system...")
        log.info(f"Using reward config: {reward_config}")
        runs = filter_best_runs_per_system(runs, reward_config=reward_config)
        log.info(f"Filtered to {len(runs)} best runs for training\n")
        if len(runs) == 0:
            log.error("Error: No valid runs after filtering")
            return None
    else:
        log.info(f"\nUsing all {len(runs)} valid runs for pretraining (no filtering)\n")

    # Transition-count filter takes highest precedence -- a direct data
    # quality criterion; then stratified negative sampling; then the reward
    # threshold.
    if opts.min_transitions is not None:
        log.info(f"\nApplying transition filtering (min per site: >= {opts.min_transitions})...")
        runs = filter_runs_by_min_transitions(runs, min_transitions=opts.min_transitions)
        if len(runs) == 0:
            log.error("Error: No runs remaining after transition filtering")
            log.info("Try lowering --min-transitions")
            return None
    elif opts.stratified_negative_fraction is not None:
        log.info("\nApplying stratified negative sampling "
                 f"(max_fraction={opts.stratified_negative_fraction:.0%}, quadratic ramp)...")
        runs = sample_runs_stratified_negative(
            runs, fraction_per_bucket=opts.stratified_negative_fraction,
            reward_config=reward_config)
        if len(runs) == 0:
            log.error("Error: No runs remaining after stratified negative sampling")
            return None
    elif opts.min_reward_threshold is not None:
        log.info(f"\nApplying reward filtering (threshold: >= {opts.min_reward_threshold})...")
        runs = filter_runs_by_reward(runs, min_reward=opts.min_reward_threshold,
                                     reward_config=reward_config)
        if len(runs) == 0:
            log.error("Error: No runs remaining after reward filtering")
            log.info("Try lowering min_reward_threshold or disable reward filtering")
            return None

    if opts.filter_outliers:
        log.info(f"\nApplying statistical filtering (±{opts.outlier_std_threshold}σ threshold)...")
        runs = filter_runs_by_coefficient_range(runs, n_std=opts.outlier_std_threshold)
        if len(runs) == 0:
            log.error("Error: No runs remaining after statistical filtering")
            log.info("Try increasing outlier_std_threshold or set filter_outliers=False")
            return None
    return runs
