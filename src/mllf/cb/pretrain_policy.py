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


def build_fully_connected_graph_for_pretraining(run_dir: Path, toppar_dir=None, toppar_files=None, warn_missing_types=True):
    """Build fully-connected PyG graph for pretraining from saved variables.py.
    
    Unlike the standard graph builder which only creates edges for non-zero coefficients,
    this function creates edges for ALL pairs within each site. This provides richer
    training data for pretraining, especially for linear biases which can be estimated
    for all pairs.
    
    **Key differences from standard graph builder:**
    1. Creates ALL directed pairs within each site (O(N²) edges per site)
    2. Computes linear coefficients correctly: linear_ij = b[j] - b[i]
    3. Reads nonlinear coefficients from matrices (may be 0.0 for many pairs)
    
    **Why this helps pretraining:**
    - More training data: ~10x more edges for typical systems
    - Proper linear bias encoding: antisymmetric by construction
    - Better gradient signal: model learns from all pairwise relationships
    - Matches pairwise MLP approach: same data representation
    
    Args:
        run_dir: Directory containing variables.py and graph_info.json
        toppar_dir: Path to toppar directory (None uses package default)
        toppar_files: List of specific toppar filenames to include
        warn_missing_types: If True, warn when sub RTF files contain atom types not in vocabulary
    
    Returns:
        Tuple of (data, targets, extras) where:
        - data: PyG Data with fully-connected edges within each site
        - targets: List of [linear, quadratic, skew, end] for each directed edge
        - extras: Dict with relation_names, base_relation_map, etc.
    """
    import json
    import yaml
    from torch_geometric.data import Data
    from mllf.cb import graph_utils
    
    run_dir = Path(run_dir)
    
    # Load graph_info.json to get node features and site structure
    graph_info_path = run_dir / "graph_info.json"
    if not graph_info_path.exists():
        raise FileNotFoundError(f"graph_info.json not found in {run_dir}")
    
    with open(graph_info_path, 'r') as f:
        graph_info = json.load(f)
    
    # Load node features using existing graph_utils
    from mllf.cb.graph import Graph
    g = Graph.from_graph_info(graph_info)
    
    # Build node features
    data_sparse, extras = graph_utils.build_pyg_graph_from_mllf_graph(
        g, toppar_dir=toppar_dir, toppar_files=toppar_files, 
        warn_missing_types=warn_missing_types
    )
    
    # Extract site structure
    nsubs_per_site = graph_info.get('nsubs_per_site', [])
    if not nsubs_per_site:
        # Compute from sites data (backward compatibility for old graph_info.json files)
        from collections import defaultdict
        site_counts = defaultdict(int)
        sites_info = graph_info.get('sites', {})
        for key, node_info in sites_info.items():
            site = node_info.get('site')
            if site is not None:
                site_counts[site] += 1
        if site_counts:
            nsubs_per_site = [site_counts[site] for site in sorted(site_counts.keys())]
        else:
            raise ValueError(f"Could not determine nsubs_per_site from graph_info.json in {run_dir}")
    
    # Load bias coefficients from variables.py
    variables_path = run_dir / "variables.py"
    if not variables_path.exists():
        raise FileNotFoundError(f"variables.py not found in {run_dir}")
    
    with open(variables_path, 'r') as f:
        content = f.read()
    
    # Extract YAML string
    yaml_start = content.find('bias_string = """') + len('bias_string = """')
    if yaml_start < len('bias_string = """'):
        yaml_start = content.find('bias_string="""') + len('bias_string="""')
    if yaml_start < len('bias_string="""'):
        yaml_start = content.find("bias_string = '''") + len("bias_string = '''")
    if yaml_start < len("bias_string = '''"):
        yaml_start = content.find("bias_string='''") + len("bias_string='''")
    
    yaml_end = content.find('"""', yaml_start)
    if yaml_end == -1:
        yaml_end = content.find("'''", yaml_start)
    
    yaml_str = content[yaml_start:yaml_end]
    bias_data = yaml.safe_load(yaml_str)
    
    # Extract bias matrices
    b_list = bias_data['b']
    if isinstance(b_list[0], list):
        b_vector = np.array(b_list[0], dtype=float)
    else:
        b_vector = np.array(b_list, dtype=float)
    
    c_matrix = np.array(bias_data['c'], dtype=float)
    x_matrix = np.array(bias_data['x'], dtype=float)
    s_matrix = np.array(bias_data['s'], dtype=float)
    
    # Build directed pairs for ALL substituents within each site
    from mllf.cb.pairwise_utils import build_directed_pairs
    pairs = build_directed_pairs(nsubs_per_site)
    
    # Get relation names from extras
    relation_names = extras['relation_names']
    base_relation_map = extras['base_relation_map']
    rel_to_idx = extras['relation_map']
    
    # Build edge_index, edge_type, and edge_attr for all pairs
    src_list = []
    dst_list = []
    edge_type_list = []
    edge_attr_list = []
    
    # For each directed pair, create edges for all 4 bias types
    for (i, j) in pairs:
        for bias in ['linear', 'quadratic', 'skew', 'end']:
            fwd_name, bwd_name = base_relation_map[bias]
            
            # Determine which relation (fwd or bwd) this edge represents
            # For i<j: use forward relation; for i>j: use backward relation
            if i < j:
                rel_name = fwd_name
            else:
                rel_name = bwd_name
            
            rel_idx = rel_to_idx[rel_name]
            
            src_list.append(i)
            dst_list.append(j)
            edge_type_list.append(rel_idx)
            
            # One-hot edge attribute
            k = len(relation_names)
            one_hot = torch.zeros((k,), dtype=torch.get_default_dtype())
            one_hot[rel_idx] = 1.0
            edge_attr_list.append(one_hot)
    
    # Create PyG Data object
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    edge_attr = torch.stack(edge_attr_list, dim=0) if edge_attr_list else torch.zeros((0, len(relation_names)), dtype=torch.get_default_dtype())
    
    data = Data(
        x=data_sparse.x,  # Use node features from sparse graph
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr
    )
    
    # Build targets for each edge: [linear, quadratic, skew, end]
    # Each edge should only predict its specific coefficient type
    base_order = list(base_relation_map.keys())  # ['linear', 'quadratic', 'skew', 'end']
    
    targets = []
    for (i, j) in pairs:
        # Linear: b[j] - b[i] (proper antisymmetric conversion)
        # This naturally gives antisymmetric values: linear_ji = b[i] - b[j] = -linear_ij
        linear = float(b_vector[j] - b_vector[i])
        
        # Quadratic: Antisymmetric, only upper triangle stored
        # If i < j: use c[i,j] directly
        # If i > j: use -c[j,i] (negate the opposite direction)
        if i < j:
            quadratic = float(c_matrix[i, j])
        else:
            quadratic = -float(c_matrix[j, i])
        
        # Skew and End: NOT antisymmetric, both directions stored independently
        skew = float(x_matrix[i, j])
        end = float(s_matrix[i, j])
        
        # Create 4 edges per pair, each with ONLY its target coefficient
        # Edge 0 (linear type): predict [linear, 0, 0, 0]
        # Edge 1 (quadratic type): predict [0, quadratic, 0, 0]
        # Edge 2 (skew type): predict [0, 0, skew, 0]
        # Edge 3 (end type): predict [0, 0, 0, end]
        targets.append([linear, 0.0, 0.0, 0.0])
        targets.append([0.0, quadratic, 0.0, 0.0])
        targets.append([0.0, 0.0, skew, 0.0])
        targets.append([0.0, 0.0, 0.0, end])
    
    return data, targets, extras


def filter_best_runs_per_system(runs: List[Dict], reward_config: Optional[Dict] = None) -> List[Dict]:
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
    
    from collections import defaultdict
    
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
            grandparent_dir = run_dir.parent.parent.name  # e.g., "pretraining" or "14benz_good_runs"
            
            # Check if parent looks like a combo directory (starts with "comb_")
            if parent_dir.startswith('comb_'):
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
            
            # Compute reward using config parameters (filter out legacy/unknown parameters)
            valid_params = {
                'w_P', 'w_T', 'w_U', 'gamma', 'P_baseline', 'T_baseline',
                'min_transitions_per_site', 'min_coverage_ratio', 'entropy_bonus',
                'concentration_penalty_threshold'
            }
            filtered_config = {k: v for k, v in reward_config.items() if k in valid_params}
            
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
            print(f"  {system_name}: selected run with reward {best_reward:.2f}")
    
    return best_runs


def load_pretraining_runs(pretraining_dir: Path) -> List[Dict]:
    """Load all collected pretraining runs.
    
    Supports both flat and nested directory structures:
    - Flat: pretraining_dir/run_001/, pretraining_dir/run_002/, ...
    - Nested: pretraining_dir/combo_XXX/run_001/, pretraining_dir/combo_XXX/run_002/, ...
    
    Args:
        pretraining_dir: Directory with collected run data
    
    Returns:
        List of dicts with run_dir, metadata, sim_results for each run
    """
    runs = []
    
    def _load_run(run_dir: Path):
        """Helper to load a single run directory."""
        # Load metadata
        metadata_file = run_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Skip if simulation didn't terminate normally
        if not metadata.get("terminated_normally", False):
            print(f"  Skipping {run_dir.name}: did not terminate normally")
            return None
        
        # Load simulation results
        results_file = run_dir / "simulation_results.json"
        if not results_file.exists():
            print(f"  Skipping {run_dir.name}: no simulation_results.json")
            return None
        
        with open(results_file, 'r') as f:
            sim_results = json.load(f)
        
        return {
            "run_dir": run_dir,
            "source_dir": metadata.get("source_run_dir"),
            "metadata": metadata,
            "sim_results": sim_results,
        }
    
    # Iterate through top-level directories
    for entry in sorted(pretraining_dir.iterdir()):
        if not entry.is_dir():
            continue
        
        # Check if this is a run directory (has metadata.json)
        if (entry / "metadata.json").exists():
            # Flat structure: entry is a run directory
            run_data = _load_run(entry)
            if run_data:
                runs.append(run_data)
        else:
            # Nested structure: entry might contain run subdirectories
            # Look one level deeper for run directories
            for subentry in sorted(entry.iterdir()):
                if subentry.is_dir() and (subentry / "metadata.json").exists():
                    run_data = _load_run(subentry)
                    if run_data:
                        runs.append(run_data)
    
    return runs


def compute_reward_from_sim_results(
    sim_results: Dict,
    num_sites: int,
    nsubs_per_site: List[int],
    w_P: float = 0.5,
    w_T: float = 0.75,
    w_U: float = 0.3,
    gamma: float = 4.0,
    P_baseline: float = 500.0,  # Updated to higher_rewards_v1 config
    T_baseline: float = 50.0,   # Updated to higher_rewards_v1 config
    min_transitions_per_site: int = 10,
    min_coverage_ratio: float = 0.5,
    entropy_bonus: float = 8.0,  # Updated to higher_rewards_v1 config
    concentration_penalty_threshold: float = 0.8,
) -> float:
    """Compute reward from simulation results dict using improved reward logic.
    
    This implements the same logic as train_improved.py but works with cached
    simulation results instead of reading from output files.
    
    **Confidence Factor (C_F)**: Scales population reward by data reliability
    
    Uses a **graduated transition penalty system** to provide continuous feedback:
    - **0 transitions**: -40.0 penalty (death floor - zero activity)
    - **1 transition**: -32.0 penalty (very low activity)
    - **2 transitions**: -24.0 penalty (very low activity)
    - **3-9 transitions**: -2.0 - (2.0 × deficit) penalty (climbing ramp, ranges -16 to -4)
    - **≥10 transitions**: 0.0 penalty, unlocks R_T reward (success zone)
    
    Default parameters match the updated reward configuration which achieved
    improved gradient signal for low-transition runs (graduated penalties help
    1-2 transition runs compared to uniform -40 penalty).
    
    Args:
        sim_results: Dict with 'populations' and 'transitions' keys
        num_sites: Number of sites in the system
        nsubs_per_site: List of number of substituents per site
        w_P, w_T, w_U: Reward weights for populations, transitions, and uniformity
        gamma: Scaling factor for rewards (used in coverage/concentration penalties)
        P_baseline: Normalization baseline for populations
        T_baseline: Normalization baseline for transitions
        min_transitions_per_site: Minimum transitions required per site (default: 10)
        min_coverage_ratio: Minimum ratio of substituents that must be visited
        entropy_bonus: Bonus for uniform distributions
        concentration_penalty_threshold: Threshold for concentration penalty (e.g., 0.8 = 80%)
    
    Returns:
        Scalar reward value
    """
    populations = sim_results.get("populations", {})
    transitions = sim_results.get("transitions", {})
    
    # Extract population counts (use only HIGHEST lambda value)
    pop_list = []
    for block_id in sorted([int(k) for k in populations.keys()]):
        block_data = populations[str(block_id)]
        counts = block_data.get("counts", {})
        if counts:
            # Use only the highest lambda value
            max_lambda = max(counts.keys(), key=lambda x: float(x))
            pop_list.append(counts[max_lambda])
    
    if not pop_list:
        return -50.0  # No population data (capped at penalty limit)
    
    pop_array = np.array(pop_list, dtype=float)
    total_pop = pop_array.sum()
    
    if total_pop == 0:
        return -50.0  # No sampling occurred (capped at penalty limit)
    
    # Extract transition counts per site (use only HIGHEST lambda value)
    trans_per_site = []
    for site_id in sorted([int(k) for k in transitions.keys()]):
        site_data = transitions[str(site_id)]
        if site_data and isinstance(site_data, dict):
            # Use only the highest lambda value
            max_lambda = max(site_data.keys(), key=lambda x: float(x))
            total_trans = site_data[max_lambda]
            trans_per_site.append(total_trans)
        else:
            trans_per_site.append(0)
    
    # Ensure we have transition data for all sites
    while len(trans_per_site) < num_sites:
        trans_per_site.append(0)
    
    trans_array = np.array(trans_per_site[:num_sites], dtype=float)
    total_trans = trans_array.sum()
    
    # Track minimum transitions across all sites for Confidence Factor
    min_transitions_across_sites = int(trans_array.min())
    
    # === STRICT REQUIREMENTS (penalties) ===
    # Match train_improved.py penalty calculation
    penalties = 0.0
    
    # 1. Tiered transition penalty system
    # Replaces binary threshold with continuous gradient feedback
    sites_below_threshold = 0
    
    for site_idx, trans_count in enumerate(trans_array):
        if trans_count < min_transitions_per_site:
            sites_below_threshold += 1
            
            if trans_count == 0:
                # Tier 1: "Death Floor" - zero activity (max penalty)
                penalties -= 40.0
            elif trans_count == 1:
                # Tier 1: "Death Floor" - very low activity
                penalties -= 32.0
            elif trans_count == 2:
                # Tier 1: "Death Floor" - very low activity
                penalties -= 24.0
            elif trans_count < min_transitions_per_site:
                # Tier 2: "Climbing Ramp" - softened gradient from ~-16 to -4
                # Formula: -2.0 - (2.0 * deficit)
                deficit = min_transitions_per_site - trans_count
                penalties -= (2.0 + 2.0 * deficit)
    
    # 2. Coverage requirement (minimum % of substituents visited)
    num_populated = np.count_nonzero(pop_array)
    total_subs = sum(nsubs_per_site)
    coverage_ratio = num_populated / total_subs if total_subs > 0 else 0.0
    
    if coverage_ratio < min_coverage_ratio:
        deficit = min_coverage_ratio - coverage_ratio
        penalties -= gamma * 20.0 * deficit
    
    # 3. Concentration penalty (per-site check)
    pop_idx = 0
    for site_idx, nsubs in enumerate(nsubs_per_site):
        site_pops = pop_array[pop_idx:pop_idx + nsubs]
        site_total = site_pops.sum()
        
        if site_total > 0:
            concentration_ratio = site_pops.max() / site_total
            if concentration_ratio > concentration_penalty_threshold:
                penalties -= gamma * 5.0 * (concentration_ratio - concentration_penalty_threshold)
        
        pop_idx += nsubs
    
    # === REWARD COMPONENTS ===
    
    # Confidence Factor (C_F): Scale population reward by data reliability
    # Low-transition runs have unreliable population distributions
    confidence_factor = min(1.0, min_transitions_across_sites / (2.0 * min_transitions_per_site))
    
    # R_P: Population balance (coefficient of variation - lower is better/more uniform)
    pop_probs = pop_array / total_pop  # Needed for entropy calculation below
    nonzero_pops = pop_array[pop_array > 0]
    
    R_P = 0.0
    if len(nonzero_pops) > 1:
        # Require meaningful coverage: at least 2 subs per site on average
        # If we have 8 subs total and only 2 are visited (one per site), that's degenerate
        min_meaningful_coverage = max(2, num_sites * 1.5)  # At least 1.5 subs per site
        
        if len(nonzero_pops) >= min_meaningful_coverage:
            # Use coefficient of variation (std/mean) for balance
            pop_mean = np.mean(nonzero_pops)
            pop_std = np.std(nonzero_pops)
            cv = pop_std / pop_mean if pop_mean > 0 else 1.0
            
            # Balance factor: exp(-cv) ranges from ~0.37 (CV=1) to 1.0 (CV=0)
            balance_factor = np.exp(-cv)
            
            # Normalized population reward (only count non-zero populations)
            total_pop_normalized = sum(p / P_baseline for p in nonzero_pops)
            
            # Apply Confidence Factor to scale R_P by data reliability
            R_P = w_P * total_pop_normalized * balance_factor * confidence_factor
        else:
            # Insufficient coverage: minimal reward proportional to coverage
            R_P = w_P * 0.01 * coverage_ratio * confidence_factor
    
    # R_T: Transitions (Tier 3: "Success Zone" - only if all sites >= min_transitions_per_site)
    if sites_below_threshold == 0:
        R_T = w_T * (total_trans / T_baseline)
        avg_trans_per_site = total_trans / num_sites if num_sites > 0 else 0
        if avg_trans_per_site > min_transitions_per_site * 2:
            R_T *= 1.5  # 50% bonus for very active sampling
    else:
        R_T = 0.0  # No transition reward if some sites below threshold (Tier 1 or 2)
    
    # R_U: Coverage uniformity reward (matching train_improved.py)
    R_U = w_U * coverage_ratio
    
    # R_entropy: Shannon entropy bonus for uniform distributions
    entropy = -np.sum(pop_probs * np.log(pop_probs + 1e-10))
    max_entropy = np.log(len(pop_probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    R_entropy = entropy_bonus * normalized_entropy
    
    # ========== PENALTY CLAMPING ==========
    # Prevent gradient explosion by capping maximum negative penalty
    max_penalty = 50.0
    if penalties < -max_penalty:
        penalties = -max_penalty
    
    # Total reward (matching train_improved.py)
    reward = R_P + R_T + R_U + R_entropy + penalties  # penalties are already negative
    
    return reward


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
    from collections import defaultdict
    
    # Collect all coefficient values by type
    coeff_values = {
        'linear': [],
        'quadratic': [],
        'skew': [],
        'end': []
    }
    
    print(f"Computing coefficient statistics across {len(runs)} runs...")
    
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
        
        except Exception as e:
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
    print("\n" + "="*80)
    print("Coefficient Statistics Across All Runs")
    print("="*80)
    for bias_type, stat in stats.items():
        print(f"{bias_type:12s}: mean={stat['mean']:7.2f}, std={stat['std']:7.2f}, "
              f"p95={stat['p95']:7.2f}, max={stat['max']:7.2f}, n={stat['count']}")
    print("="*80 + "\n")
    
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
        print(f"Warning: Only {len(runs)} runs available, skipping statistical filtering")
        return runs
    
    # Compute statistics across all runs
    stats = compute_coefficient_statistics(runs)
    
    # Compute thresholds
    thresholds = {}
    print(f"Filtering thresholds (mean ± {n_std}σ):")
    for bias_type, stat in stats.items():
        threshold = stat['mean'] + n_std * stat['std']
        thresholds[bias_type] = threshold
        print(f"  {bias_type:12s}: ≤ {threshold:.2f}")
    print()
    
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
                excluded_runs.append((run_dir.parent.name, run_dir.name, violated_type, max_violation))
            else:
                filtered_runs.append(run)
        
        except Exception as e:
            # On error, include the run (conservative approach)
            filtered_runs.append(run)
    
    # Report results
    print(f"\n{'='*80}")
    print(f"Statistical Filtering Results (±{n_std}σ threshold)")
    print(f"{'='*80}")
    print(f"  Total runs: {len(runs)}")
    print(f"  Kept: {len(filtered_runs)} ({100*len(filtered_runs)/len(runs):.1f}%)")
    print(f"  Excluded: {len(excluded_runs)} ({100*len(excluded_runs)/len(runs):.1f}%)")
    
    if excluded_runs:
        print(f"\n  Top 10 excluded runs:")
        excluded_runs.sort(key=lambda x: x[3], reverse=True)
        for system, run, bias_type, max_val in excluded_runs[:10]:
            print(f"    {system:30s} {run:15s} ({bias_type:10s}: {max_val:.2f})")
    
    print(f"{'='*80}\n")
    
    return filtered_runs


def filter_runs_by_reward(
    runs: List[Dict],
    min_reward: float = 0.0,
) -> List[Dict]:
    """Filter runs by minimum reward threshold.
    
    Computes reward for each run using compute_reward_from_sim_results() and
    excludes runs below the threshold.
    
    Args:
        runs: List of run dicts from load_pretraining_runs
        min_reward: Minimum reward threshold (default: 0.0)
    
    Returns:
        Filtered list of runs meeting reward threshold
    """
    from pathlib import Path
    import json
    
    print(f"\n{'='*80}")
    print(f"Reward Filtering (threshold: >= {min_reward})")
    print(f"{'='*80}")
    
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
                site_num = block_data['site']
                site_counts[site_num] = site_counts.get(site_num, 0) + 1
            
            num_sites = len(site_counts)
            nsubs_per_site = [site_counts[site] for site in sorted(site_counts.keys())]
            
            # Compute reward
            reward = compute_reward_from_sim_results(sim_results, num_sites, nsubs_per_site)
            
            # Filter by threshold
            if reward >= min_reward:
                filtered_runs.append(run)
            else:
                excluded_runs.append((run_dir.parent.name, run_dir.name, reward, "Below threshold"))
        
        except Exception as e:
            # On error, exclude the run (conservative approach for reward filtering)
            error_reason = f"Error: {str(e)}"
            error_counts[error_reason] = error_counts.get(error_reason, 0) + 1
            excluded_runs.append((run_dir.parent.name, run_dir.name, None, error_reason))
    
    # Report results
    print(f"  Total runs: {len(runs)}")
    print(f"  Kept: {len(filtered_runs)} ({100*len(filtered_runs)/len(runs):.1f}%)")
    print(f"  Excluded: {len(excluded_runs)} ({100*len(excluded_runs)/len(runs):.1f}%)")
    
    # Show error breakdown
    if error_counts:
        print(f"\n  Error breakdown:")
        for error_reason, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {error_reason}: {count} runs")
    
    if excluded_runs:
        # Show distribution of excluded runs
        reward_excluded = [x for x in excluded_runs if isinstance(x[2], (int, float))]
        if reward_excluded:
            print(f"\n  Reward statistics for excluded runs:")
            rewards = [x[2] for x in reward_excluded]
            print(f"    Mean: {sum(rewards)/len(rewards):.2f}")
            print(f"    Min: {min(rewards):.2f}")
            print(f"    Max: {max(rewards):.2f}")
        
        print(f"\n  Top 10 excluded runs (by reward):")
        # Sort by reward (descending) - show the "best" of the excluded runs
        reward_excluded_sorted = sorted(reward_excluded, key=lambda x: x[2], reverse=True)
        for system, run, reward, reason in reward_excluded_sorted[:10]:
            print(f"    {system:30s} {run:15s} reward={reward:7.2f}")
    
    print(f"{'='*80}\n")
    
    return filtered_runs


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
    use_fully_connected=True,
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
        use_fully_connected: If True, use fully-connected graph with all pairs within sites.
                            If False, use sparse graph with only non-zero coefficient edges.
                            Fully-connected provides more training data and proper linear bias encoding.
    
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
            if use_fully_connected:
                # Use fully-connected graph for richer training data
                data, targets, extras = build_fully_connected_graph_for_pretraining(
                    run_dir, 
                    toppar_dir=toppar_dir,
                    toppar_files=toppar_files,
                    warn_missing_types=warn_missing_types
                )
            else:
                # Use sparse graph (only edges with non-zero coefficients)
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
        
        # Log high losses with run information
        if mse_loss.item() > 500.0:
            run_name = f"{run_dir.parent.name}/{run_dir.name}"
            print(f"  ⚠️  HIGH LOSS: Run {run_idx+1}/{len(runs)} ({run_name}): loss={mse_loss.item():.2f}")
        
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
    use_best_only: bool = False,
    use_fully_connected: bool = True,
    filter_outliers: bool = True,
    outlier_std_threshold: float = 3.0,
    min_reward_threshold: Optional[float] = None,
):
    """Run policy pretraining with provided runs.
    
    Args:
        runs: List of run dicts (from load_pretraining_runs)
        output_dir: Directory to save pretrained policy
        config: Configuration dict (same format as workflow_sample.yaml)
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device ('cuda' or 'cpu'). If None, auto-detect.
        use_best_only: If True, filter to best run per system. If False, use all valid runs.
        use_fully_connected: If True (default), build fully-connected graphs with all pairs
                            within sites for richer training data and proper linear bias encoding.
                            If False, use sparse graphs with only non-zero coefficient edges.
        filter_outliers: If True (default), exclude runs with coefficients outside statistical range
        outlier_std_threshold: Number of standard deviations for outlier threshold (default: 3.0)
        min_reward_threshold: If set, exclude runs with reward below this threshold (e.g., 5.0)
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
    
    # Optionally filter to keep only best run per system for behavior cloning
    if use_best_only:
        # Use reward config from workflow yaml for consistent reward calculation
        reward_config = config.get('reward', {})
        print(f"\nFiltering {len(runs)} runs to keep only best per system...")
        print(f"Using reward config: {reward_config}")
        best_runs = filter_best_runs_per_system(runs, reward_config=reward_config)
        print(f"Filtered to {len(best_runs)} best runs for training\n")
        
        if len(best_runs) == 0:
            print("Error: No valid runs after filtering")
            return
        
        # Update runs to use filtered best runs
        runs = best_runs
    else:
        print(f"\nUsing all {len(runs)} valid runs for pretraining (no filtering)\n")
    
    # Optionally filter runs by minimum reward threshold (applied first)
    if min_reward_threshold is not None:
        print(f"\nApplying reward filtering (threshold: >= {min_reward_threshold})...")
        runs = filter_runs_by_reward(runs, min_reward=min_reward_threshold)
        
        if len(runs) == 0:
            print("Error: No runs remaining after reward filtering")
            print("Try lowering min_reward_threshold or disable reward filtering")
            return
    
    # Optionally filter outlier runs based on coefficient statistics (applied second)
    if filter_outliers:
        print(f"\nApplying statistical filtering (±{outlier_std_threshold}σ threshold)...")
        runs = filter_runs_by_coefficient_range(runs, n_std=outlier_std_threshold)
        
        if len(runs) == 0:
            print("Error: No runs remaining after statistical filtering")
            print("Try increasing outlier_std_threshold or set filter_outliers=False")
            return
    
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
    
    # Optimizer: policy.parameters() already includes encoder since encoder is a submodule
    optimizer = optim.Adam(
        policy.parameters(),
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
            warn_missing_types=warn_missing_types,
            use_fully_connected=use_fully_connected
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
    parser.add_argument(
        "--use-best-only",
        action="store_true",
        help="If set, filter to best run per system. Otherwise use all valid runs (default: use all)",
    )
    parser.add_argument(
        "--use-sparse-graphs",
        action="store_true",
        help="If set, use sparse graphs (only non-zero coefficient edges). By default, uses fully-connected graphs with all pairs within sites for richer training data.",
    )
    parser.add_argument(
        "--no-filter-outliers",
        action="store_true",
        help="If set, disable statistical filtering of outlier runs. By default, runs with coefficients outside ±N standard deviations are excluded.",
    )
    parser.add_argument(
        "--outlier-std-threshold",
        type=float,
        default=3.0,
        help="Number of standard deviations for outlier threshold (default: 3.0). Runs with any coefficient exceeding mean ± N*std are excluded.",
    )
    parser.add_argument(
        "--min-reward-threshold",
        type=float,
        default=None,
        help="Minimum reward threshold for filtering runs (e.g., 5.0). If set, only runs with reward >= threshold are used for pretraining.",
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
        use_best_only=args.use_best_only,
        use_fully_connected=not args.use_sparse_graphs,  # Default to fully-connected
        filter_outliers=not args.no_filter_outliers,  # Default to True (filter enabled)
        outlier_std_threshold=args.outlier_std_threshold,
        min_reward_threshold=args.min_reward_threshold,
    )


if __name__ == "__main__":
    main()
