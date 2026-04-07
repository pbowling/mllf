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
import collections
import math
import random
import re
import threading
import warnings
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


# ── per-system graph cache helpers ──────────────────────────────────────────

def _build_graph_structure(prep_dir: str, graph_info_path: Path,
                           toppar_dir, toppar_files, warn_missing_types,
                           deepset_model, solvent_state) -> tuple:
    """Build the *structure-only* part of a pretraining graph (node features + edges).

    This is the expensive step (DeepSet AEV computation).  All runs that share
    the same prep directory will produce an identical structure, so we call this
    once per prep_dir and reuse the result for every run in that group.

    Returns (data_structure, extras, nsubs_per_site) where data_structure holds
    node features, edge_index, edge_type, edge_attr but NO target-specific data.
    """
    import json
    from torch_geometric.data import Data
    from mllf.cb import graph_utils
    from mllf.cb.graph import Graph
    from mllf.cb.graph_utils import build_directed_pairs

    with open(graph_info_path) as f:
        graph_info = json.load(f)

    g = Graph.from_graph_info(graph_info)

    rtf_results = None
    if deepset_model is not None and prep_dir is not None:
        from mllf.file_handling.read_rtf import parse_rtf_dir
        rtf_results = parse_rtf_dir(prep_dir)

    data_sparse, extras = graph_utils.build_pyg_graph_from_mllf_graph(
        g, toppar_dir=toppar_dir, toppar_files=toppar_files,
        warn_missing_types=warn_missing_types,
        deepset_model=deepset_model,
        pdb_dir=prep_dir,
        pdb_pattern='site{site}_sub{sub}_frag.pdb',
        rtf_results=rtf_results,
        prep_dir=prep_dir,
        solvent_state=solvent_state,
    )

    nsubs_per_site = graph_info.get('nsubs_per_site', [])
    if not nsubs_per_site:
        from collections import defaultdict
        site_counts = defaultdict(int)
        for _, node_info in graph_info.get('sites', {}).items():
            s = node_info.get('site')
            if s is not None:
                site_counts[s] += 1
        if site_counts:
            nsubs_per_site = [site_counts[s] for s in sorted(site_counts)]
        else:
            raise ValueError(f"Could not determine nsubs_per_site from {graph_info_path}")

    relation_names   = extras['relation_names']
    base_relation_map = extras['base_relation_map']
    rel_to_idx       = extras['relation_map']
    pairs = build_directed_pairs(nsubs_per_site)

    src_list, dst_list, edge_type_list, edge_attr_list = [], [], [], []
    for (i, j) in pairs:
        for bias in ['linear', 'quadratic', 'skew', 'end']:
            fwd_name, bwd_name = base_relation_map[bias]
            rel_name = fwd_name if i < j else bwd_name
            rel_idx  = rel_to_idx[rel_name]
            src_list.append(i)
            dst_list.append(j)
            edge_type_list.append(rel_idx)
            k = len(relation_names)
            one_hot = torch.zeros((k,), dtype=torch.get_default_dtype())
            one_hot[rel_idx] = 1.0
            edge_attr_list.append(one_hot)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type  = torch.tensor(edge_type_list, dtype=torch.long)
    edge_attr  = (torch.stack(edge_attr_list, dim=0) if edge_attr_list
                  else torch.zeros((0, len(relation_names)), dtype=torch.get_default_dtype()))

    data = Data(
        x=data_sparse.x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
    )
    return data.cpu(), extras, nsubs_per_site, pairs


def _extract_targets_from_variables(run_dir, nsubs_per_site: list, pairs: list):
    """Extract bias-coefficient targets from a single run's variables.py.

    This is cheap (pure Python, no AEV computation).  Call once per run after
    the shared graph structure has been built via ``_build_graph_structure``.

    Returns a list of [linear, quadratic, skew, end] targets (one per directed
    edge), or None if the variables file cannot be parsed, is missing, or the
    bias matrix size does not match nsubs_per_site.
    """
    import yaml
    try:
        variables_path = Path(run_dir) / 'variables.py'
        if not variables_path.exists():
            return None
        content = variables_path.read_text()

        # Find the YAML block — try all quoting styles; track the matching closer
        yaml_start = -1
        close_delim = None
        for open_d, close_d in (('bias_string = """', '"""'),
                                 ('bias_string="""',   '"""'),
                                 ("bias_string = '''", "'''"),
                                 ("bias_string='''",   "'''")):
            pos = content.find(open_d)
            if pos != -1:
                yaml_start = pos + len(open_d)
                close_delim = close_d
                break
        if yaml_start == -1:
            return None

        yaml_end = content.find(close_delim, yaml_start)
        if yaml_end == -1:
            return None

        bias_data = yaml.safe_load(content[yaml_start:yaml_end])
        b_list = bias_data['b']
        b_vector = np.array(b_list[0] if isinstance(b_list[0], list) else b_list, dtype=float)
        n_nodes = sum(nsubs_per_site)
        if len(b_vector) != n_nodes:
            return None  # size mismatch — incompatible run
        c_matrix = np.array(bias_data['c'], dtype=float)
        x_matrix = np.array(bias_data['x'], dtype=float)
        s_matrix = np.array(bias_data['s'], dtype=float)
        targets = []
        for (i, j) in pairs:
            linear    = float(b_vector[j] - b_vector[i])
            quadratic = float(c_matrix[i, j]) if i < j else -float(c_matrix[j, i])
            skew      = float(x_matrix[i, j])
            end       = float(s_matrix[i, j])
            targets.append([linear,    0.0, 0.0, 0.0])
            targets.append([0.0, quadratic, 0.0, 0.0])
            targets.append([0.0,       0.0, skew, 0.0])
            targets.append([0.0,       0.0, 0.0,  end])
        return targets
    except Exception:
        return None


def build_fully_connected_graph_for_pretraining(run_dir: Path, toppar_dir=None, toppar_files=None,
                                                 warn_missing_types=True, deepset_model=None,
                                                 pdb_dir=None, prep_dir=None, solvent_state=None,
                                                 pdb_pattern='site{site}_sub{sub}_frag.pdb'):
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
        deepset_model: Optional PretrainedDeepSet model for 3D structural node features.
            When provided, node features are DeepSet embeddings instead of atom-type encodings.
        pdb_dir: Directory containing _frag.pdb files (required when deepset_model is provided).
        prep_dir: Prep directory for MIC/context-aware AEV computation (usually same as pdb_dir).
        solvent_state: Environment type ('gas'/'vacuum', 'protein', 'solvent'/'water').
        pdb_pattern: Filename pattern for substituent PDB files (default: site{site}_sub{sub}_frag.pdb).
    
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
    
    # Build node features (standard or DeepSet embedding mode)
    # When using DeepSet, load RTF results from the prep directory so that
    # partial charges are available for AEV feature construction.
    rtf_results = None
    if deepset_model is not None and pdb_dir is not None:
        from mllf.file_handling.read_rtf import parse_rtf_dir
        rtf_results = parse_rtf_dir(pdb_dir)

    data_sparse, extras = graph_utils.build_pyg_graph_from_mllf_graph(
        g, toppar_dir=toppar_dir, toppar_files=toppar_files,
        warn_missing_types=warn_missing_types,
        deepset_model=deepset_model,
        pdb_dir=pdb_dir,
        pdb_pattern=pdb_pattern,
        rtf_results=rtf_results,
        prep_dir=prep_dir,
        solvent_state=solvent_state,
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

    # Sanity-check: bias matrix size must match the node count from graph_info.
    # A mismatch means the run was simulated with a different number of active
    # substituents than graph_info.json records (e.g. cmet runs with 6 active
    # subs but graph_info listing 11).  These runs cannot be used because we
    # cannot reliably map node indices to bias-matrix rows.
    n_nodes = sum(nsubs_per_site)
    if len(b_vector) != n_nodes:
        raise ValueError(
            f"Bias matrix size ({len(b_vector)}) does not match node count "
            f"from nsubs_per_site ({n_nodes} = {nsubs_per_site}). "
            f"Run has inconsistent graph_info / variables data."
        )

    # Build directed pairs for ALL substituents within each site
    from mllf.cb.graph_utils import build_directed_pairs
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
    
    # 1. Multi-site aware transition penalty system
    # Apply base penalty once based on worst site, then add scaling for additional bad sites
    # This prevents unfair accumulation when multiple sites are degenerate
    sites_below_threshold = 0
    
    # Count sites below threshold
    for trans_count in trans_array:
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
    elif sites_below_threshold == 1:
        penalties -= base_penalty
    
    # 2. Coverage requirement (minimum % of substituents visited)
    num_populated = np.count_nonzero(pop_array)
    total_subs = sum(nsubs_per_site)
    coverage_ratio = num_populated / total_subs if total_subs > 0 else 0.0
    
    # Adaptive coverage requirement: scales with system size to encourage visiting multiple subs
    # Formula: min_subs = 1 + 0.5*(total-1)
    # Examples: 2 subs→1.5 (75%), 3 subs→2.0 (67%), 4 subs→2.5 (62.5%), 6 subs→3.5 (58%)
    min_subs_required = 1.0 + 0.5 * (total_subs - 1) if total_subs > 1 else 0.5
    adaptive_min_coverage = min_subs_required / total_subs if total_subs > 0 else 0.0
    
    # NO DOUBLE JEOPARDY: Don't penalize coverage if transitions are too low for reliable statistics
    # Coverage is only meaningful when there are enough transitions to have statistical confidence
    # Only apply coverage penalty if transitions are at or above the success threshold
    if coverage_ratio < adaptive_min_coverage and min_transitions_across_sites >= min_transitions_per_site:
        # System has sufficient transitions but poor coverage - penalize the sampling inefficiency
        deficit = adaptive_min_coverage - coverage_ratio
        penalty_scale = np.sqrt(total_subs) if total_subs > 1 else 1.0
        penalties -= gamma * 20.0 * deficit / penalty_scale
    
    # 3. Concentration penalty (per-site check)
    pop_idx = 0
    for site_idx, nsubs in enumerate(nsubs_per_site):
        site_pops = pop_array[pop_idx:pop_idx + nsubs]
        site_total = site_pops.sum()
        
        if site_total > 0:
            concentration_ratio = site_pops.max() / site_total
            if concentration_ratio > concentration_penalty_threshold:
                # Reduced coefficient (2.0 instead of 5.0) to prevent excessive accumulation in multi-site systems
                penalties -= gamma * 2.0 * (concentration_ratio - concentration_penalty_threshold)
        
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
    # Increased from 50 to 60 to preserve gradient information with multi-site systems
    max_penalty = 60.0
    if penalties < -max_penalty:
        penalties = -max_penalty
    
    # Completeness gate: if any substituent was never visited, replace the positive
    # reward components with -0.01 so the total is always negative. Penalties are
    # still added to preserve gradient signal (worse behaviour = more negative).
    if num_populated < total_subs:
        reward = -0.01 + penalties
    else:
        reward = R_P + R_T + R_U + R_entropy + penalties

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
    import json

    print(f"\n{'='*80}")
    print(f"Transition Filtering (min transitions per site: >= {min_transitions})")
    print(f"{'='*80}")

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

    print(f"  Total runs : {len(runs)}")
    print(f"  Kept       : {len(filtered_runs)} ({100*len(filtered_runs)/max(len(runs),1):.1f}%)")
    print(f"  Excluded   : {len(excluded_runs)} ({100*len(excluded_runs)/max(len(runs),1):.1f}%)")
    if n_errors:
        print(f"  Errors     : {n_errors}")

    # Histogram of worst-site transition counts in excluded runs
    excl_counts = [t for _, _, t, _ in excluded_runs if isinstance(t, int)]
    if excl_counts:
        from collections import Counter
        hist = Counter(excl_counts)
        print(f"  Excluded by worst-site transition count:")
        for k in sorted(hist):
            print(f"    {k:3d} transitions: {hist[k]} runs")

    print(f"{'='*80}\n")
    return filtered_runs


def sample_runs_stratified_negative(
    runs: List[Dict],
    fraction_per_bucket: float = 0.25,
    seed: int = 42,
) -> List[Dict]:
    """Keep all positive-reward runs and randomly sample a fraction of each
    negative-reward bucket.

    Buckets (left-exclusive, right-inclusive except the last):
        (-inf, -50], (-50, -40], (-40, -30], (-30, -20], (-20, -10], (-10, 0)

    All runs with reward >= 0 are always kept in full.
    Runs whose reward cannot be computed are excluded.

    Args:
        runs: List of run dicts from load_pretraining_runs.
        fraction_per_bucket: Fraction of each negative bucket to sample (default: 0.25).
            Must be in (0, 1]. A value of 1.0 keeps all runs in each bucket.
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Filtered + sampled list of runs.
    """
    # Upper bounds for negative buckets; the implicit lower bound of the first
    # bucket is -inf.  For a reward r < 0 we assign it to the first bucket
    # whose upper bound satisfies r <= upper_bound.
    BUCKET_UPPER_BOUNDS = [-50, -40, -30, -20, -10, 0]

    print(f"\n{'='*80}")
    print(f"Stratified Negative Sampling  (fraction_per_bucket={fraction_per_bucket:.0%})")
    print(f"{'='*80}")

    # Compute reward for every run -----------------------------------------
    scored: List[tuple] = []  # (reward, run_dict)
    n_error = 0
    for run in runs:
        try:
            run_dir = Path(run['run_dir'])

            sim_results_path = run_dir / "simulation_results.json"
            if not sim_results_path.exists():
                n_error += 1
                continue
            with open(sim_results_path) as f:
                sim_results = json.load(f)

            graph_info_path = run_dir / "graph_info.json"
            if not graph_info_path.exists():
                n_error += 1
                continue
            with open(graph_info_path) as f:
                graph_info = json.load(f)

            sites = graph_info.get('sites', {})
            if not sites:
                n_error += 1
                continue

            site_counts: Dict = {}
            for block_data in sites.values():
                s = block_data['site']
                site_counts[s] = site_counts.get(s, 0) + 1
            num_sites = len(site_counts)
            nsubs_per_site = [site_counts[s] for s in sorted(site_counts)]

            reward = compute_reward_from_sim_results(sim_results, num_sites, nsubs_per_site)
            scored.append((reward, run))
        except Exception:
            n_error += 1

    print(f"  Total runs scored: {len(scored):,}")
    if n_error:
        print(f"  Skipped (scoring error): {n_error:,}")

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

    # Sample from each bucket -----------------------------------------------
    rng = random.Random(seed)
    sampled_negative: List[Dict] = []
    print(f"\n  {'Bucket':<18} {'Available':>10} {'Sampled':>9}")
    print(f"  {'-'*18} {'-'*10} {'-'*9}")
    for i, label in enumerate(bucket_labels):
        available = buckets[i]
        n_avail = len(available)
        n_sample = max(1, int(math.ceil(n_avail * fraction_per_bucket))) if n_avail > 0 else 0
        if n_avail <= n_sample:
            selected = available
        else:
            selected = rng.sample(available, n_sample)
        sampled_negative.extend(selected)
        print(f"  {label:<18} {n_avail:>10,} {len(selected):>9,}")

    result = positive_runs + sampled_negative
    print(f"\n  Positive (kept all):   {len(positive_runs):>7,}")
    print(f"  Negative (sampled):    {len(sampled_negative):>7,}")
    print(f"  Total after sampling:  {len(result):>7,}")
    print(f"{'='*80}\n")
    return result


from mllf.cb.workflow_utils import build_edge_weights as _build_edge_weights  # noqa: E402


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
    deepset_model=None,
    graph_cache: Optional[List] = None,
    ddg_no_transition_weight: float = 0.2,
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
        deepset_model: Optional PretrainedDeepSet model. When provided, node features are
                      64-dim DeepSet embeddings computed from 3D atomic structure + AEVs
                      instead of standard atom-type encodings.
    
    Returns:
        Dict with epoch statistics
    """
    policy.train()
    encoder.train()
    
    epoch_loss = 0.0
    num_updates = 0
    
    for run_idx, run in enumerate(runs):
        run_dir = run["run_dir"]

        if graph_cache is not None:
            # Use pre-built cached graph — skip all file I/O and AEV computation
            cached = graph_cache[run_idx]
            if cached is None:
                continue
            data_cpu, targets_list = cached
            data    = data_cpu.to(device)
            targets = torch.tensor(targets_list, dtype=torch.float32, device=device)
        else:
            # Resolve prep directory for DeepSet AEV computation
            pdb_dir = None
            prep_dir = None
            solvent_state = None
            if deepset_model is not None:
                source_dir = run.get("source_dir")
                if source_dir:
                    candidate = Path(source_dir) / "prep"
                    if candidate.is_dir():
                        pdb_dir = str(candidate)
                        prep_dir = pdb_dir
                # Fallback: prep may be at the combo level (parent of run_dir)
                if pdb_dir is None:
                    candidate = run_dir.parent / "prep"
                    if candidate.is_dir():
                        pdb_dir = str(candidate)
                        prep_dir = pdb_dir
                solvent_state = run.get("metadata", {}).get("solvent_state")

            # Build graph from saved data AND get target coefficients
            try:
                if use_fully_connected:
                    data, targets, extras = build_fully_connected_graph_for_pretraining(
                        run_dir,
                        toppar_dir=toppar_dir,
                        toppar_files=toppar_files,
                        warn_missing_types=warn_missing_types,
                        deepset_model=deepset_model,
                        pdb_dir=pdb_dir,
                        prep_dir=prep_dir,
                        solvent_state=solvent_state,
                    )
                else:
                    data, targets, extras = build_graph_from_saved_data(
                        run_dir,
                        toppar_dir=toppar_dir,
                        toppar_files=toppar_files,
                        warn_missing_types=warn_missing_types
                    )

                data = data.to(device)

                if targets is None or len(targets) == 0:
                    print(f"  Warning: No target coefficients for {run_dir.name}, skipping")
                    continue

                targets = torch.tensor(targets, dtype=torch.float32, device=device)

            except Exception as e:
                print(f"  Error building graph for {run_dir.name}: {e}")
                continue
        
        # Get predicted coefficients from policy (deterministic mean)
        _, _, predicted_means, _ = policy.get_actions(
            data.x, data.edge_index, data.edge_type, data.edge_attr,
            deterministic=True  # Use mean predictions, not sampled
        )
        
        # Behavior Cloning: weighted & masked MSE.
        # active_mask selects the one non-zero target per edge.
        # edge_weights (from DDG data) down-weight edges whose substituent pair
        # had no observed transitions (unreliable bias targets).
        active_mask = targets.abs() > 1e-8            # [E, 4], one True per row
        if active_mask.any():
            ddg_pairs = run.get('sim_results', {}).get('ddg_pairs', {})
            edge_weights = _build_edge_weights(
                data.edge_index, ddg_pairs, ddg_no_transition_weight, device
            )  # [E]
            # Expand weights to match [E, 4] targets shape then select active
            weights_2d = edge_weights.unsqueeze(1).expand_as(targets)  # [E, 4]
            sq_err = (predicted_means - targets) ** 2                  # [E, 4]
            run_loss = (sq_err * weights_2d)[active_mask].mean()
        else:
            run_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Check for NaN/inf
        if torch.isnan(run_loss) or torch.isinf(run_loss):
            print(f"  Warning: NaN/inf loss for {run['run_dir'].name}, skipping")
            continue
        
        # Update
        optimizer.zero_grad()
        run_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += run_loss.item()
        num_updates += 1
        
        # Log high losses with run information
        if run_loss.item() > 500.0:
            run_name = f"{run_dir.parent.name}/{run_dir.name}"
            print(f"  ⚠️  HIGH LOSS: Run {run_idx+1}/{len(runs)} ({run_name}): loss={run_loss.item():.2f}")
        
        if (run_idx + 1) % 10 == 0:
            avg_loss = epoch_loss / num_updates
            print(f"  Run {run_idx+1}/{len(runs)}: loss={run_loss.item():.4f}, avg_loss={avg_loss:.4f}")
    
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
    min_transitions: Optional[int] = None,
    stratified_negative_fraction: Optional[float] = None,
    deepset_model=None,
    patience: int = 10,
    ddg_no_transition_weight: float = 0.2,
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
        stratified_negative_fraction: If set, keep all positive-reward runs and randomly
                            sample this fraction from each negative-reward bucket
                            ((-inf,-50], (-50,-40], ..., (-10,0)).  Applied instead of
                            min_reward_threshold when both are specified.
        deepset_model: Optional PretrainedDeepSet for 3D atomic node features. When provided,
                      the RGCN in_dim is set to deepset_model.embedding_dim (64) rather than
                      the standard atom-type encoding dimension.
        patience: Early stopping patience (default: 10). Training stops if the MSE loss
                  does not improve for this many consecutive epochs.
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
    
    # Transition-count filter takes highest precedence — direct data quality criterion
    if min_transitions is not None:
        print(f"\nApplying transition filtering (min per site: >= {min_transitions})...")
        runs = filter_runs_by_min_transitions(runs, min_transitions=min_transitions)

        if len(runs) == 0:
            print("Error: No runs remaining after transition filtering")
            print("Try lowering --min-transitions")
            return

    # Stratified negative sampling (takes precedence over min_reward_threshold)
    elif stratified_negative_fraction is not None:
        print(f"\nApplying stratified negative sampling (fraction_per_bucket={stratified_negative_fraction:.0%})...")
        runs = sample_runs_stratified_negative(runs, fraction_per_bucket=stratified_negative_fraction)

        if len(runs) == 0:
            print("Error: No runs remaining after stratified negative sampling")
            return

    # Optionally filter runs by minimum reward threshold (applied when stratified sampling not used)
    elif min_reward_threshold is not None:
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
    
    # Get a sample run to infer graph structure (num_relations, relation_names)
    # Node feature dim is taken from deepset_model.embedding_dim when available.
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

    # When using DeepSet, node features are embedding_dim-dimensional regardless
    # of what the standard graph builder returns.
    node_feat_dim = (
        deepset_model.embedding_dim if deepset_model is not None
        else sample_data.x.size(1)
    )
    if deepset_model is not None:
        print(f"\nDeepSet mode: node features = {node_feat_dim}-dim embeddings "
              f"(replaced standard {sample_data.x.size(1)}-dim atom-type encoding)")
    
    # Create model using config (same as workflow)
    train_config = config.get('training', {})
    encoder_config = train_config.get('encoder', {})
    policy_config = train_config.get('policy', {})
    
    encoder = RGCNEncoder(
        in_dim=node_feat_dim,
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
    # Cosine annealing LR schedule: decays smoothly from lr → lr/100 over all epochs.
    # Prevents the optimizer from escaping good basins found in early epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 100.0
    )

    print(f"\nModel architecture:")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())} params")
    print(f"  Policy: {sum(p.numel() for p in policy.parameters())} params")
    print(f"  LR schedule: cosine annealing, {learning_rate} → {learning_rate/100:.6f} over {epochs} epochs")
    print(f"  Early stopping patience: {patience} epochs")
    print(f"  Graph caching: all {len(runs)} graphs pre-built once, reused across all epochs")

    # Training loop
    reward_config = config.get('reward', {})
    best_loss = float('inf')
    epochs_without_improvement = 0

    # ------------------------------------------------------------------
    # Suppress per-occurrence warnings during training and collect them
    # for a deduplicated summary printed at the end.
    # Warnings are grouped by category + a path-stripped message template,
    # so "Element mismatch in /long/path/foo.pdb" and the same mismatch in
    # every other run appear as a single entry with a total count.
    # ------------------------------------------------------------------
    _warning_counts  = collections.Counter()   # (category, template) -> count
    _warning_example = {}                       # (category, template) -> full message
    _warning_lock    = threading.Lock()          # thread-safe updates during parallel build
    _path_re = re.compile(r'/[^\s:*?"<>|]+')

    _orig_showwarning = warnings.showwarning

    def _collect_warning(message, category, filename, lineno, file=None, line=None):
        text     = str(message)
        template = _path_re.sub('<path>', text)
        key      = (category.__name__, template)
        with _warning_lock:
            _warning_counts[key]  += 1
            if key not in _warning_example:
                _warning_example[key] = (text, filename, lineno)

    warnings.showwarning = _collect_warning

    # ------------------------------------------------------------------
    # Pre-build all graphs once (serially) and cache them.
    # Graph building (file IO + AEV computation) dominates per-epoch time;
    # caching eliminates it from epochs 2 onward — the primary benefit.
    # Building in parallel via ThreadPoolExecutor was benchmarked and found
    # to be no faster: AEV computation is CPU-bound and PyTorch already
    # saturates all available cores per build, so multiple workers just
    # contend for the same threads.
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Pre-building {len(runs)} graphs (serial)...")
    print(f"{'='*60}")

    graph_cache = [None] * len(runs)

    # Group run indices by their resolved prep_dir so the expensive graph
    # structure (DeepSet AEVs + topology) is built once per unique structure.
    # All runs in the same dataset/combo dir share the same prep/ and thus
    # produce identical node features; only the target coefficients differ.
    from collections import defaultdict as _dd
    groups = _dd(list)   # group_key -> [(global_idx, run)]
    for idx, run in enumerate(runs):
        _run_dir = run["run_dir"]
        if deepset_model is not None:
            # Check local prep first (most reliable — always present after copy_prep_to_local.py).
            # This ensures all runs in the same pretraining system directory share one
            # graph-structure build regardless of whether external source paths exist.
            _pdir = None
            _local = Path(_run_dir).parent / "prep"
            if _local.is_dir():
                _pdir = str(_local)
            if _pdir is None:
                # Fall back to source_dir/prep (e.g. combo comb_*/prep)
                _src = run.get("source_dir")
                if _src:
                    _cand = Path(_src) / "prep"
                    if _cand.is_dir():
                        _pdir = str(_cand)
            group_key = _pdir if _pdir else str(_run_dir)
        else:
            group_key = str(_run_dir)   # no sharing without DeepSet
        groups[group_key].append((idx, run))

    n_unique = len(groups)
    n_ok = n_fail = 0
    done = 0
    print(f"  {len(runs)} runs => {n_unique} unique graph structures")

    for struct_num, (group_key, group_runs) in enumerate(groups.items(), 1):
        first_run_dir = group_runs[0][1]["run_dir"]
        gi_path = first_run_dir / "graph_info.json"
        if not gi_path.exists():
            for gidx, _ in group_runs:
                n_fail += 1
                done += 1
            continue

        prep_for_build = (group_key
                          if (deepset_model is not None and group_key != str(first_run_dir))
                          else None)
        _solvent = group_runs[0][1].get("metadata", {}).get("solvent_state")

        if use_fully_connected:
            try:
                struct_data, _extras, _nsubs, _pairs = _build_graph_structure(
                    prep_for_build, gi_path,
                    toppar_dir=toppar_dir, toppar_files=toppar_files,
                    warn_missing_types=warn_missing_types,
                    deepset_model=deepset_model,
                    solvent_state=_solvent,
                )
            except Exception as exc:
                print(f"  Error building structure {struct_num}/{n_unique} "
                      f"({Path(group_key).name}): {exc}")
                n_fail += len(group_runs)
                done   += len(group_runs)
                continue

            for gidx, run in group_runs:
                _tgts = _extract_targets_from_variables(
                    run["run_dir"], _nsubs, _pairs)
                if _tgts:
                    graph_cache[gidx] = (struct_data, _tgts)
                    n_ok += 1
                else:
                    n_fail += 1
                done += 1
                if done % 500 == 0 or done == len(runs):
                    print(f"  {done}/{len(runs)} runs cached "
                          f"({struct_num}/{n_unique} structures, {n_fail} failed)...")
        else:
            for gidx, run in group_runs:
                try:
                    _data, _tgts, _ = build_graph_from_saved_data(
                        run["run_dir"], toppar_dir=toppar_dir, toppar_files=toppar_files,
                        warn_missing_types=warn_missing_types,
                    )
                    if _tgts is None or len(_tgts) == 0:
                        n_fail += 1
                    else:
                        graph_cache[gidx] = (_data.cpu(), _tgts)
                        n_ok += 1
                except Exception as exc:
                    print(f"  Error building graph for {run['run_dir'].name}: {exc}")
                    n_fail += 1
                done += 1
                if done % 500 == 0 or done == len(runs):
                    print(f"  {done}/{len(runs)} graphs built ({n_fail} failed)...")

    print(f"Graph cache ready: {n_ok} built, {n_fail} skipped")

    print(f"\n{'='*60}")
    print(f"Starting behavior cloning for {epochs} epochs")
    print(f"Training on {len(runs)} runs ({n_ok} with valid graphs)")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        stats = pretrain_epoch(
            encoder, policy, optimizer, runs, reward_config, device,
            toppar_dir=toppar_dir,
            toppar_files=toppar_files,
            warn_missing_types=warn_missing_types,
            use_fully_connected=use_fully_connected,
            deepset_model=deepset_model,
            graph_cache=graph_cache,
            ddg_no_transition_weight=ddg_no_transition_weight,
        )
        
        print(f"  MSE Loss: {stats['loss']:.4f}")
        print(f"  Runs processed: {stats['num_runs']}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Step LR scheduler after each epoch
        scheduler.step()
        
        # Save best model (lowest loss)
        if stats['loss'] < best_loss:
            best_loss = stats['loss']
            epochs_without_improvement = 0
            
            best_path = output_dir / "best_policy.pt"
            torch.save({
                'encoder_state': encoder.state_dict(),
                'policy_state': policy.state_dict(),
                'epoch': epoch + 1,
                'loss': stats['loss'],
            }, best_path)
            print(f"  Saved best model (loss: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        torch.save({
            'encoder_state': encoder.state_dict(),
            'policy_state': policy.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch + 1,
            'stats': stats,
        }, checkpoint_path)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs (best loss: {best_loss:.4f})")
            break
    
    # Save final model
    final_path = output_dir / "final_policy.pt"
    torch.save({
        'encoder_state': encoder.state_dict(),
        'policy_state': policy.state_dict(),
        'epoch': epochs,
    }, final_path)
    
    # Save metadata
    metadata = {
        'node_feat_dim': node_feat_dim,
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

    # Restore original warning handler
    warnings.showwarning = _orig_showwarning

    # Print deduplicated warning summary
    if _warning_counts:
        print(f"\n{'='*60}")
        print(f"Warning Summary ({sum(_warning_counts.values())} total occurrences, "
              f"{len(_warning_counts)} unique types)")
        print(f"{'='*60}")
        for (cat, template), count in sorted(_warning_counts.items(),
                                              key=lambda kv: -kv[1]):
            example_text, ex_file, ex_line = _warning_example[(cat, template)]
            short_file = Path(ex_file).name if ex_file else '?'
            print(f"  [{count:5d}x] {cat} ({short_file}:{ex_line})")
            print(f"           {template[:120]}")
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
        "--min-transitions",
        type=int,
        default=None,
        metavar="N",
        help="Keep only runs where every site has >= N transitions at the highest lambda "
             "value (e.g. --min-transitions 3). Uses the raw transition counts from "
             "simulation_results.json — no reward formula involved. Takes precedence over "
             "--min-reward-threshold and --stratified-negative-fraction when set.",
    )
    parser.add_argument(
        "--min-reward-threshold",
        type=float,
        default=None,
        help="Minimum reward threshold for filtering runs (e.g., 5.0). If set, only runs with reward >= threshold are used for pretraining. Ignored when --stratified-negative-fraction is set.",
    )
    parser.add_argument(
        "--stratified-negative-fraction",
        type=float,
        default=None,
        metavar="F",
        help="If set, keep all positive-reward runs and randomly sample this fraction of each "
             "negative-reward bucket: (-inf,-50], (-50,-40], (-40,-30], (-30,-20], (-20,-10], (-10,0). "
             "E.g. 0.25 samples 25%% of each bucket. "
             "When specified, --min-reward-threshold is ignored. Default: disabled.",
    )
    parser.add_argument(
        "--deepset-encoder",
        type=str,
        default=None,
        help="Path to a pretrained DeepSet encoder checkpoint (best_encoder.pt). "
             "When provided, node features are replaced by 64-dim DeepSet embeddings "
             "computed from 3D atomic structure + AEVs (the full DeepSet → max-pool → RGCN pipeline).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10). Training stops if the MSE loss does "
             "not improve for this many consecutive epochs.",
    )
    parser.add_argument(
        "--ddg-no-transition-weight",
        type=float,
        default=None,
        metavar="W",
        help="Weight (0–1) applied to edges whose substituent pair had no observed "
             "lambda-space transitions (NaN or Inf DDG). Default: read from config "
             "reward.ddg_no_transition_weight, falling back to 0.2.",
    )

    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ddg_no_transition_weight: CLI flag takes precedence, then config, then hardcoded default
    reward_cfg = config.get('reward', {})
    ddg_no_transition_weight = (
        args.ddg_no_transition_weight
        if args.ddg_no_transition_weight is not None
        else reward_cfg.get('ddg_no_transition_weight', 0.2)
    )

    # Load pretrained DeepSet encoder if specified
    deepset_model = None
    if args.deepset_encoder:
        from mllf.cb.deepset_autoencoder import load_pretrained_deepset
        print(f"\nLoading pretrained DeepSet encoder from {args.deepset_encoder}...")
        deepset_model = load_pretrained_deepset(args.deepset_encoder, freeze_weights=True)
    
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
        min_transitions=args.min_transitions,
        stratified_negative_fraction=args.stratified_negative_fraction,
        deepset_model=deepset_model,
        patience=args.patience,
        ddg_no_transition_weight=ddg_no_transition_weight,
    )


if __name__ == "__main__":
    main()
