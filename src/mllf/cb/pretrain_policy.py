"""Policy pretraining on collected MSLD simulation data via Behavior Cloning.

This script trains the policy to directly predict bias coefficients from successful
simulations using supervised learning (behavior cloning). This is fundamentally
different from REINFORCE training:

**Behavior Cloning Approach:**
- Extract bias coefficients from successful runs as training targets
- Train policy to predict these coefficients using MSE loss
- Filter data to use only the best runs (highest rewards per system)
- Requires 50-100 epochs for convergence

**Data Requirements:**
- Must have bias coefficient matrices (c, x, s, b) in variables.py
- Must have simulation results to compute rewards for filtering
- RTF files used to build graph structure

**Multi-Structure Support:**
- Minimized structures: MD-equilibrated, energy-minimized (pretraining data)
- Unrelaxed structures: Crystallographic/experimental conformations from training data
- Random selection: Per-run mixing of minimized and unrelaxed
- Supports caching of Uni-Mol embeddings for fast reuse

This allows the policy to learn good bias coefficient predictions from
successful simulations before running expensive RL episodes, and supports
training with diverse structural conformations.

Usage (single structure type):
    python -m mllf.cb.pretrain_policy \\
        --pretraining-dir pretraining/14benz_solv \\
        --output-dir models/pretrained_policy \\
        --config examples/workflow_pretrain.yaml \\
        --epochs 20

Usage (with unrelaxed structures):
    # Update workflow_pretrain.yaml:
    # unimol:
    #   structure_selection: random
    #   unrelaxed_system_mappings:
    #     14benz_solv: /path/to/training/systems/14benz_solv/prep
    
    python -m mllf.cb.pretrain_policy \\
        --pretraining-dir pretraining/14benz_solv \\
        --output-dir models/pretrained_policy \\
        --config examples/workflow_pretrain.yaml \\
        --epochs 20
"""
import argparse
import collections
import csv
import math
import random
import re
import threading
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data

from mllf.cb.policy import UnimolPolicy
from mllf.file_handling.read_pdb import parse_pdb_file


# ── Representation tracking ────────────────────────────────────────────────────

def initialize_representation_tracker(output_dir: Path) -> Path:
    """Initialize CSV file for tracking which representations are used.
    
    Args:
        output_dir: Output directory for pretraining
        
    Returns:
        Path to tracking CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracking_file = output_dir / 'pretraining_representations_used.csv'
    
    # Create header if file doesn't exist
    if not tracking_file.exists():
        with open(tracking_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'system', 'run_idx', 'structure_type', 'source', 
                'prep_dir_used', 'explicit_mapping', 'timestamp'
            ])
            writer.writeheader()
    
    return tracking_file


def track_representation_choice(
    tracking_file: Path,
    system_name: str,
    run_idx: int,
    structure_type: str,
    source: str,
    prep_dir_used: Path,
    explicit_mapping: bool = False
) -> None:
    """Log which representation (minimized/unrelaxed) was used for a run.
    
    Args:
        tracking_file: Path to tracking CSV file
        system_name: System name (e.g., 'indolizine_prot')
        run_idx: Run index
        structure_type: 'minimized' or 'unrelaxed'
        source: Where structures came from (e.g., 'pretraining/prep', 'training/systems')
        prep_dir_used: Full path to prep directory used
        explicit_mapping: Whether an explicit mapping was used
    """
    tracking_file = Path(tracking_file)
    
    with open(tracking_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'system', 'run_idx', 'structure_type', 'source',
            'prep_dir_used', 'explicit_mapping', 'timestamp'
        ])
        writer.writerow({
            'system': system_name,
            'run_idx': run_idx,
            'structure_type': structure_type,
            'source': source,
            'prep_dir_used': str(prep_dir_used.resolve()),
            'explicit_mapping': str(explicit_mapping),
            'timestamp': datetime.now().isoformat(),
        })


def read_representation_tracking(tracking_file: Path) -> List[Dict]:
    """Read and parse representation tracking CSV.
    
    Args:
        tracking_file: Path to tracking CSV file
        
    Returns:
        List of tracking records (dicts)
    """
    tracking_file = Path(tracking_file)
    records = []
    
    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            reader = csv.DictReader(f)
            records = list(reader)
    
    return records


# ── Structure type selection and unrelaxed path detection ───────────────────

# ── Structure type selection and path detection ────────────────────────────


def _select_structure_type_for_run(
    run_idx: int, 
    strategy: str, 
    system_name: Optional[str] = None,
    skip_patterns: Optional[List[str]] = None,
    seed: int = 42
) -> str:
    """Determine which structure type to use for a given run.
    
    **Strategies:**
    - 'minimized': Always use minimized (pretraining) structures
    - 'unrelaxed': Always use unrelaxed (crystallographic) structures  
    - 'random': Randomly alternate between minimized and unrelaxed per run
    
    **System-Specific Skipping:**
    If skip_patterns is provided and the system_name matches any pattern,
    unrelaxed structures are skipped regardless of strategy (falls back to minimized).
    
    Args:
        run_idx: Index of the run (used for random seeding if needed)
        strategy: Strategy to use ('minimized', 'unrelaxed', 'random')
        system_name: Name of the system (for pattern matching against skip list)
        skip_patterns: List of patterns to match against system_name (case-insensitive)
                      Uses 'if pattern.lower() in system_name.lower()'
        seed: Random seed for reproducibility in 'random' mode
        
    Returns:
        str: 'minimized' or 'unrelaxed'
    """
    strategy = strategy.lower()
    
    # Check if this system should skip unrelaxed
    should_skip_unrelaxed = False
    if system_name is not None and skip_patterns:
        system_name_lower = system_name.lower()
        for pattern in skip_patterns:
            if pattern.lower() in system_name_lower:
                should_skip_unrelaxed = True
                break
    
    if strategy == 'minimized' or should_skip_unrelaxed:
        return 'minimized'
    elif strategy == 'unrelaxed':
        return 'unrelaxed'
    elif strategy == 'random':
        rng = np.random.RandomState(seed + run_idx)  # Deterministic per run
        return 'minimized' if rng.rand() < 0.5 else 'unrelaxed'
    else:
        raise ValueError(f"Unknown structure selection strategy: {strategy}. Use 'minimized', 'unrelaxed', or 'random'.")


def build_unrelaxed_system_mappings(
    pretraining_base_dir: Path,
    training_systems_base: Optional[Path] = None,
    config_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:
    """Build mapping from pretraining system directories to unrelaxed (training) directories.
    
    Supports three modes:
    1. Config-provided mappings: Use explicitly provided paths from config
    2. Automatic detection: Match pretraining system names to training/systems/
    3. Hybrid: Config mappings override automatic detection
    
    **Example:**
    If pretraining has:
        - pretraining/indolizine_prot/prep
        - pretraining/indolizine_solv/prep
    
    This function returns:
        {
            'indolizine_prot': Path('/mllf/training/systems/indolizine_prot/prep'),
            'indolizine_solv': Path('/mllf/training/systems/indolizine_solv/prep'),
        }
    
    Args:
        pretraining_base_dir: Base pretraining directory containing system subdirs
        training_systems_base: Base training/systems directory (auto-detected if None)
        config_mappings: Dict of explicit system_name -> unrelaxed_path mappings from config
        
    Returns:
        Dict mapping system names to online prep directories
    """
    pretraining_base_dir = Path(pretraining_base_dir)
    
    # Auto-detect training base if not provided
    if training_systems_base is None:
        training_systems_base = pretraining_base_dir.parent / 'training' / 'systems'
    else:
        training_systems_base = Path(training_systems_base)
    
    mappings = {}
    
    # First, scan pretraining directory for systems
    if pretraining_base_dir.exists():
        for sys_dir in pretraining_base_dir.iterdir():
            if sys_dir.is_dir() and (sys_dir / 'prep').exists():
                sys_name = sys_dir.name
                
                # Check config mappings first
                if config_mappings and sys_name in config_mappings:
                    unrelaxed_path = Path(config_mappings[sys_name])
                else:
                    # Try automatic detection
                    unrelaxed_path = Path(training_systems_base) / sys_name / 'prep'
                
                if unrelaxed_path.exists():
                    mappings[sys_name] = unrelaxed_path
                else:
                    print(f"    Warning: Could not find unrelaxed path for {sys_name}: {unrelaxed_path}")
    
    return mappings


def prepare_pretraining_config(
    config: Dict,
    pretraining_base_dir: Path,
    training_systems_base: Optional[Path] = None,
) -> Dict:
    """Prepare and validate pretraining configuration for multi-structure support.
    
    Processes config to:
    - Build unrelaxed system mappings if needed
    - Validate structure selection strategy
    - Set up cache directory if requested
    
    Args:
        config: Loaded workflow_pretrain.yaml configuration dict
        pretraining_base_dir: Base pretraining directory
        training_systems_base: Base training/systems directory
        
    Returns:
        Updated config dict with resolved mappings and paths
    """
    config = config.copy()
    
    # Extract Uni-Mol configuration
    unimol_cfg = config.get('unimol', {})
    structure_selection = unimol_cfg.get('structure_selection', 'minimized').lower()
    
    # Validate strategy
    if structure_selection not in ['minimized', 'unrelaxed', 'random']:
        raise ValueError(
            f"Invalid structure_selection: {structure_selection}. "
            f"Must be 'minimized', 'unrelaxed', or 'random'."
        )
    
    # Build unrelaxed mappings if needed
    if structure_selection in ['unrelaxed', 'random']:
        config_mappings = unimol_cfg.get('unrelaxed_system_mappings')
        unrelaxed_mappings = build_unrelaxed_system_mappings(
            pretraining_base_dir,
            training_systems_base=training_systems_base,
            config_mappings=config_mappings,
        )
        config['_unrelaxed_system_mappings'] = unrelaxed_mappings  # Store resolved mappings
        
        if not unrelaxed_mappings:
            print(f"    Warning: No unrelaxed system mappings found. Check paths!")
    
    # Setup cache directory
    if unimol_cfg.get('cache_embeddings', False):
        cache_dir = unimol_cfg.get('cache_dir')
        if cache_dir is None:
            cache_dir = Path(pretraining_base_dir) / 'embeddings_cache'
        config['_cache_dir'] = Path(cache_dir)
    
    return config


# ── Uni-Mol embedding computation for pretraining ────────────────────────────

def _compute_unimol_embeddings_and_edges(
    run_dir: Path,
    graph_info_path: Path,
    structure_type: str = 'minimized',
    unrelaxed_system_mappings: Optional[Dict[str, Path]] = None,
    custom_search_paths: Optional[Dict[str, List[str]]] = None,
    cache_dir: Optional[Path] = None,
    run_idx: int = 0,
    env_cutoff: float = 8.0,
    use_environment_difference: bool = True,
    consensus_dict: Optional[Dict[str, Dict[str, Optional[set]]]] = None,
) -> tuple:
    """Compute Uni-Mol embeddings for all substituents in a run.
    
    Supports both minimized (pretraining) and unrelaxed (crystallographic) structures.
    Both representations are constructed from the same prep directory - the choice only
    affects semantic interpretation (e.g., which PDB files represent the initial vs relaxed state).
    
    **Embedding Modes:**
    - **Standard mode** (use_environment_difference=False): Single 512D embedding per node
      - Input to policy: [ligand+environment diff, ligand+environment mean]
      - Backward compatible with original UnimolPolicy design
    - **Dual embedding mode** (use_environment_difference=True): Two separate 512D embeddings per node
      - Ligand-only: core + sub (captures substituent-specific information)
      - Full: core + sub + environment + ref_subs (captures ligand+environment context)
      - Edge input to policy: [diff_ligand (antisymmetric), mean_full (symmetric)]
      - This allows MLPs to learn substituent-dependent and environment-dependent info in parallel
    
    **Structure Types:**
    - 'minimized': Use minimized (relaxed, MD-equilibrated) representation
    - 'unrelaxed': Use unrelaxed (crystallographic/experimental) representation
    
    Both use the same prep directory files; the selection is logged for tracking.
    
    **Consensus Filtering:**
    If consensus_dict is provided, environment atoms are filtered to only atoms present in
    ALL substituents at a site. Each system has its own consensus built from its substituents.
    
    Args:
        run_dir: Directory containing variables.py and graph_info.json
        graph_info_path: Path to graph_info.json
        structure_type: 'minimized' or 'unrelaxed' (default: 'minimized')
        unrelaxed_system_mappings: Dict mapping system names to alternate prep paths (optional override)
        custom_search_paths: Dict mapping system solvent_state to custom PDB search paths
                           e.g., {'protein': ['pdb/protein.pdb'], 'solv': ['solvent.pdb']}
        cache_dir: Optional directory to cache/load embeddings
        run_idx: Run index (for reference in tracking)
        env_cutoff: Environment distance cutoff in Ångströms (default: 8.0)
        use_environment_difference: If True, compute dual embeddings: [ligand-only(512), full(512)] = 1024D per node.
                                   If False, use standard embedding: ligand+environment(512) per node.
        consensus_dict: Optional nested dict mapping systems to site consensus atoms.
                       Structure: {prep_dir_str: {site_key: {(resnum, chain, atomname), ...}, ...}, ...}
                       If provided, environment atoms are filtered to consensus atoms for each site.
                       Use None for sites with no consensus (environment not filtered for that site).
    
    Returns:
        (unimol_embeddings [N, emb_dim], edge_index [2, E], nsubs_per_site, representation_source)
        where:
        - N = total substituents
        - E = directed pairs within sites
        - emb_dim = 1024 when use_environment_difference=True (dual embeddings)
        - emb_dim = 512 when use_environment_difference=False (standard embeddings)
        - representation_source = dict with 'structure_type', 'prep_dir_used', 'source', 'explicit_mapping'
    """
    import json
    from mllf.cb.unimol_representation import (
        get_substituent_unimol_with_environment,
        get_substituent_dual_embeddings,
        construct_full_ligand,
        get_unimol_representation,
    )
    from mllf.cb.graph_utils import build_directed_pairs
    from mllf.file_handling.read_pdb import parse_pdb_file
    
    # Load graph_info to get site structure
    with open(graph_info_path) as f:
        graph_info = json.load(f)
    
    sites = graph_info.get("sites", {})
    
    # Get nsubs_per_site from graph_info if available (preferred)
    nsubs_per_site = graph_info.get('nsubs_per_site', None)
    
    if nsubs_per_site is None:
        # Reconstruct from sites structure: count subs per site
        # Sites can be keyed as "site1_sub1" or as "1" with "subs" list
        from collections import defaultdict
        site_sub_counts = defaultdict(int)
        for site_key, site_info in sites.items():
            site_id = site_info.get("site")
            if site_id is not None:
                site_sub_counts[site_id] += 1
        if site_sub_counts:
            nsubs_per_site = [site_sub_counts[s] for s in sorted(site_sub_counts.keys())]
        else:
            raise ValueError(f"Could not determine nsubs_per_site from {graph_info_path}")
    
    n_nodes = sum(nsubs_per_site)
    
    # Build mapping from (site_id, sub_id) to global node index
    # Iterate through sites and map each (site, sub) to its position
    sub_to_global_idx = {}
    global_idx = 0
    for site_id in sorted(set(site_info["site"] for site_info in sites.values())):
        subs_for_site = sorted([
            site_info["sub"]
            for site_info in sites.values()
            if site_info["site"] == site_id
        ])
        for local_idx, sub_id in enumerate(subs_for_site):
            sub_to_global_idx[(site_id, sub_id)] = global_idx
            global_idx += 1
    
    # Compute Uni-Mol embeddings for each substituent
    prep_dir = run_dir.parent / "prep"
    if not prep_dir.is_dir():
        raise FileNotFoundError(f"Prep dir not found: {prep_dir}")
    
    # Determine which prep directory to use
    # Both minimized and unrelaxed use the same prep_dir; only the choice is tracked
    # Handle nested combos: need to look TWO levels up if in comb_* directory
    sys_name = prep_dir.parent.name
    if sys_name.startswith('comb_'):
        sys_name = prep_dir.parent.parent.name
    active_prep_dir = prep_dir
    explicit_mapping = False
    source = 'pretraining/prep'
    
    # Check if explicit mapping provided for unrelaxed
    if structure_type == 'unrelaxed' and unrelaxed_system_mappings and sys_name in unrelaxed_system_mappings:
        active_prep_dir = Path(unrelaxed_system_mappings[sys_name])
        explicit_mapping = True
        source = 'training/systems'
        if not active_prep_dir.exists():
            raise FileNotFoundError(f"Mapped prep dir not found: {active_prep_dir}")
    
    # Initialize representation source tracking
    representation_source = {
        'structure_type': structure_type,
        'prep_dir_used': str(active_prep_dir.resolve()),
        'source': source,
        'explicit_mapping': explicit_mapping,
    }
    
    # Determine embedding dimension based on representation type
    # Dual embeddings (ligand-only + full) = 1024D, standard = 512D
    emb_dim = 1024 if use_environment_difference else 512
    unimol_embeddings = torch.zeros((n_nodes, emb_dim), dtype=torch.float32)
    
    # Read core PDB once (needed for environment loading)
    core_pdb = active_prep_dir / "core.pdb"
    if not core_pdb.exists():
        raise FileNotFoundError(f"Core PDB not found: {core_pdb}")
    
    # Get solvent state from graph_info if available (for environment path selection)
    solvent_state = graph_info.get('solvent_state', 'solv')
    
    # Determine custom search paths for this system
    sub_custom_paths = None
    if custom_search_paths is not None:
        sub_custom_paths = custom_search_paths.get(solvent_state)
    
    # Compute embedding for each substituent with environment context
    prep_dir_str = str(prep_dir)  # Key for consensus lookup
    
    for site_key, site_info in sites.items():
        site_id = site_info.get("site")
        sub_id = site_info.get("sub")
        
        if site_id is None or sub_id is None:
            continue
        
        try:
            sub_pdb = active_prep_dir / f"site{site_id}_sub{sub_id}_frag.pdb"
            if not sub_pdb.exists():
                print(f"  Warning: Sub PDB not found: {sub_pdb}, using zeros")
                continue
            
            # Get consensus atoms for this site if available.
            # consensus_dict is keyed by [prep_dir_str][site_name] where
            # site_name is per-SITE (e.g. "site1"), NOT per-node like
            # `site_key` (e.g. "site1_sub1") — derive the site-level lookup
            # key from site_id instead of using site_key directly.
            consensus_atoms = None
            if consensus_dict is not None:
                system_consensus = consensus_dict.get(prep_dir_str)
                if system_consensus is not None:
                    consensus_atoms = system_consensus.get(f"site{site_id}")
            
            # Compute Uni-Mol embeddings
            if use_environment_difference:
                # Compute dual embeddings: ligand-only and full
                # Stack as [512 ligand + 512 full] = 1024D for edge computation
                emb_ligand, emb_full = get_substituent_dual_embeddings(
                    sub_pdb=str(sub_pdb),
                    core_pdb=str(core_pdb),
                    prep_dir=active_prep_dir,
                    env_cutoff=env_cutoff,
                    atom_limit=256,
                    custom_search_paths=sub_custom_paths,
                    skip_minimized=(structure_type == 'unrelaxed'),
                    consensus_atoms=consensus_atoms,  # Filter by consensus if provided
                )
                # Concatenate for storage: UnimolPolicy._forward_edges() will split [ligand, full]
                emb = np.concatenate([emb_ligand, emb_full])  # [1024]
            else:
                # Compute standard embedding with environment context (512D)
                emb = get_substituent_unimol_with_environment(
                    sub_pdb=str(sub_pdb),
                    core_pdb=str(core_pdb),
                    prep_dir=active_prep_dir,
                    env_cutoff=env_cutoff,
                    atom_limit=256,
                    custom_search_paths=sub_custom_paths,
                    cache_dir=cache_dir,
                    save_cache=False,  # Save caching is optional; set by caller
                    skip_minimized=(structure_type == 'unrelaxed'),
                    consensus_atoms=consensus_atoms,  # Filter by consensus if provided
                )
            global_idx = sub_to_global_idx[(site_id, sub_id)]
            unimol_embeddings[global_idx] = torch.from_numpy(emb).float()
            
        except Exception as e:
            print(f"  Warning: Could not compute embedding for site{site_id}_sub{sub_id}: {e}")
    
    # Build edge_index for all directed pairs within each site
    pairs = build_directed_pairs(nsubs_per_site)
    src_list = [p[0] for p in pairs]
    dst_list = [p[1] for p in pairs]
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    
    return unimol_embeddings, edge_index, nsubs_per_site, representation_source


# ── Pretraining representation tracking ──────────────────────────────────────

def read_representation_tracking(tracking_file: Path) -> List[Dict]:
    """Read pretraining representation tracking CSV.
    
    Args:
        tracking_file: Path to pretraining_representations_used.csv
    
    Returns:
        List of dicts with tracking information
    """
    import csv
    
    tracking_file = Path(tracking_file)
    if not tracking_file.exists():
        return []
    
    results = []
    with open(tracking_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    return results


# ── per-system graph cache helpers ──────────────────────────────────────────


def _extract_targets_from_variables(run_dir, nsubs_per_site: list, pairs: list):
    """Extract bias-coefficient targets from a single run's variables.py.

    This is cheap (pure Python, no AEV computation).  Call once per run after
    the shared graph structure has been built via ``_build_graph_structure``.

    For UnimolPolicy, each edge gets ONE target row with 4 values: [linear, quadratic, skew, end].
    This matches the policy architecture where each edge produces 4 bias predictions.

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
            quadratic = float(c_matrix[i, j]) if i < j else float(c_matrix[j, i])
            skew      = float(x_matrix[i, j])
            end       = float(s_matrix[i, j])
            # One row per edge with all 4 bias types (matches UnimolPolicy output)
            targets.append([linear, quadratic, skew, end])
        return targets
    except Exception:
        return None


def compute_pairwise_confidence_weights(
    sim_results: Dict,
    nsubs_per_site: List[int],
    pairs: List[tuple],
) -> Optional[List[float]]:
    """Compute per-edge confidence weights from pairwise population balance and DDG data.
    
    For each directed edge (i, j), extracts populations and DDG values from simulation results
    to compute a confidence weight. This weight reflects how well-sampled and reliable the
    pairwise interaction is.
    
    **Confidence Computation:**
    - If either population is 0 or missing: weight = 0 (no data)
    - If DDG value is NaN or infinite: weight = 0 (unreliable)
    - Otherwise: weight = min(pop_i, pop_j) / max(pop_i, pop_j)
      (population balance, ranges 0-1, perfect when populations are equal)
    
    **Node to Block Mapping:**
    MSLD block numbering starts at 1 (reference), with block 2 being the first real block.
    In the graph, nodes are 0-indexed (node 0 = block 2, node 1 = block 3, etc.).
    
    Args:
        sim_results: Dict with 'populations' and 'ddg_pairs' from simulation_results.json
        nsubs_per_site: List of number of substituents per site (for validation)
        pairs: List of (i, j) directed edge tuples from build_directed_pairs()
    
    Returns:
        List of per-edge confidence weights [0.0, 1.0], or None if cannot extract populations
    """
    populations = sim_results.get("populations", {})
    ddg_pairs = sim_results.get("ddg_pairs", {})
    
    if not populations:
        return None
    
    # Extract population counts (use only HIGHEST lambda value, matching reward logic)
    pop_dict = {}  # node_idx (0-based) -> population count
    for block_id_str in populations.keys():
        try:
            block_id = int(block_id_str)
        except ValueError:
            continue
        
        # Convert block ID to node index: node_idx = block_id - 2
        # (block 2 = node 0, block 3 = node 1, etc.)
        node_idx = block_id - 2
        if node_idx < 0:
            continue
        
        block_data = populations[block_id_str]
        counts = block_data.get("counts", {})
        if counts:
            # Use only the highest lambda value (same as reward computation)
            max_lambda = max(counts.keys(), key=lambda x: float(x))
            pop_dict[node_idx] = counts[max_lambda]
    
    if not pop_dict:
        return None
    
    # Compute confidence weight for each edge
    weights = []
    for (i, j) in pairs:
        pop_i = pop_dict.get(i, 0)
        pop_j = pop_dict.get(j, 0)
        
        # Zero weight if either population is missing or zero
        if pop_i <= 0 or pop_j <= 0:
            weights.append(0.0)
            continue
        
        # Check DDG value for this pair
        # DDG key format: "{i+2}_{j+2}" (convert node indices back to block IDs)
        # For reverse pairs: ΔG(j→i) = -ΔG(i→j), so if forward key missing,
        # look for reverse key and negate its value
        ddg_key_fwd = f"{i+2}_{j+2}"
        ddg_key_rev = f"{j+2}_{i+2}"
        
        ddg_value = None
        if ddg_key_fwd in ddg_pairs:
            ddg_value = ddg_pairs[ddg_key_fwd]
        elif ddg_key_rev in ddg_pairs:
            # Use negation of reverse pair (antisymmetric property)
            reverse_val = ddg_pairs[ddg_key_rev]
            try:
                ddg_value = -float(reverse_val)
            except (ValueError, TypeError):
                ddg_value = None
        
        # Zero weight if DDG is missing, NaN, or infinite
        if ddg_value is None:
            weights.append(0.0)
            continue
        
        try:
            ddg_val = float(ddg_value)
            if not (np.isfinite(ddg_val)):  # catches NaN and inf
                weights.append(0.0)
                continue
        except (ValueError, TypeError):
            weights.append(0.0)
            continue
        
        # Population balance weight: min / max (symmetric around 0.5 for equal pops)
        balance = min(pop_i, pop_j) / max(pop_i, pop_j)
        weights.append(float(balance))
    
    return weights


def build_fully_connected_graph_for_pretraining(run_dir: Path, toppar_dir=None, toppar_files=None,
                                                 warn_missing_types=True,
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
        pdb_dir: Directory containing _frag.pdb files (required to construct Uni-Mol representations).
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
    
    data_sparse, extras = graph_utils.build_pyg_graph_from_mllf_graph(
        g, toppar_dir=toppar_dir, toppar_files=toppar_files,
        warn_missing_types=warn_missing_types,
        pdb_dir=pdb_dir,
        pdb_pattern=pdb_pattern,
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
    pairs_to_process = pairs

    for (i, j) in pairs_to_process:
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
        edge_attr=edge_attr,
        site_index=data_sparse.site_index,
    )
    
    # Build targets for each edge: [linear, quadratic, skew, end]
    # Each edge should only predict its specific coefficient type
    base_order = list(base_relation_map.keys())  # ['linear', 'quadratic', 'skew', 'end']
    
    targets = []
    
    # Process original pairs
    for (i, j) in pairs:
        # Linear: b[j] - b[i] (proper antisymmetric conversion)
        # This naturally gives antisymmetric values: linear_ji = b[i] - b[j] = -linear_ij
        linear = float(b_vector[j] - b_vector[i])
        
        # Quadratic: Symmetric, only upper triangle stored
        # If i < j: use c[i,j] directly
        # If i > j: use c[j,i] (use the opposite direction)
        if i < j:
            quadratic = float(c_matrix[i, j])
        else:
            quadratic = float(c_matrix[j, i])
        
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
        w_P, w_T: Reward weights for populations and transitions
        w_U: Accepted for API compatibility but unused in current formula;
             coverage is handled by the quadratic coverage_factor multiplier.
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
    
    # 2. Coverage requirement
    # NOTE: adaptive coverage penalty removed — replaced by coverage_factor below
    num_populated = np.count_nonzero(pop_array)
    total_subs = sum(nsubs_per_site)
    coverage_ratio = num_populated / total_subs if total_subs > 0 else 0.0
    
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
    
    # R_P: Population balance
    pop_probs = pop_array / total_pop  # needed for entropy below
    nonzero_pops = pop_array[pop_array > 0]
    
    R_P = 0.0
    if len(nonzero_pops) > 1:
        min_meaningful_coverage = max(2, num_sites * 1.5)
        
        if len(nonzero_pops) >= min_meaningful_coverage:
            # balance_factor removed: R_entropy captures within-visited uniformity
            total_pop_normalized = sum(p / P_baseline for p in nonzero_pops)
            R_P = w_P * total_pop_normalized * confidence_factor
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
    
    # R_entropy: Shannon entropy bonus for uniform distributions
    entropy = -np.sum(pop_probs * np.log(pop_probs + 1e-10))
    max_entropy = np.log(len(pop_probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    R_entropy = entropy_bonus * normalized_entropy
    
    # ========== PENALTY CLAMPING ==========
    max_penalty = 60.0
    if penalties < -max_penalty:
        penalties = -max_penalty
    
    # coverage_factor: smooth quadratic multiplier replacing hard completeness gate
    coverage_factor = (num_populated / total_subs) ** 2 if total_subs > 0 else 0.0
    reward = coverage_factor * (R_P + R_T + R_entropy) + penalties

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
                site_num = extract_site_number(block_name, block_data)
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
    raise ValueError(f"Cannot extract site number from block_id '{block_id}' and block_data has no 'site' key")


def sample_runs_stratified_negative(
    runs: List[Dict],
    fraction_per_bucket: float = 0.55,
    seed: int = 42,
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

    Returns:
        Filtered + sampled list of runs.
    """
    # Upper bounds for negative buckets; the implicit lower bound of the first
    # bucket is -inf.  For a reward r < 0 we assign it to the first bucket
    # whose upper bound satisfies r <= upper_bound.
    BUCKET_UPPER_BOUNDS = [-50, -40, -30, -20, -10, 0]

    print(f"\n{'='*80}")
    print(f"Stratified Negative Sampling  (max_fraction={fraction_per_bucket:.0%}, quadratic ramp)")
    print(f"{'='*80}")

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

            reward = compute_reward_from_sim_results(sim_results, num_sites, nsubs_per_site)
            scored.append((reward, run))
        except Exception as e:
            n_error += 1
            error_log.append((str(run_dir), f"{type(e).__name__}: {str(e)[:80]}"))

    print(f"  Total runs scored: {len(scored):,}")
    if n_error:
        print(f"  Skipped (scoring error): {n_error:,}")
        # Print top error types
        error_types = {}
        for run_path, error_msg in error_log:
            if ":" in error_msg:
                error_type = error_msg.split(":")[0]
            else:
                error_type = error_msg
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print(f"\n  Error Type Breakdown:")
        for error_type in sorted(error_types.keys(), key=lambda x: error_types[x], reverse=True):
            count = error_types[error_type]
            print(f"    {error_type:<40} {count:>6,}")
        
        # Log to file for detailed analysis
        error_log_path = Path("scoring_errors.log")
        with open(error_log_path, 'w') as f:
            f.write(f"Scoring Errors Log\n")
            f.write(f"Total errors: {n_error}\n\n")
            for run_path, error_msg in error_log[:20]:  # Log first 20
                f.write(f"{run_path}\n  {error_msg}\n")
        print(f"\n  Full error log saved to: {error_log_path}")

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
    print(f"\n  {'Bucket':<18} {'Fraction':>8} {'Available':>10} {'Sampled':>9}")
    print(f"  {'-'*18} {'-'*8} {'-'*10} {'-'*9}")
    for i, label in enumerate(bucket_labels):
        available = buckets[i]
        n_avail = len(available)
        # Quadratic ramp: bucket 0 → 0%, bucket (n_buckets-1) → fraction_per_bucket
        bucket_frac = fraction_per_bucket * (i / max(n_buckets - 1, 1)) ** 2
        n_sample = max(1, int(math.ceil(n_avail * bucket_frac))) if (n_avail > 0 and bucket_frac > 0) else 0
        if n_avail <= n_sample:
            selected = available
        else:
            selected = rng.sample(available, n_sample) if n_sample > 0 else []
        sampled_negative.extend(selected)
        print(f"  {label:<18} {bucket_frac:>7.0%} {n_avail:>10,} {len(selected):>9,}")

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

    print(f"\n  Positive (kept all):   {len(positive_runs):>7,}")
    print(f"  {'Bucket':<18} {'Count':>10}")
    print(f"  {'-'*18} {'-'*10}")
    for i, label in enumerate(pos_labels):
        print(f"  {label:<18} {pos_buckets[i]:>10,}")

    print(f"\n  Negative (sampled):    {len(sampled_negative):>7,}")
    print(f"  Total after sampling:  {len(result):>7,}")
    print(f"{'='*80}\n")
    return result


def pretrain_epoch(
    policy: nn.Module,
    optimizer: optim.Optimizer,
    runs: List[Dict],
    reward_config: Dict,
    device: torch.device,
    toppar_dir=None,
    toppar_files=None,
    warn_missing_types=True,
    graph_cache: Optional[List] = None,
    groups: Optional[Dict] = None,
    reward_weighted: bool = False,
    awr_temperature: float = 0.5,
    pairwise_weights_cache: Optional[List] = None,
    unimol_config: Optional[Dict] = None,
    consensus_dict: Optional[Dict[str, Optional[set]]] = None,
) -> Dict[str, float]:
    """Run one behavior cloning AWR epoch for UnimolPolicy with optional pairwise weighting.
    
    Args:
        policy: UnimolPolicy model
        optimizer: Optimizer
        runs: List of run dicts with 'run_dir', targets, etc.
        reward_config: Reward configuration dict
        device: Torch device
        toppar_dir: Optional toppar directory
        toppar_files: Optional toppar files
        warn_missing_types: Whether to warn about missing atom types
        graph_cache: Optional pre-built graph cache (list of tuples)
        groups: Optional run grouping for grouped updates
        reward_weighted: Whether to use AWR weighting
        awr_temperature: Temperature for AWR reweighting
        pairwise_weights_cache: Optional per-run pairwise edge weights
        unimol_config: Optional dict with 'environment_cutoff' and 'use_environment_difference'
    
    Returns:
        Dict with 'loss', 'num_runs', 'num_updates' keys
    """
    if unimol_config is None:
        unimol_config = {}
    
    env_cutoff = unimol_config.get('environment_cutoff', 8.0)
    use_environment_difference = unimol_config.get('use_environment_difference', True)
    
    policy.train()

    # Build normalised reward weights (mean = 1.0) so the learning rate is unchanged.
    if reward_weighted:
        raw_w = [max(0.0, run.get("_bc_reward", 0.0)) for run in runs]
        total_w = sum(raw_w)
        if total_w > 1e-10:
            n_runs = len(runs)
            norm_weights = [w * n_runs / total_w for w in raw_w]
        else:
            reward_weighted = False
            norm_weights = None
    else:
        norm_weights = None

    epoch_loss = 0.0
    num_updates = 0
    total_runs = 0

    if groups is not None and graph_cache is not None:
        # Per-group gradient accumulation (one step per unique Uni-Mol embedding set)
        for group_key, group_indices in groups.items():
            valid = [
                (i, norm_weights[i] if norm_weights is not None else 1.0)
                for i in group_indices
                if (norm_weights is None or norm_weights[i] >= 1e-8)
                and graph_cache[i] is not None
            ]
            if not valid:
                continue
            n_valid = len(valid)
            optimizer.zero_grad()
            group_loss = 0.0
            
            for run_idx, rw in valid:
                unimol_emb, edge_idx, targets_list = graph_cache[run_idx]
                targets = torch.tensor(targets_list, dtype=torch.float32, device=device)
                
                # Move embeddings and edge_index to device
                unimol_emb = unimol_emb.to(device)
                edge_idx = edge_idx.to(device)
                
                # Forward pass through UnimolPolicy
                mean, log_std = policy._forward_edges(unimol_emb, edge_idx)
                
                active_mask = targets.abs() > 1e-8
                if not active_mask.any():
                    continue
                
                # AWR loss: -exp(r/β) · log π(a | s) over active elements
                std = torch.exp(log_std)
                logp_per = torch.distributions.Normal(mean, std).log_prob(targets)  # [E, D]
                raw_r = runs[run_idx].get("_bc_reward", 0.0)
                awr_w = min(math.exp(raw_r / awr_temperature), 20.0)
                
                # Apply per-edge pairwise confidence weighting if available
                if pairwise_weights_cache is not None and pairwise_weights_cache[run_idx] is not None:
                    edge_weights = torch.tensor(
                        pairwise_weights_cache[run_idx],
                        dtype=torch.float32,
                        device=device
                    )  # [E]
                    
                    # Apply edge weights: zero out low-confidence edges
                    # Only include edges with non-zero confidence
                    edge_mask = edge_weights > 1e-8
                    
                    if edge_mask.any():
                        # Weight loss by pairwise confidence (normalized per edge)
                        # Shape: logp_per is [E, D], edge_weights is [E]
                        weighted_logp = logp_per * edge_weights.unsqueeze(-1)  # [E, D]
                        
                        # Apply both active_mask and edge_mask
                        combined_mask = active_mask & edge_mask.unsqueeze(-1)
                        if combined_mask.any():
                            run_loss = -awr_w * weighted_logp[combined_mask].mean() / n_valid
                        else:
                            continue
                    else:
                        # No edges with sufficient confidence, skip this run
                        continue
                else:
                    # No pairwise weighting, use original computation
                    run_loss = -awr_w * logp_per[active_mask].mean() / n_valid
                
                if torch.isnan(run_loss) or torch.isinf(run_loss):
                    continue
                run_loss.backward()
                group_loss += run_loss.item()
            
            if group_loss > 0.0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += group_loss
                num_updates += 1
                total_runs += n_valid
            else:
                optimizer.zero_grad()
        
        avg_loss = epoch_loss / num_updates if num_updates > 0 else 0.0
        return {'loss': avg_loss, 'num_runs': total_runs, 'num_updates': num_updates}

    # Legacy per-run updates (when graph_cache provided but groups not)
    for run_idx, run in enumerate(runs):
        rw = norm_weights[run_idx] if norm_weights is not None else 1.0
        if rw < 1e-8:
            continue

        if graph_cache is not None:
            cached = graph_cache[run_idx]
            if cached is None:
                continue
            unimol_emb, edge_idx, targets_list = cached
            targets = torch.tensor(targets_list, dtype=torch.float32, device=device)
        else:
            # Fallback: compute embeddings on-the-fly (slow)
            run_dir = Path(run["run_dir"])
            gi_path = run_dir / "graph_info.json"
            try:
                unimol_emb, edge_idx, nsubs_per_site, _ = _compute_unimol_embeddings_and_edges(
                    run_dir, gi_path,
                    env_cutoff=env_cutoff,
                    use_environment_difference=use_environment_difference,
                    consensus_dict=consensus_dict,
                )
                from mllf.cb.graph_utils import build_directed_pairs
                pairs = build_directed_pairs(nsubs_per_site)
                targets_list = _extract_targets_from_variables(run_dir, nsubs_per_site, pairs)
                if not targets_list:
                    continue
                targets = torch.tensor(targets_list, dtype=torch.float32, device=device)
            except Exception as e:
                print(f"  Error computing Uni-Mol embeddings for {run_dir.name}: {e}")
                continue

        # Move to device
        unimol_emb = unimol_emb.to(device)
        edge_idx = edge_idx.to(device)
        
        # Forward pass
        mean, log_std = policy._forward_edges(unimol_emb, edge_idx)
        
        active_mask = targets.abs() > 1e-8
        if not active_mask.any():
            continue
        
        # Compute AWR loss
        optimizer.zero_grad()
        std = torch.exp(log_std)
        logp_per = torch.distributions.Normal(mean, std).log_prob(targets)
        raw_r = run.get("_bc_reward", 0.0)
        awr_w = min(math.exp(raw_r / awr_temperature), 20.0)
        
        # Apply per-edge pairwise confidence weighting if available
        if pairwise_weights_cache is not None and pairwise_weights_cache[run_idx] is not None:
            edge_weights = torch.tensor(
                pairwise_weights_cache[run_idx],
                dtype=torch.float32,
                device=device
            )  # [E]
            
            # Apply edge weights: zero out low-confidence edges
            edge_mask = edge_weights > 1e-8
            
            if edge_mask.any():
                weighted_logp = logp_per * edge_weights.unsqueeze(-1)  # [E, D]
                combined_mask = active_mask & edge_mask.unsqueeze(-1)
                if combined_mask.any():
                    loss = -awr_w * weighted_logp[combined_mask].mean()
                else:
                    continue
            else:
                # No edges with sufficient confidence, skip this run
                continue
        else:
            # No pairwise weighting, use original computation
            loss = -awr_w * logp_per[active_mask].mean()
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_updates += 1
            total_runs += 1

    avg_loss = epoch_loss / num_updates if num_updates > 0 else 0.0
    return {'loss': avg_loss, 'num_runs': total_runs, 'num_updates': num_updates}


def _extract_highest_lambda_counts(populations) -> list:
    """Convert simulation_results.json populations dict to a flat list of raw counts.

    ``simulation_results.json`` stores populations as a dict::

        {"2": {"counts": {"0.95": N, "0.99": M}, "site": S}, ...}

    Each block's counts sub-dict holds MSLD population counts sampled at
    different lambda values (e.g. 0.95 and 0.99).  These are **not** averaged
    or normalised — only the single highest-lambda count is used (e.g. 0.99
    when both 0.95 and 0.99 are present), since it is the most physically
    meaningful value and the values at different lambdas are incommensurable.

    Returns a list indexed by node_idx where::

        result[node_idx] = highest-lambda population count for block (node_idx + 2)

    (MSLD block_offset=2 convention: block 1 is the reference.)

    If *populations* is already a list it is returned unchanged.
    """
    if not populations:
        return []
    if isinstance(populations, list):
        return populations
    block_offset = 2
    try:
        max_node = max(int(k) - block_offset for k in populations.keys())
    except (ValueError, TypeError):
        return []
    result = [0] * (max_node + 1)
    for block_id_str, block_info in populations.items():
        try:
            node_idx = int(block_id_str) - block_offset
        except ValueError:
            continue
        if node_idx < 0:
            continue
        counts = block_info.get('counts', {}) if isinstance(block_info, dict) else {}
        if counts:
            # Take only the highest-lambda value — do not average across lambdas.
            max_lambda = max(counts.keys(), key=lambda x: float(x))
            result[node_idx] = counts[max_lambda]
    return result


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
    patience: int = 10,
    reward_weighted: bool = False,
    freeze_encoder_after: Optional[int] = None,
    q_epochs: int = 0,
    q_lr: float = 1e-3,
    q_stratified_fraction: Optional[float] = None,
    awr_temperature: float = 0.5,
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
        patience: Early stopping patience (default: 10). Training stops if the MSE loss
                  does not improve for this many consecutive epochs.
        reward_weighted: If True, weight each run's MSE loss by its reward (clamped >= 0
                        and normalised so the mean weight = 1.0).  High-reward runs drive
                        more of the gradient signal; zero-reward runs are skipped entirely.
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

    # Snapshot the full run pool before any BC-specific selection.
    all_runs_unfiltered = list(runs)

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
        print(f"\nApplying stratified negative sampling (max_fraction={stratified_negative_fraction:.0%}, quadratic ramp)...")
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
    
    # For Uni-Mol pretraining, schema is fixed: 4 bias types (linear, quadratic, skew, end)
    # with 2 directions each = 8 relations. We don't need to build a sample graph.
    # (Unlike the old approach which required toppar files and node features.)
    
    relation_names = [
        'linear_ij', 'linear_ji',
        'quadratic_ij', 'quadratic_ji',
        'skew_ij', 'skew_ji',
        'end_ij', 'end_ji',
    ]
    base_relation_map = {
        'linear': 0, 'quadratic': 1, 'skew': 2, 'end': 3
    }
    
    sample_extras = {
        'relation_names': relation_names,
        'base_relation_map': base_relation_map,
    }
    
    print(f"\nSchema for Uni-Mol pretraining:")
    print(f"  Bias types: 4 (linear, quadratic, skew, end)")
    print(f"  Relations: {len(relation_names)} (2 directions per type)")
    print(f"  Relation names: {relation_names}")
    
    # Create policy using Uni-Mol representations only
    from mllf.cb.policy import UnimolPolicy
    
    train_config = config.get('training', {})
    policy_config = train_config.get('unimol', {})
    
    # Extract Uni-Mol settings from top-level unimol config
    unimol_cfg = config.get('unimol', {})
    env_cutoff = unimol_cfg.get('environment_cutoff', 8.0)
    use_environment_difference = unimol_cfg.get('use_environment_difference', True)
    
    policy = UnimolPolicy(
        unimol_dim=policy_config.get('unimol_dim', 512),
        mlp_hidden=policy_config.get('mlp_hidden', 64),
        mlp_out_dim=len(sample_extras['relation_names']) // 2,
        use_dual_embeddings=use_environment_difference,
    ).to(device)
    
    print(f"\nModel architecture:")
    if use_environment_difference:
        print(f"  Policy: UnimolPolicy (dual embeddings: ligand-only + full)")
        print(f"  Input: [diff_ligand(512D), mean_full(512D)] = 1024D edge features")
        print(f"  diff_ligand captures substituent variation (sub-dependent)")
        print(f"  mean_full captures environment effects (environment-dependent)")
    else:
        print(f"  Policy: UnimolPolicy (ligand+environment)")
        print(f"  Dimensionality: 1024D (2×512D) → 512D → 256D → 64D (EdgeValueMLP)")
    print(f"  Total parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Optimizer: cosine-annealed Adam over all policy parameters.
    optimizer = optim.Adam(
        policy.parameters(),
        lr=learning_rate
    )
    # Cosine annealing LR schedule: decays smoothly from lr → lr/100 over all epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 100.0
    )
    active_opt = optimizer
    active_sched = scheduler

    print(f"  Learning rate: {learning_rate} (cosine annealed to {learning_rate/100:.6f})")
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
    # Pre-build per-system Uni-Mol graphs (shared across all runs from that system).
    # All runs from the same system (prep directory) have identical substitutent
    # coordinates, so Uni-Mol embeddings and edge indices are identical.
    # Only the targets differ per run (different variables.py).
    # 
    # NOTE: When using 'random' structure selection, this cache is skipped
    # and representations are computed per-run instead, allowing different structure types
    # to be selected for each run. For 'minimized' and 'unrelaxed', per-system caching
    # is used since the structure type is deterministic per system.
    # ------------------------------------------------------------------
    
    # Check structure selection strategy
    unimol_cfg = config.get('unimol', {})
    structure_selection = unimol_cfg.get('structure_selection', 'minimized').lower()
    use_system_cache = (structure_selection in ['minimized', 'unrelaxed'])  # Use cache for deterministic structure types
    unrelaxed_mappings = config.get('_unrelaxed_system_mappings', {})
    skip_unrelaxed_for_systems = unimol_cfg.get('skip_unrelaxed_for_systems', [])
    env_cutoff = unimol_cfg.get('environment_cutoff', 8.0)  # Distance cutoff for environment filtering
    use_environment_difference = unimol_cfg.get('use_environment_difference', True)  # Use (env+lig)-(lig) representation
    tracking_file = initialize_representation_tracker(output_dir)
    
    # Initialize consensus_dict - will be populated if consensus building is enabled
    consensus_dict = None
    
    if structure_selection != 'minimized':
        print(f"\n{'='*60}")
        print(f"Structure selection: {structure_selection}")
        print(f"Computing representations per-run (system cache disabled)")
        print(f"Tracking: {tracking_file}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"Grouping {len(runs)} runs by system and building per-system graphs...")
        print(f"{'='*60}")
    
    # Group runs by prep directory (system)
    system_to_runs = {}  # prep_dir -> list of run indices
    system_graph_cache = {}  # prep_dir -> (unimol_emb, edge_idx, nsubs_per_site)
    system_structure_cache = {}  # prep_dir -> nsubs_per_site (always populated, for pairwise weights)
    
    for run_idx, run in enumerate(runs):
        run_dir = Path(run["run_dir"])
        prep_dir = run_dir.parent / "prep"
        prep_dir_str = str(prep_dir)
        
        if prep_dir_str not in system_to_runs:
            system_to_runs[prep_dir_str] = []
        system_to_runs[prep_dir_str].append(run_idx)
    
    print(f"Found {len(system_to_runs)} unique systems")
    
    # Build environment consensus atoms if configured
    # Structure: consensus_dict[prep_dir_str][site_name] = set of consensus atoms
    use_consensus = unimol_cfg.get('use_environment_consensus', True)
    consensus_dict = {}  # Will be populated per-system
    
    if use_consensus:
        print("\nBuilding environment consensus atoms per system/site...")
        from mllf.cb.environment_consensus import build_site_consensus
        
        # Build consensus for each unique system (prep_dir)
        for prep_dir_str in sorted(system_to_runs.keys()):
            prep_dir = Path(prep_dir_str)
            
            # Skip if prep directory doesn't exist
            if not prep_dir.is_dir():
                continue
            
            # Find all site_* PDB files in this system's prep directory
            # Handle both regular files and combo subdirectories
            site_to_subs = {}
            
            # Search for PDB files directly in prep_dir
            for sub_pdb in sorted(prep_dir.glob('site*_sub*_frag.pdb')):
                site_num = sub_pdb.stem.split('_')[0]  # e.g., "site1"
                if site_num not in site_to_subs:
                    site_to_subs[site_num] = []
                if sub_pdb not in site_to_subs[site_num]:
                    site_to_subs[site_num].append(sub_pdb)
            
            # Also search in combo subdirectories (for combo structures)
            for comb_dir in sorted(prep_dir.glob('comb_*/')):
                for sub_pdb in sorted(comb_dir.glob('site*_sub*_frag.pdb')):
                    site_num = sub_pdb.stem.split('_')[0]  # e.g., "site1"
                    if site_num not in site_to_subs:
                        site_to_subs[site_num] = []
                    if sub_pdb not in site_to_subs[site_num]:
                        site_to_subs[site_num].append(sub_pdb)
            
            # Build consensus for each site in this system
            if site_to_subs:
                consensus_dict[prep_dir_str] = {}
                
                # Find core.pdb for this system (try direct path first, then combo)
                core_pdb = None
                if (prep_dir / 'core.pdb').exists():
                    core_pdb = str(prep_dir / 'core.pdb')
                else:
                    # Try first combo directory
                    for comb_dir in sorted(prep_dir.glob('comb_*/')):
                        if (comb_dir / 'core.pdb').exists():
                            core_pdb = str(comb_dir / 'core.pdb')
                            break
                
                if core_pdb:
                    # Extract system name for logging
                    # Handle nested combos: system/comb_ID/prep or system/prep
                    system_path = Path(prep_dir_str)
                    if system_path.parent.name.startswith('comb_'):
                        # Nested combo: .../system/comb_ID/prep
                        system_label = system_path.parent.parent.name
                    else:
                        # Regular: .../system/prep
                        system_label = system_path.parent.name
                    
                    # SKIP consensus building for vacuum systems (no environment to define)
                    if 'vac' in system_label.lower() or 'vacuum' in system_label.lower():
                        print(f"\n[CONSENSUS] Skipping consensus for vacuum system: {system_label} (no environment)")
                        # Set all sites to None for this system
                        for site_name in sorted(site_to_subs.keys()):
                            consensus_dict[prep_dir_str][site_name] = None
                    else:
                        # Build consensus for solvent/protein systems
                        for site_name in sorted(site_to_subs.keys()):
                            sub_pdbs = [str(p) for p in sorted(site_to_subs[site_name])]
                            try:
                                consensus = build_site_consensus(
                                    site_name,
                                    sub_pdbs,
                                    core_pdb,
                                    prep_dir,
                                    env_cutoff=env_cutoff,
                                    system_name=system_label,
                                )
                                consensus_dict[prep_dir_str][site_name] = consensus
                            except Exception as e:
                                print(f"  Warning: Failed to build consensus for {prep_dir_str}/{site_name}: {e}")
                                consensus_dict[prep_dir_str][site_name] = None
        
        # Save consensus atoms for reference/debugging
        if consensus_dict:
            import json
            consensus_json = {}
            for prep_dir_str, sites in consensus_dict.items():
                if sites:
                    consensus_json[prep_dir_str] = {}
                    for site_key, atoms in sites.items():
                        if atoms is not None:
                            # Convert set of tuples to list of lists for JSON serialization
                            consensus_json[prep_dir_str][site_key] = sorted([list(atom) for atom in atoms])
                        else:
                            consensus_json[prep_dir_str][site_key] = None
            
            if consensus_json:
                consensus_file = output_dir / 'environment_consensus.json'
                with open(consensus_file, 'w') as f:
                    json.dump(consensus_json, f, indent=2)
                print(f"  Saved consensus to {consensus_file.name}\n")
        else:
            print()
    
    # Build graph cache per system (only if using minimized mode)
    n_systems_ok = 0
    n_systems_fail = 0
    
    if use_system_cache:
        print(f"\nBuilding per-system graphs ({len(system_to_runs)} systems)...")
        for system_idx, (prep_dir_str, run_indices) in enumerate(sorted(system_to_runs.items())):
            try:
                prep_dir = Path(prep_dir_str)
                
                # Use first run in system to get graph_info (all runs from same system have identical structure)
                first_run_idx = run_indices[0]
                first_run = runs[first_run_idx]
                run_dir = Path(first_run["run_dir"])
                gi_path = run_dir / "graph_info.json"
                
                if not gi_path.exists():
                    n_systems_fail += 1
                    continue
                
                # Compute Uni-Mol embeddings using configured structure selection
                unimol_emb, edge_idx, nsubs_per_site, _ = _compute_unimol_embeddings_and_edges(
                    run_dir, gi_path,
                    structure_type=structure_selection,
                    unrelaxed_system_mappings=unrelaxed_mappings,
                    env_cutoff=env_cutoff,
                    use_environment_difference=use_environment_difference,
                    run_idx=0,
                    consensus_dict=consensus_dict,
                )
                
                # Cache the system graph (shared across all runs)
                system_graph_cache[prep_dir_str] = (unimol_emb, edge_idx, nsubs_per_site)
                system_structure_cache[prep_dir_str] = nsubs_per_site
                n_systems_ok += 1
                
                # Handle nested combos: look TWO levels up if in comb_* directory
                display_name = prep_dir.parent.name
                if display_name.startswith('comb_'):
                    display_name = prep_dir.parent.parent.name + '/' + display_name
                print(f"  [{system_idx + 1}/{len(system_to_runs)}] {display_name}: "
                      f"{len(run_indices)} runs, {unimol_emb.size(0)} subs, {edge_idx.size(1)} edges")
            
            except Exception as e:
                print(f"  System {prep_dir_str}: {e}")
                n_systems_fail += 1
        
        print(f"\nPer-system graph cache ready: {n_systems_ok} systems built, {n_systems_fail} failed")
    else:
        print(f"Skipping system cache (using per-run structure selection)")
        n_systems_ok = 0
        for prep_dir_str in system_to_runs.keys():
            n_systems_ok += 1
        # Still build structure cache for pairwise weights calculation
        print(f"Building per-system structure cache (for pairwise weights)...")
        for prep_dir_str, run_indices in sorted(system_to_runs.items()):
            try:
                first_run_idx = run_indices[0]
                first_run = runs[first_run_idx]
                run_dir = Path(first_run["run_dir"])
                gi_path = run_dir / "graph_info.json"
                
                if gi_path.exists():
                    with open(gi_path) as f:
                        gi = json.load(f)
                    # Extract nsubs_per_site from graph_info
                    nsubs_per_site = gi.get('nsubs_per_site', [])
                    system_structure_cache[prep_dir_str] = nsubs_per_site
            except Exception:
                pass  # Graceful degradation
    
    # Now build per-run targets cache
    graph_cache = [None] * len(runs)
    n_ok = 0
    n_fail = 0
    
    print(f"\n{'='*60}")
    if use_system_cache:
        print(f"Pre-building per-run targets (using system cache)...")
    else:
        print(f"Pre-building per-run embeddings and targets...")
        print(f"(structure selection: {structure_selection})")
    print(f"{'='*60}")
    
    for run_idx, run in enumerate(runs):
        try:
            run_dir = Path(run["run_dir"])
            prep_dir = run_dir.parent / "prep"
            prep_dir_str = str(prep_dir)
            gi_path = run_dir / "graph_info.json"
            
            if not gi_path.exists():
                n_fail += 1
                continue
            
            # Determine structure type for this run
            if use_system_cache:
                # Minimized mode: use cached system embeddings
                if prep_dir_str not in system_graph_cache:
                    n_fail += 1
                    continue
                
                unimol_emb, edge_idx, nsubs_per_site = system_graph_cache[prep_dir_str]
                struct_type_used = 'minimized'
                source_info = 'system_cache'
                
            else:
                # Random or unrelaxed mode: compute per-run with selected structure
                # Handle nested combos: look TWO levels up if in comb_* directory
                system_name = prep_dir.parent.name
                if system_name.startswith('comb_'):
                    system_name = prep_dir.parent.parent.name
                struct_type_used = _select_structure_type_for_run(
                    run_idx, 
                    structure_selection,
                    system_name=system_name,
                    skip_patterns=skip_unrelaxed_for_systems
                )
                
                unimol_emb, edge_idx, nsubs_per_site, rep_source = _compute_unimol_embeddings_and_edges(
                    run_dir, gi_path,
                    structure_type=struct_type_used,
                    unrelaxed_system_mappings=unrelaxed_mappings,
                    env_cutoff=env_cutoff,
                    use_environment_difference=use_environment_difference,
                    run_idx=run_idx,
                    consensus_dict=consensus_dict,
                )
                
                # Track representation choice
                track_representation_choice(
                    tracking_file,
                    system_name=system_name,
                    run_idx=run_idx,
                    structure_type=struct_type_used,
                    source=rep_source['source'],
                    prep_dir_used=Path(rep_source['prep_dir_used']),
                    explicit_mapping=rep_source['explicit_mapping'],
                )
                source_info = rep_source['source']
            
            # Extract run-specific targets from variables.py
            from mllf.cb.graph_utils import build_directed_pairs
            pairs = build_directed_pairs(nsubs_per_site)
            targets = _extract_targets_from_variables(run_dir, nsubs_per_site, pairs)
            
            if targets and edge_idx.size(1) > 0:
                # Cache: (unimol_embeddings [N, 512], edge_index [2, E], targets [E, 4])
                graph_cache[run_idx] = (unimol_emb, edge_idx, targets)
                n_ok += 1
            else:
                n_fail += 1
            
            if (run_idx + 1) % 100 == 0 or (run_idx + 1) == len(runs):
                pct = 100 * n_ok / (run_idx + 1) if (run_idx + 1) > 0 else 0
                print(f"  {run_idx + 1}/{len(runs)} runs ({n_ok} cached, {n_fail} failed, {pct:.0f}%)...")
        
        except Exception as e:
            print(f"  Run {run_idx}: {e}")
            n_fail += 1
    
    print(f"\nRun cache ready: {n_ok} runs cached, {n_fail} failed")
    if not use_system_cache:
        print(f"Representation tracking: {tracking_file}")
    
    # Build per-run pairwise confidence weights cache (based on population balance and DDG reliability)
    print(f"\n{'='*60}")
    print(f"Building per-run pairwise confidence weights...")
    print(f"{'='*60}")
    
    pairwise_weights_cache = [None] * len(runs)
    n_weights_ok = 0
    
    for run_idx, run in enumerate(runs):
        try:
            run_dir = Path(run["run_dir"])
            prep_dir = run_dir.parent / "prep"
            prep_dir_str = str(prep_dir)
            
            # Get system structure info (always available)
            if prep_dir_str not in system_structure_cache:
                continue
            
            nsubs_per_site = system_structure_cache[prep_dir_str]
            
            # Build pairs for this system
            from mllf.cb.graph_utils import build_directed_pairs
            pairs = build_directed_pairs(nsubs_per_site)
            
            # Compute pairwise confidence weights from simulation results
            if "sim_results" in run:
                weights = compute_pairwise_confidence_weights(
                    run["sim_results"], nsubs_per_site, pairs
                )
                if weights is not None:
                    pairwise_weights_cache[run_idx] = weights
                    n_weights_ok += 1
        
        except Exception as e:
            # If pairwise weighting fails, just skip for this run (graceful degradation)
            pass
        
        if (run_idx + 1) % 100 == 0 or (run_idx + 1) == len(runs):
            print(f"  {run_idx + 1}/{len(runs)} runs ({n_weights_ok} with pairwise weights)...")
    
    print(f"\nPairwise confidence weights ready: {n_weights_ok} runs with weights")
    if n_weights_ok > 0:
        print(f"  Pairwise AWR weighting: ENABLED")
    
    # For compatibility with pretrain_epoch, use a simple grouping (one run per group)
    # This allows us to reuse the per-graph gradient accumulation logic if needed
    epoch_groups = {f"run_{i}": [i] for i in range(len(runs)) if graph_cache[i] is not None}

    # Always pre-compute per-run rewards: AWR uses exp(r/β) weighting directly.
    # Stored as run["_bc_reward"] = max(0, reward) so pretrain_epoch can weight by exp(r/β).
    _VALID_RC = {
        "w_P", "w_T", "w_U", "gamma", "P_baseline", "T_baseline",
        "min_transitions_per_site", "min_coverage_ratio",
        "entropy_bonus", "concentration_penalty_threshold",
    }
    filtered_rc = {k: v for k, v in reward_config.items() if k in _VALID_RC}
    print("\nPre-computing per-run rewards for AWR loss weighting...")
    n_pos_w = 0
    total_pos_w = 0.0
    for run in runs:
        try:
            run_dir = Path(run["run_dir"])
            with open(run_dir / "graph_info.json") as _f:
                gi = json.load(_f)
            sites = gi.get("sites", {})
            site_cnts: Dict[int, int] = {}
            for bd in sites.values():
                s = bd["site"]
                site_cnts[s] = site_cnts.get(s, 0) + 1
            nsubs_per_site = [site_cnts[s] for s in sorted(site_cnts)]
            r = compute_reward_from_sim_results(
                run["sim_results"], len(site_cnts), nsubs_per_site, **filtered_rc
            )
            run["_bc_reward"] = max(0.0, r)
            if r > 0:
                n_pos_w += 1
                total_pos_w += r
        except Exception:
            run["_bc_reward"] = 0.0
    print(f"  Positive-reward runs: {n_pos_w}/{len(runs)}  "
          f"(total positive reward mass: {total_pos_w:.2f})")
    if reward_weighted and n_pos_w == 0:
        print("  Warning: all rewards zero or negative — disabling reward weighting.")
        reward_weighted = False

    print(f"\n{'='*60}")
    print(f"Starting behavior cloning for {epochs} epochs")
    print(f"Training on {len(runs)} runs ({n_ok} with valid graphs)")
    if reward_weighted:
        print("Reward-weighted loss: ENABLED")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        stats = pretrain_epoch(
            policy, active_opt, runs, reward_config, device,
            toppar_dir=toppar_dir,
            toppar_files=toppar_files,
            warn_missing_types=warn_missing_types,
            graph_cache=graph_cache,
            groups=epoch_groups,
            reward_weighted=reward_weighted,
            awr_temperature=awr_temperature,
            pairwise_weights_cache=pairwise_weights_cache,
            unimol_config={'environment_cutoff': env_cutoff, 'use_environment_difference': use_environment_difference},
            consensus_dict=consensus_dict,  # Pass consensus for environment filtering
        )

        print(f"  AWR Loss: {stats['loss']:.4f}")
        print(f"  Runs processed: {stats['num_runs']} ({stats.get('num_updates', stats['num_runs'])} grad steps)")
        print(f"  LR: {active_sched.get_last_lr()[0]:.6f}")

        # Step LR scheduler after each epoch
        active_sched.step()

        # Save best model (lowest loss)
        if stats['loss'] < best_loss:
            best_loss = stats['loss']
            epochs_without_improvement = 0

            best_path = output_dir / "best_policy.pt"
            _ckpt = {'policy_state': policy.state_dict(), 'epoch': epoch + 1, 'loss': stats['loss']}
            torch.save(_ckpt, best_path)
            print(f"  Saved best model (loss: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
        _epoch_ckpt = {
            'policy_state': policy.state_dict(),
            'optimizer_state': active_opt.state_dict(),
            'scheduler_state': active_sched.state_dict(),
            'epoch': epoch + 1,
            'stats': stats,
        }
        torch.save(_epoch_ckpt, checkpoint_path)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs (best loss: {best_loss:.4f})")
            break
    
    # Save final model
    final_path = output_dir / "final_policy.pt"
    _final_ckpt = {'policy_state': policy.state_dict(), 'epoch': epochs}
    torch.save(_final_ckpt, final_path)
    
    # Save metadata
    metadata = {
        'policy_type': 'unimol',
        'unimol_dim': policy_config.get('unimol_dim', 512),
        'mlp_hidden': policy_config.get('mlp_hidden', 64),
        'num_relations': len(sample_extras['relation_names']),
        'num_pretraining_runs': len(runs),
        'num_runs_with_graphs': n_ok,
        'epochs': epochs,
        'best_loss': best_loss,
        'training_method': 'behavior_cloning',
        'device': str(device),
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
        default="examples/workflow_pretrain.yaml",
        help="Config file (same format as workflow_pretrain.yaml)",
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
        help="If set, keep all positive-reward runs and sample negative-reward buckets with a "
             "quadratic ramp: worst bucket (-inf,-50] keeps 0%%, best bucket (-10,0) keeps this fraction. "
             "E.g. 0.55 ramps from 0%% to 55%% (quadratic schedule). "
             "When specified, --min-reward-threshold is ignored. Default: disabled.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10). Training stops if the MSE loss does "
             "not improve for this many consecutive epochs.",
    )
    parser.add_argument(
        "--reward-weighted",
        action="store_true",
        default=False,
        help="Weight each run's MSE loss by its reward (clamped >= 0, normalised so the mean "
             "weight = 1.0). High-reward runs contribute proportionally more gradient signal; "
             "zero/negative-reward runs are skipped. Useful when the training set contains "
             "many low-quality runs that compress the predicted coefficient range.",
    )
    parser.add_argument(
        "--freeze-encoder-after",
        type=int,
        default=None,
        metavar="N",
        help="(Unused — kept for config file compatibility.)",
    )
    parser.add_argument(
        "--awr-temperature",
        type=float,
        default=1.0,
        help="Temperature β for AWR loss: each run is weighted by exp(r/β). Higher β → "
             "more uniform weighting; lower β → sharper emphasis on high-reward runs. "
             "Default: 0.5.",
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
        min_transitions=args.min_transitions,
        stratified_negative_fraction=args.stratified_negative_fraction,
        patience=args.patience,
        reward_weighted=args.reward_weighted,
        freeze_encoder_after=args.freeze_encoder_after,
        awr_temperature=args.awr_temperature,
    )


if __name__ == "__main__":
    main()
