"""Utilities for pairwise MLP approach: feature extraction and integration.

This module provides helper functions to:
1. Extract substituent features from RTF data
2. Build directed pair lists for a combination
3. Convert MLP predictions to variables.py format
4. Integrate with existing CB workflow infrastructure
"""
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import numpy as np

from mllf.cb.atom_vocab import get_atom_type_vocab
from mllf.cb.graph_utils import _node_feature_from_meta
from mllf.file_handling.read_rtf import parse_rtf_dir


def load_substituent_features_from_graph_info(
    graph_info_path: str,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Dict]:
    """Load substituent features from graph_info.json file.
    
    This is used for pretraining where graph_info.json contains all the 
    substituent information (atom types, charges, etc.) without needing
    the original RTF files.
    
    Args:
        graph_info_path: Path to graph_info.json file
    
    Returns:
        Tuple of (features, pairs, metadata):
        - features: [N_subs, feature_dim] tensor of substituent features
        - pairs: List of (sub_i, sub_j) tuples for directed pairs within sites
        - metadata: Dict with 'nsubs_per_site', 'feature_dim', etc.
    """
    import json
    
    graph_info_path = Path(graph_info_path)
    with open(graph_info_path, 'r') as f:
        graph_info = json.load(f)
    
    # Load vocabularies (CGenFF only)
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf']
    )
    
    # Extract substituent info from graph_info
    sites_dict = graph_info.get('sites', {})
    solvent_state = graph_info.get('solvent_state', 'solvent')
    
    # Parse substituent keys and sort by (site, sub)
    subs = []
    for key, sub_info in sites_dict.items():
        site = sub_info.get('site')
        sub = sub_info.get('sub')
        if site is not None and sub is not None:
            subs.append((site, sub, key, sub_info))
    
    subs.sort(key=lambda x: (x[0], x[1]))  # Sort by (site, sub)
    
    if not subs:
        raise ValueError("No substituents found in graph_info.json")
    
    # Group by site to determine nsubs_per_site
    from collections import defaultdict
    site_subs = defaultdict(list)
    for site, sub, key, sub_info in subs:
        site_subs[site].append((sub, key, sub_info))
    
    # Build nsubs_per_site in site order
    nsubs_per_site = []
    for site in sorted(site_subs.keys()):
        nsubs_per_site.append(len(site_subs[site]))
    
    # Extract features for each substituent
    # _node_feature_from_meta expects 'solvent' as a string key
    features_list = []
    for site, sub, key, sub_info in subs:
        atom_types = sub_info.get('atom_types', [])
        total_charge = sub_info.get('total_charge', 0.0)
        
        # Build feature using same logic as extract_substituent_features
        # Note: _node_feature_from_meta expects 'distinct_atom_types' and 'solvent' (string) keys
        meta = {
            'total_charge': total_charge,
            'solvent': solvent_state,  # Pass as string: 'solvent', 'protein', or 'gas'
            'distinct_atom_types': atom_types,
        }
        
        feat = _node_feature_from_meta(
            meta, atom_type_vocab, element_vocab, atom_to_element
        )
        features_list.append(feat)
    
    features = torch.stack(features_list, dim=0)  # [N_subs, feature_dim]
    
    # Build directed pairs (both directions for all i≠j within same site)
    pairs = build_directed_pairs(nsubs_per_site)
    
    metadata = {
        'nsubs_per_site': nsubs_per_site,
        'feature_dim': features.shape[1],
        'vocab_info': {
            'num_atom_types': len(atom_type_vocab),
            'num_elements': len(element_vocab),
        },
        'solvent_state': solvent_state,
    }
    
    return features, pairs, metadata


def extract_substituent_features(
    rtf_results: Dict[str, Dict],
    solvent_override: Optional[str] = None
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Dict]:
    """Extract substituent features and pair information from RTF results.
    
    Args:
        rtf_results: Dictionary mapping keys to parsed RTF data (from parse_rtf_dir)
        solvent_override: Optional environment override ('gas', 'solv', or 'protein')
    
    Returns:
        Tuple of (features, pairs, metadata):
        - features: [N_subs, feature_dim] tensor of substituent features
        - pairs: List of (site, sub) tuples in order
        - metadata: Dict with 'nsubs_per_site', 'feature_dim', 'vocab_info'
    """
    # Load vocabularies (CGenFF only)
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf']
    )
    
    # Sort substituents by site and sub for deterministic ordering
    subs = []
    for key, parsed in rtf_results.items():
        site = parsed.get('site')
        sub = parsed.get('sub')
        if site is not None and sub is not None:
            subs.append((site, sub, key, parsed))
    
    subs.sort(key=lambda x: (x[0], x[1]))  # Sort by (site, sub)
    
    if not subs:
        raise ValueError("No substituents found in rtf_results")
    
    # Helper to detect and normalize solvent state
    def detect_solvent_state(filename: str) -> str:
        """Detect solvent from filename or return default."""
        fn_lower = filename.lower()
        if any(x in fn_lower for x in ('gas', 'vacuum', 'vac')):
            return 'gas'
        elif any(x in fn_lower for x in ('solv', 'water', 'aq')):
            return 'solv'
        elif 'prot' in fn_lower:
            return 'protein'
        return 'gas'  # default
    
    # Extract features for each substituent
    features_list = []
    pair_list = []
    
    for site, sub, key, parsed in subs:
        # Determine solvent state
        if solvent_override:
            solvent = solvent_override.lower()
        else:
            filename = parsed.get('rtf', '')
            solvent = detect_solvent_state(filename)
        
        # Build metadata for feature extraction
        # RTF parser returns 'atom_types' (list with duplicates)
        # Pass full list to _node_feature_from_meta for count-based encoding
        atom_types_list = parsed.get('atom_types', [])
        
        meta = {
            'total_charge': parsed.get('total_charge', 0.0),
            'solvent': solvent,
            'distinct_atom_types': atom_types_list  # Full list with duplicates for counts
        }
        
        # Extract features using existing infrastructure
        features = _node_feature_from_meta(meta, atom_type_vocab, element_vocab, atom_to_element)
        features_list.append(features)
        pair_list.append((site, sub))
    
    # Stack features into tensor
    features_tensor = torch.stack(features_list)  # [N_subs, feature_dim]
    
    # Compute nsubs_per_site
    site_counts = {}
    for site, sub in pair_list:
        site_counts[site] = site_counts.get(site, 0) + 1
    nsubs_per_site = [site_counts[site] for site in sorted(site_counts.keys())]
    
    # Build directed pairs (0-based indices)
    pairs = build_directed_pairs(nsubs_per_site)
    
    # Build metadata
    metadata = {
        'nsubs_per_site': nsubs_per_site,
        'feature_dim': features_tensor.shape[1],
        'vocab_info': {
            'atom_type_vocab_size': len(atom_type_vocab),
            'element_vocab_size': len(element_vocab)
        },
        'substituents': pair_list  # List of (site, sub) in order
    }
    
    return features_tensor, pairs, metadata


def build_directed_pairs(nsubs_per_site: List[int]) -> List[Tuple[int, int]]:
    """Build list of directed pairs for pairwise bias predictions.
    
    Generates BOTH directions (i→j and j→i) within each site to allow
    independent predictions for skew and end biases.
    No cross-site pairs are generated.
    
    Args:
        nsubs_per_site: List of substituent counts per site
    
    Returns:
        List of (i, j) pairs including both directions within the same site
    """
    pairs = []
    offset = 0
    
    for nsubs in nsubs_per_site:
        # Generate both directions for all pairs within site
        for i in range(nsubs):
            for j in range(nsubs):
                if i != j:  # Skip diagonal (self-pairs)
                    pairs.append((offset + i, offset + j))
        offset += nsubs
    
    return pairs


def predictions_to_bias_dict(
    actions: torch.Tensor,
    pairs: List[Tuple[int, int]],
    nsubs_per_site: List[int]
) -> Dict[str, List[List[float]]]:
    """Convert MLP predictions to bias dictionary format for variables.py.
    
    Predictions include both directions (i→j and j→i) for all pairs.
    - Linear (b) and Quadratic (c): Enforce antisymmetry by averaging
    - Skew (x) and End (s): Use independent predictions for each direction
    
    Args:
        actions: [N_pairs, 4] tensor of bias coefficients [linear, quadratic, skew, end]
        pairs: List of (i, j) directed pair indices matching actions (includes both directions)
        nsubs_per_site: List of substituent counts per site
    
    Returns:
        Dictionary with keys 'b', 'c', 'x', 's' containing bias matrices
    """
    total_subs = sum(nsubs_per_site)
    
    # Initialize matrices
    b_matrix = [[0.0 for _ in range(total_subs)] for _ in range(total_subs)]
    c_matrix = [[0.0 for _ in range(total_subs)] for _ in range(total_subs)]
    x_matrix = [[0.0 for _ in range(total_subs)] for _ in range(total_subs)]
    s_matrix = [[0.0 for _ in range(total_subs)] for _ in range(total_subs)]
    
    # Fill in predictions
    actions_np = actions.detach().cpu().numpy()
    
    # First pass: collect all predictions
    linear_vals = {}
    quad_vals = {}
    
    for (i, j), action in zip(pairs, actions_np):
        linear, quadratic, skew, end = action
        
        # For skew and end: use predictions directly (independent directions)
        x_matrix[i][j] = float(skew)
        s_matrix[i][j] = float(end)
        
        # For linear and quadratic: collect values to enforce antisymmetry
        linear_vals[(i, j)] = float(linear)
        quad_vals[(i, j)] = float(quadratic)
    
    # Second pass: enforce antisymmetry for linear and quadratic
    # Use the average: M[i][j] = (pred_ij - pred_ji) / 2
    processed = set()
    for (i, j) in pairs:
        if (i, j) in processed or (j, i) in processed:
            continue
        
        fwd_linear = linear_vals.get((i, j), 0.0)
        rev_linear = linear_vals.get((j, i), 0.0)
        antisym_linear = (fwd_linear - rev_linear) / 2.0
        b_matrix[i][j] = antisym_linear
        b_matrix[j][i] = -antisym_linear
        
        fwd_quad = quad_vals.get((i, j), 0.0)
        rev_quad = quad_vals.get((j, i), 0.0)
        antisym_quad = (fwd_quad - rev_quad) / 2.0
        # IMPORTANT: Store only upper triangle for quadratic to prevent cancellation
        # If i < j, store in c[i][j]; otherwise store in c[j][i]
        # Lower triangle remains zero
        if i < j:
            c_matrix[i][j] = antisym_quad
            # c_matrix[j][i] remains 0.0
        else:
            c_matrix[j][i] = antisym_quad
            # c_matrix[i][j] remains 0.0
        
        processed.add((i, j))
        processed.add((j, i))
    
    return {
        'b': b_matrix,
        'c': c_matrix,
        'x': x_matrix,
        's': s_matrix
    }


def save_pairwise_graph_info(combo_dir: str, metadata: Dict) -> None:
    """Save graph_info.json from pairwise metadata.
    
    This ensures the reward function can correctly handle the combination structure.
    
    Args:
        combo_dir: Path to combination directory
        metadata: Metadata dict from extract_substituent_features
    """
    import json
    
    combo_path = Path(combo_dir)
    graph_info_path = combo_path / 'graph_info.json'
    
    # Build graph_info structure
    graph_info = {
        'nsubs_per_site': metadata['nsubs_per_site'],
        'total_substituents': sum(metadata['nsubs_per_site']),
        'feature_dim': metadata['feature_dim']
    }
    
    # Save to file
    with open(graph_info_path, 'w') as f:
        json.dump(graph_info, f, indent=2)


def load_substituent_features_from_combo(
    combo_dir: str,
    solvent_override: Optional[str] = None
) -> Tuple[torch.Tensor, List[Tuple[int, int]], Dict]:
    """Load substituent features from a combination directory.
    
    Args:
        combo_dir: Path to combination directory containing RTF files
        solvent_override: Optional environment override
    
    Returns:
        Tuple of (features, pairs, metadata) as in extract_substituent_features
    """
    combo_path = Path(combo_dir)
    
    # Look for RTF files in combo_dir or combo_dir/prep
    prep_path = combo_path / "prep"
    if prep_path.exists():
        combo_path = prep_path
    
    # Parse RTF files in the combination directory
    rtf_results = parse_rtf_dir(str(combo_path))
    
    if not rtf_results:
        raise ValueError(f"No RTF files found in {combo_dir} or {combo_dir}/prep")
    
    # Extract features
    return extract_substituent_features(rtf_results, solvent_override)


def write_variables_from_pairwise_predictions(
    combo_dir: str,
    actions: torch.Tensor,
    pairs: List[Tuple[int, int]],
    nsubs_per_site: List[int],
    output_filename: str = "variables.py",
    bias_clip: float = 1000.0
) -> None:
    """Write variables.py file from pairwise MLP predictions.
    
    Args:
        combo_dir: Path to combination directory
        actions: [N_pairs, 4] tensor of predictions
        pairs: List of (i, j) directed pair indices
        nsubs_per_site: List of substituent counts per site
        output_filename: Name of output file (default: "variables.py")
        bias_clip: Maximum absolute value for bias coefficients
    """
    import yaml
    
    combo_path = Path(combo_dir)
    output_path = combo_path / output_filename
    
    # Convert predictions to bias dict
    bias_dict = predictions_to_bias_dict(actions, pairs, nsubs_per_site)
    
    # Apply clipping
    for key in ['b', 'c', 'x', 's']:
        matrix = bias_dict[key]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = max(-bias_clip, min(bias_clip, matrix[i][j]))
    
    # Convert b matrix to per-node vector (average of incident edges)
    total_subs = sum(nsubs_per_site)
    b_matrix = bias_dict['b']
    b_vec = [0.0] * total_subs
    for i in range(total_subs):
        incident_vals = []
        for j in range(total_subs):
            if i != j:
                incident_vals.append(b_matrix[i][j])
        b_vec[i] = sum(incident_vals) / len(incident_vals) if incident_vals else 0.0
    
    # Enforce constraint: first substituent of each site must have b=0.0
    idx = 0
    for count in nsubs_per_site:
        if count > 0:
            b_vec[idx] = 0.0
        idx += count
    
    c_mat = bias_dict['c']
    x_mat = bias_dict['x']
    s_mat = bias_dict['s']
    
    # Format bias_string manually to match expected YAML structure
    # b: single row with first element using '- -' then rest using just '-'
    # c, x, s: NxN matrices with each row starting '- -' for first element, then '-' for rest
    lines = []
    
    # Format b vector: [- - val1, - val2, - val3, ...]
    lines.append('b:')
    if b_vec:
        lines.append(f'- - {b_vec[0]}')
        for val in b_vec[1:]:
            lines.append(f'  - {val}')
    else:
        lines.append('- - 0.0')
    
    # Format c matrix
    lines.append('c:')
    for row in c_mat:
        lines.append(f'- - {row[0]}')
        for val in row[1:]:
            lines.append(f'  - {val}')
    
    # Format x matrix
    lines.append('x:')
    for row in x_mat:
        lines.append(f'- - {row[0]}')
        for val in row[1:]:
            lines.append(f'  - {val}')
    
    # Format s matrix
    lines.append('s:')
    for row in s_mat:
        lines.append(f'- - {row[0]}')
        for val in row[1:]:
            lines.append(f'  - {val}')
    
    yaml_block = '\n'.join(lines)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("# Auto-generated variables.py — bias_string contains YAML for bias matrices\n")
        f.write("bias_string = '''\n")
        f.write(yaml_block)
        f.write("\n\n'''\n")
