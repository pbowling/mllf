"""High-level workflow utilities for preparing combos, training and running sims.

This module centralizes common steps used for contextual bandit training workflows:
  1. Generate combination directories from site/sub fragment files
  2. Split manifests into train/validation sets
  3. Build PyG graphs from RTF fragments or variables.py files
  4. Run quick training epochs for testing
  5. Execute simulations concurrently with Slurm support
  6. Archive completed runs

"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import random
import shutil
import tarfile
import os

import torch

from mllf.file_handling.generate_combinations import create_combination_dirs
from mllf.cb.graph import Graph, EdgeCoeffs
from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.cb import graph_utils
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy
from mllf.cli.sim import run_simulation_batch, parse_simulation_results


def load_manifest(manifest_path: str) -> List[str]:
    """Load list of combo directories from manifest file.
    
    Args:
        manifest_path: Path to manifest file with one combo directory per line.
    
    Returns:
        List of combo directory paths as strings.
    """
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def load_bias_from_variables(py_path: str) -> Dict[str, Any]:
    """Load the YAML bias mapping embedded inside a `variables.py` file.

    Args:
        py_path: Path to a `variables.py` file containing a triple-quoted
                 `bias_string` variable with YAML content.

    Returns:
        Dict parsed from the `bias_string` YAML block with keys 'b', 'c', 'x', 's',
        or {} if the bias_string is not found or cannot be parsed.
    """
    text = Path(py_path).read_text(encoding='utf-8')
    m = __import__('re').search(r"bias_string\s*=\s*(?:\"\"\"|'''')([\s\S]*?)(?:\"\"\"|'''')", text, __import__('re').S)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


def graph_from_bias(bias: Dict[str, Any]) -> Graph:
    """Build an mllf Graph from a bias dict.

    The returned Graph will have explicit EdgeCoeffs only for pairs where the
    input bias matrices contain non-zero values. This allows the graph structure
    to be determined by existing bias values while leaving coefficient predictions
    to the policy/MLP.

    Args:
        bias: Dict with keys:
              - 'b': per-node linear biases (flat list or NxN matrix)
              - 'c': NxN quadratic bias matrix
              - 'x': NxN skew bias matrix
              - 's': NxN end bias matrix

    Returns:
        Graph instance sized by the length of 'b' (or 1 if missing).
        Edges are only created for node pairs with non-zero c/x/s values.
    """
    b = bias.get('b', [])
    if isinstance(b, list) and b and isinstance(b[0], list):
        flat_b = [float(x) for row in b for x in row]
    elif isinstance(b, list):
        flat_b = [float(x) for x in b]
    else:
        flat_b = []

    N = len(flat_b) if flat_b else 1
    c = bias.get('c', [])
    x = bias.get('x', [])
    s = bias.get('s', [])

    g = Graph(int(N))
    for i in range(int(N)):
        for j in range(i + 1, int(N)):
            try:
                cval = float(c[i][j]) if c and len(c) > i and len(c[i]) > j else 0.0
            except Exception:
                cval = 0.0
            try:
                xval = float(x[i][j]) if x and len(x) > i and len(x[i]) > j else 0.0
            except Exception:
                xval = 0.0
            try:
                sval = float(s[i][j]) if s and len(s) > i and len(s[i]) > j else 0.0
            except Exception:
                sval = 0.0
            if any((cval, xval, sval)):
                g.set_edge(i, j, EdgeCoeffs(linear=0.0, quadratic=cval, skew=xval, end=sval))
    return g


def save_graph_info_from_rtf(combo_dir: str, g) -> None:
    """Save graph_info.json from a Graph object built from RTF files.
    
    This is critical for the reward function to correctly handle asymmetric
    molecules by extracting actual nsubs_per_site instead of estimating.
    
    Args:
        combo_dir: Path to combination directory
        g: Graph object with populated node metadata
    """
    import json
    from collections import defaultdict
    
    combo_path = Path(combo_dir)
    graph_info_path = combo_path / 'graph_info.json'
    
    # Extract site information from node metadata
    sites_info = {}
    solvent_state = None
    site_counts = defaultdict(int)
    
    for node_idx, node_info in g.nodes.items():
        if node_info:
            site = node_info.get('site')
            sub = node_info.get('sub')
            if site is not None and sub is not None:
                key = f"site{site}_sub{sub}"
                sites_info[key] = {
                    'site': site,
                    'sub': sub,
                    'total_charge': node_info.get('total_charge', 0.0),
                    'atom_types': node_info.get('atom_types', []),
                    'distinct_atom_types': node_info.get('distinct_atom_types', []),
                }
                # Count substituents per site
                site_counts[site] += 1
                # Extract solvent state (should be consistent across nodes)
                if solvent_state is None:
                    solvent_state = node_info.get('solvent', 'unknown')
    
    # Build nsubs_per_site list in site order
    nsubs_per_site = [site_counts[site] for site in sorted(site_counts.keys())]
    
    graph_info = {
        'solvent_state': solvent_state or 'unknown',
        'sites': sites_info,
        'nsubs_per_site': nsubs_per_site
    }
    
    with open(graph_info_path, 'w') as f:
        json.dump(graph_info, f, indent=2)


def build_data_and_targets_from_combo(combo_dir: str, base_bias: str = 'quadratic', verify_graph: bool = False,
                                      toppar_dir: str = None, toppar_files: list = None, warn_missing_types: bool = True,
                                      solvent_state: str = None):
    """Build PyG Data object and per-edge targets from a combo directory.

    This function prefers RTF fragments when available (Graph.from_rtf_results).
    If no RTF fragments are found, it falls back to reading variables.py and
    constructing a Graph from the embedded YAML bias_string.

    Args:
        combo_dir: Path to a combo directory containing either RTF fragments
                   (site*_sub*_*_pres.rtf files) or a variables.py file.
        base_bias: Legacy parameter (currently unused, kept for compatibility).
        verify_graph: If True, verify that PyG edge expansion matches Graph.edge_mask.
                      Useful for debugging but adds runtime overhead.
        toppar_dir: Path to toppar directory for vocabulary (None uses package default)
        toppar_files: List of specific toppar filenames to include (e.g., ['top_all36_cgenff.rtf'])
        warn_missing_types: If True, warn when sub RTF files contain atom types not in vocabulary
        solvent_state: Environment type for the system ('solv', 'gas', or 'protein').
                      If None, attempts to detect from filenames.

    Returns:
        Tuple of (data, targets, extras) where:
        - data: PyG Data object with node features, edge_index, edge_type, edge_attr
        - targets: List of per-directed-edge multi-dimensional target vectors
                   (one D-length vector per directed edge in data.edge_index)
        - extras: Dict with 'relation_names' (list of relation names) and
                  'base_relation_map' (dict mapping base -> (fwd_name, bwd_name))

    Raises:
        FileNotFoundError: If neither RTF fragments nor variables.py are present.
        RuntimeError: If verify_graph=True and graph verification fails.
    """
    bias: Dict[str, Any] = {}
    combo_path = Path(combo_dir)
    
    # Check for RTF files in both combo_dir and combo_dir/prep
    rtf_results = parse_rtf_dir(combo_dir)
    rtf_dir = combo_dir
    if not rtf_results:
        prep_dir = combo_path / 'prep'
        if prep_dir.exists() and prep_dir.is_dir():
            rtf_results = parse_rtf_dir(str(prep_dir))
            rtf_dir = str(prep_dir)
    
    if rtf_results:
        # Pass the directory path for solvent state detection from folder name
        g = Graph.from_rtf_results(rtf_results, solvent_override=solvent_state, directory=rtf_dir)
        # Save graph_info.json for reward function to extract actual nsubs_per_site
        save_graph_info_from_rtf(combo_dir, g)
    else:
        # Try loading from graph_info.json (saved during pretraining data collection)
        graph_info_path = combo_path / 'graph_info.json'
        if graph_info_path.exists():
            import json
            with open(graph_info_path, 'r') as f:
                graph_info = json.load(f)
            g = Graph.from_graph_info(graph_info)
        else:
            # Fall back to creating graph from bias matrices (no node metadata)
            vpy = combo_path / 'variables.py'
            if vpy.exists():
                bias = load_bias_from_variables(str(vpy))
                g = graph_from_bias(bias)
            else:
                raise FileNotFoundError(f'No RTF fragments, graph_info.json, or variables.py found in {combo_dir} or {combo_dir}/prep')

    data, extras = graph_utils.build_pyg_graph_from_mllf_graph(g, toppar_dir=toppar_dir, toppar_files=toppar_files, 
                                                                 warn_missing_types=warn_missing_types)

    # Optional verification: ensure PyG edges correspond to Graph.edge_mask
    if verify_graph:
        rel_names = extras.get('relation_names', [])
        rel_map = extras.get('base_relation_map', {})
        # build a set of directed edges present in data keyed by (src,dst,rel_name)
        present = set()
        ei = data.edge_index
        et = data.edge_type
        for k in range(ei.shape[1]):
            s = int(ei[0, k].item())
            d = int(ei[1, k].item())
            ridx = int(et[k].item()) if et.numel() > k else None
            rname = rel_names[ridx] if ridx is not None and ridx < len(rel_names) else None
            present.add((s, d, rname))

        for (i, j), mask in getattr(g, 'edge_mask', {}).items():
            for base in ('linear', 'quadratic', 'skew', 'end'):
                allowed = bool(mask.get(base, False))
                if not allowed:
                    # ensure no forward/backward edges for this base in present
                    fwd, bwd = rel_map.get(base, (f"{base}_fwd", f"{base}_bwd"))
                    if (i, j, fwd) in present or (j, i, bwd) in present:
                        raise RuntimeError(f"Graph verification failed: unexpected directed edge for base {base} on ({i},{j})")
                else:
                    fwd, bwd = rel_map.get(base, (f"{base}_fwd", f"{base}_bwd"))
                    if (i, j, fwd) not in present or (j, i, bwd) not in present:
                        raise RuntimeError(f"Graph verification failed: missing directed edges for base {base} on ({i},{j})")

    # build per-edge multi-dimensional targets aligned to data.edge_index.
    rel_names = extras.get('relation_names', [])
    base_map = extras.get('base_relation_map', {})
    # base_order determines output ordering for multi-dim targets
    base_order = list(base_map.keys()) if isinstance(base_map, dict) else ['quadratic', 'skew', 'end']
    if 'linear' not in base_order:
        base_order.append('linear')

    D = len(base_order)

    # map relation name -> base index
    relname_to_baseidx = {}
    for b_idx, (base, (fwd, bwd)) in enumerate(base_map.items()):
        relname_to_baseidx[fwd] = b_idx
        relname_to_baseidx[bwd] = b_idx

    base_to_matrix = {
        'quadratic': bias.get('c', []),
        'skew': bias.get('x', []),
        'end': bias.get('s', []),
        'linear': bias.get('b', []),
    }

    targets = []
    ei = data.edge_index
    for k in range(ei.shape[1]):
        src = int(ei[0, k].item())
        dst = int(ei[1, k].item())
        rel_idx = int(data.edge_type[k].item()) if hasattr(data, 'edge_type') and data.edge_type.numel() > k else None
        rel_name = rel_names[rel_idx] if rel_idx is not None and rel_idx < len(rel_names) else None

        vec = [0.0 for _ in range(D)]
        if rel_name is not None:
            base_idx = relname_to_baseidx.get(rel_name)
            if base_idx is not None:
                base_name = base_order[base_idx]
                mat = base_to_matrix.get(base_name, [])
                try:
                    if base_name == 'linear':
                        bmat = mat
                        if isinstance(bmat, list) and bmat:
                            if isinstance(bmat[0], list):
                                val = float(bmat[src][dst]) if len(bmat) > src and len(bmat[src]) > dst else 0.0
                            else:
                                try:
                                    lhs = float(bmat[src])
                                except Exception:
                                    lhs = 0.0
                                try:
                                    rhs = float(bmat[dst])
                                except Exception:
                                    rhs = 0.0
                                val = 0.5 * (lhs + rhs)
                        else:
                            val = 0.0
                    else:
                        val = float(mat[src][dst]) if mat and len(mat) > src and len(mat[src]) > dst else 0.0
                except Exception:
                    val = 0.0
                vec[base_idx] = val

        targets.append(vec)

    return data, targets, extras


def write_variables_from_actions(combo_dir: str, data, extras: dict, actions: torch.Tensor, out_name: str = 'variables.py', bias_clip: float = 1000.0) -> None:
    """Write a variables.py file from per-directed-edge policy actions.

    This function maps directed relation actions back to base biases (quadratic, skew,
    end, linear). Quadratic (symmetric) uses upper triangle only (i < j). Skew and
    end (NOT symmetric) store BOTH directions independently - each directed edge
    (i→j) has its own value. Linear predictions are antisymmetric (edge (i,j)
    approximates b[j] - b[i], matching the pretraining target) and are inverted
    relative to each site's reference substituent (b[ref] = 0) to recover the
    per-node 'b' vector — they must NOT be averaged as if symmetric.

    Args:
        combo_dir: Path to combo directory where variables.py will be written.
        data: PyG Data object with edge_index and edge_type (from build_data_and_targets_from_combo).
        extras: Dict with 'relation_names' and 'base_relation_map' (from build_data_and_targets_from_combo).
        actions: Tensor of per-directed-edge scalar actions (shape [E] where E = data.edge_index.shape[1]).
                 Each action corresponds to one directed edge in data.edge_index.
        out_name: Name of output file (default: 'variables.py').
        bias_clip: Maximum absolute value for bias coefficients (default: 1000.0).
                   All bias values will be clipped to [-bias_clip, bias_clip].

    Returns:
        None. Writes a Python file containing a triple-quoted YAML bias_string with keys:
        - 'b': per-node linear bias vector (length N)
        - 'c': NxN quadratic bias matrix (upper triangular only, symmetric)
        - 'x': NxN skew bias matrix (full matrix, both directions, NOT symmetric)
        - 's': NxN end bias matrix (full matrix, both directions, NOT symmetric)
    """
    combo_dir = Path(combo_dir)
    N = int(data.x.shape[0])
    base_map = extras.get('base_relation_map', {})
    base_order = list(base_map.keys()) if isinstance(base_map, dict) else ['quadratic', 'skew', 'end']
    if 'linear' not in base_order:
        base_order.append('linear')

    # relation names produced by graph_utils (index -> name)
    rel_names = extras.get('relation_names', []) if isinstance(extras, dict) else []
    # reverse mapping from relation name to base (e.g. 'quadratic_fwd' -> 'quadratic')
    rel_to_base = {}
    if isinstance(base_map, dict):
        for base, pair in base_map.items():
            try:
                fwd, bwd = pair
            except Exception:
                continue
            rel_to_base[fwd] = base
            rel_to_base[bwd] = base

    # Collect values for each base type
    # UnimolPolicy outputs all 4 bias types [linear, quadratic, skew, end] for each edge
    # We extract all dimensions and map them to the appropriate matrices
    per_base_forward = {'quadratic': {}}  # Quadratic is symmetric: undirected canonical pairs
    per_base_directed = {'skew': {}, 'end': {}, 'linear': {}}  # NOT symmetric: directed pairs
    
    # Index into policy output for each base type: [linear, quadratic, skew, end]
    bias_type_index = {'linear': 0, 'quadratic': 1, 'skew': 2, 'end': 3}
    
    ei = data.edge_index
    et = data.edge_type
    for k in range(ei.shape[1]):
        src = int(ei[0, k].item())
        dst = int(ei[1, k].item())
        
        # Extract action values for this edge
        try:
            a = actions[k]
            # Handle different action tensor shapes
            if hasattr(a, 'dim') and a.dim() == 0:
                # Scalar action - only one output
                action_vals = {'linear': float(a.item()), 'quadratic': 0.0, 'skew': 0.0, 'end': 0.0}
            elif hasattr(a, 'shape') and len(a.shape) > 0 and a.shape[-1] == 4:
                # Multi-output action: extract all 4 bias types
                action_vals = {}
                for base_name in ['linear', 'quadratic', 'skew', 'end']:
                    idx = bias_type_index[base_name]
                    try:
                        action_vals[base_name] = float(a[idx].item() if hasattr(a[idx], 'item') else a[idx])
                    except Exception:
                        action_vals[base_name] = 0.0
            else:
                # Fallback: treat as scalar
                vlist = a.detach().cpu().numpy().tolist() if hasattr(a, 'detach') else list(a)
                if isinstance(vlist, list) and len(vlist) == 4:
                    action_vals = {base_name: float(vlist[bias_type_index[base_name]]) 
                                   for base_name in ['linear', 'quadratic', 'skew', 'end']}
                else:
                    scalar_val = float(vlist) if not isinstance(vlist, list) else float(vlist[0])
                    action_vals = {base_name: (scalar_val if base_name == 'quadratic' else 0.0)
                                   for base_name in ['linear', 'quadratic', 'skew', 'end']}
        except Exception:
            try:
                val = float(actions[k])
                action_vals = {base_name: (val if base_name == 'quadratic' else 0.0)
                               for base_name in ['linear', 'quadratic', 'skew', 'end']}
            except Exception:
                action_vals = {base_name: 0.0 for base_name in ['linear', 'quadratic', 'skew', 'end']}
        
        # Store all 4 bias type values for this edge
        for base in ['linear', 'quadratic', 'skew', 'end']:
            val = action_vals.get(base, 0.0)
            
            if base in ['skew', 'end', 'linear']:
                # Skew, end, and linear are NOT symmetric — store directed pairs
                # (preserve both directions independently). Linear predictions
                # approximate b[dst] - b[src] (antisymmetric), so averaging
                # forward/backward values here would wipe out the signal.
                per_base_directed[base][(src, dst)] = val
            else:
                # Quadratic: undirected canonical pairs.
                # This is a genuinely symmetric matrix, so predictions from BOTH
                # directions (i→j and j→i) are averaged together. This matches
                # pretraining where both edges share the same target value.
                pair = (min(src, dst), max(src, dst))
                
                if pair not in per_base_forward[base]:
                    # First edge for this pair: initialize with (value, count)
                    per_base_forward[base][pair] = (val, 1)
                else:
                    # Second edge for this pair: accumulate for averaging
                    prev_val, prev_count = per_base_forward[base][pair]
                    per_base_forward[base][pair] = (prev_val + val, prev_count + 1)

    # Assemble bias matrices for nonlinear terms
    # IMPORTANT: Quadratic is symmetric, so we store ONLY the upper triangle (i < j).
    # Skew and end are NOT symmetric - they need BOTH directions stored independently.
    def build_mat_for_quadratic():
        """Build quadratic matrix (symmetric, upper triangle only).
        
        For each canonical pair, average predictions from both directions if available.
        """
        mat = [[0.0 for _ in range(N)] for _ in range(N)]
        vals_map = per_base_forward.get('quadratic', {})
        for (i, j), val_data in vals_map.items():
            # val_data is either a float (legacy) or (value, count) tuple (new)
            try:
                if isinstance(val_data, tuple):
                    sum_val, count = val_data
                    v = sum_val / count  # Average both directions
                else:
                    v = float(val_data)
            except Exception:
                v = 0.0
            # Clip to prevent extreme values
            v = max(-bias_clip, min(bias_clip, v))
            # Only set the canonical forward entry (i < j means store in upper triangle)
            # This ensures each undirected pair has exactly ONE bias value
            if i < j:
                mat[i][j] = v
                # mat[j][i] remains 0.0 (not -v)
            else:
                mat[j][i] = v
                # mat[i][j] remains 0.0 (not -v)
        return mat
    
    def build_mat_for_bidirectional(base_name: str):
        """Build skew/end matrix (NOT symmetric, both directions stored)."""
        mat = [[0.0 for _ in range(N)] for _ in range(N)]
        directed_vals = per_base_directed.get(base_name, {})
        
        # Populate matrix from directed pairs
        for (src, dst), val in directed_vals.items():
            try:
                v = float(val)
            except Exception:
                v = 0.0
            v = max(-bias_clip, min(bias_clip, v))
            mat[src][dst] = v
        
        return mat

    c_mat = build_mat_for_quadratic()
    x_mat = build_mat_for_bidirectional('skew')
    s_mat = build_mat_for_bidirectional('end')

    # Derive per-node linear bias 'b' from directed per-edge linear predictions.
    #
    # During pretraining the linear target for directed edge (i, j) is
    # b[j] - b[i] — an ANTISYMMETRIC quantity (edge (j,i) targets b[i] - b[j] =
    # -(b[j]-b[i])), unlike quadratic which is genuinely symmetric. Reconstructing
    # 'b' therefore requires inverting this difference relative to each site's
    # reference substituent (b[ref] = 0), NOT averaging raw forward/backward
    # predictions together (that would cancel the antisymmetric signal to ~0).
    b_vec = [0.0 for _ in range(N)]
    linear_directed = per_base_directed.get('linear', {})

    # Determine site groups: ordered lists of node indices per site, with the
    # reference substituent (sub == 1) first. Prefer graph_info.json (accurate
    # site/sub metadata); fall back to contiguous nsubs_per_site blocks.
    site_groups: List[List[int]] = []
    graph_info_path = combo_dir / 'graph_info.json'
    if graph_info_path.exists():
        import json
        try:
            with open(graph_info_path, 'r') as f:
                graph_info = json.load(f)
            sites_info = graph_info.get('sites', {})

            all_nodes = []
            for key, node_info in sites_info.items():
                site = node_info.get('site')
                sub = node_info.get('sub')
                if site is not None and sub is not None:
                    all_nodes.append((site, sub))
            all_nodes.sort(key=lambda x: (x[0], x[1]))  # Sort by (site, sub)

            grouped: Dict[Any, List[int]] = {}
            for idx, (site, sub) in enumerate(all_nodes):
                if idx < N:
                    grouped.setdefault(site, []).append(idx)
            # Each group is already ordered by ascending sub, so sub==1 (the
            # reference) is first.
            site_groups = [grouped[s] for s in sorted(grouped.keys())]
        except Exception:
            site_groups = []

    if not site_groups:
        nsubs_per_site = extras.get('nsubs_per_site') if isinstance(extras, dict) else None
        if nsubs_per_site:
            offset = 0
            for nsubs in nsubs_per_site:
                site_groups.append(list(range(offset, offset + nsubs)))
                offset += nsubs
        elif N > 0:
            site_groups = [list(range(N))]

    for site_nodes in site_groups:
        if not site_nodes:
            continue
        ref = site_nodes[0]
        b_vec[ref] = 0.0
        for j in site_nodes[1:]:
            fwd = linear_directed.get((ref, j))
            bwd = linear_directed.get((j, ref))
            if fwd is not None and bwd is not None:
                # Average the two independent antisymmetric estimates:
                # fwd ≈ b[j]-b[ref] = b[j],  -bwd ≈ b[j]-b[ref] = b[j]
                val = 0.5 * (fwd - bwd)
            elif fwd is not None:
                val = fwd
            elif bwd is not None:
                val = -bwd
            else:
                val = 0.0
            b_vec[j] = max(-bias_clip, min(bias_clip, val))

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
    content = f"""# Auto-generated variables.py — bias_string contains YAML for bias matrices
bias_string = '''
{yaml_block}

'''
"""
    (combo_dir / out_name).write_text(content, encoding='utf-8')


def default_env_reward(actions: torch.Tensor, target_vals: List[float]) -> float:
    """Compute reward as negative MSE between actions and target values.

    This is a simple supervised-style reward used as a fallback when simulation
    results are not available or fail. Real training typically uses simulation
    outputs (transition counts, population metrics) for reward.

    Args:
        actions: Tensor of predicted actions (typically per-edge coefficients).
        target_vals: List of target values (same length as actions).

    Returns:
        Scalar reward: -MSE(actions, targets). Returns -inf on error.
    """
    try:
        targ = torch.tensor(target_vals, dtype=actions.dtype, device=actions.device)
        mse = torch.mean((actions - targ) ** 2).item()
        return -mse
    except Exception:
        return float('-inf')


def create_and_manifest(input_dir: str, out_dir: str, dry_run: bool = False) -> str:
    """Generate combination directories and create a manifest file.

    This function scans input_dir for site/sub files (e.g., site1_sub2_*_pres.rtf),
    generates all valid combinations (respecting the constraint that combinations
    differing only in tail permutations are treated as duplicates), and creates
    a manifest listing all generated combo directories.

    Args:
        input_dir: Directory containing site{n}_sub{m}_*_{pres|frag}.{rtf|pdb} files.
        out_dir: Output directory where combo subdirectories will be created.
        dry_run: If True, print actions without creating directories or copying files.

    Returns:
        Path to the generated manifest.txt file (one combo directory path per line).
    """
    created = create_combination_dirs(Path(input_dir), Path(out_dir), dry_run=dry_run)
    manifest_path = Path(out_dir) / 'manifest.txt'
    with manifest_path.open('w') as fh:
        for p in created:
            fh.write(str(p) + '\n')
    return str(manifest_path)


def split_manifest(manifest: str, train_frac: float = 0.8, seed: int = 0) -> Tuple[str, str]:
    """Split a manifest file into training and validation sets.

    Args:
        manifest: Path to manifest.txt file listing combo directories (one per line).
        train_frac: Fraction of combos to use for training (default: 0.8 = 80%).
        seed: Random seed for shuffling (default: 0 for reproducibility).

    Returns:
        Tuple of (train_manifest_path, val_manifest_path).
        Creates manifest.train.txt and manifest.val.txt in the same directory as manifest.
    """
    with open(manifest, 'r', encoding='utf-8') as fh:
        combos = [ln.strip() for ln in fh if ln.strip()]
    random.Random(seed).shuffle(combos)
    n = int(len(combos) * train_frac)
    train = combos[:n]
    val = combos[n:]
    mtrain = Path(manifest).parent / 'manifest.train.txt'
    mval = Path(manifest).parent / 'manifest.val.txt'
    mtrain.write_text('\n'.join(train) + ('\n' if train else ''), encoding='utf-8')
    mval.write_text('\n'.join(val) + ('\n' if val else ''), encoding='utf-8')
    return str(mtrain), str(mval)


def run_quick_epoch_for_combo(combo_dir: str, base_bias: str = 'quadratic') -> Dict[str, Any]:
    """Run a single training epoch for demonstration/testing purposes.

    This function builds a small RGCN encoder and EdgePolicy, performs one forward
    pass with action sampling, computes a supervised reward (negative MSE vs targets),
    and updates the policy with REINFORCE. It's intended for quick validation and
    testing, not for full training runs.

    Args:
        combo_dir: Path to combo directory with RTF fragments or variables.py.
        base_bias: Legacy parameter (currently unused, kept for compatibility).

    Returns:
        Dict with key 'reward' containing the scalar reward from the epoch.
    """
    data, targets, extras = build_data_and_targets_from_combo(combo_dir, base_bias=base_bias)
    sample_data = data
    in_dim = sample_data.x.shape[1]
    num_rels = int(sample_data.edge_attr.shape[1]) if hasattr(sample_data, 'edge_attr') else 1
    encoder = RGCNEncoder(in_dim=in_dim, hidden_dims=[32], out_dim=16, num_relations=num_rels)
    base_map = extras.get('base_relation_map', {}) if isinstance(extras, dict) else {}
    edge_out_dim = len(list(base_map.keys())) if isinstance(base_map, dict) else 1
    policy = EdgePolicy.from_pyg_data(encoder, 16, sample_data, mlp_hidden=32, mlp_out_dim=edge_out_dim)
    policy.train()
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # one pass: sample actions, write variables, compute reward against targets, and update
    node_emb = policy.forward_node_embeddings(sample_data.x, sample_data.edge_index, getattr(sample_data, 'edge_type', None))
    edge_actions, edge_logp, edge_mean, edge_logstd = policy.get_actions(sample_data.x, sample_data.edge_index, getattr(sample_data, 'edge_type', None), getattr(sample_data, 'edge_attr', None), deterministic=False)
    # write variables and run sim could be done here; for quick epoch compute reward from targets
    reward = default_env_reward(edge_actions.detach(), targets)
    loss = -(edge_logp.sum() * float(reward))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return {'reward': float(reward)}


def compress_runs(manifest: str, out_tar: str) -> str:
    """Create a gzipped tar archive of the directory containing combo runs.

    Args:
        manifest: Path to manifest file. The parent directory of this file will be archived.
        out_tar: Desired output archive path (e.g., 'combos_archive'). The '.tar.gz'
                 extension will be added automatically if not present.

    Returns:
        Path to the created .tar.gz archive file.
    """
    base = Path(manifest).parent
    tar = Path(out_tar)
    # make a tar.gz of the directory containing combos
    shutil.make_archive(str(tar.with_suffix('')), 'gztar', root_dir=str(base))
    return str(tar.with_suffix('.tar.gz'))


def run_from_config(config_path: str) -> Dict[str, Any]:
    """Execute a complete workflow based on a YAML configuration file.

    This is the main entry point for running the full pipeline: combo generation,
    train/val split, quick epoch demo, concurrent simulations, and archiving.

    YAML Configuration Options:
        create_combos:          # (optional) Generate combo directories
          input_dir: str        # Directory with site_sub files
          out_dir: str          # Output directory for combos
          dry_run: bool         # If true, print actions without creating files
        
        manifest: str           # (alternative to create_combos) Path to existing manifest
        
        split:                  # (optional) Split manifest into train/val
          train_frac: float     # Fraction for training (e.g., 0.8)
          seed: int             # Random seed for reproducibility
        
        base_bias: str          # Legacy parameter (currently unused)
        
        run_sims: bool          # If true, run simulations concurrently
        sim_cmd: str            # Shell command to run in each combo (default: './run.sh')
        max_workers: int        # Max concurrent simulations (default: 4)
        timeout: int            # Per-simulation timeout in seconds (optional)
        
        compress_after:         # (optional) Create archive after completion
          out_tar: str          # Output archive path

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dict with keys:
        - 'manifest': Path to manifest file
        - 'train_manifest': Path to train manifest (if split requested)
        - 'val_manifest': Path to val manifest (if split requested)
        - 'example_combo': Path to first combo (used for quick epoch)
        - 'quick_epoch': Dict with 'reward' from quick training pass
        - 'sim_results': Dict with simulation results (if run_sims=True)
        - 'archive': Path to created archive (if compress_after specified)
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    results: Dict[str, Any] = {}
    # Step 1: create combos
    if cfg.get('create_combos'):
        manifest = create_and_manifest(cfg['create_combos']['input_dir'], cfg['create_combos'].get('out_dir', 'combos'), dry_run=cfg['create_combos'].get('dry_run', False))
    else:
        manifest = cfg.get('manifest')

    results['manifest'] = manifest
    # Step 2: split
    if cfg.get('split') and manifest:
        train_m, val_m = split_manifest(manifest, cfg['split'].get('train_frac', 0.8), cfg['split'].get('seed', 0))
        results['train_manifest'] = train_m
        results['val_manifest'] = val_m

    # Step 3: pick example combo
    example_combo = None
    with open(manifest, 'r', encoding='utf-8') as fh:
        combos = [ln.strip() for ln in fh if ln.strip()]
    if combos:
        example_combo = combos[0]
        results['example_combo'] = example_combo

    # Step 4: build graph & run quick epoch
    if example_combo:
        results['quick_epoch'] = run_quick_epoch_for_combo(example_combo, base_bias=cfg.get('base_bias', 'quadratic'))

    # Step 5: optionally start simulations concurrently
    if cfg.get('run_sims'):
        sim_cmd = cfg.get('sim_cmd')
        max_workers = cfg.get('max_workers', 4)
        timeout = cfg.get('timeout')
        sim_results = run_simulation_batch(manifest, sim_cmd=sim_cmd, max_workers=max_workers, timeout=timeout)
        results['sim_results'] = sim_results

    # Step 6: compress if requested
    if cfg.get('compress_after'):
        out_tar = cfg['compress_after'].get('out_tar', str(Path(manifest).parent) + '.tar.gz')
        results['archive'] = compress_runs(manifest, out_tar)

    return results


if __name__ == '__main__':
    # Command-line interface: run the full workflow from a YAML config file
    # Usage: python -m mllf.cli.workflow config.yaml
    # Or: python src/mllf/cli/workflow.py config.yaml
    import argparse
    p = argparse.ArgumentParser(
        description='Run contextual bandit workflow from YAML config',
        epilog='Example: python -m mllf.cli.workflow examples/workflow_sample.yaml'
    )
    p.add_argument('config', help='Path to YAML config file describing workflow steps')
    args = p.parse_args()
    out = run_from_config(args.config)
    # Print results as YAML for easy inspection
    print(yaml.safe_dump(out, sort_keys=False))
