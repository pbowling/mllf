"""Write the ``variables.py`` bias files the simulator reads.

``write_variables_from_actions`` turns policy actions into a combo's
``variables.py``; ``write_variables_py_from_inp`` converts an old-style
``.inp`` (``set <param> = <value>`` lines) to that format.

Sign convention: flat ``cs*``/``xs*``/``ss*`` values (what CHARMM receives via
``ldbv ... @coeff``) are the negatives of the c/x/s matrix coefficients;
``lams*`` equals b. Every conversion here goes through
``read_bias_coeff.FLAT_TO_MATRIX_NONLINEAR``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .read_bias_coeff import (
    FLAT_TO_MATRIX_LINEAR,
    FLAT_TO_MATRIX_NONLINEAR,
    NONLINEAR_SIGN_KEY,
    parse_old,
)


def write_variables_py_from_inp(inp_path: str, out_path: str):
    """Convert an old-style `.inp` variables file into a `variables.py`-style
    file containing scalar entries that `parse_new` can read.

    This is a simple translator used for tests and for creating variables.py
    files from ALF/ALF-generated `.inp` outputs. It reads the `.inp` using the
    existing `parse_old` helper and writes out a small Python file containing
    scalar lines of the form `param: value` (one per line). A minimal
    `bias_string` is included to be compatible with existing templates.

    Signs follow ALF's variablesflat.py layout: the ``b``/``c``/``x``/``s``
    matrices are in the matrix convention (c/x/s = NEGATED flat cs/xs/ss, b =
    lams), while the scalar ``lams``/``cs``/``xs``/``ss`` entries are copied as
    written (flat CHARMM convention). The file is marked with
    ``_nonlinear_sign_convention: matrix``.
    """
    if not os.path.exists(inp_path):
        raise FileNotFoundError(inp_path)

    data = parse_old(inp_path)

    # Prepare header (imports + docstring to match pretraining file format)
    header = '"""Auto-generated variables file for pretraining data."""\nimport yaml\n\n'

    lines = [header]

    # Build bias_string matrices (b, c, x, s) in YAML list-of-lists format.
    # Map: lams -> b, cs -> c, xs -> x, ss -> s
    def _extract_site_sub_map(group_keys):
        """Return dict site -> dict(sub -> value) parsed from parameter keys like lams1s2
        or cs1s1s1s2."""
        site_map = {}
        for k, v in group_keys.items():
            nums = re.findall(r"(\d+)", k)
            if not nums:
                continue
            # For lams: pattern lams{site}s{sub} -> nums [site, sub]
            # For cs/xs/ss: pattern grp{site}s{a}s{site}s{b} or similar -> take
            # primary site from first number
            if k.startswith('lams') and len(nums) >= 2:
                site = int(nums[0])
                sub = int(nums[1])
            else:
                # for cs/xs/ss pick primary site as first number and sub index as second
                # when possible
                site = int(nums[0])
                sub = int(nums[1]) if len(nums) >= 2 else 1
            site_map.setdefault(site, {})[sub] = float(v)
        return site_map

    lams_map = _extract_site_sub_map(data.get('lams', {}))
    cs_map = _extract_site_sub_map(data.get('cs', {}))
    xs_map = _extract_site_sub_map(data.get('xs', {}))
    ss_map = _extract_site_sub_map(data.get('ss', {}))

    # determine list of sites and per-site counts
    sites = sorted(set(lams_map.keys()) | set(cs_map.keys())
                   | set(xs_map.keys()) | set(ss_map.keys()))
    per_site_counts = {}
    for s in sites:
        count = max(
            max(lams_map.get(s, {}).keys()) if lams_map.get(s) else 0,
            max(cs_map.get(s, {}).keys()) if cs_map.get(s) else 0,
            max(xs_map.get(s, {}).keys()) if xs_map.get(s) else 0,
            max(ss_map.get(s, {}).keys()) if ss_map.get(s) else 0,
        )
        per_site_counts[s] = count

    # create flattened global ordering of substituents across sites
    global_index = {}
    idx = 1
    for s in sites:
        for sub in range(1, per_site_counts[s] + 1):
            global_index[(s, sub)] = idx
            idx += 1
    total_subs = idx - 1

    # build b vector (flattened lams in global order)
    b_vec = [0.0] * total_subs
    for (s, sub), g in global_index.items():
        b_vec[g - 1] = FLAT_TO_MATRIX_LINEAR * float(lams_map.get(s, {}).get(sub, 0.0))

    # helper to initialize NxN matrix and fill from group dict by parsing keys
    def build_matrix_from_group(group_dict, sign):
        """NxN matrix from flat cs/xs/ss entries, converted to the matrix convention."""
        mat = [[0.0 for _ in range(total_subs)] for _ in range(total_subs)]
        for k, v in group_dict.items():
            nums = re.findall(r"(\d+)", k)
            if len(nums) >= 4:
                s1 = int(nums[0])
                a = int(nums[1])
                s2 = int(nums[2])
                b = int(nums[3])
                i = global_index.get((s1, a))
                j = global_index.get((s2, b))
                if i is not None and j is not None:
                    mat[i - 1][j - 1] = (sign * float(v)) or 0.0
            elif len(nums) >= 2:
                # fallback: treat as site/sub -> value mapping (diagonal)
                s1 = int(nums[0])
                a = int(nums[1])
                i = global_index.get((s1, a))
                if i is not None:
                    mat[i - 1][i - 1] = (sign * float(v)) or 0.0
        return mat

    c_mat = build_matrix_from_group(data.get('cs', {}), FLAT_TO_MATRIX_NONLINEAR)
    x_mat = build_matrix_from_group(data.get('xs', {}), FLAT_TO_MATRIX_NONLINEAR)
    s_mat = build_matrix_from_group(data.get('ss', {}), FLAT_TO_MATRIX_NONLINEAR)

    # Compose bias_string block with b (single row) and full NxN matrices for c/x/s
    bias_lines = ["bias_string=\"\"\"\n"]
    # b: single row
    bias_lines.append("b:\n")
    if total_subs > 0:
        bias_lines.append(f"- - {b_vec[0]}\n")
        for v in b_vec[1:]:
            bias_lines.append(f"  - {v}\n")
    else:
        bias_lines.append("- - 0.0\n")

    def _mat_to_lines(mat):
        out = []
        for row in mat:
            out.append(f"- - {row[0]}\n")
            for val in row[1:]:
                out.append(f"  - {val}\n")
        return out

    bias_lines.append("c:\n")
    bias_lines.extend(_mat_to_lines(c_mat))
    bias_lines.append("x:\n")
    bias_lines.extend(_mat_to_lines(x_mat))
    bias_lines.append("s:\n")
    bias_lines.extend(_mat_to_lines(s_mat))
    bias_lines.append(f"{NONLINEAR_SIGN_KEY}: matrix\n")

    # include textual scalar entries inside the bias_string (single copy),
    # as written -- flat CHARMM convention, like ALF's variablesflat.py
    for group in ('lams', 'cs', 'xs', 'ss'):
        grp = data.get(group, {})
        if grp:
            for k in sorted(grp.keys()):
                # write as YAML scalar entries inside the bias_string block
                bias_lines.append(f"{k}: {repr(float(grp[k]))}\n")

    # close bias_string block
    bias_lines.append('"""\n\n')

    # Add the yaml.safe_load line to make the bias dict available
    bias_lines.append('bias = yaml.safe_load(bias_string)\n')

    # write header + bias_string
    lines = [header] + bias_lines

    # (scalars are included inside the bias_string above; do not duplicate them here)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(out_path)) or '.'
    os.makedirs(out_dir, exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.writelines(lines)

    return out_path


def fit_site_b(site_nodes: List[int], linear_directed: Dict[Tuple[int, int], float],
               edge_var: Optional[Dict[Tuple[int, int], float]] = None
               ) -> Tuple[Dict[int, float], Dict[str, float]]:
    """Per-substituent b for one site from every directed edge's b difference.

    Each directed edge (i -> j) predicts b[j] - b[i]; (j -> i) predicts the
    negative. With the site's reference (``site_nodes[0]``) at 0, the other
    b values are the weighted least-squares fit to all of them (weight
    1 / ``edge_var``, default equal), so a difference measured along any path
    (1 -> 3 -> 5 as well as 1 -> 5) informs b. For a two-substituent site this
    is the mean of the two directions.

    Returns (b by node, {'rms', 'max', 'n_pairs'}): the misfit between each
    pair's combined estimate and the fitted b[j] - b[i] (kcal/mol), i.e. how
    far the predicted differences are from closing around cycles; 0 with no
    cycles.
    """
    others = list(site_nodes[1:])
    col = {n: c for c, n in enumerate(others)}
    in_site = set(site_nodes)
    pairs: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for (src, dst), val in linear_directed.items():
        if src not in in_site or dst not in in_site or src == dst:
            continue
        var = (edge_var or {}).get((src, dst), 1.0)
        w = 1.0 / var if var and np.isfinite(var) and var > 0 else 1.0
        lo, hi = min(src, dst), max(src, dst)
        pairs.setdefault((lo, hi), []).append((float(val) if src == lo else -float(val), w))
    b = {n: 0.0 for n in site_nodes}
    if not others or not pairs:
        return b, {'rms': 0.0, 'max': 0.0, 'n_pairs': 0}
    rows, rhs, est = [], [], {}
    for (lo, hi), vals in pairs.items():
        w = sum(v[1] for v in vals)
        d = sum(v[0] * v[1] for v in vals) / w
        est[(lo, hi)] = d
        row = np.zeros(len(others))
        if hi in col:
            row[col[hi]] += 1.0
        if lo in col:
            row[col[lo]] -= 1.0
        sw = np.sqrt(w)
        rows.append(row * sw)
        rhs.append(d * sw)
    sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    for n, c in col.items():
        b[n] = float(sol[c])
    res = [est[(lo, hi)] - (b[hi] - b[lo]) for (lo, hi) in est]
    return b, {'rms': float(np.sqrt(np.mean(np.square(res)))),
               'max': float(np.max(np.abs(res))), 'n_pairs': len(res)}


def write_variables_from_actions(combo_dir: str, data, extras: dict, actions,
                                 out_name: str = 'variables.py', bias_clip: float = 1000.0,
                                 linear_var=None) -> None:
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
        data: graph data with edge_index and edge_type (``build_graph_and_data``).
        extras: Dict with 'relation_names', 'base_relation_map' and 'nsubs_per_site'
            (``build_graph_and_data``).
        actions: Per-directed-edge actions, one row per edge of data.edge_index
            ([E] scalars or [E, 4] for linear, quadratic, skew, end).
        out_name: Name of output file (default: 'variables.py').
        bias_clip: Maximum absolute value for bias coefficients (default: 1000.0).
                   All bias values will be clipped to [-bias_clip, bias_clip].
        linear_var: Optional [E] variance of each edge's linear action (e.g.
                   ``hindsight.linear_edge_variance``), weighting the per-site
                   b fit (``fit_site_b``); equal weights if None.

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

        # Determine which bias type this edge corresponds to based on edge_type
        edge_base = None
        if hasattr(et, '__len__') and et.numel() > k:
            rel_idx = int(et[k].item())
            if rel_idx < len(rel_names):
                rel_name = rel_names[rel_idx]
                edge_base = rel_to_base.get(rel_name)

        # Extract action values for this edge
        try:
            a = actions[k]
            # Handle different action tensor shapes
            if hasattr(a, 'dim') and a.dim() == 0:
                # Scalar action - route to the appropriate bias type based on edge_type
                # If edge_type indicates a specific bias type, use that; otherwise default
                # to quadratic
                action_vals = {
                    base_name: 0.0 for base_name in [
                        'linear', 'quadratic', 'skew', 'end']}
                scalar_val = float(a.item())
                if edge_base and edge_base in action_vals:
                    action_vals[edge_base] = scalar_val
                else:
                    # Fallback: default to quadratic for backwards compatibility
                    action_vals['quadratic'] = scalar_val
            elif hasattr(a, 'shape') and len(a.shape) > 0 and a.shape[-1] == 4:
                # Multi-output action: extract all 4 bias types
                action_vals = {}
                for base_name in ['linear', 'quadratic', 'skew', 'end']:
                    idx = bias_type_index[base_name]
                    try:
                        action_vals[base_name] = float(
                            a[idx].item() if hasattr(
                                a[idx], 'item') else a[idx])
                    except Exception:
                        action_vals[base_name] = 0.0
            else:
                # Fallback: treat as scalar or list
                vlist = a.detach().cpu().numpy().tolist() if hasattr(a, 'detach') else list(a)
                if isinstance(vlist, list) and len(vlist) == 4:
                    action_vals = {base_name: float(vlist[bias_type_index[base_name]])
                                   for base_name in ['linear', 'quadratic', 'skew', 'end']}
                else:
                    scalar_val = float(vlist) if not isinstance(vlist, list) else float(vlist[0])
                    action_vals = {
                        base_name: 0.0 for base_name in [
                            'linear', 'quadratic', 'skew', 'end']}
                    if edge_base and edge_base in action_vals:
                        action_vals[edge_base] = scalar_val
                    else:
                        # Fallback: default to quadratic
                        action_vals['quadratic'] = scalar_val
        except Exception:
            try:
                val = float(actions[k])
                action_vals = {
                    base_name: 0.0 for base_name in [
                        'linear', 'quadratic', 'skew', 'end']}
                if edge_base and edge_base in action_vals:
                    action_vals[edge_base] = val
                else:
                    # Fallback: default to quadratic
                    action_vals['quadratic'] = val
            except Exception:
                action_vals = {
                    base_name: 0.0 for base_name in [
                        'linear', 'quadratic', 'skew', 'end']}

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

    # Every within-site edge, not only those from the reference, informs b:
    # fit_site_b solves for one b per substituent (reference 0) from all of
    # them, weighted by linear_var.
    edge_var = None
    if linear_var is not None:
        edge_var = {(int(ei[0, k]), int(ei[1, k])): float(linear_var[k])
                    for k in range(min(ei.shape[1], len(linear_var)))}
    b_fit = {}
    for s_idx, site_nodes in enumerate(site_groups):
        if not site_nodes:
            continue
        b_site, fit = fit_site_b(site_nodes, linear_directed, edge_var)
        for j, val in b_site.items():
            b_vec[j] = max(-bias_clip, min(bias_clip, val))
        if len(site_nodes) > 2:
            b_fit[f'site{s_idx + 1}'] = fit
    if b_fit:
        # Cycle-closure misfit of the predicted b differences, per site.
        (combo_dir / 'b_fit.json').write_text(json.dumps(b_fit, indent=1))

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
