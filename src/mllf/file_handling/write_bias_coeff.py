"""Write bias coefficient files in the old `.inp` or `.py` style used by the simulator.

This module provides helpers to write a simple `.inp` or `.py` file with lines of the
form `set <param> = <value>` so the existing parser can read it back.

"""
from __future__ import annotations

from typing import Union, Sequence, Optional
import os
import re

from .read_bias_coeff import parse_old  # kept for tests/debug


def write_bias_inp_from_graph(graph, filename: str, sub_counts: Optional[Sequence[int]] = None,
                              mapping: Optional[dict] = None, header_source: Optional[str] = None) -> None:
    """Write a `.inp` file from a Graph instance.

    This writes single-sub (lams) and two-sub (cs) entries.

    Arguments:
        graph: instance with `num_nodes` and `edges` mapping (EdgeCoeffs dataclass)
        filename: output path to write
        sub_counts: optional sequence of length num_nodes with number of substituents
            per site. If None, defaults to 1 substituent per site.
        coeff_key: which EdgeCoeffs attribute to use for pair values
            (one of 'linear','quadratic','skew','end'). Default 'linear'.
    """
    n = graph.num_nodes
    if sub_counts is None:
        sub_counts = [1] * n
    if len(sub_counts) != n:
        raise ValueError("sub_counts must have length equal to graph.num_nodes")

    # default mapping per user: lams -> linear, cs -> quadratic, xs -> skew, ss -> end
    # mapping dict maps pair-key to EdgeCoeffs attribute
    if mapping is None:
        mapping = {"cs": "quadratic", "ss": "end", "xs": "skew"}

    def edge_coeff_key(i: int, j: int, key: str) -> float:
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        e = graph.get_edge(a, b)
        return float(getattr(e, key, 0.0))

    lines = []

    # If a header_source is provided, copy non-bias lines verbatim from it.
    # Lines considered "bias" start with a 'set ' and a name beginning with
    # one of the bias prefixes (lams, cs, ss, xs). Those are skipped when
    # copying so we can append freshly generated bias lines below.
    if header_source is not None:
        if not os.path.exists(header_source):
            raise FileNotFoundError(f"header_source not found: {header_source}")
        with open(header_source, "r", encoding="utf-8") as hf:
            for ln in hf:
                s = ln.strip()
                if not s:
                    # preserve blank lines
                    lines.append(ln if ln.endswith("\n") else ln + "\n")
                    continue
                if not s.startswith("set "):
                    lines.append(ln if ln.endswith("\n") else ln + "\n")
                    continue
                # extract parameter name
                parts = ln.split("=")
                left = parts[0].strip()
                try:
                    _, name = left.split(None, 1)
                except ValueError:
                    # unexpected format, copy verbatim
                    lines.append(ln if ln.endswith("\n") else ln + "\n")
                    continue
                # skip bias coefficients (we'll generate them below)
                if name.startswith(("lams", "cs", "ss", "xs")):
                    continue
                lines.append(ln if ln.endswith("\n") else ln + "\n")

    # Single-sub (lams): write one line per substituent per site (example file lists all)
    for i in range(n):
        # compute site-level proxy as average of incident edge 'linear' values
        incident_vals = []
        for j in range(n):
            if i == j:
                continue
            incident_vals.append(edge_coeff_key(i, j, "linear"))
        site_val = float(sum(incident_vals) / len(incident_vals)) if incident_vals else 0.0
        for s in range(1, sub_counts[i] + 1):
            lines.append(f"set lams{i+1}s{s} = {site_val:12.8f}\n")

    # Pair biases: for each ordered pair of (site,sub) x (site2,sub2) excluding identical
    # site+sub, write cs/ss/xs using mapped edge coefficient components.
    for i in range(n):
        for j in range(i, n):
            if i < j:
                # all combinations for different sites
                for sa in range(1, sub_counts[i] + 1):
                    for sb in range(1, sub_counts[j] + 1):
                        cs_val = edge_coeff_key(i, j, mapping.get("cs", "quadratic"))
                        ss_val = edge_coeff_key(i, j, mapping.get("ss", "end"))
                        xs_val = edge_coeff_key(i, j, mapping.get("xs", "skew"))
                        lines.append(f"set cs{i+1}s{sa}s{j+1}s{sb} = {cs_val:12.8f}\n")
                        # ss/xs forward
                        lines.append(f"set ss{i+1}s{sa}s{j+1}s{sb} = {ss_val:12.8f}\n")
                        lines.append(f"set xs{i+1}s{sa}s{j+1}s{sb} = {xs_val:12.8f}\n")
                        # ss/xs reverse (example contains both orientations for ss/xs)
                        lines.append(f"set ss{j+1}s{sb}s{i+1}s{sa} = {ss_val:12.8f}\n")
                        lines.append(f"set xs{j+1}s{sb}s{i+1}s{sa} = {xs_val:12.8f}\n")
            else:
                # same site: only sa < sb (strict upper triangle)
                for sa in range(1, sub_counts[i] + 1):
                    for sb in range(sa + 1, sub_counts[j] + 1):
                        cs_val = edge_coeff_key(i, j, mapping.get("cs", "quadratic"))
                        ss_val = edge_coeff_key(i, j, mapping.get("ss", "end"))
                        xs_val = edge_coeff_key(i, j, mapping.get("xs", "skew"))
                        lines.append(f"set cs{i+1}s{sa}s{j+1}s{sb} = {cs_val:12.8f}\n")
                        # ss/xs forward
                        lines.append(f"set ss{i+1}s{sa}s{j+1}s{sb} = {ss_val:12.8f}\n")
                        lines.append(f"set xs{i+1}s{sa}s{j+1}s{sb} = {xs_val:12.8f}\n")
                        # ss/xs reverse
                        lines.append(f"set ss{j+1}s{sb}s{i+1}s{sa} = {ss_val:12.8f}\n")
                        lines.append(f"set xs{j+1}s{sb}s{i+1}s{sa} = {xs_val:12.8f}\n")

    # ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def create_variables_py_from_template(template_path: str, out_path: str, minimizeflag: bool = False):
    """Create a variables#.py file by copying a template and setting minimizeflag.

    This is a simple textual transform: it replaces an assignment like
    'minimizeflag=True' (possibly with surrounding whitespace) with the
    requested value. If the template is not a .py file or the token isn't
    found, the template is copied verbatim with a appended 'minimizeflag'
    assignment.
    """

    if not os.path.exists(template_path):
        raise FileNotFoundError(template_path)

    with open(template_path, 'r', encoding='utf-8') as fh:
        content = fh.read()

    # replace existing minimizeflag assignment
    flag_str = 'minimizeflag=True'
    flag_false = 'minimizeflag=False'
    # tolerant regex to match minimizeflag = True with optional spaces
    new_content, nsub = re.subn(r"minimizeflag\s*=\s*True", flag_false if not minimizeflag else flag_str, content)
    if nsub == 0:
        # if not present, append at end
        new_content = content + f"\n{flag_false if not minimizeflag else flag_str}\n"

    # write out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(new_content)


def write_variables_py_from_inp(inp_path: str, out_path: str):
    """Convert an old-style `.inp` variables file into a `variables.py`-style
    file containing scalar entries that `parse_new` can read.

    This is a simple translator used for tests and for creating variables.py
    files from ALF/ALF-generated `.inp` outputs. It reads the `.inp` using the
    existing `parse_old` helper and writes out a small Python file containing
    scalar lines of the form `param: value` (one per line). A minimal
    `bias_string` is included to be compatible with existing templates.
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
        """Return dict site -> dict(sub -> value) parsed from parameter keys like lams1s2 or cs1s1s1s2."""
        site_map = {}
        for k, v in group_keys.items():
            nums = re.findall(r"(\d+)", k)
            if not nums:
                continue
            # For lams: pattern lams{site}s{sub} -> nums [site, sub]
            # For cs/xs/ss: pattern grp{site}s{a}s{site}s{b} or similar -> take primary site from first number
            if k.startswith('lams') and len(nums) >= 2:
                site = int(nums[0])
                sub = int(nums[1])
            else:
                # for cs/xs/ss pick primary site as first number and sub index as second when possible
                site = int(nums[0])
                sub = int(nums[1]) if len(nums) >= 2 else 1
            site_map.setdefault(site, {})[sub] = float(v)
        return site_map

    lams_map = _extract_site_sub_map(data.get('lams', {}))
    cs_map = _extract_site_sub_map(data.get('cs', {}))
    xs_map = _extract_site_sub_map(data.get('xs', {}))
    ss_map = _extract_site_sub_map(data.get('ss', {}))

    # determine list of sites and per-site counts
    sites = sorted(set(lams_map.keys()) | set(cs_map.keys()) | set(xs_map.keys()) | set(ss_map.keys()))
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
        b_vec[g - 1] = float(lams_map.get(s, {}).get(sub, 0.0))

    # helper to initialize NxN matrix and fill from group dict by parsing keys
    def build_matrix_from_group(group_dict):
        mat = [[0.0 for _ in range(total_subs)] for _ in range(total_subs)]
        for k, v in group_dict.items():
            nums = re.findall(r"(\d+)", k)
            if len(nums) >= 4:
                s1 = int(nums[0]); a = int(nums[1]); s2 = int(nums[2]); b = int(nums[3])
                i = global_index.get((s1, a))
                j = global_index.get((s2, b))
                if i is not None and j is not None:
                    mat[i - 1][j - 1] = float(v)
            elif len(nums) >= 2:
                # fallback: treat as site/sub -> value mapping (diagonal)
                s1 = int(nums[0]); a = int(nums[1])
                i = global_index.get((s1, a))
                if i is not None:
                    mat[i - 1][i - 1] = float(v)
        return mat

    c_mat = build_matrix_from_group(data.get('cs', {}))
    x_mat = build_matrix_from_group(data.get('xs', {}))
    s_mat = build_matrix_from_group(data.get('ss', {}))

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


    # include textual scalar entries inside the bias_string (single copy)
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
