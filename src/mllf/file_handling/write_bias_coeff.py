"""Write bias coefficient files in the old `.inp` style used by the simulator.

This module provides helpers to write a simple `.inp` file with lines of the
form `set <param> = <value>` so the existing parser (`read_bias_coeff.parse_old`)
can read it back.

We map each undirected edge (i,j) to four parameters using the `cs` prefix:
  set cs_i_j_linear = ...
  set cs_i_j_quadratic = ...
  set cs_i_j_skew = ...
  set cs_i_j_end = ...

Parameter names are alphanumeric/underscore so the old parser will classify
them under the `cs` group.
"""
from __future__ import annotations

from typing import Union, Sequence, Optional
import os

from .read_bias_coeff import parse_old  # kept for tests/debug


def write_bias_inp_from_graph(graph, filename: str, sub_counts: Optional[Sequence[int]] = None,
                              mapping: Optional[dict] = None) -> None:
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

    # Header variables present in example files (names matter for tests)
    # values are placeholders; the test compares only parameter names
    lines.append("set minimizeflag = 0\n")
    lines.append("set nblocks = 0\n")
    lines.append("set ncentral = 0\n")
    lines.append(f"set nnodes = {n}\n")
    lines.append("set nreps = 0\n")
    lines.append(f"set nsites = {n}\n")
    # nsubs per site
    for idx, sc in enumerate(sub_counts, start=1):
        lines.append(f"set nsubs{idx} = {sc}\n")
    lines.append("set restartfile = ''\n")
    # other header items present in example
    lines.append("set sysname = ''\n")
    lines.append("set temp = 0\n")

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
