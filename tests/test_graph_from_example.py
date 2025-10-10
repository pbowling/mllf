import os
from pathlib import Path

from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.rl.graph import Graph


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / 'pyproject.toml').exists():
            return parent
    raise RuntimeError('Could not find repository root (pyproject.toml)')


def test_graph_construction_from_example():
    repo = _find_repo_root(Path(__file__))
    examples_dir = repo / 'examples' / 'rl' / '14benz_solv_5.5'
    assert examples_dir.is_dir(), f"example dir not found: {examples_dir}"

    rtf_results = parse_rtf_dir(str(examples_dir))
    assert rtf_results, "parse_rtf_dir returned no entries"

    g = Graph.from_rtf_results(rtf_results)

    # For this example we expect one node per substituent (11 nodes total)
    assert g.num_nodes == 11, f"expected 11 nodes, got {g.num_nodes}"

    # ensure each node has required metadata fields
    for i in range(g.num_nodes):
        info = g.get_node_info(i)
        assert 'site' in info and 'sub' in info or 'rtf' in info, f"node {i} missing metadata"
