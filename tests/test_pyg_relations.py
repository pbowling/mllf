import numpy as np
from pathlib import Path

from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.rl.graph import Graph
from mllf.cb.graph_utils import build_pyg_graph_from_mllf_graph


def test_pyg_relations_match_graph():
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / 'examples' / 'cb' / '14benz_solv_5.5'
    assert example_dir.exists(), f"Example directory not found: {example_dir}"

    rtf_results = parse_rtf_dir(str(example_dir))
    g = Graph.from_rtf_results(rtf_results)

    data, extras = build_pyg_graph_from_mllf_graph(g)
    relation_names = extras['relation_names']
    rel_map = extras['relation_map']
    k = len(relation_names)

    # Basic shapes
    assert data.x.shape[0] == g.num_nodes
    assert data.edge_attr.shape[1] == k + 1

    # Build a set of actual directed triples (i,j,rel)
    actual = set()
    for idx in range(data.edge_index.shape[1]):
        i = int(data.edge_index[0, idx].item())
        j = int(data.edge_index[1, idx].item())
        rel = int(data.edge_type[idx].item())
        actual.add((i, j, rel))
        # check one-hot
        oh = data.edge_attr[idx, :k].cpu().numpy()
        assert np.isclose(oh.sum(), 1.0)
        assert oh[rel] == 1.0
        # check coefficient value matches graph's stored coeff for undirected pair
        ui, uj = (i, j) if i < j else (j, i)
        if (ui, uj) in g.edges:
            coeffs = g.get_edge(ui, uj)
            bias_name = relation_names[rel]
            expected_val = float(getattr(coeffs, bias_name))
            actual_val = float(data.edge_attr[idx, -1].item())
            assert np.isclose(actual_val, expected_val)

    # For every undirected allowed pair in g.edge_mask, ensure both directed edges exist
    expected_directed = set()
    for (i, j), mask in g.edge_mask.items():
        for bias, allowed in mask.items():
            if not allowed:
                continue
            r = rel_map[bias]
            expected_directed.add((i, j, r))
            expected_directed.add((j, i, r))

    # expected_directed should be subset of actual (they should match)
    assert expected_directed.issubset(actual)
    # and at least one relation exists
    assert len(actual) > 0
