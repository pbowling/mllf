from pathlib import Path

from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.rl.graph import Graph


def test_build_graph_from_example_print():
    """Build a Graph from the example RTF files and print summary for manual inspection.

    This test is intentionally verbose so you can confirm the node metadata and
    connectivity masks match expectations described in the project.
    """
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / 'examples' / 'cb' / '14benz_solv_5.5'
    assert example_dir.exists(), f"Example directory not found: {example_dir}"

    # parse rtf files from the example directory (this will include siteX_subY files)
    rtf_results = parse_rtf_dir(str(example_dir))
    print('\nParsed RTF keys:', sorted(list(rtf_results.keys())))

    # build graph
    g = Graph.from_rtf_results(rtf_results)

    print(f'Graph: num_nodes={g.num_nodes}, num_edges={len(g.edges)}')

    # print node metadata for each node
    for n in range(g.num_nodes):
        info = g.get_node_info(n)
        print(f'Node {n}: site={info.get("site")}, sub={info.get("sub")}, total_charge={info.get("total_charge")}, distinct_atoms={len(info.get("distinct_atom_types", []))}')

    # print allowed edges per bias
    for bias in ('linear', 'quadratic', 'skew', 'end'):
        allowed = g.get_allowed_edges_for_bias(bias)
        print(f'Allowed edges for {bias} ({len(allowed)}): {sorted(allowed)}')

    # sanity assertions (basic)
    assert g.num_nodes > 0
    # ensure at least one linear edge exists when subs per site > 1
    lin = g.get_allowed_edges_for_bias('linear')
    # don't force a strict number; just assert mask structure is present (list)
    assert isinstance(lin, list)

    # --- explicit connectivity assertions ---
    # build mapping site -> list of node indices
    site_nodes = {}
    for n in range(g.num_nodes):
        info = g.get_node_info(n)
        s = info.get('site')
        site_nodes.setdefault(s, []).append(n)

    # expected: no inter-site edges (verify by sampling one inter-site pair if multiple sites)
    sites = sorted(k for k in site_nodes.keys() if k is not None)
    if len(sites) > 1:
        a = site_nodes[sites[0]][0]
        b = site_nodes[sites[1]][0]
        # inter-site masks should be all False
        for bias in ('linear', 'quadratic', 'skew', 'end'):
            assert (a, b) not in g.get_allowed_edges_for_bias(bias)

    # linear: only edges connecting sub==1 to other subs within same site
    expected_linear = set()
    for s, nodes in site_nodes.items():
        # find node with sub == 1
        sub1 = None
        for n in nodes:
            if g.get_node_info(n).get('sub') == 1:
                sub1 = n
                break
        if sub1 is None:
            continue
        for n in nodes:
            if n == sub1:
                continue
            i, j = (sub1, n) if sub1 < n else (n, sub1)
            expected_linear.add((i, j))

    assert set(g.get_allowed_edges_for_bias('linear')) == expected_linear

    # quadratic/skew/end: fully connected among nodes within each site
    def complete_pairs(nodes):
        out = set()
        nodes = sorted(nodes)
        for i_idx in range(len(nodes)):
            for j_idx in range(i_idx + 1, len(nodes)):
                out.add((nodes[i_idx], nodes[j_idx]))
        return out

    expected_quadratic = set()
    for nodes in site_nodes.values():
        expected_quadratic.update(complete_pairs(nodes))

    assert set(g.get_allowed_edges_for_bias('quadratic')) == expected_quadratic
    assert set(g.get_allowed_edges_for_bias('skew')) == expected_quadratic
    assert set(g.get_allowed_edges_for_bias('end')) == expected_quadratic
