import os
from mllf.rl.graph import Graph
from mllf.file_handling.write_bias_coeff import write_bias_inp_from_graph


def _read_set_names(path):
    names = []
    with open(path, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            if not ln.startswith('set '):
                continue
            # extract parameter name (between 'set ' and ' =')
            parts = ln.split('=')
            left = parts[0].strip()
            _, name = left.split(None, 1)
            # only keep actual bias coefficient names (lams, cs, ss, xs)
            if name.startswith(('lams', 'cs', 'ss', 'xs')):
                names.append(name)
    return names


def test_write_matches_example(tmp_path):
    # Build a simple graph with 5 sites and dummy coeffs (values don't matter for names)
    g = Graph(5)
    # arbitrary coeffs
    for i in range(5):
        for j in range(i+1, 5):
            g.set_edge(i, j, [0.0, 0.0, 0.0, 0.0])

    subs = [3, 4, 8, 8, 8]
    out = tmp_path / "generated.inp"
    write_bias_inp_from_graph(g, str(out), sub_counts=subs)

    example = os.path.join('examples', 'rl', 'variables85.inp')
    gen_names = set(_read_set_names(str(out)))
    ex_names = set(_read_set_names(example))

    # assert generated parameter names equal the example's parameter names
    assert gen_names == ex_names
