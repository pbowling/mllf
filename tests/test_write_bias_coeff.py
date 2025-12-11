import os
from pathlib import Path
import yaml
import numpy as np
from mllf.cb.graph import Graph
from mllf.file_handling.write_bias_coeff import write_bias_inp_from_graph, write_variables_py_from_inp


def _get_example_path(filename):
	"""Get absolute path to example file, relative to repository root."""
	test_file = Path(__file__)
	repo_root = test_file.parent.parent
	return repo_root / "examples" / "cb" / filename


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
    example = _get_example_path('variables85.inp')
    
    if not example.exists():
        import pytest
        pytest.skip(f"Example file not found: {example}")
    
    # use the example file as the header source so non-bias lines are copied verbatim
    write_bias_inp_from_graph(g, str(out), sub_counts=subs, header_source=str(example))
    gen_names = set(_read_set_names(str(out)))
    ex_names = set(_read_set_names(str(example)))

    # assert generated parameter names equal the example's parameter names
    assert gen_names == ex_names


def test_variables_py_contains_bias(tmp_path):
    """Generate a variables.py from an .inp and assert:

    - a triple-quoted `bias_string` exists
    - the YAML inside contains keys 'b','c','x','s'
    - b flattens to the same length as the c/x/s matrices
    - no textual scalar lines (lams/cs/xs/ss) appear after the closing triple quotes
    """

    inp = _get_example_path('variables85.inp')
    
    if not inp.exists():
        import pytest
        pytest.skip(f"Example file not found: {inp}")
    
    out = tmp_path / "variables.py"
    write_variables_py_from_inp(str(inp), str(out))

    s = out.read_text(encoding='utf-8')
    assert 'bias_string="""' in s

    start = s.find('bias_string="""') + len('bias_string="""')
    end = s.find('"""', start)
    assert end != -1, "closing triple quotes for bias_string not found"

    bias_text = s[start:end]
    bias = yaml.load(bias_text, Loader=yaml.Loader)

    # ensure required keys exist
    for key in ('b', 'c', 'x', 's'):
        assert key in bias

    # flatten b and compare with c matrix shape
    flat_b = []
    for row in bias.get('b', []):
        flat_b.extend(row if isinstance(row, list) else [row])

    c = np.array(bias.get('c', []))
    assert c.ndim == 2 and c.shape[0] == c.shape[1]
    assert len(flat_b) == c.shape[0]

    # ensure no duplicate scalar lines after the bias_string closing quotes
    after = s[end + 3 :]
    dup_lines = [ln for ln in after.splitlines() if ln.strip().startswith(('lams', 'cs', 'xs', 'ss'))]
    assert dup_lines == []