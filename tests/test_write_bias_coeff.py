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
	return repo_root / "tests" / "samples" / filename


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
    
    # ensure bias = yaml.safe_load(bias_string) line is present
    assert 'bias = yaml.safe_load(bias_string)' in s, "Missing yaml.safe_load line"

# ---------------------------------------------------------------------------
# Sign convention: flat cs/xs/ss (CHARMM, .inp) = -matrix c/x/s; lams = b.
# ---------------------------------------------------------------------------

def _load_bias_string(path):
    import re
    text = Path(path).read_text(encoding='utf-8')
    m = re.search(r'bias_string\s*=\s*"""(.*?)"""', text, re.S)
    return yaml.safe_load(m.group(1))


def _two_site_graph(lin=1.5, quad=2.0, skew=0.5, end=0.3):
    g = Graph(2)
    g.set_edge(0, 1, [lin, quad, skew, end])
    return g


def test_inp_writer_negates_nonlinear_terms(tmp_path):
    """Graph coefficients are matrix convention; the .inp must hold what CHARMM gets."""
    from mllf.file_handling.read_bias_coeff import parse_old
    out = tmp_path / "variables.inp"
    write_bias_inp_from_graph(_two_site_graph(), str(out), sub_counts=[1, 1])
    d = parse_old(str(out))
    assert d['lams']['lams1s1'] == 1.5 and d['lams']['lams2s1'] == 1.5    # linear: same sign
    assert d['cs']['cs1s1s2s1'] == -2.0                                   # quadratic: negated
    assert d['xs']['xs1s1s2s1'] == -0.5 and d['xs']['xs2s1s1s1'] == -0.5  # skew: negated
    assert d['ss']['ss1s1s2s1'] == -0.3 and d['ss']['ss2s1s1s1'] == -0.3  # end: negated


def test_inp_writer_round_trips_through_collector_reader(tmp_path):
    """graph -> .inp -> parse_bias_from_inp recovers the original (matrix) coefficients."""
    from mllf.cli.collect_pretraining_data import parse_bias_from_inp
    out = tmp_path / "variables.inp"
    write_bias_inp_from_graph(_two_site_graph(), str(out), sub_counts=[1, 1])
    r = parse_bias_from_inp(out)
    assert np.allclose(r['b'], [1.5, 1.5])
    assert r['c'][0][1] == 2.0
    assert r['x'][0][1] == 0.5 and r['x'][1][0] == 0.5
    assert r['s'][0][1] == 0.3 and r['s'][1][0] == 0.3


def test_inp_writer_zero_coefficients_not_written_as_negative_zero(tmp_path):
    out = tmp_path / "variables.inp"
    write_bias_inp_from_graph(_two_site_graph(quad=0.0, skew=0.0, end=0.0), str(out), sub_counts=[1, 1])
    assert '-0.00000000' not in out.read_text()


def test_variables_py_from_inp_uses_matrix_signs(tmp_path):
    """Matrices negate the flat cs/xs/ss (b unchanged); flat scalars kept as written."""
    inp = _get_example_path('variables85.inp')
    if not inp.exists():
        import pytest
        pytest.skip(f"Example file not found: {inp}")
    out = tmp_path / "variables.py"
    write_variables_py_from_inp(str(inp), str(out))
    bias = _load_bias_string(out)
    # sample file: lams1s2 = 0.13, cs1s1s1s2 = -39.95, xs1s1s1s2 = -5.93, xs1s2s1s1 = -1.46
    b = bias['b'][0] if isinstance(bias['b'][0], list) else bias['b']
    assert b[1] == 0.13
    assert bias['c'][0][1] == 39.95
    assert bias['x'][0][1] == 5.93 and bias['x'][1][0] == 1.46
    assert bias['cs1s1s1s2'] == -39.95 and bias['xs1s1s1s2'] == -5.93   # flat keys as written
    assert bias['_nonlinear_sign_convention'] == 'matrix'


def test_variables_py_from_inp_end_term_sign(tmp_path):
    inp = tmp_path / "variables.inp"
    inp.write_text("set lams1s1 = 0.00\nset lams1s2 = -2.00\n"
                   "set ss1s1s1s2 = -1.10\nset ss1s2s1s1 = 0.25\n")
    out = tmp_path / "variables.py"
    write_variables_py_from_inp(str(inp), str(out))
    bias = _load_bias_string(out)
    assert bias['s'][0][1] == 1.10 and bias['s'][1][0] == -0.25


def test_both_inp_readers_agree(tmp_path):
    """write_variables_py_from_inp and the collector's parse_bias_from_inp give the same matrices."""
    from mllf.cli.collect_pretraining_data import parse_bias_from_inp
    inp = _get_example_path('variables85.inp')
    if not inp.exists():
        import pytest
        pytest.skip(f"Example file not found: {inp}")
    out = tmp_path / "variables.py"
    write_variables_py_from_inp(str(inp), str(out))
    via_py = _load_bias_string(out)
    via_collector = parse_bias_from_inp(inp)
    b_py = via_py['b'][0] if isinstance(via_py['b'][0], list) else via_py['b']
    assert np.allclose(b_py, via_collector['b'])
    for key in ('c', 'x', 's'):
        assert np.allclose(np.array(via_py[key]), np.array(via_collector[key])), key
