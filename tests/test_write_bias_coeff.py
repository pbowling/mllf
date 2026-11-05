"""variables.py written from an old-style .inp: structure and sign convention."""
from pathlib import Path

import numpy as np
import yaml

from mllf.file_handling.write_bias_coeff import write_variables_py_from_inp


def _get_example_path(filename):
    """Get absolute path to example file, relative to repository root."""
    test_file = Path(__file__)
    repo_root = test_file.parent.parent
    return repo_root / "tests" / "samples" / filename


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
    after = s[end + 3:]
    dup_lines = [
        ln for ln in after.splitlines() if ln.strip().startswith(
            ('lams', 'cs', 'xs', 'ss'))]
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
    """write_variables_py_from_inp and the collector's parse_bias_from_inp give the same
    matrices."""
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
