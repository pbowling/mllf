"""Tests for collect_pretraining_data CLI helpers and related utilities.

Covers:
  - find_run_directories      (flat MSLD and nested combo structures)
  - parse_bias_from_py        (variables*.py → matrix keys only)
  - parse_bias_from_inp       (variables*.inp → matrix format)
  - detect_solvent_state      (path-based heuristic)
  - _extract_nsubs (indirectly via read_bias_coeff.parse_new)
  - _read_box_from_prep_script (box= extraction from prep .py file)
  - filter_best_runs_per_system (grouping & best-reward selection)
"""
import json
import math
import textwrap
from pathlib import Path

import pytest
import torch

from mllf.cli.collect_pretraining_data import (
    find_run_directories,
    parse_bias_from_py,
    parse_bias_from_inp,
    detect_solvent_state,
)
from mllf.cb.aev_processor import _read_box_from_prep_script
from mllf.cb.pretrain_policy import filter_best_runs_per_system
from mllf.file_handling.read_bias_coeff import read_bias_coeff


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_run_dir(base: Path, name: str, has_vars: bool = True, has_output: bool = True,
                  failed: bool = False) -> Path:
    """Create a minimal run directory with optional stub files."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    if has_vars:
        (d / 'variables.py').write_text('# stub\n')
    if has_output:
        (d / 'output').write_text('stub\n')
    return d


def _variables_py_content(n: int = 3) -> str:
    """Minimal variables.py with an n-block 1-site bias_string."""
    # Build b as a YAML nested list [[0.0, v1, v2, ...]]
    vals = ', '.join(f'{i * 0.1:.1f}' for i in range(n))
    row = ', '.join('0.0' for _ in range(n))
    rows = '\n'.join(f'- [{row}]' for _ in range(n))
    lines = [
        'bias_string = """',
        'b:',
        f'- [{vals}]',
        'c:',
        rows,
        'x:',
        rows,
        's:',
        rows,
        '"""',
        f'nsubs = [{n}]',
    ]
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# TestFindRunDirectories
# ---------------------------------------------------------------------------

class TestFindRunDirectories:
    """Test find_run_directories for flat and nested combo structures."""

    def test_flat_structure_returns_run_dirs(self, tmp_path):
        """Finds run# directories at the top level."""
        _make_run_dir(tmp_path, 'run1')
        _make_run_dir(tmp_path, 'run2')
        _make_run_dir(tmp_path, 'run3')
        result = find_run_directories(tmp_path)
        assert len(result) == 3
        assert all(d.name.startswith('run') for d in result)

    def test_flat_sorted_numerically(self, tmp_path):
        """Flat structure is sorted by run number, not lexicographically."""
        _make_run_dir(tmp_path, 'run10')
        _make_run_dir(tmp_path, 'run2')
        _make_run_dir(tmp_path, 'run1')
        result = find_run_directories(tmp_path)
        nums = [int(d.name.replace('run', '')) for d in result]
        assert nums == sorted(nums)

    def test_flat_excludes_failed(self, tmp_path):
        """Directories containing '_failed' in the name are excluded."""
        _make_run_dir(tmp_path, 'run1')
        _make_run_dir(tmp_path, 'run2_failed')
        result = find_run_directories(tmp_path)
        names = [d.name for d in result]
        assert 'run1' in names
        assert 'run2_failed' not in names

    def test_flat_excludes_dirs_without_data(self, tmp_path):
        """Directories without variables or output files are skipped."""
        _make_run_dir(tmp_path, 'run1')
        empty_run = tmp_path / 'run2'
        empty_run.mkdir()
        result = find_run_directories(tmp_path)
        assert len(result) == 1
        assert result[0].name == 'run1'

    def test_combo_structure_returns_run_dirs(self, tmp_path):
        """Nested comb_*/run_*/ structure: run dirs from inside each combo."""
        for combo in ('comb_001', 'comb_002'):
            for run in ('run_001', 'run_002'):
                _make_run_dir(tmp_path / combo, run)
        result = find_run_directories(tmp_path)
        assert len(result) == 4
        assert all(d.name.startswith('run_') for d in result)

    def test_combo_excludes_failed(self, tmp_path):
        """Failed run dirs inside combo dirs are skipped."""
        _make_run_dir(tmp_path / 'comb_001', 'run_001')
        _make_run_dir(tmp_path / 'comb_001', 'run_002_failed')
        result = find_run_directories(tmp_path)
        assert len(result) == 1
        assert result[0].name == 'run_001'

    def test_returns_empty_for_empty_dir(self, tmp_path):
        """Empty directory → empty list (no crash)."""
        assert find_run_directories(tmp_path) == []


# ---------------------------------------------------------------------------
# TestParseBiasFromPy
# ---------------------------------------------------------------------------

class TestParseBiasFromPy:
    """Test parse_bias_from_py: delegates to read_bias_coeff & strips scalar dicts."""

    def test_returns_matrix_keys(self, tmp_path):
        """Result contains b, c, x, s keys."""
        f = tmp_path / 'variables.py'
        f.write_text(_variables_py_content(3))
        result = parse_bias_from_py(f)
        assert set(result.keys()) >= {'b', 'c', 'x', 's'}

    def test_b_is_flat_list(self, tmp_path):
        """b vector is a flat 1-D Python list after normalization."""
        f = tmp_path / 'variables.py'
        f.write_text(_variables_py_content(4))
        result = parse_bias_from_py(f)
        b = result['b']
        assert isinstance(b, list)
        assert len(b) == 4
        assert all(isinstance(v, float) for v in b)

    def test_c_matrix_dimensions(self, tmp_path):
        """c matrix is n×n for an n-substituent system."""
        n = 5
        f = tmp_path / 'variables.py'
        f.write_text(_variables_py_content(n))
        result = parse_bias_from_py(f)
        c = result['c']
        assert len(c) == n
        assert all(len(row) == n for row in c)

    def test_scalar_subdicts_stripped(self, tmp_path):
        """lams/cs/xs/ss scalar subdicts are NOT present in the returned dict."""
        f = tmp_path / 'variables.py'
        f.write_text(_variables_py_content(3))
        result = parse_bias_from_py(f)
        for key in ('lams', 'cs', 'xs', 'ss'):
            assert key not in result

    def test_raises_when_b_missing(self, tmp_path):
        """Raises ValueError when no bias_string can be parsed."""
        f = tmp_path / 'variables.py'
        f.write_text('# no bias_string here\nx = 1\n')
        with pytest.raises(ValueError, match="Could not parse bias vector"):
            parse_bias_from_py(f)

    def test_parses_real_example_file(self):
        """Parses the checked-in tests/fixtures/variables.py without error."""
        example = Path(__file__).parent / 'samples' / 'variables.py'
        if not example.exists():
            pytest.skip('tests/samples/variables.py not found')
        result = parse_bias_from_py(example)
        assert 'b' in result and isinstance(result['b'], list)
        assert len(result['b']) > 0


# ---------------------------------------------------------------------------
# TestParseBiasFromInp
# ---------------------------------------------------------------------------

class TestParseBiasFromInp:
    """Test parse_bias_from_inp: converts old .inp scalar format to matrices."""

    def _write_inp(self, path: Path, n: int = 2) -> None:
        """Write a minimal variables.inp with n sites x n subs each."""
        lines = []
        for s in range(1, n + 1):
            for sub in range(1, n + 1):
                lines.append(f'set lams{s}s{sub} = {(s - 1) * n + sub - 1:.2f}')
        for s in range(1, n + 1):
            for sub_i in range(1, n + 1):
                for sub_j in range(1, n + 1):
                    val = (s * 10 + sub_i + sub_j) * 0.1
                    lines.append(f'set cs{s}s{sub_i}s{s}s{sub_j} = {val:.4f}')
        path.write_text('\n'.join(lines) + '\n')

    def test_returns_required_keys(self, tmp_path):
        """Result contains b, c, x, s, and _nsubs_per_site keys."""
        f = tmp_path / 'variables85.inp'
        self._write_inp(f, n=2)
        result = parse_bias_from_inp(f)
        assert set(result.keys()) >= {'b', 'c', 'x', 's', '_nsubs_per_site'}

    def test_b_length_matches_total_subs(self, tmp_path):
        """b has one entry per substituent block (n_sites × n_subs)."""
        n = 2
        f = tmp_path / 'variables.inp'
        self._write_inp(f, n=n)
        result = parse_bias_from_inp(f)
        # 2 sites × 2 subs = 4 blocks
        assert len(result['b']) == n * n

    def test_c_matrix_square(self, tmp_path):
        """c is a square (n_blocks × n_blocks) matrix."""
        n = 2
        f = tmp_path / 'variables.inp'
        self._write_inp(f, n=n)
        result = parse_bias_from_inp(f)
        c = result['c']
        assert len(c) == n * n
        assert all(len(row) == n * n for row in c)

    def test_parses_real_example_inp(self):
        """Parses the checked-in tests/fixtures/variables85.inp."""
        example = Path(__file__).parent / 'samples' / 'variables85.inp'
        if not example.exists():
            pytest.skip('tests/samples/variables85.inp not found')
        result = parse_bias_from_inp(example)
        assert 'b' in result
        assert len(result['b']) > 0


# ---------------------------------------------------------------------------
# TestDetectSolventState
# ---------------------------------------------------------------------------

class TestDetectSolventState:
    """Test the path-based solvent state heuristic."""

    @pytest.mark.parametrize('path_fragment,expected', [
        ('pretraining/1_FAAH_solvent_group1/run1', 'solv'),
        ('pretraining/14benz_solv/run001', 'solv'),
        ('pretraining/aq_system/run1', 'solv'),
        ('pretraining/1benz_vac_group1/run1', 'gas'),
        ('pretraining/vacuum_system/run1', 'gas'),
        ('pretraining/gas_phase/run1', 'gas'),
        ('pretraining/1_FAAH_protein_group1/run_001', 'protein'),
        ('pretraining/abl_prot_runs/run_001', 'protein'),
        ('pretraining/generic_system/run1', 'unknown'),
    ])
    def test_heuristic(self, tmp_path, path_fragment, expected):
        """detect_solvent_state returns expected category for known path patterns."""
        run_dir = tmp_path / path_fragment
        run_dir.mkdir(parents=True, exist_ok=True)
        assert detect_solvent_state(run_dir) == expected


# ---------------------------------------------------------------------------
# TestExtractNsubsViaReadBiasCoeff
# ---------------------------------------------------------------------------

class TestExtractNsubsViaReadBiasCoeff:
    """Test _extract_nsubs indirectly through read_bias_coeff / parse_new.

    parse_new returns _nsubs_per_site in the result dict whenever
    _extract_nsubs can infer it from the file content.
    """

    def _write_vars(self, path: Path, content_extra: str = '', n: int = 3) -> None:
        """Write a variables.py with optional extra content before bias_string."""
        rows = '\n'.join(f'- [{", ".join("0.0" for _ in range(n))}]' for _ in range(n))
        bias = textwrap.dedent(f"""\
            b:
            - {" ".join(["- 0.0"] + [f"- {i * 0.1:.1f}" for i in range(1, n)])}
            c:
            {rows}
            x:
            {rows}
            s:
            {rows}
            """)
        path.write_text(content_extra + f'\nbias_string = """\n{bias}"""\n')

    def test_nsubs_from_module_level_assignment(self, tmp_path):
        """Finds nsubs via top-level `nsubs = [...]` in the file."""
        f = tmp_path / 'variables.py'
        self._write_vars(f, content_extra='nsubs = [3, 4]\n', n=7)
        data = read_bias_coeff(str(f))
        assert data.get('_nsubs_per_site') == [3, 4]

    def test_nsubs_from_alf_info_string(self, tmp_path):
        """Finds nsubs from embedded alf_info_string YAML block."""
        alf_yaml = textwrap.dedent("""\
            alf_info_string = \"\"\"
            nsubs:
            - 5
            - 6
            \"\"\"
            """)
        f = tmp_path / 'variables.py'
        self._write_vars(f, content_extra=alf_yaml, n=11)
        data = read_bias_coeff(str(f))
        assert data.get('_nsubs_per_site') == [5, 6]

    def test_nsubs_from_alf_info_py(self, tmp_path):
        """Finds nsubs from alf_info.py in the same directory."""
        alf_info = tmp_path / 'alf_info.py'
        alf_info.write_text("alf_info['nsubs'] = [2, 3]\n")
        f = tmp_path / 'variables.py'
        self._write_vars(f, n=5)
        data = read_bias_coeff(str(f), prep_dir=tmp_path)
        assert data.get('_nsubs_per_site') == [2, 3]

    def test_nsubs_absent_when_not_found(self, tmp_path):
        """Returns no _nsubs_per_site key when none of the sources are present."""
        f = tmp_path / 'variables.py'
        self._write_vars(f)   # no alf_info, no nsubs line, no alf_info_string
        data = read_bias_coeff(str(f))
        # No nsubs information in the file → key absent or None
        assert data.get('_nsubs_per_site') is None


# ---------------------------------------------------------------------------
# TestReadBoxFromPrepScript
# ---------------------------------------------------------------------------

class TestReadBoxFromPrepScript:
    """Test _read_box_from_prep_script via the public extract_environment path."""

    def test_reads_box_from_py_file(self, tmp_path):
        """Parses `box = 40.0` from a prep .py file."""
        (tmp_path / 'prep.py').write_text('box = 40.0\n# comment\n')
        box = _read_box_from_prep_script(tmp_path)
        assert box is not None
        assert math.isclose(box, 40.0, rel_tol=1e-9)

    def test_reads_box_integer_value(self, tmp_path):
        """Parses `box = 35` (integer, no decimal point) correctly."""
        (tmp_path / 'prep.py').write_text('box = 35\n')
        box = _read_box_from_prep_script(tmp_path)
        assert box is not None
        assert math.isclose(box, 35.0, rel_tol=1e-9)

    def test_returns_none_when_no_py_file(self, tmp_path):
        """Returns None when no .py files are present."""
        assert _read_box_from_prep_script(tmp_path) is None

    def test_skips_alf_info_py(self, tmp_path):
        """alf_info.py is excluded; returns None when it is the only .py."""
        (tmp_path / 'alf_info.py').write_text("alf_info['nsubs'] = [3]\nbox = 99.0\n")
        # Only alf_info.py present; should be skipped
        assert _read_box_from_prep_script(tmp_path) is None

    def test_returns_none_when_box_line_absent(self, tmp_path):
        """Returns None when a .py file exists but contains no `box = ...` line."""
        (tmp_path / 'prep.py').write_text('nsubs = [3]\n# no box assignment\n')
        assert _read_box_from_prep_script(tmp_path) is None


# ---------------------------------------------------------------------------
# TestFilterBestRunsPerSystem
# ---------------------------------------------------------------------------

def _make_pretraining_run(tmp_path: Path, parent_parts: tuple, run_name: str,
                          n_transitions: int = 50) -> dict:
    """Build a minimal run dict with enough data for reward computation."""
    run_dir = tmp_path
    for part in parent_parts:
        run_dir = run_dir / part
    run_dir = run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # graph_info.json with 1 site × 3 subs
    graph_info = {
        'sites': {f'site1_sub{i}': {'site': 1, 'sub': i}
                  for i in range(1, 4)},
        'solvent_state': 'solv',
    }
    (run_dir / 'graph_info.json').write_text(json.dumps(graph_info))
    # sim_results: format matching collect_run_data / compute_reward_from_sim_results expectations
    # populations: {block_str: {"counts": {lambda_str: count}, "site": site_int}}
    pops = {
        str(i): {"counts": {"0.95": 100, "0.99": 100}, "site": 1}
        for i in range(1, 4)
    }
    # transitions: {site_str: {lambda_str: count}}
    trans = {"1": {"0.95": n_transitions, "0.99": n_transitions}}
    return {
        'run_dir': run_dir,
        'source_dir': str(run_dir),
        'metadata': {'num_sites': 1, 'num_substituents': 3},
        'sim_results': {'populations': pops, 'transitions': trans},
    }


class TestFilterBestRunsPerSystem:
    """Test filter_best_runs_per_system grouping and best-run selection."""

    def test_single_run_per_system_always_selected(self, tmp_path):
        """One run per system → all runs returned unchanged."""
        runs = [
            _make_pretraining_run(tmp_path, ('14benz_solv',), 'run_001'),
            _make_pretraining_run(tmp_path, ('indole_solv',), 'run_001'),
        ]
        result = filter_best_runs_per_system(runs)
        assert len(result) == 2

    def test_best_run_selected_per_flat_system(self, tmp_path):
        """Two runs from the same flat parent dir → only the best-reward one kept."""
        # Use a neutral grandparent name that doesn't contain 'best' or 'combo'
        # (those keywords trigger special legacy-combo grouping in the function).
        base = tmp_path / 'pretraining'
        low  = _make_pretraining_run(base, ('14benz_solv',), 'run_001', n_transitions=0)
        high = _make_pretraining_run(base, ('14benz_solv',), 'run_002', n_transitions=100)
        result = filter_best_runs_per_system([low, high])
        assert len(result) == 1
        assert result[0]['run_dir'].name == 'run_002'

    def test_combo_structure_grouped_per_combo(self, tmp_path):
        """Runs inside comb_* parent dirs are grouped per combo, not per grandparent."""
        run_a1 = _make_pretraining_run(tmp_path, ('dataset', 'comb_001'), 'run_001',
                                       n_transitions=0)
        run_a2 = _make_pretraining_run(tmp_path, ('dataset', 'comb_001'), 'run_002',
                                       n_transitions=100)
        run_b  = _make_pretraining_run(tmp_path, ('dataset', 'comb_002'), 'run_001',
                                       n_transitions=50)
        result = filter_best_runs_per_system([run_a1, run_a2, run_b])
        # 2 distinct combos → 2 runs selected (best from comb_001 + only from comb_002)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        """Empty run list → empty result (no crash)."""
        assert filter_best_runs_per_system([]) == []
