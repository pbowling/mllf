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


# ---------------------------------------------------------------------------
# TestBuildEdgeWeights
# ---------------------------------------------------------------------------

class TestBuildEdgeWeights:
    """Tests for build_edge_weights in mllf.cb.workflow_utils."""

    def setup_method(self):
        from mllf.cb.workflow_utils import build_edge_weights
        self.build_edge_weights = build_edge_weights
        self.device = torch.device('cpu')

    def _edge_index(self, edges):
        """Build a [2, E] tensor from a list of (src, dst) tuples."""
        return torch.tensor(edges, dtype=torch.long).T

    def test_empty_ddg_pairs_returns_all_ones(self):
        """When ddg_pairs is empty, every edge gets weight 1.0."""
        ei = self._edge_index([(0, 1), (1, 0), (0, 2)])
        w = self.build_edge_weights(ei, {}, 0.2, self.device)
        assert w.shape == (3,)
        assert torch.all(w == 1.0)

    def test_none_entry_gets_no_transition_weight(self):
        """Pair with None in ddg_pairs (NaN DDG) receives no_transition_weight."""
        # node 0 → block 2, node 1 → block 3; key "2_3"
        ei = self._edge_index([(0, 1)])
        w = self.build_edge_weights(ei, {"2_3": None}, 0.2, self.device)
        assert torch.isclose(w[0], torch.tensor(0.2))

    def test_finite_entry_gets_full_weight(self):
        """Pair with a finite float (transitions observed) receives weight 1.0."""
        ei = self._edge_index([(0, 1)])
        w = self.build_edge_weights(ei, {"2_3": -1.23}, 0.2, self.device)
        assert torch.isclose(w[0], torch.tensor(1.0))

    def test_missing_key_treated_as_full_weight(self):
        """Pair absent from ddg_pairs (old data) receives weight 1.0."""
        ei = self._edge_index([(0, 1)])
        # no "2_3" key in dict
        w = self.build_edge_weights(ei, {"3_4": None}, 0.2, self.device)
        assert torch.isclose(w[0], torch.tensor(1.0))

    def test_direction_independent(self):
        """Edge (1, 0) and (0, 1) should produce the same weight (lo/hi normalised)."""
        ddg = {"2_3": None}
        w_fwd = self.build_edge_weights(self._edge_index([(0, 1)]), ddg, 0.3, self.device)
        w_rev = self.build_edge_weights(self._edge_index([(1, 0)]), ddg, 0.3, self.device)
        assert torch.isclose(w_fwd[0], w_rev[0])

    def test_mixed_edges(self):
        """Multiple edges with different ddg_pairs entries are weighted correctly."""
        # node 0→blk2, node 1→blk3, node 2→blk4
        ddg = {"2_3": None, "2_4": -0.5}
        ei = self._edge_index([(0, 1), (0, 2), (1, 2)])
        w = self.build_edge_weights(ei, ddg, 0.1, self.device)
        assert torch.isclose(w[0], torch.tensor(0.1))   # 2_3 → None
        assert torch.isclose(w[1], torch.tensor(1.0))   # 2_4 → finite
        assert torch.isclose(w[2], torch.tensor(1.0))   # 3_4 → missing → 1.0

    def test_custom_no_transition_weight(self):
        """Respects the caller-supplied no_transition_weight."""
        ei = self._edge_index([(0, 1)])
        w = self.build_edge_weights(ei, {"2_3": None}, 0.5, self.device)
        assert torch.isclose(w[0], torch.tensor(0.5))

    def test_output_dtype_float32(self):
        """Output tensor has dtype float32."""
        ei = self._edge_index([(0, 1)])
        w = self.build_edge_weights(ei, {}, 0.2, self.device)
        assert w.dtype == torch.float32


# ---------------------------------------------------------------------------
# TestParseSimulationMetrics
# ---------------------------------------------------------------------------

class TestParseSimulationMetrics:
    """Tests for parse_simulation_metrics in mllf.cb.workflow_utils."""

    def setup_method(self):
        from mllf.cb.workflow_utils import parse_simulation_metrics
        self.parse_simulation_metrics = parse_simulation_metrics

    def test_returns_ddg_pairs_key(self, tmp_path):
        """Result dict always contains 'ddg_pairs' key."""
        # Write a minimal output file with no DDG content
        f = tmp_path / 'output.out'
        f.write_text('nothing here\n')
        result = self.parse_simulation_metrics(f)
        assert 'ddg_pairs' in result

    def test_ddg_pairs_empty_for_no_section(self, tmp_path):
        """ddg_pairs is empty dict when output has no SINGLE DDG section."""
        f = tmp_path / 'output.out'
        f.write_text('nothing here\n')
        result = self.parse_simulation_metrics(f)
        assert result['ddg_pairs'] == {}

    def test_ddg_pairs_populated_from_real_sample(self):
        """ddg_pairs from the sample file contains the expected pairs."""
        sample = Path(__file__).parent / 'samples' / '14benz_solv_5.5' / 'output.txt'
        if not sample.exists():
            pytest.skip('Sample output file not found')
        result = self.parse_simulation_metrics(sample)
        assert 'ddg_pairs' in result
        # Sample file has NaN/Inf for all pairs → all None
        assert len(result['ddg_pairs']) > 0
        assert all(v is None for v in result['ddg_pairs'].values())

    def test_ddg_pairs_keys_are_strings(self, tmp_path):
        """ddg_pairs keys follow "lo_hi" string format (JSON-serialisable)."""
        sample = Path(__file__).parent / 'samples' / '14benz_solv_5.5' / 'output.txt'
        if not sample.exists():
            pytest.skip('Sample output file not found')
        result = self.parse_simulation_metrics(sample)
        for key in result['ddg_pairs']:
            assert isinstance(key, str)
            parts = key.split('_')
            assert len(parts) == 2
            assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# TestBackfillDdgPairs
# ---------------------------------------------------------------------------

class TestBackfillDdgPairs:
    """Tests for backfill_ddg_pairs in mllf.cli.collect_pretraining_data."""

    def setup_method(self):
        from mllf.cli.collect_pretraining_data import backfill_ddg_pairs
        self.backfill_ddg_pairs = backfill_ddg_pairs

    def _make_run(self, root: Path, name: str, source_output: str,
                  already_has_ddg: bool = False) -> Path:
        """Create a pretraining run dir with metadata.json + simulation_results.json."""
        run_dir = root / name
        run_dir.mkdir(parents=True)
        source_dir = root / f'{name}_source'
        source_dir.mkdir(parents=True)
        (source_dir / 'output.out').write_text(source_output)

        sim = {'populations': [], 'transitions': [], 'terminated_normally': True}
        if already_has_ddg:
            sim['ddg_pairs'] = {}
        (run_dir / 'simulation_results.json').write_text(json.dumps(sim))

        metadata = {'source_run_dir': str(source_dir)}
        (run_dir / 'metadata.json').write_text(json.dumps(metadata))
        return run_dir

    def _simple_ddg_output(self):
        """Minimal CHARMM output with a SINGLE DDG block (all NaN)."""
        return (
            "             BLK(I)..BLK(J).....> 0.950 ....> 0.990 .....> 0.950 ....> 0.990\n"
            "SINGLE DDG>       2      3         NaN         NaN         NaN         NaN\n"
        )

    def test_updates_runs_without_ddg_pairs(self, tmp_path):
        """Runs missing 'ddg_pairs' in simulation_results.json are updated."""
        self._make_run(tmp_path, 'run1', self._simple_ddg_output())
        n_updated, n_skipped = self.backfill_ddg_pairs(tmp_path)
        assert n_updated == 1
        assert n_skipped == 0
        sim = json.loads((tmp_path / 'run1' / 'simulation_results.json').read_text())
        assert 'ddg_pairs' in sim

    def test_skips_runs_already_having_ddg_pairs(self, tmp_path):
        """Runs that already have 'ddg_pairs' are not reprocessed."""
        self._make_run(tmp_path, 'run1', self._simple_ddg_output(), already_has_ddg=True)
        n_updated, n_skipped = self.backfill_ddg_pairs(tmp_path)
        assert n_updated == 0
        assert n_skipped == 1

    def test_dry_run_does_not_modify_files(self, tmp_path):
        """With dry_run=True, simulation_results.json is not modified."""
        self._make_run(tmp_path, 'run1', self._simple_ddg_output())
        original = (tmp_path / 'run1' / 'simulation_results.json').read_text()
        self.backfill_ddg_pairs(tmp_path, dry_run=True)
        after = (tmp_path / 'run1' / 'simulation_results.json').read_text()
        assert original == after

    def test_skips_run_without_metadata(self, tmp_path):
        """A run directory with no metadata.json is counted as skipped."""
        run_dir = tmp_path / 'run1'
        run_dir.mkdir()
        sim = {'populations': [], 'transitions': []}
        (run_dir / 'simulation_results.json').write_text(json.dumps(sim))
        # No metadata.json
        n_updated, n_skipped = self.backfill_ddg_pairs(tmp_path)
        assert n_updated == 0
        assert n_skipped >= 1

    def test_ddg_values_in_updated_file(self, tmp_path):
        """After backfill, ddg_pairs keys follow 'lo_hi' format."""
        self._make_run(tmp_path, 'run1', self._simple_ddg_output())
        self.backfill_ddg_pairs(tmp_path)
        sim = json.loads((tmp_path / 'run1' / 'simulation_results.json').read_text())
        ddg = sim['ddg_pairs']
        assert isinstance(ddg, dict)
        # NaN → None round-trips through JSON as null
        for key, val in ddg.items():
            parts = key.split('_')
            assert len(parts) == 2 and all(p.isdigit() for p in parts)
            assert val is None  # NaN DDG → None

    def test_mixed_runs_updated_and_skipped(self, tmp_path):
        """Correctly counts updated vs skipped in a mixed batch."""
        self._make_run(tmp_path, 'run1', self._simple_ddg_output())
        self._make_run(tmp_path, 'run2', self._simple_ddg_output(), already_has_ddg=True)
        n_updated, n_skipped = self.backfill_ddg_pairs(tmp_path)
        assert n_updated == 1
        assert n_skipped == 1

