"""Tests for collect_pretraining_data CLI helpers and related utilities.

Covers:
  - find_run_directories      (flat MSLD and nested combo structures)
  - parse_bias_from_py        (variables*.py -> matrix keys only)
  - parse_bias_from_inp       (variables*.inp -> matrix format)
  - detect_solvent_state      (path-based heuristic)
  - _extract_nsubs (indirectly via read_bias_coeff.parse_new)
  - _read_box_from_prep_script (box= extraction from prep .py file)
  - filter_best_runs_per_system (grouping & best-reward selection)
"""
import json
import re
import textwrap
from pathlib import Path

import pytest
import torch
import yaml

from mllf.cb.unimol_representation import _read_box_from_prep_script
from mllf.cli.collect_pretraining_data import (
    backfill_barrier_pairs,
    detect_solvent_state,
    find_run_directories,
    fix_inp_nonlinear_signs,
    parse_bias_from_inp,
    parse_bias_from_py,
)
from mllf.file_handling.read_bias_coeff import read_bias_coeff
from mllf.pretrain.filters import filter_best_runs_per_system


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


# find_run_directories

def test_find_run_directories_flat_layout(tmp_path):
    """run# directories, sorted by number; failed and empty ones skipped."""
    for name in ('run10', 'run2', 'run1', 'run3_failed'):
        _make_run_dir(tmp_path, name)
    (tmp_path / 'run4').mkdir()                      # no variables or output
    assert [d.name for d in find_run_directories(tmp_path)] == ['run1', 'run2', 'run10']


def test_find_run_directories_combo_layout(tmp_path):
    """comb_*/run_* directories, failed ones skipped; an empty root gives none."""
    assert find_run_directories(tmp_path) == []
    for combo in ('comb_001', 'comb_002'):
        _make_run_dir(tmp_path / combo, 'run_001')
    _make_run_dir(tmp_path / 'comb_001', 'run_002_failed')
    found = find_run_directories(tmp_path)
    assert len(found) == 2 and all(d.name == 'run_001' for d in found)


# parse_bias_from_py / parse_bias_from_inp

def test_parse_bias_from_py_returns_matrices_only(tmp_path):
    f = tmp_path / 'variables.py'
    f.write_text(_variables_py_content(4))
    r = parse_bias_from_py(f)
    assert len(r['b']) == 4 and all(isinstance(v, float) for v in r['b'])
    assert len(r['c']) == 4 and all(len(row) == 4 for row in r['c'])
    assert not {'lams', 'cs', 'xs', 'ss'} & set(r)
    assert parse_bias_from_py(Path(__file__).parent / 'samples' / 'variables.py')['b']


def test_parse_bias_from_py_raises_without_bias_string(tmp_path):
    f = tmp_path / 'variables.py'
    f.write_text('# no bias_string here\nx = 1\n')
    with pytest.raises(ValueError, match="Could not parse bias vector"):
        parse_bias_from_py(f)


def test_parse_bias_from_inp_shapes(tmp_path):
    """One b entry per block and a square c (2 sites x 2 subs here)."""
    lines = [f'set lams{s}s{u} = {(s - 1) * 2 + u - 1:.2f}' for s in (1, 2) for u in (1, 2)]
    lines += [f'set cs{s}s{i}s{s}s{j} = {(s * 10 + i + j) * 0.1:.4f}'
              for s in (1, 2) for i in (1, 2) for j in (1, 2)]
    f = tmp_path / 'variables.inp'
    f.write_text('\n'.join(lines) + '\n')
    r = parse_bias_from_inp(f)
    assert '_nsubs_per_site' in r and len(r['b']) == 4
    assert len(r['c']) == 4 and all(len(row) == 4 for row in r['c'])
    assert parse_bias_from_inp(Path(__file__).parent / 'samples' / 'variables85.inp')['b']


def test_parse_bias_from_inp_nonlinear_signs_match_matrix_convention(tmp_path):
    """Flat cs/xs/ss values are -matrix (ALF convention); lams is b unchanged.

    Mirrors an ALF variablesflat.py that stores both forms: c[0][1] = +19.53
    with cs1s1s1s2 = -19.53, x[0][1] = -0.49 with xs1s1s1s2 = +0.49, and so on.
    """
    f = tmp_path / 'variables.inp'
    f.write_text('\n'.join([
        'set lams1s1 = 0.00', 'set lams1s2 = -7.94', 'set cs1s1s1s2 = -19.53',
        'set xs1s1s1s2 = 0.49', 'set xs1s2s1s1 = -1.60',
        'set ss1s1s1s2 = -1.11', 'set ss1s2s1s1 = 0.25',
    ]) + '\n')
    r = parse_bias_from_inp(f)
    assert r['b'] == pytest.approx([0.0, -7.94])
    assert r['c'][0][1] == pytest.approx(19.53)
    assert (r['x'][0][1], r['x'][1][0]) == pytest.approx((-0.49, 1.60))
    assert (r['s'][0][1], r['s'][1][0]) == pytest.approx((1.11, -0.25))


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
        """b has one entry per substituent block (n_sites x n_subs)."""
        n = 2
        f = tmp_path / 'variables.inp'
        self._write_inp(f, n=n)
        result = parse_bias_from_inp(f)
        # 2 sites x 2 subs = 4 blocks
        assert len(result['b']) == n * n

    def test_c_matrix_square(self, tmp_path):
        """c is a square (n_blocks x n_blocks) matrix."""
        n = 2
        f = tmp_path / 'variables.inp'
        self._write_inp(f, n=n)
        result = parse_bias_from_inp(f)
        c = result['c']
        assert len(c) == n * n
        assert all(len(row) == n * n for row in c)

    def test_nonlinear_signs_match_matrix_convention(self, tmp_path):
        """Flat cs/xs/ss values are -matrix (ALF convention); lams is b unchanged.

        Mirrors an ALF variablesflat.py that stores both forms:
        c[0][1] = +19.53 with cs1s1s1s2 = -19.53, x[0][1] = -0.49 with
        xs1s1s1s2 = +0.49, x[1][0] = +1.60 with xs1s2s1s1 = -1.60,
        s[0][1] = +1.11 with ss1s1s1s2 = -1.11, b[1] = lams1s2 = -7.94.
        """
        f = tmp_path / 'variables.inp'
        f.write_text('\n'.join([
            'set lams1s1 = 0.00', 'set lams1s2 = -7.94',
            'set cs1s1s1s2 = -19.53',
            'set xs1s1s1s2 = 0.49', 'set xs1s2s1s1 = -1.60',
            'set ss1s1s1s2 = -1.11', 'set ss1s2s1s1 = 0.25',
        ]) + '\n')
        r = parse_bias_from_inp(f)
        assert r['b'] == pytest.approx([0.0, -7.94])
        assert r['c'][0][1] == pytest.approx(19.53)
        assert r['x'][0][1] == pytest.approx(-0.49)
        assert r['x'][1][0] == pytest.approx(1.60)
        assert r['s'][0][1] == pytest.approx(1.11)
        assert r['s'][1][0] == pytest.approx(-0.25)

    def test_parses_real_example_inp(self):
        """Parses the checked-in tests/fixtures/variables85.inp."""
        example = Path(__file__).parent / 'samples' / 'variables85.inp'
        if not example.exists():
            pytest.skip('tests/samples/variables85.inp not found')
        result = parse_bias_from_inp(example)
        assert 'b' in result
        assert len(result['b']) > 0


class TestFixInpNonlinearSigns:
    """fix_inp_nonlinear_signs: repair runs collected from .inp before the sign fix."""

    @staticmethod
    def _make_run(root, name, source_files, c01, x01, x10, s01, b1=-3.0):
        source = root / 'sources' / name
        source.mkdir(parents=True)
        for fn in source_files:
            (source / fn).write_text('set lams1s1 = 0.00\n')
        run = root / 'pretraining' / name
        run.mkdir(parents=True)
        (run / 'metadata.json').write_text(json.dumps({'source_run_dir': str(source)}))
        bias = {'b': [0.0, b1], 'c': [[0.0, c01], [0.0, 0.0]],
                'x': [[0.0, x01], [x10, 0.0]], 's': [[0.0, s01], [0.0, 0.0]]}
        (run / 'variables.py').write_text(
            '"""Auto-generated variables file for pretraining data."""\nimport yaml\n\n'
            f'bias_string = """\n{yaml.dump(bias, sort_keys=False).strip()}\n"""\n\n'
            'bias = yaml.safe_load(bias_string)\n')
        return run

    @staticmethod
    def _bias(run):
        text = (run / 'variables.py').read_text()
        return yaml.safe_load(re.search(r'bias_string\s*=\s*"""(.*?)"""', text, re.S).group(1))

    def test_inp_sourced_run_is_negated_once(self, tmp_path):
        run = self._make_run(tmp_path, 'inp_run', ['variablesflat.inp'], -19.5, 0.5, -1.6, -1.1)
        assert fix_inp_nonlinear_signs(tmp_path / 'pretraining') == (1, 0, 0)
        b = self._bias(run)
        assert b['c'][0][1] == pytest.approx(19.5)
        assert b['x'][0][1] == pytest.approx(-0.5) and b['x'][1][0] == pytest.approx(1.6)
        assert b['s'][0][1] == pytest.approx(1.1)
        assert b['b'] == pytest.approx([0.0, -3.0])          # linear bias untouched
        assert b['_nonlinear_sign_convention'] == 'matrix'
        # rerunning does not flip it back
        assert fix_inp_nonlinear_signs(tmp_path / 'pretraining') == (0, 1, 0)
        assert self._bias(run)['c'][0][1] == pytest.approx(19.5)

    def test_py_sourced_run_untouched(self, tmp_path):
        run = self._make_run(
            tmp_path, 'py_run', [
                'variablesflat.py', 'variables3.inp'], 19.5, -0.5, 1.6, 1.1)
        before = (run / 'variables.py').read_text()
        assert fix_inp_nonlinear_signs(tmp_path / 'pretraining') == (0, 1, 0)
        assert (run / 'variables.py').read_text() == before

    def test_missing_source_reported_not_changed(self, tmp_path):
        run = self._make_run(tmp_path, 'gone', [], 19.5, -0.5, 1.6, 1.1)
        before = (run / 'variables.py').read_text()
        assert fix_inp_nonlinear_signs(tmp_path / 'pretraining') == (0, 0, 1)
        assert (run / 'variables.py').read_text() == before

    def test_dry_run_writes_nothing(self, tmp_path):
        run = self._make_run(tmp_path, 'inp_run', ['variables5.inp'], -19.5, 0.5, -1.6, -1.1)
        before = (run / 'variables.py').read_text()
        assert fix_inp_nonlinear_signs(tmp_path / 'pretraining', dry_run=True) == (1, 0, 0)
        assert (run / 'variables.py').read_text() == before

    def test_new_inp_collections_are_marked(self, tmp_path):
        f = tmp_path / 'variables.inp'
        f.write_text('set lams1s1 = 0.00\nset lams1s2 = -1.00\nset cs1s1s1s2 = -5.00\n')
        assert parse_bias_from_inp(f)['_nonlinear_sign_convention'] == 'matrix'


class TestBackfillLambdaProfile:
    """backfill_barrier_pairs: add barrier/skew/end to existing simulation_results.json."""

    @staticmethod
    def _output_text():
        rows = ([[1.0, 0.0]] * 400 + [[0.97, 0.03]] * 40 + [[0.7, 0.3]] * 20 + [[0.5, 0.5]] * 30
                + [[0.3, 0.7]] * 20 + [[0.03, 0.97]] * 40 + [[0.0, 1.0]] * 400)
        lines = [" LAMBDA.......TIME(ps).. .BLOCKS   2   THRU BLOCK    3"]
        lines += [f"LAMBDA>  {0.02 * (i + 1):10.4f}  {a:.4f}  {b:.4f}" for i,
                  (a, b) in enumerate(rows)]
        lines += ["Temperature is:  298.150 K", "Site:     1",
                  "SINGLE DDG>  2  3   0.0  0.0  -1.0  -1.0"]
        return "\n".join(lines)

    def _make(self, root):
        with_out = root / 'sys' / 'comb_0001' / 'run_001'
        no_out = root / 'sys' / 'comb_0001' / 'run_002'
        for r in (with_out, no_out):
            r.mkdir(parents=True)
            (r / 'simulation_results.json').write_text(json.dumps({'ddg_pairs': {'2_3': -1.0}}))
        (with_out / 'output.out').write_text(self._output_text())
        return with_out, no_out

    @pytest.mark.parametrize('workers', [1, 2])
    def test_fills_fields_and_skips_missing_output(self, tmp_path, workers):
        with_out, no_out = self._make(tmp_path)
        assert backfill_barrier_pairs(tmp_path, workers=workers) == (1, 1)
        sr = json.loads((with_out / 'simulation_results.json').read_text())
        assert set(sr) >= {'barrier_pairs', 'skew_pairs', 'end_pairs'}
        assert sr['barrier_pairs']['2_3'] is not None
        assert sr['skew_pairs']['2_3'] == pytest.approx(0.0, abs=1e-9)   # symmetric profile
        assert sr['end_pairs']['2_3'] == pytest.approx(0.0, abs=1e-9)
        assert sr['simulated_ns'] is None              # output has no dynamics header
        assert 'barrier_pairs' not in json.loads((no_out / 'simulation_results.json').read_text())
        # second pass skips the run that already has the fields
        assert backfill_barrier_pairs(tmp_path, workers=workers) == (0, 2)

    def test_adds_only_the_run_length_to_profiled_runs(self, tmp_path):
        with_out, _ = self._make(tmp_path)
        profiled = {'ddg_pairs': {'2_3': -1.0}, 'barrier_pairs': {'2_3': 9.9}, 'skew_pairs': {},
                    'end_pairs': {}, 'transition_pairs': {}}
        (with_out / 'simulation_results.json').write_text(json.dumps(profiled))
        header = ("   NSTEP =    13000    NSAVC =     1000    NSAVV =       10\n"
                  "  TIME STEP =  1.6E-01 AKMA       4.00000E-03 PS\n"
                  "   NSTEP =    13000    NSAVC =     1000    NSAVV =       10\n"
                  "  TIME STEP =  1.6E-01 AKMA       4.00000E-03 PS\n")
        (with_out / 'output.out').write_text(header + self._output_text())
        assert backfill_barrier_pairs(tmp_path)[0] == 1
        sr = json.loads((with_out / 'simulation_results.json').read_text())
        assert sr['simulated_ns'] == pytest.approx(0.104)
        assert sr['barrier_pairs'] == {'2_3': 9.9}      # profile left as it was

    def test_dry_run_writes_nothing(self, tmp_path):
        with_out, _ = self._make(tmp_path)
        before = (with_out / 'simulation_results.json').read_text()
        assert backfill_barrier_pairs(tmp_path, dry_run=True)[0] == 1
        assert (with_out / 'simulation_results.json').read_text() == before


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
        # No nsubs information in the file -> key absent or None
        assert data.get('_nsubs_per_site') is None


# ---------------------------------------------------------------------------
# TestReadBoxFromPrepScript
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('files, box', [
    ({'prep.py': 'box = 40.0\n# comment\n'}, 40.0),
    ({'prep.py': 'box = 35\n'}, 35.0),                              # integer value
    ({}, None),                                                     # no .py file
    ({'alf_info.py': "alf_info['nsubs'] = [3]\nbox = 99.0\n"}, None),  # alf_info.py is skipped
    ({'prep.py': 'nsubs = [3]\n'}, None),                           # no box line
])
def test_read_box_from_prep_script(tmp_path, files, box):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    got = _read_box_from_prep_script(tmp_path)
    assert got == (pytest.approx(box) if box is not None else None)


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
    # graph_info.json with 1 site x 3 subs
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
        """One run per system -> all runs returned unchanged."""
        runs = [
            _make_pretraining_run(tmp_path, ('14benz_solv',), 'run_001'),
            _make_pretraining_run(tmp_path, ('indole_solv',), 'run_001'),
        ]
        result = filter_best_runs_per_system(runs)
        assert len(result) == 2

    def test_best_run_selected_per_flat_system(self, tmp_path):
        """Two runs from the same flat parent dir -> only the best-reward one kept."""
        # Use a neutral grandparent name that doesn't contain 'best' or 'combo'
        # (those keywords trigger special legacy-combo grouping in the function).
        base = tmp_path / 'pretraining'
        low = _make_pretraining_run(base, ('14benz_solv',), 'run_001', n_transitions=0)
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
        run_b = _make_pretraining_run(tmp_path, ('dataset', 'comb_002'), 'run_001',
                                      n_transitions=50)
        result = filter_best_runs_per_system([run_a1, run_a2, run_b])
        # 2 distinct combos -> 2 runs selected (best from comb_001 + only from comb_002)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        """Empty run list -> empty result (no crash)."""
        assert filter_best_runs_per_system([]) == []


# ---------------------------------------------------------------------------
# TestBuildEdgeWeights
# ---------------------------------------------------------------------------

def test_build_edge_weights():
    """Weight 1 for a pair with transitions or no DDG entry (old data);
    no_transition_weight for a pair with a None DDG, in either direction."""
    from mllf.cb.workflow_utils import build_edge_weights
    cpu = torch.device('cpu')
    # nodes 0, 1, 2 are blocks 2, 3, 4
    ei = torch.tensor([(0, 1), (1, 0), (0, 2), (1, 2)], dtype=torch.long).T
    w = build_edge_weights(ei, {"2_3": None, "2_4": -0.5}, 0.3, cpu)
    assert w.dtype == torch.float32
    assert w.tolist() == pytest.approx([0.3, 0.3, 1.0, 1.0])
    assert build_edge_weights(ei, {}, 0.3, cpu).tolist() == [1.0] * 4


def test_parse_simulation_metrics_ddg_pairs(tmp_path):
    """ddg_pairs is always present, keyed "lo_hi"; empty without a SINGLE DDG section."""
    from mllf.cb.workflow_utils import parse_simulation_metrics
    f = tmp_path / 'output.out'
    f.write_text('nothing here\n')
    assert parse_simulation_metrics(f)['ddg_pairs'] == {}
    sample = Path(__file__).parent / 'samples' / '14benz_solv_5.5' / 'output.txt'
    ddg = parse_simulation_metrics(sample)['ddg_pairs']
    # the sample has NaN/Inf for every pair
    assert ddg and all(v is None for v in ddg.values())
    assert all(re.fullmatch(r'\d+_\d+', k) for k in ddg)


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

    def test_updates_only_runs_without_ddg_pairs(self, tmp_path):
        """Runs without ddg_pairs get them (NaN -> None, keys "lo_hi"); runs
        that have them, or have no metadata.json, are skipped."""
        self._make_run(tmp_path, 'run1', self._simple_ddg_output())
        self._make_run(tmp_path, 'run2', self._simple_ddg_output(), already_has_ddg=True)
        (tmp_path / 'run3').mkdir()
        (tmp_path / 'run3' / 'simulation_results.json').write_text('{}')
        assert self.backfill_ddg_pairs(tmp_path) == (1, 2)
        sim = json.loads((tmp_path / 'run1' / 'simulation_results.json').read_text())
        assert sim['ddg_pairs'] == {'2_3': None}

    def test_dry_run_does_not_modify_files(self, tmp_path):
        self._make_run(tmp_path, 'run1', self._simple_ddg_output())
        original = (tmp_path / 'run1' / 'simulation_results.json').read_text()
        self.backfill_ddg_pairs(tmp_path, dry_run=True)
        assert (tmp_path / 'run1' / 'simulation_results.json').read_text() == original


# ---------------------------------------------------------------------------
# Active-substituent mapping (which master sub carries each bias position)
# ---------------------------------------------------------------------------
from mllf.cli.collect_pretraining_data import (  # noqa: E402
    parse_active_subs_from_inp,
    parse_active_subs_from_output,
)


# master sub -> atom names in its _pres.rtf
_MASTERS = {
    'site1_sub1': ['C010', 'H011'], 'site1_sub2': ['C020', 'H021'], 'site1_sub3': ['C030', 'H031'],
    'site1_sub7': ['C070', 'H071'], 'site1_sub8': ['C080', 'H081'],
    'site2_sub1': ['C110', 'H111'], 'site2_sub6': ['C160', 'H161'], 'site2_sub2': ['C120', 'H121'],
}


def _prep(tmp_path):
    prep = tmp_path / 'prep'
    prep.mkdir()
    for master, names in _MASTERS.items():
        s, j = re.match(r'site(\d+)_sub(\d+)', master).groups()
        atoms = '\n'.join(f'ATOM {n} CT00 0.0' for n in names)
        (prep / f'{master}_pres.rtf').write_text(f'PRES p{s}_{j} 0.0\nGROUP\n{atoms}\n')
    return prep


def _script(blocks, reads, echo=False):
    """CHARMM script (or its output echo) reading `reads`, defining local
    labels -> master atoms, and calling them in `blocks` order."""
    c = ' CHARMM>    ' if echo else ''
    b = '  BLOCK>       ' if echo else '   '
    lines = [f'{c}read rtf append card name @builddir/{m}_pres.rtf' for m in reads]
    for label, master in blocks:
        lines.append(f'{c}define {label} -')
        lines.append(f'{c}   select ( -')
        lines += [f'{c}   atom @ligseg @resnum {n} .or. -' for n in _MASTERS[master]]
        lines.append(f'{c}   none ) end')
        if echo:
            lines.append(' Parameter: LIGSEG -> "LIG"')
    lines.append(f'{c}BLOCK 6')
    lines += [f'{b}Call {k + 2} sele {label} show end' for k, (label, _) in enumerate(blocks)]
    return '\n'.join(lines) + '\n'


# p38 group2-like: blocks site1 = (1, 7, 8), site2 = (1, 6); a second .inp
# reads every sub, and the selections are defined out of label order.
_BLOCKS = [('site1_sub1', 'site1_sub1'), ('site1_sub2', 'site1_sub7'),
           ('site1_sub3', 'site1_sub8'), ('site2_sub1', 'site2_sub1'),
           ('site2_sub2', 'site2_sub6')]
_EXPECTED = {
    'site1': [
        'site1_sub1',
        'site1_sub7',
        'site1_sub8'],
    'site2': [
        'site2_sub1',
        'site2_sub6']}


def test_active_subs_from_output_follow_block_calls(tmp_path):
    prep = _prep(tmp_path)
    out = tmp_path / 'output'
    shuffled = [_BLOCKS[2], _BLOCKS[0], _BLOCKS[4], _BLOCKS[1], _BLOCKS[3]]
    text = _script(shuffled, reads=['site1_sub8', 'site1_sub1', 'site1_sub7', 'site2_sub6',
                                    'site2_sub1'], echo=True)
    # calls are in label order regardless of define order
    text = re.sub(r'(BLOCK>\s+Call \d+ sele \S+ show end\n)+',
                  lambda m: ''.join(f'  BLOCK>       Call {k + 2} sele {lab} show end\n'
                                    for k, (lab, _) in enumerate(_BLOCKS)), text)
    out.write_text(text)
    assert parse_active_subs_from_output(out, prep, [3, 2]) == _EXPECTED


def test_active_subs_from_output_none_without_echo_or_count_mismatch(tmp_path):
    prep = _prep(tmp_path)
    out = tmp_path / 'output'
    out.write_text('no block setup here\n')
    assert parse_active_subs_from_output(out, prep, [3, 2]) is None
    out.write_text(_script(_BLOCKS, reads=[], echo=True))
    assert parse_active_subs_from_output(out, prep, [2, 2]) is None


def test_active_subs_from_inp_ignores_extra_inp_listing_all_subs(tmp_path):
    # benz.inp sorts first and lists every sub; p38.inp is the script that ran.
    prep = _prep(tmp_path)
    (prep / 'benz.inp').write_text('\n'.join(
        f'read rtf append card name @builddir/{m}_pres.rtf' for m in sorted(_MASTERS)) + '\n')
    (prep / 'p38.inp').write_text(_script(_BLOCKS, reads=[m for _, m in _BLOCKS]))
    assert parse_active_subs_from_inp(prep, [3, 2]) == _EXPECTED


def test_active_subs_from_inp_reads_when_no_block_setup(tmp_path):
    prep = _prep(tmp_path)
    (prep / 'a_all.inp').write_text('\n'.join(
        f'read rtf append card name @builddir/{m}_pres.rtf' for m in sorted(_MASTERS)) + '\n')
    (prep / 'b_ran.inp').write_text('\n'.join(
        ['! read rtf append card name @builddir/site1_sub2_pres.rtf'] +
        [f'read rtf append card name @builddir/{m}_pres.rtf'
         for m in ('site1_sub1', 'site1_sub7', 'site1_sub8', 'site2_sub1', 'site2_sub6')]) + '\n')
    assert parse_active_subs_from_inp(prep, [3, 2]) == _EXPECTED


def test_rekey_uses_msld_site_and_records_master():
    from mllf.cli.collect_pretraining_data import rekey_graph_info
    gi = {'solvent_state': 'solvent', 'sites': {
        'site2_sub21': {'site': 2, 'sub': 21, 'atom_types': ['CT00'],
                        'filename': 'site2_sub21_pres.rtf'},
        'site2_sub1': {'site': 2, 'sub': 1, 'atom_types': ['CT00'],
                       'filename': 'site2_sub1_pres.rtf'},
        'site2_sub2': {'site': 2, 'sub': 2, 'atom_types': ['NG00'],
                       'filename': 'site2_sub2_pres.rtf'}}}
    out = rekey_graph_info(
        gi, {
            'site1': ['site2_sub21'], 'site2': [
                'site2_sub2', 'site2_sub1']}, [
            1, 2])
    assert {k: (v['site'], v['sub'], v['master']) for k, v in out['sites'].items()} == {
        'site1_sub1': (1, 1, 'site2_sub21'), 'site2_sub1': (2, 1, 'site2_sub2'),
        'site2_sub2': (2, 2, 'site2_sub1')}
    assert out['sites']['site2_sub1']['atom_types'] == ['NG00']


def test_node_fragment_is_the_master_sub():
    from mllf.pretrain.features import _node_master_sub
    assert _node_master_sub({'master': 'site1_sub12'}, 1, 2) == 'site1_sub12'
    assert _node_master_sub({'filename': 'site1_sub12_pres.rtf'}, 1, 2) == 'site1_sub12'
    assert _node_master_sub({}, 1, 2) == 'site1_sub2'
