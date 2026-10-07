import os
import math
import pytest

from mllf.file_handling.read_bias_coeff import read_bias_coeff
from mllf.file_handling.read_output import parse_lambda_trajectory, parse_pair_barriers
from mllf.file_handling.read_rtf import parse_rtf_file, parse_rtf_dir


def _lambda_output(rows, first_block=2, temperature="298.150"):
    """Minimal CHARMM-like output: lambda trajectory + SINGLE DDG + temperature."""
    last = first_block + len(rows[0]) - 1
    lines = [" Temperature is:    0.000 K",   # setup-section line that must be ignored
             f" LAMBDA.......TIME(ps).. .BLOCKS   {first_block}   THRU BLOCK    {last}"]
    lines += [f"LAMBDA>  {0.02 * (i + 1):10.4f}  " + "  ".join(f"{v:.4f}" for v in r)
              for i, r in enumerate(rows)]
    lines += [f"Temperature is:  {temperature} K", "Site:     1"]
    blocks = list(range(first_block, last + 1))
    lines += [f"SINGLE DDG>  {a}  {b}   0.0  0.0  -1.0  -1.0"
              for i, a in enumerate(blocks) for b in blocks[i + 1:]]
    return "\n".join(lines)


def test_parse_lambda_trajectory_reads_header_and_frames():
    text = _lambda_output([[1.0, 0.0], [0.5, 0.5]])
    first, frames = parse_lambda_trajectory(text)
    assert first == 2
    assert frames == [[1.0, 0.0], [0.5, 0.5]]


def test_parse_pair_barriers_two_subs_matches_counts():
    # 400 frames at each end, 100 at the midpoint -> barrier = -kT ln(100/400)
    rows = [[1.0, 0.0]] * 400 + [[0.0, 1.0]] * 400 + [[0.5, 0.5]] * 100
    barrier = parse_pair_barriers(_lambda_output(rows))[(2, 3)]
    kT = 0.0019872041 * 298.15
    expected = -kT * (math.log(100.5) - math.log(400.5))
    assert barrier == pytest.approx(expected, abs=1e-6)
    assert barrier > 0


def test_parse_pair_barriers_is_insensitive_to_endpoint_imbalance():
    # A linear-bias tilt moves population between ends; the midpoint measured
    # against the geometric mean of the ends is unchanged.
    even = [[1.0, 0.0]] * 400 + [[0.0, 1.0]] * 400 + [[0.5, 0.5]] * 100
    tilted = [[1.0, 0.0]] * 1600 + [[0.0, 1.0]] * 100 + [[0.5, 0.5]] * 100
    b_even = parse_pair_barriers(_lambda_output(even))[(2, 3)]
    b_tilt = parse_pair_barriers(_lambda_output(tilted))[(2, 3)]
    assert b_tilt == pytest.approx(b_even, abs=0.01)


def test_parse_pair_barriers_multisub_uses_pair_edge_only():
    # Three subs: frames dominated by block 4 are off the (2,3) edge and ignored.
    rows = ([[1.0, 0.0, 0.0]] * 300 + [[0.0, 1.0, 0.0]] * 300 + [[0.5, 0.5, 0.0]] * 30
            + [[0.0, 0.0, 1.0]] * 5000 + [[0.3, 0.0, 0.7]] * 200)
    res = parse_pair_barriers(_lambda_output(rows))
    kT = 0.0019872041 * 298.15
    assert res[(2, 3)] == pytest.approx(-kT * (math.log(30.5) - math.log(300.5)), abs=1e-6)
    assert set(res) == {(2, 3), (2, 4), (3, 4)}


def test_parse_pair_barriers_none_when_endpoint_never_visited():
    rows = [[1.0, 0.0]] * 500 + [[0.5, 0.5]] * 50
    assert parse_pair_barriers(_lambda_output(rows))[(2, 3)] is None


def test_parse_pair_barriers_empty_without_trajectory():
    assert parse_pair_barriers("no lambda output here") == {}


def test_resolve_output_file_prefers_local_then_source(tmp_path):
    import json
    from mllf.file_handling.read_output import resolve_output_file
    source = tmp_path / "source_run"
    source.mkdir()
    (source / "output").write_text("source output")      # original sims may use 'output'
    run = tmp_path / "collected_run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"source_run_dir": str(source)}))
    # no local copy -> found via metadata.json source_run_dir
    assert resolve_output_file(run) == source / "output"
    # a local copy takes precedence
    (run / "output.out").write_text("local output")
    assert resolve_output_file(run) == run / "output.out"


def test_resolve_output_file_none_when_missing(tmp_path):
    from mllf.file_handling.read_output import resolve_output_file
    assert resolve_output_file(tmp_path) is None
    (tmp_path / "metadata.json").write_text('{"source_run_dir": "/does/not/exist"}')
    assert resolve_output_file(tmp_path) is None


def test_read_bias_coeff_parses_old_file():
	"""Ensure read_bias_coeff can parse the example variables85.inp (old ALF format)."""
	here = os.path.dirname(__file__)
	fn = os.path.join(here, 'samples', 'variables85.inp')

	data = read_bias_coeff(fn)

	# top-level groups should exist
	assert set(data.keys()) == {'lams', 'cs', 'xs', 'ss'}

	# check a few known values from the example file
	# lams1s2 = 0.13
	assert 'lams1s2' in data['lams']
	assert math.isclose(data['lams']['lams1s2'], 0.13, rel_tol=1e-6)

	# lams3s8 = -14.20
	assert math.isclose(data['lams']['lams3s8'], -14.20, rel_tol=1e-6)

	# cs1s1s1s2 = -39.95
	assert 'cs1s1s1s2' in data['cs']
	assert math.isclose(data['cs']['cs1s1s1s2'], -39.95, rel_tol=1e-6)

	# some entries should be zero values parsed as floats
	assert math.isclose(data['cs'].get('cs1s1s2s1', 0.0), -0.0, rel_tol=1e-6)

def test_read_bias_coeff_parses_new_file():
	"""Ensure read_bias_coeff can parse the example variables16.py (new ALF format).

	The `b:` nested list encodes linear biases for multiple i blocks. Each block
	begins with a 0.0 value. We assert some sample lams assignments to ensure
	the zero-handling logic is correct.
	"""
	here = os.path.dirname(__file__)
	fn = os.path.join(here, 'samples', 'variables.py')

	data = read_bias_coeff(fn)

	# top-level groups should exist (scalar entries)
	assert set(data.keys()) >= {'lams', 'cs', 'xs', 'ss'}
	
	# matrices should also be present
	assert 'b' in data and 'c' in data and 'x' in data and 's' in data
	
	# b is normalised to a flat 1-D list (parse_new unwraps the [[...]] wrapper)
	b = data['b']
	assert isinstance(b, list), "b should be a list"
	assert len(b) == 31, "b should have 31 values for this example"
	assert isinstance(b[0], float), "b[0] should be a scalar float"
	assert b[0] == 0.0 and b[1] == 0.13, "b vector values should match"
	
	# c should be a 31x31 matrix
	c = data['c']
	assert isinstance(c, list) and len(c) == 31
	assert all(isinstance(row, list) and len(row) == 31 for row in c)

	assert 'lams1s2' in data['lams']
	assert math.isclose(data['lams']['lams1s2'], 0.13, rel_tol=1e-6)

	# lams3s8 = -14.20
	assert math.isclose(data['lams']['lams3s8'], -14.20, rel_tol=1e-6)

	# cs1s1s1s2 = -39.95
	assert 'cs1s1s1s2' in data['cs']
	assert math.isclose(data['cs']['cs1s1s1s2'], -39.95, rel_tol=1e-6)

	# some entries should be zero values parsed as floats
	assert math.isclose(data['cs'].get('cs1s1s2s1', 0.0), -0.0, rel_tol=1e-6)


def test_parse_rtf_file_and_dir():
	"""Test parse_rtf_file and parse_rtf_dir on the example PRES file."""
	here = os.path.dirname(__file__)
	rtf_path = os.path.join(here, 'samples', '14benz_solv_5.5', 'site1_sub1_pres.rtf')

	parsed = parse_rtf_file(rtf_path)

	# site and sub should be detected from the filename
	assert parsed['site'] == 1
	assert parsed['sub'] == 1

	# atom_types should include the two types in the file
	assert 'C261' in parsed['atom_types']
	assert 'HG61' in parsed['atom_types']

	# total_charge should be the sum of -0.115000 and 0.115000 -> 0.0
	assert math.isclose(parsed['total_charge'], 0.0, abs_tol=1e-9)

	# parse directory and ensure key exists
	d = os.path.join(here, 'samples', '14benz_solv_5.5')
	results = parse_rtf_dir(d)
	assert 'site1_sub1' in results



def _profile_rows(counts):
    """Two-sub lambda rows [lambda_lo, lambda_hi] from {r: n_frames}."""
    rows = []
    for r, n in counts.items():
        rows += [[1.0 - r, r]] * n
    return rows


_SYMMETRIC = {0.0: 400, 0.03: 40, 0.3: 20, 0.5: 30, 0.7: 20, 0.97: 40, 1.0: 400}


def test_parse_pair_profile_symmetric_profile_has_no_asymmetry():
    from mllf.file_handling.read_output import parse_pair_profile
    prof = parse_pair_profile(_lambda_output(_profile_rows(_SYMMETRIC)))[(2, 3)]
    assert prof["skew"] == pytest.approx(0.0, abs=1e-9)
    assert prof["end"] == pytest.approx(0.0, abs=1e-9)
    assert prof["barrier"] is not None


def test_parse_pair_profile_end_asymmetry_softer_hi_end():
    # Four times the shoulder population next to the hi endpoint -> hi end less sharp.
    from mllf.file_handling.read_output import parse_pair_profile
    counts = {**_SYMMETRIC, 0.97: 160}
    prof = parse_pair_profile(_lambda_output(_profile_rows(counts)))[(2, 3)]
    kT = 0.0019872041 * 298.15
    assert prof["end"] == pytest.approx(-kT * math.log(160.5 / 40.5), abs=1e-9)


def test_parse_pair_profile_skew_heavier_hi_flank():
    # Hi-side flank (r ~ 0.7) four times the lo-side flank, ends symmetric.
    from mllf.file_handling.read_output import parse_pair_profile
    counts = {**_SYMMETRIC, 0.7: 80}
    prof = parse_pair_profile(_lambda_output(_profile_rows(counts)))[(2, 3)]
    kT = 0.0019872041 * 298.15
    assert prof["end"] == pytest.approx(0.0, abs=1e-9)
    assert prof["skew"] == pytest.approx(-kT * math.log(80.5 / 20.5), abs=1e-9)


def test_parse_pair_profile_skew_end_none_without_endpoint_cores():
    # Endpoints only reached to within 5% -> no endpoint-core frames for skew/end,
    # but the (wider-binned) barrier is still defined.
    from mllf.file_handling.read_output import parse_pair_profile
    counts = {0.03: 400, 0.3: 20, 0.5: 30, 0.7: 20, 0.97: 400}
    prof = parse_pair_profile(_lambda_output(_profile_rows(counts)))[(2, 3)]
    assert prof["skew"] is None and prof["end"] is None
    assert prof["barrier"] is not None


def test_parse_pair_profile_endpoint_occupancy():
    from mllf.file_handling.read_output import parse_pair_profile
    ep = parse_pair_profile(_lambda_output(_profile_rows(_SYMMETRIC)))[(2, 3)]["endpoints"]
    # r=0.0 -> at_lo, 0.03 -> near_lo, 0.97 -> near_hi, 1.0 -> at_hi; 0.3/0.5/0.7 in neither
    assert ep == {"edge": 950, "at_lo": 400, "near_lo": 40, "near_hi": 40, "at_hi": 400}


def test_parse_pair_profile_endpoint_held_off_signature():
    # lambda crowds up to the lo end (r ~ 0.1) but never reaches it, hi end untouched:
    # the mup1 failure. Occupancy is reported although no barrier/skew/end is defined.
    from mllf.file_handling.read_output import parse_pair_profile
    prof = parse_pair_profile(_lambda_output(_profile_rows({0.1: 800, 0.3: 200})))[(2, 3)]
    assert prof["endpoints"] == {"edge": 1000, "at_lo": 0, "near_lo": 800, "near_hi": 0, "at_hi": 0}
    assert prof["barrier"] is None and prof["end"] is None


def test_parse_pair_barriers_matches_profile():
    from mllf.file_handling.read_output import parse_pair_profile
    text = _lambda_output(_profile_rows(_SYMMETRIC))
    assert parse_pair_barriers(text)[(2, 3)] == parse_pair_profile(text)[(2, 3)]["barrier"]


def test_pair_transitions_credit_each_state_change_to_its_pair():
    # site with 3 subs (blocks 2,3,4): states 2 → 3 → 2 → 4 → 3, with non-physical frames between
    from mllf.file_handling.read_output import parse_pair_profile
    seq = [[1, 0, 0], [0.5, 0.5, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0.4, 0, 0.6], [0, 0, 1], [0, 1, 0]]
    rows = []
    for r in seq:
        rows += [r] * 30          # keep every pair above the minimum-frames threshold
    prof = parse_pair_profile(_lambda_output(rows))
    assert prof[(2, 3)]["transitions"] == 2      # 2→3, 3→2
    assert prof[(2, 4)]["transitions"] == 1      # 2→4
    assert prof[(3, 4)]["transitions"] == 1      # 4→3
    # CHARMM's site total would be 4 + 1 (first entry into a physical state)
    assert sum(v["transitions"] for v in prof.values()) == 4


def test_pair_transitions_ignore_other_sites():
    # two sites (2,3) and (4,5): site-2 changes must not count for site 1
    from mllf.file_handling.read_output import parse_pair_profile
    rows = [[1, 0, 1, 0]] * 200 + [[1, 0, 0, 1]] * 200 + [[1, 0, 1, 0]] * 200
    text = _lambda_output(rows).replace(
        "SINGLE DDG>  2  4   0.0  0.0  -1.0  -1.0\n", "").replace(
        "SINGLE DDG>  2  5   0.0  0.0  -1.0  -1.0\n", "").replace(
        "SINGLE DDG>  3  4   0.0  0.0  -1.0  -1.0\n", "").replace(
        "SINGLE DDG>  3  5   0.0  0.0  -1.0  -1.0\n", "")
    prof = parse_pair_profile(text)
    assert set(prof) == {(2, 3), (4, 5)}
    assert prof[(2, 3)]["transitions"] == 0 and prof[(4, 5)]["transitions"] == 2
