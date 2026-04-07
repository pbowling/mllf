import math
from pathlib import Path
from mllf.file_handling.read_output import (
	terminated_normally,
	parse_single_population,
	parse_transitions_and_rates,
	parse_single_ddg,
)


def load_example():
	# Find the repository root and construct path to example file
	test_file = Path(__file__)
	repo_root = test_file.parent.parent
	p = repo_root / "tests" / "samples" / "14benz_solv_5.5" / "output.txt"
	
	with open(p, "r", encoding="utf-8", errors="ignore") as fh:
		return fh.read()


def test_terminated_normally():
	txt = load_example()
	assert terminated_normally(txt)


def test_single_population_parsing():
	txt = load_example()
	pops = parse_single_population(txt)
	# from the example, block 3 final column is 3093 associated with lambda 0.99
	assert 2 in pops
	assert pops[3]["counts"][0.99] == 3093
	# block 2 should be associated with site 1 (per the SINGLE DDG block listing)
	assert pops[2]["site"] == 1


def test_transitions_and_rates_parsing():
	txt = load_example()
	transitions, rates = parse_transitions_and_rates(txt)
	# site 1 has 1 transitions at lambda 0.99 and rate 0.01282
	assert 1 in transitions
	assert transitions[1][0.99] == 1
	assert 1 in rates
	# allow small float rounding tolerance
	assert abs(rates[1][0.99] - 0.01282) < 1e-6


# ---------------------------------------------------------------------------
# Tests for parse_single_ddg
# ---------------------------------------------------------------------------

_DDG_HEADER = "             BLK(I)..BLK(J).....> 0.950 ....> 0.990 .....> 0.950 ....> 0.990"

def _make_ddg_text(*rows):
	"""Build a minimal output snippet containing a SINGLE DDG section."""
	lines = [_DDG_HEADER] + [f"SINGLE DDG>  {r}" for r in rows]
	return "\n".join(lines) + "\n"


def test_parse_single_ddg_returns_empty_for_no_section():
	"""Returns empty dict when output contains no SINGLE DDG lines."""
	assert parse_single_ddg("some text\nno ddg here\n") == {}


def test_parse_single_ddg_nan_returns_none():
	"""NaN values are mapped to None (no usable crossings)."""
	txt = _make_ddg_text("2  3   NaN   NaN   NaN   NaN")
	result = parse_single_ddg(txt)
	assert (2, 3) in result
	assert result[(2, 3)] is None


def test_parse_single_ddg_infinity_returns_none():
	"""Infinity (only one crossing direction) is mapped to None."""
	txt = _make_ddg_text("2  3  Infinity  Infinity  Infinity  Infinity")
	result = parse_single_ddg(txt)
	assert result[(2, 3)] is None


def test_parse_single_ddg_negative_infinity_returns_none():
	"""-Infinity is mapped to None (no round-trip free energy)."""
	txt = _make_ddg_text("2  3  -Infinity  -Infinity  -Infinity  -Infinity")
	result = parse_single_ddg(txt)
	assert result[(2, 3)] is None


def test_parse_single_ddg_finite_value_returned():
	"""A pair with finite bias DDG has its highest-lambda value returned."""
	# 4 columns: nobias_0.95, nobias_0.99, bias_0.95, bias_0.99
	# highest bias col (last) = -3.14
	txt = _make_ddg_text("2  3   1.0   1.1   -2.5   -3.14")
	result = parse_single_ddg(txt)
	assert (2, 3) in result
	assert math.isclose(result[(2, 3)], -3.14, rel_tol=1e-6)


def test_parse_single_ddg_key_ordering():
	"""Keys are always (lo, hi) regardless of the order they appear in the file."""
	txt = _make_ddg_text("4  2   1.0   1.1   -2.5   -3.14")
	result = parse_single_ddg(txt)
	assert (2, 4) in result
	assert (4, 2) not in result


def test_parse_single_ddg_multiple_pairs():
	"""Multiple pairs are all parsed, with correct None / float values."""
	txt = _make_ddg_text(
		"2  3   NaN   NaN   NaN   NaN",
		"2  4   1.0   1.1   0.5   0.55",
		"3  4   Infinity  Infinity  Infinity  Infinity",
	)
	result = parse_single_ddg(txt)
	assert result[(2, 3)] is None
	assert math.isclose(result[(2, 4)], 0.55, rel_tol=1e-6)
	assert result[(3, 4)] is None


def test_parse_single_ddg_sample_file():
	"""All pairs in the example output file are NaN or ±Inf → all None."""
	txt = load_example()
	result = parse_single_ddg(txt)
	# The sample file has DDG rows for two sites; all values are NaN or ±Inf
	assert len(result) > 0, "Expected DDG pairs from sample file"
	assert all(v is None for v in result.values()), (
		"All sample DDG values should be None (NaN/Inf)"
	)

