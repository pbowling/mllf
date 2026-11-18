import math
from pathlib import Path

import pytest

from mllf.file_handling.read_output import (
    parse_single_ddg,
    parse_single_population,
    parse_transitions_and_rates,
    terminated_normally,
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


@pytest.mark.parametrize('row, pair, expected', [
    ("2  3   NaN   NaN   NaN   NaN", (2, 3), None),               # no usable crossings
    ("2  3  Infinity  Infinity  Infinity  Infinity", (2, 3), None),   # one direction only
    ("2  3  -Infinity  -Infinity  -Infinity  -Infinity", (2, 3), None),
    ("2  3   1.0   1.1   -2.5   -3.14", (2, 3), -3.14),           # highest-lambda bias column
    ("4  2   1.0   1.1   -2.5   -3.14", (2, 4), -3.14),           # keys are (lo, hi)
])
def test_parse_single_ddg_values(row, pair, expected):
    result = parse_single_ddg(_make_ddg_text(row))
    assert list(result) == [pair]
    if expected is None:
        assert result[pair] is None
    else:
        assert math.isclose(result[pair], expected, rel_tol=1e-6)


def test_parse_single_ddg_multiple_pairs():
    result = parse_single_ddg(_make_ddg_text(
        "2  3   NaN   NaN   NaN   NaN",
        "2  4   1.0   1.1   0.5   0.55",
        "3  4   Infinity  Infinity  Infinity  Infinity",
    ))
    assert result[(2, 3)] is None and result[(3, 4)] is None
    assert math.isclose(result[(2, 4)], 0.55, rel_tol=1e-6)


def test_parse_single_ddg_sample_file():
    """All pairs in the example output file are NaN or +/-Inf -> all None."""
    txt = load_example()
    result = parse_single_ddg(txt)
    # The sample file has DDG rows for two sites; all values are NaN or +/-Inf
    assert len(result) > 0, "Expected DDG pairs from sample file"
    assert all(v is None for v in result.values()), (
        "All sample DDG values should be None (NaN/Inf)"
    )
