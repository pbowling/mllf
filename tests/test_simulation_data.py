from pathlib import Path
from mllf.file_handling.read_output import (
	terminated_normally,
	parse_single_population,
	parse_transitions_and_rates,
)


def load_example():
	# Find the repository root and construct path to example file
	test_file = Path(__file__)
	repo_root = test_file.parent.parent
	p = repo_root / "examples" / "cb" / "14benz_solv_5.5" / "output.txt"
	
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

