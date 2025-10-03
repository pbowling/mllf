from rl.file_handling.read_output import (
	terminated_normally,
	parse_single_population,
	parse_transitions_and_rates,
)


def load_example():
	p = "examples/rl/14benz_vac_output.txt"
	with open(p, "r", encoding="utf-8", errors="ignore") as fh:
		return fh.read()


def test_terminated_normally():
	txt = load_example()
	assert terminated_normally(txt)


def test_single_population_parsing():
	txt = load_example()
	pops = parse_single_population(txt)
	# from the example, block 2 final column is 76 associated with lambda 0.99
	assert 2 in pops
	assert pops[2]["counts"][0.99] == 76
	# block 2 should be associated with site 1 (per the SINGLE DDG block listing)
	assert pops[2]["site"] == 1


def test_transitions_and_rates_parsing():
	txt = load_example()
	transitions, rates = parse_transitions_and_rates(txt)
	# site 1 has 24 transitions at lambda 0.99 and rate 0.30769
	assert 1 in transitions
	assert transitions[1][0.99] == 24
	assert 1 in rates
	# allow small float rounding tolerance
	assert abs(rates[1][0.99] - 0.30769) < 1e-6

