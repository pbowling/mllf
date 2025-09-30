import os
import math

from src.file_handling.read_bias_coeff import read_bias_coeff


def test_read_bias_coeff_parses_old_file():
	"""Ensure read_bias_coeff can parse the example variables85.inp (old format)."""
	here = os.path.dirname(os.path.dirname(__file__))
	fn = os.path.join(here, 'examples', 'variables85.inp')

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

