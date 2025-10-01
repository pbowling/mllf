import os
import math

from mllf.file_handling.read_bias_coeff import read_bias_coeff


def test_read_bias_coeff_parses_old_file():
	"""Ensure read_bias_coeff can parse the example variables85.inp (old ALF format)."""
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

def test_read_bias_coeff_parses_new_file():
	"""Ensure read_bias_coeff can parse the example variables16.py (new ALF format).

	The `b:` nested list encodes linear biases for multiple i blocks. Each block
	begins with a 0.0 value. We assert some sample lams assignments to ensure
	the zero-handling logic is correct.
	"""
	here = os.path.dirname(os.path.dirname(__file__))
	fn = os.path.join(here, 'examples', 'variables16.py')

	data = read_bias_coeff(fn)

	# top-level groups should exist
	assert set(data.keys()) == {'lams', 'cs', 'xs', 'ss'}

	# According to the file, the first block (i=1) starts with 0.0 then -11.4, -6.58, ...
	assert math.isclose(data['lams']['lams1s1'], 0.0, rel_tol=1e-6)
	assert math.isclose(data['lams']['lams1s2'], -11.4, rel_tol=1e-6)
	assert math.isclose(data['lams']['lams1s3'], -6.58, rel_tol=1e-6)

	# The second block (i=2) should start with 0.0 as well (after encountering a 0.0 that isn't at j=1)
	# find a sample from the second block: according to file lams2s1 == 0.0 and lams2s2 == -11.629999999999999
	assert math.isclose(data['lams']['lams2s1'], 0.0, rel_tol=1e-6)
	assert math.isclose(data['lams']['lams2s2'], -11.629999999999999, rel_tol=1e-6)

