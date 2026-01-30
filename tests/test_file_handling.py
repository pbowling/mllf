import os
import math

from mllf.file_handling.read_bias_coeff import read_bias_coeff
from mllf.file_handling.read_rtf import parse_rtf_file, parse_rtf_dir


def test_read_bias_coeff_parses_old_file():
	"""Ensure read_bias_coeff can parse the example variables85.inp (old ALF format)."""
	here = os.path.dirname(os.path.dirname(__file__))
	fn = os.path.join(here, 'examples', 'cb', 'variables85.inp')

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
	fn = os.path.join(here, 'examples', 'cb', 'variables.py')

	data = read_bias_coeff(fn)

	# top-level groups should exist (scalar entries)
	assert set(data.keys()) >= {'lams', 'cs', 'xs', 'ss'}
	
	# matrices should also be present
	assert 'b' in data and 'c' in data and 'x' in data and 's' in data
	
	# b should be a nested list (single row vector)
	b = data['b']
	assert isinstance(b, list), "b should be a list"
	assert len(b) == 1, "b should be a single row (nested list)"
	assert isinstance(b[0], list), "b[0] should be a list of values"
	assert len(b[0]) == 31, "b should have 31 values for this example"
	assert b[0][0] == 0.0 and b[0][1] == 0.13, "b vector values should match"
	
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
	here = os.path.dirname(os.path.dirname(__file__))
	rtf_path = os.path.join(here, 'examples', 'cb', '14benz_solv_5.5', 'site1_sub1_pres.rtf')

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
	d = os.path.join(here, 'examples', 'cb', '14benz_solv_5.5')
	results = parse_rtf_dir(d)
	assert 'site1_sub1' in results

