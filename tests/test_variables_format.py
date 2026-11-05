"""Storage conventions of an ALF variables file (variables85.inp)."""
from pathlib import Path

from mllf.file_handling.read_bias_coeff import parse_old


def test_alf_variables_storage_conventions():
    d = parse_old(str(Path(__file__).parent / 'samples' / 'variables85.inp'))
    lams, cs, xs, ss = d['lams'], d['cs'], d['xs'], d['ss']
    # linear biases are relative to each site's first substituent
    assert abs(lams['lams1s1']) < 0.01 and abs(lams['lams2s1']) < 0.01
    # quadratic: upper triangle only
    assert 'cs1s1s1s2' in cs and 'cs1s2s1s1' not in cs
    # skew and end: both directions, and not antisymmetric
    for terms, key in ((xs, 'xs'), (ss, 'ss')):
        fwd, rev = terms[f'{key}1s1s1s2'], terms[f'{key}1s2s1s1']
        assert abs(fwd + rev) > 0.01
