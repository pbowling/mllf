import os
import math

from mllf.mlp.setup_pairs import assemble_pairs


def test_assemble_pairs_has_lams_and_cs():
    root = os.path.join('examples', 'training_files')
    results = assemble_pairs(root)

    # find a run that contains site1_sub1 (examples may vary)
    run_name = None
    for rn, pairs in results.items():
        if 'site1_sub1' in pairs:
            run_name = rn
            break
    assert run_name is not None, f"no run containing site1_sub1 found in {list(results.keys())}"
    run_pairs = results[run_name]
    p = run_pairs['site1_sub1']

    # Expect a lams_vector for this site
    assert 'lams_vector' in p['biases']
    assert isinstance(p['biases']['lams_vector'], list)

    # Expect some cs entries mapped to this site (cs keys include site numbers)
    assert 'cs' in p['biases']
    assert any(k.startswith('cs') for k in p['biases']['cs'].keys())


def test_pair_linear_bias_between_subs():
    """Check that the linear bias between two subs is the difference of their
    per-sub linear coefficients (i.e. relative to one another, not to sub 1).

    Uses site2_sub2 and site2_sub3 from the example training run where
    variables62.py contains the lams for site 2. Expect lams2s3 - lams2s2 ≈ 9.68
    """
    root = os.path.join('examples', 'training_files')
    results = assemble_pairs(root)

    # find a run that contains both site2_sub2 and site2_sub3
    run_name = None
    for rn, pairs in results.items():
        if 'site2_sub2' in pairs and 'site2_sub3' in pairs:
            run_name = rn
            break
    assert run_name is not None, f"no run containing site2_sub2/site2_sub3 found in {list(results.keys())}"
    run_pairs = results[run_name]

    p2 = run_pairs['site2_sub2']
    p3 = run_pairs['site2_sub3']

    # both should carry the site's lams_vector
    assert 'lams_vector' in p2['biases']
    lams = p2['biases']['lams_vector']
    # sub indices are 1-based; lams_vector is ordered by sub index
    idx2 = p2['sub'] - 1
    idx3 = p3['sub'] - 1
    # compute pair linear bias as lams(sub3) - lams(sub2)
    pair_bias = float(lams[idx3]) - float(lams[idx2])

    # expected from variables62.py: lams2s2 = -16.07, lams2s3 = -6.39 -> diff ≈ 9.68
    import math
    assert math.isclose(pair_bias, 9.679999999999998, rel_tol=1e-6)


def test_pair_sign_inversion_on_reverse():
    """Ensure that reversing an ordered pair in pairwise_lams flips the sign."""
    root = os.path.join('examples', 'training_files')
    results = assemble_pairs(root)

    # find a run that contains both site2_sub2 and site2_sub3
    run_name = None
    for rn, pairs in results.items():
        if 'site2_sub2' in pairs and 'site2_sub3' in pairs:
            run_name = rn
            break
    assert run_name is not None
    run_pairs = results[run_name]

    p2 = run_pairs['site2_sub2']
    p3 = run_pairs['site2_sub3']

    pw_all = p2['biases'].get('pairwise_biases')
    assert pw_all is not None

    # lams must be present and invert on reversal
    assert 'lams' in pw_all, 'lams pairwise biases missing'
    lm = pw_all['lams']
    assert 'pair_2_3' in lm and 'pair_3_2' in lm
    assert math.isclose(float(lm['pair_2_3']), -float(lm['pair_3_2']), rel_tol=1e-12)

    # For other groups (cs, xs, ss) only assert sign inversion when both keys exist
    for group in ('cs', 'xs', 'ss'):
        mapping = pw_all.get(group)
        if not mapping:
            continue
        if 'pair_2_3' in mapping and 'pair_3_2' in mapping:
            assert math.isclose(float(mapping['pair_2_3']), -float(mapping['pair_3_2']), rel_tol=1e-12), f"sign inversion failed for {group}"
