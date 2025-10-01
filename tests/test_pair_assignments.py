import os
import math

from mllf.mlp.setup_pairs import assemble_pairs


def test_faah_site1_pairs_and_cs_key():
    root = os.path.join('examples', 'training_files')
    results = assemble_pairs(root)

    # find the faah solv run (we want the solvent run which contains variables117.py)
    run_name = None
    for rn in results:
        if rn.startswith('faah') and 'solv' in rn:
            run_name = rn
            break
    assert run_name is not None, 'faah run not found'

    pairs = results[run_name]

    # collect site 1 subs
    subs = [p for k, p in pairs.items() if p.get('site') == 1]
    # expect 6 subs
    subs_ids = sorted({p.get('sub') for p in subs})
    assert len(subs_ids) == 6, f'expected 6 subs in site1, found {len(subs_ids)}'

    # number of unordered AB pairs = n*(n-1)/2 -> 15 for n=6
    n = len(subs_ids)
    expected_pairs = n * (n - 1) // 2
    assert expected_pairs == 15

    # check cs1s1s1s2 mapping: look at site1_sub1 biases
    p1 = pairs.get('site1_sub1')
    assert p1 is not None
    pw = p1['biases'].get('pairwise_biases')
    assert pw is not None
    # direct mapping should put cs value under pw['cs']['pair_1_2']
    csmap = pw.get('cs')
    assert csmap is not None
    v = csmap.get('pair_1_2')
    assert v is not None
    assert math.isclose(v, -21.97, rel_tol=1e-6)
