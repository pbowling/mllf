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
