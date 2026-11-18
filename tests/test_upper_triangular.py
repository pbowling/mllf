"""Quadratic biases fill only the upper triangle (no cancellation between directions)."""
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from mllf.file_handling.write_bias_coeff import write_variables_from_actions


def test_quadratic_matrix_is_upper_triangular(tmp_path):
    # 3 nodes, every directed edge; both directions of a pair predict the same
    # value, which is averaged into c[i][j] for i < j
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    edge_type = torch.tensor([0, 0, 1, 0, 1, 1])        # quadratic fwd / bwd
    data = SimpleNamespace(x=torch.randn(3, 10), edge_index=edge_index, edge_type=edge_type)
    extras = {
        'relation_names': ['quadratic_fwd', 'quadratic_bwd', 'skew_fwd', 'skew_bwd',
                           'end_fwd', 'end_bwd', 'linear_fwd', 'linear_bwd'],
        'base_relation_map': {k: (f'{k}_fwd', f'{k}_bwd')
                              for k in ('quadratic', 'skew', 'end', 'linear')},
    }
    actions = torch.tensor([10.0, 20.0, 10.0, 30.0, 20.0, 30.0])
    write_variables_from_actions(str(tmp_path), data, extras, actions,
                                 out_name='variables.py', bias_clip=1000.0)
    ns = {}
    exec((tmp_path / 'variables.py').read_text(), ns)
    c = np.array(yaml.safe_load(ns['bias_string'])['c'])
    assert np.allclose(c, [[0, 10, 20], [0, 0, 30], [0, 0, 0]], atol=0.01)
