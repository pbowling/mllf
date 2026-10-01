"""Tests for pretraining data hygiene (per-pair, per-bias-type measured-target mask)."""
import numpy as np
import torch

from mllf.cb.graph_utils import build_directed_pairs
from mllf.cb.pretrain_policy import (
    _compute_physical_residual_variance,
    _edge_confidence,
    compute_pairwise_confidence_weights,
    compute_pairwise_measured_mask,
)

PAIRS = build_directed_pairs([3])  # (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)


def _sim_results(pops, ddg, barrier=None, skew=None, end=None):
    return {
        'populations': {str(b): {'counts': {'0.99': p}} for b, p in zip((2, 3, 4), pops)},
        'ddg_pairs': ddg,
        'barrier_pairs': barrier or {},
        'skew_pairs': skew or {},
        'end_pairs': end or {},
    }


def test_mask_follows_measured_signals():
    sr = _sim_results(
        pops=[100, 80, 0],
        ddg={'2_3': 0.5, '2_4': None, '3_4': None},
        barrier={'2_3': 1.1, '2_4': None, '3_4': None},
        skew={'2_3': None}, end={'2_3': 0.2},
    )
    w = compute_pairwise_confidence_weights(sr, [3], PAIRS)
    mask = compute_pairwise_measured_mask(sr, PAIRS, w)
    by_pair = dict(zip(PAIRS, mask))
    assert by_pair[(0, 1)] == [1.0, 1.0, 0.0, 1.0]
    assert by_pair[(1, 0)] == [1.0, 1.0, 0.0, 1.0]   # undirected signals, both directions
    assert by_pair[(0, 2)] == [0.0, 0.0, 0.0, 0.0]   # sub 3 never sampled
    assert by_pair[(1, 2)] == [0.0, 0.0, 0.0, 0.0]


def test_mask_without_profile_fields_keeps_only_linear():
    sr = _sim_results(pops=[100, 80, 90], ddg={'2_3': 0.5, '2_4': 0.1, '3_4': -0.3})
    for k in ('barrier_pairs', 'skew_pairs', 'end_pairs'):
        sr.pop(k)
    w = compute_pairwise_confidence_weights(sr, [3], PAIRS)
    mask = compute_pairwise_measured_mask(sr, PAIRS, w)
    assert all(row == [1.0, 0.0, 0.0, 0.0] for row in mask)


def test_mask_rejects_nonfinite_values():
    sr = _sim_results(pops=[100, 80, 90], ddg={'2_3': 0.5},
                      barrier={'2_3': float('nan')}, end={'2_3': float('inf')})
    w = compute_pairwise_confidence_weights(sr, [3], PAIRS)
    row = dict(zip(PAIRS, compute_pairwise_measured_mask(sr, PAIRS, w)))[(0, 1)]
    assert row == [1.0, 0.0, 0.0, 0.0]


def test_edge_confidence_shapes_and_hygiene_skip():
    logp = torch.zeros(2, 4)
    flat = [[0.5, 0.2]]
    per_dim = [[[0.5, 0.5, 0.0, 0.0], [0.2, 0.0, 0.2, 0.2]]]
    assert _edge_confidence(flat, 0, logp, 'cpu', True).shape == (2, 1)
    assert _edge_confidence(per_dim, 0, logp, 'cpu', True).shape == (2, 4)
    assert _edge_confidence([None], 0, logp, 'cpu', True) is None
    legacy = _edge_confidence([None], 0, logp, 'cpu', False)
    assert torch.equal(legacy, torch.ones(2, 1))


class _Constant:
    """Stand-in deterministic policy predicting zeros."""
    def eval(self):
        return self

    def _forward_edges(self, emb, edge_index, edge_type=None):
        E = edge_index.size(1)
        return torch.zeros(E, 4), torch.zeros(E, 4)


def test_residuals_use_only_measured_targets():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    targets = [[1.0, 2.0, 3.0, 15.0], [-1.0, 2.0, 3.0, 16.0]]
    cache = [(torch.zeros(2, 4), edge_index, targets), (torch.zeros(2, 4), edge_index, targets)]
    weights = [[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]], None]
    res, n = _compute_physical_residual_variance(_Constant(), cache, 'cpu', weights_cache=weights)
    assert n == 1                         # second run has no weights -> skipped
    assert res[3] == []                   # unmeasured end targets excluded
    assert sorted(res[0]) == [-1.0, 1.0]
    res_all, n_all = _compute_physical_residual_variance(_Constant(), cache, 'cpu')
    assert n_all == 2 and len(res_all[3]) == 4


def test_hygiene_loss_scales_with_absolute_weight():
    from mllf.cb.pretrain_policy import _awr_run_loss
    logp = -torch.ones(2, 4)
    active = torch.ones(2, 4, dtype=torch.bool)
    strong = torch.ones(2, 4)
    weak = torch.full((2, 4), 0.1)
    for w in (strong, weak):
        assert _awr_run_loss(w, logp, active, active, w.sum(), False).item() == 1.0
    # Hygiene: a run of poorly resolved targets counts 10x less, not equally.
    s = _awr_run_loss(strong, logp, active, active, strong.sum(), True).item()
    k = _awr_run_loss(weak, logp, active, active, weak.sum(), True).item()
    assert abs(s - 1.0) < 1e-6 and abs(k - 0.1) < 1e-6
