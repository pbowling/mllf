"""Tests for mllf.cb.reinforce_trainer (migrated from training/workflow.py).

Covers the small, independently-testable helpers around train_epoch:
  - _policy_sample_actions  (REINFORCE vs. NeuralLinear+TS dispatch)
  - _detach_cpu_or_none
  - _save_graph_data_cache / _load_graph_data_cache (round trip)
  - format_pair_success
  - append_summary

train_epoch itself is not covered here: it drives real SLURM submission and
Uni-Mol embedding computation end to end, which is out of scope for a fast
unit test (consistent with how the rest of the suite treats the equally
heavy build_graph_and_data / run_training paths).
"""
from pathlib import Path

import torch

from mllf.cb.policy import UnimolPolicy
from mllf.cb.reinforce_trainer import (
    _CachedGraphData,
    _detach_cpu_or_none,
    _load_graph_data_cache,
    _policy_sample_actions,
    _save_graph_data_cache,
    append_summary,
    format_pair_success,
)


def _toy_graph(n_nodes=6, unimol_dim=512, dual=True, seed=0):
    torch.manual_seed(seed)
    emb_dim = 1024 if dual else unimol_dim
    emb = torch.randn(n_nodes, emb_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
                                [1, 0, 3, 2, 5, 4]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    return emb, edge_index, edge_type


# ---------------------------------------------------------------------------
# _policy_sample_actions
# ---------------------------------------------------------------------------

def test_policy_sample_actions_reinforce_mode_returns_logp():
    policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True,
                           use_bayesian_heads=False)
    emb, edge_index, edge_type = _toy_graph()
    with torch.no_grad():
        actions, logp = _policy_sample_actions(policy, emb, edge_index, edge_type)
    assert actions.shape == (edge_index.shape[1], 4)
    assert logp is not None
    # get_actions sums log-prob over the 4 bias-type dims -> one scalar per edge.
    assert logp.shape == (edge_index.shape[1],)


def test_policy_sample_actions_bayesian_mode_returns_none_logp():
    policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True,
                           use_bayesian_heads=True)
    emb, edge_index, edge_type = _toy_graph()
    with torch.no_grad():
        actions, logp = _policy_sample_actions(policy, emb, edge_index, edge_type)
    assert actions.shape == (edge_index.shape[1], 4)
    assert logp is None


def test_policy_sample_actions_deterministic_reproducible():
    policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True,
                           use_bayesian_heads=False)
    emb, edge_index, edge_type = _toy_graph()
    with torch.no_grad():
        actions1, _ = _policy_sample_actions(policy, emb, edge_index, edge_type, deterministic=True)
        actions2, _ = _policy_sample_actions(policy, emb, edge_index, edge_type, deterministic=True)
    assert torch.equal(actions1, actions2)


# ---------------------------------------------------------------------------
# _detach_cpu_or_none
# ---------------------------------------------------------------------------

def test_detach_cpu_or_none_with_tensor():
    t = torch.randn(3, requires_grad=True)
    out = _detach_cpu_or_none(t)
    assert out.device.type == 'cpu'
    assert not out.requires_grad
    assert torch.equal(out, t.detach())


def test_detach_cpu_or_none_with_none():
    assert _detach_cpu_or_none(None) is None


# ---------------------------------------------------------------------------
# _save_graph_data_cache / _load_graph_data_cache
# ---------------------------------------------------------------------------

def test_graph_data_cache_roundtrip(tmp_path):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)
    data = _CachedGraphData(x, edge_index, edge_type)

    _save_graph_data_cache(tmp_path, data)
    assert (tmp_path / 'graph_data.pt').exists()

    loaded = _load_graph_data_cache(tmp_path, device='cpu')
    assert loaded is not None
    assert torch.equal(loaded.x, x)
    assert torch.equal(loaded.edge_index, edge_index)
    assert torch.equal(loaded.edge_type, edge_type)


def test_graph_data_cache_roundtrip_with_none_edge_type(tmp_path):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = _CachedGraphData(x, edge_index, None)

    _save_graph_data_cache(tmp_path, data)
    loaded = _load_graph_data_cache(tmp_path, device='cpu')

    assert loaded.edge_type is None


def test_load_graph_data_cache_returns_none_when_absent(tmp_path):
    assert _load_graph_data_cache(tmp_path, device='cpu') is None


# ---------------------------------------------------------------------------
# format_pair_success
# ---------------------------------------------------------------------------

def test_format_pair_success_none_when_no_pairs():
    assert format_pair_success({'total': 0, 'crossed': 0, 'balanced': 0, 'both': 0}, 'train') is None


def test_format_pair_success_formats_percentages():
    ps = {'total': 4, 'crossed': 2, 'balanced': 1, 'both': 1}
    line = format_pair_success(ps, 'validation')
    assert line is not None
    assert 'validation' in line
    assert 'crossed 2/4 (50.0%)' in line
    assert 'balanced 1/4 (25.0%)' in line
    assert 'both 1/4 (25.0%)' in line


# ---------------------------------------------------------------------------
# append_summary
# ---------------------------------------------------------------------------

def test_append_summary_appends_lines(tmp_path):
    summary_path = tmp_path / 'epoch_summary.log'
    append_summary(summary_path, 'first line')
    append_summary(summary_path, 'second line')
    assert summary_path.read_text() == 'first line\nsecond line\n'


def test_append_summary_noop_when_path_is_none():
    # Should not raise.
    append_summary(None, 'ignored')


def test_append_summary_noop_on_write_failure(tmp_path):
    # A directory path can't be opened for append -- append_summary should
    # swallow the error rather than propagate it (it's a best-effort log).
    bad_path = tmp_path / 'a_directory'
    bad_path.mkdir()
    append_summary(bad_path, 'ignored')
