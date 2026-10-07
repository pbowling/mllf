"""Tests for mllf.cb.hindsight (hindsight bias targets and per-pair offsets)."""
import json
import math

import pytest
import torch

from mllf.cb.hindsight import (
    BARRIER_C_SLOPE,
    KB_KCAL,
    LINEAR_STEP_CAP,
    OFFSET_KEY,
    apply_pair_offsets,
    combo_node_labels,
    hindsight_targets,
    linear_hindsight,
    pair_offset,
    quadratic_hindsight,
    read_applied_biases,
    record_pair_targets,
)

KT = KB_KCAL * 298.15
EDGES = torch.tensor([[0, 1], [1, 0]])  # one pair, both directions


def test_linear_balances_populations():
    # p_j twice p_i: b[j] must come down by kT ln 2
    t, v = linear_hindsight(5.0, 100, 200, transitions=20, kT=KT)
    assert t == pytest.approx(5.0 - KT * math.log(2.0))
    assert 0 < v < 1.0


def test_linear_balanced_run_keeps_value():
    t, _ = linear_hindsight(5.0, 150, 150, transitions=20, kT=KT)
    assert t == pytest.approx(5.0)


def test_linear_more_transitions_lower_variance():
    _, v_few = linear_hindsight(0.0, 100, 120, transitions=4, kT=KT)
    _, v_many = linear_hindsight(0.0, 100, 120, transitions=100, kT=KT)
    assert v_many < v_few


def test_linear_trapped_steps_toward_unsampled_side():
    t, v = linear_hindsight(5.0, 1000, 0, transitions=0, kT=KT)
    assert t == pytest.approx(5.0 + LINEAR_STEP_CAP)
    t, _ = linear_hindsight(5.0, 0, 1000, transitions=0, kT=KT)
    assert t == pytest.approx(5.0 - LINEAR_STEP_CAP)
    assert math.isfinite(v)


def test_linear_no_samples_is_no_information():
    _, v = linear_hindsight(5.0, 0, 0, transitions=0, kT=KT)
    assert math.isinf(v)


def test_linear_step_is_capped():
    t, v_cap = linear_hindsight(0.0, 1, 10 ** 6, transitions=2, kT=KT)
    assert t == pytest.approx(-LINEAR_STEP_CAP)
    _, v_small = linear_hindsight(0.0, 100, 110, transitions=2, kT=KT)
    assert v_cap > v_small


def test_quadratic_high_barrier_raises_c():
    t, v = quadratic_hindsight(20.0, barrier=2.2, barrier_target=1.2)
    assert t == pytest.approx(20.0 + 1.0 / abs(BARRIER_C_SLOPE))
    assert math.isfinite(v)


def test_quadratic_negative_barrier_lowers_c_capped():
    t, _ = quadratic_hindsight(27.0, barrier=-3.0, barrier_target=1.2, step_cap=8.0)
    assert t == pytest.approx(19.0)


def test_quadratic_missing_barrier_is_no_information():
    _, v = quadratic_hindsight(20.0, barrier=None, barrier_target=1.2)
    assert math.isinf(v)


def _write_variables(run_dir, b1, c01):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'variables.py').write_text(
        "bias_string = '''\n"
        f"b:\n- - 0.0\n  - {b1}\n"
        f"c:\n- - 0.0\n  - {c01}\n- - 0.0\n  - 0.0\n"
        "x:\n- - 0.0\n  - 0.0\n- - 0.0\n  - 0.0\n"
        "s:\n- - 0.0\n  - 0.0\n- - 0.0\n  - 0.0\n"
        "'''\n")


def test_hindsight_targets_edge_directions(tmp_path):
    _write_variables(tmp_path, b1=8.0, c01=20.0)
    applied = read_applied_biases(tmp_path)
    saved = {'populations': [100, 200], 'transition_pairs': {'2_3': 30},
             'barrier_pairs': {'2_3': 2.2}}
    t, v = hindsight_targets(EDGES, applied, saved, barrier_target=1.2)
    db = 8.0 - KT * math.log(2.0)
    assert t[0, 0].item() == pytest.approx(db)       # 0 -> 1: b[1] - b[0]
    assert t[1, 0].item() == pytest.approx(-db)      # 1 -> 0: antisymmetric
    assert t[0, 1].item() == pytest.approx(t[1, 1].item())  # c symmetric
    assert t[0, 1].item() > 20.0
    assert torch.isfinite(v[:, :2]).all()
    assert torch.isinf(v[:, 2:]).all()   # no skew/end measured


def test_hindsight_targets_fall_back_to_site_transitions(tmp_path):
    _write_variables(tmp_path, b1=0.0, c01=20.0)
    saved = {'populations': [100, 100], 'transitions': [12], 'barrier_pairs': {}}
    t, v = hindsight_targets(EDGES, read_applied_biases(tmp_path), saved, barrier_target=1.2)
    assert torch.isfinite(v[:, 0]).all()
    assert torch.isinf(v[:, 1]).all()


def _write_mapping(combo, subs):
    entries = [{'new_site': 1, 'new_sub': n + 1, 'original_site': 1, 'original_sub': s}
               for n, s in enumerate(subs)]
    (combo / 'mapping.json').write_text(json.dumps(entries))


def test_offsets_follow_original_sub_order(tmp_path):
    # Same physical pair (subs 4, 8) seen with nodes in both orders.
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    _write_mapping(a, [4, 8])
    _write_mapping(b, [8, 4])
    state = {}
    targets = torch.tensor([[2.0, 30.0, 0, 0], [-2.0, 30.0, 0, 0]], dtype=torch.float64)
    var = torch.tensor([[0.01, 0.01, math.inf, math.inf]] * 2, dtype=torch.float64)
    record_pair_targets(state, 'sys', combo_node_labels(a), EDGES, targets, var)
    assert list(state[OFFSET_KEY]) == ['sys|s1|4|8']

    mean = torch.zeros(2, 4)
    out, n = apply_pair_offsets(mean, mean, state, 'sys', combo_node_labels(b), EDGES)
    assert n == 2
    # In combo b node 0 is sub 8, so edge 0 -> 1 is 8 -> 4: db = -2
    assert out[0, 0].item() == pytest.approx(-2.0, abs=0.01)
    assert out[1, 0].item() == pytest.approx(2.0, abs=0.01)
    assert out[0, 1].item() == pytest.approx(30.0, abs=0.1)


def test_offset_keeps_exploration_noise():
    state = {OFFSET_KEY: {'sys|s1|1|2': {'quadratic': [[20.0, 0.01]]}}}
    labels = [(1, 1), (1, 2)]
    mean = torch.tensor([[0.0, 15.0, 0, 0], [0.0, 15.0, 0, 0]])
    sampled = mean.clone()
    sampled[:, 1] += 0.5
    out, _ = apply_pair_offsets(sampled, mean, state, 'sys', labels, EDGES)
    assert out[0, 1].item() == pytest.approx(20.5, abs=0.05)


def test_offset_history_is_bounded():
    state = {}
    labels = [(1, 1), (1, 2)]
    for k in range(8):
        t = torch.tensor([[float(k), 20.0, 0, 0], [-float(k), 20.0, 0, 0]], dtype=torch.float64)
        record_pair_targets(state, 'sys', labels, EDGES, t, torch.ones(2, 4, dtype=torch.float64),
                            history=5)
    obs = state[OFFSET_KEY]['sys|s1|1|2']['linear']
    assert [o[0] for o in obs] == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_pair_offset_shrinks_toward_model():
    assert pair_offset(0.0, [], prior_sd=3.0) == 0.0
    tight = pair_offset(0.0, [[4.0, 0.01]], prior_sd=3.0)
    loose = pair_offset(0.0, [[4.0, 100.0]], prior_sd=3.0)
    assert tight == pytest.approx(4.0, abs=0.01)
    assert 0 < loose < 1.0


def test_apply_pair_offsets_noop_without_state():
    a = torch.ones(2, 4)
    out, n = apply_pair_offsets(a, a, None, 'sys', [(1, 1), (1, 2)], EDGES)
    assert n == 0 and out is a
