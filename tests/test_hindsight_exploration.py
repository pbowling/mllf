"""Tests for mllf.cb.hindsight: skew/end targets, held-off endpoints, and
per-pair offset sampling and exploration."""
import json
import math
import random

import pytest
import torch

from mllf.cb.hindsight import (
    END_S_SLOPE_HI,
    END_S_SLOPE_LO,
    END_STEP_CAP,
    EXPLORE_FACTOR_MAX,
    OFFSET_KEY,
    SKEW_X_SLOPE,
    apply_pair_offsets,
    end_hindsight,
    endpoint_held_off,
    held_off_end_hindsight,
    hindsight_targets,
    pair_attention,
    read_applied_biases,
    record_pair_outcomes,
    record_pair_targets,
    skew_hindsight,
    update_exploration,
)

EDGES = torch.tensor([[0, 1], [1, 0]])  # one pair, both directions
LABELS = [(1, 1), (1, 2)]
KEY = 'sys|s1|1|2'

VARIABLES = """bias_string = '''
b:
- - 0.0
  - 5.0
c:
- - 0.0
  - 20.0
- - 0.0
  - 0.0
x:
- - 0.0
  - 1.0
- - 0.5
  - 0.0
s:
- - 0.0
  - {s01}
- - {s10}
  - 0.0
'''
"""


def _write_variables(run_dir, s01, s10):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'variables.py').write_text(VARIABLES.format(s01=s01, s10=s10))


# ── skew / end / held-off endpoints ──────────────────────────────────────────

def test_skew_step_removes_asymmetry_split_by_direction():
    (x_lh, v1), (x_hl, v2) = skew_hindsight(1.0, 0.5, skew=0.2)
    d_lh, d_hl = x_lh - 1.0, x_hl - 0.5
    assert d_lh == pytest.approx(0.2 / (2 * SKEW_X_SLOPE))
    assert d_hl == pytest.approx(-d_lh)
    # predicted change in skew: -0.19 dx_lh + 0.19 dx_hl = -skew
    assert -SKEW_X_SLOPE * d_lh + SKEW_X_SLOPE * d_hl == pytest.approx(-0.2)
    assert math.isfinite(v1) and v1 == v2


def test_end_step_removes_asymmetry():
    (s_lh, _), (s_hl, _) = end_hindsight(2.0, 2.0, end=0.3)
    d_lh, d_hl = s_lh - 2.0, s_hl - 2.0
    assert -END_S_SLOPE_HI * d_lh + END_S_SLOPE_LO * d_hl == pytest.approx(-0.3)


def test_end_step_is_capped_and_missing_signal_is_no_information():
    (s_lh, _), _ = end_hindsight(2.0, 2.0, end=10.0)
    assert s_lh - 2.0 == pytest.approx(END_STEP_CAP)
    (_, v), (_, v2) = end_hindsight(2.0, 2.0, end=None)
    assert math.isinf(v) and math.isinf(v2)


def test_endpoint_held_off_detection():
    held = {'edge': 1000, 'at_lo': 0, 'near_lo': 800, 'near_hi': 0, 'at_hi': 0}
    barely = {'edge': 1000, 'at_lo': 20, 'near_lo': 980, 'near_hi': 0, 'at_hi': 0}
    fine = {'edge': 1000, 'at_lo': 300, 'near_lo': 200, 'near_hi': 200, 'at_hi': 300}
    assert endpoint_held_off(held, 'lo') and not endpoint_held_off(held, 'hi')
    assert endpoint_held_off(barely, 'lo')
    assert not endpoint_held_off(fine, 'lo') and not endpoint_held_off(fine, 'hi')
    assert not endpoint_held_off(None, 'lo')


def test_held_off_end_shrinks_geometrically_to_reference():
    s, path = 15.0, []
    for _ in range(4):
        s, v = held_off_end_hindsight(s, reference=1.5)
        path.append(s)
        assert math.isfinite(v)
    assert path == pytest.approx([8.25, 4.875, 3.1875, 2.34375])
    _, v = held_off_end_hindsight(1.0, reference=1.5)
    assert math.isinf(v)


def test_hindsight_targets_held_off_lo_end_targets_s_hi_lo(tmp_path):
    # mup1-like: lambda hovers next to the lo end and never reaches either end.
    _write_variables(tmp_path, s01=7.6, s10=15.6)
    saved = {'populations': [0, 0], 'transitions': [0], 'barrier_pairs': {},
             'endpoint_pairs': {'2_3': {'edge': 1000, 'at_lo': 0, 'near_lo': 800,
                                        'near_hi': 0, 'at_hi': 0}}}
    t, v = hindsight_targets(EDGES, read_applied_biases(tmp_path), saved, barrier_target=1.2)
    # edge 1 -> 0 carries s[hi][lo], which shapes the lo end: halfway to 1.5
    assert t[1, 3].item() == pytest.approx(15.6 - 0.5 * (15.6 - 1.5))
    assert math.isfinite(v[1, 3])
    # edge 0 -> 1 carries s[lo][hi] (hi end): no signal
    assert math.isinf(v[0, 3])
    assert math.isinf(v[0, 0])            # b: neither substituent sampled


def test_hindsight_targets_skew_end_per_direction(tmp_path):
    _write_variables(tmp_path, s01=2.0, s10=2.0)
    saved = {'populations': [100, 100], 'transition_pairs': {'2_3': 30},
             'barrier_pairs': {'2_3': 1.2}, 'skew_pairs': {'2_3': 0.2}, 'end_pairs': {'2_3': 0.3}}
    t, v = hindsight_targets(EDGES, read_applied_biases(tmp_path), saved, barrier_target=1.2)
    (x_lh, _), (x_hl, _) = skew_hindsight(1.0, 0.5, 0.2)
    (s_lh, _), (s_hl, _) = end_hindsight(2.0, 2.0, 0.3)
    assert t[0, 2].item() == pytest.approx(x_lh) and t[1, 2].item() == pytest.approx(x_hl)
    assert t[0, 3].item() == pytest.approx(s_lh) and t[1, 3].item() == pytest.approx(s_hl)
    assert torch.isfinite(v).all()


# ── offsets for x / s, sampling, exploration ─────────────────────────────────

def test_skew_end_offsets_recorded_per_direction():
    state = {}
    t = torch.tensor([[0.0, 0.0, 1.5, 3.0], [0.0, 0.0, 0.7, 9.0]], dtype=torch.float64)
    v = torch.tensor([[math.inf, math.inf, 0.1, 0.1]] * 2, dtype=torch.float64)
    record_pair_targets(state, 'sys', LABELS, EDGES, t, v)
    entry = state[OFFSET_KEY][KEY]
    assert entry['skew_fwd'] == [[1.5, 0.1, False]] and entry['skew_rev'] == [[0.7, 0.1, False]]
    assert entry['end_fwd'] == [[3.0, 0.1, False]] and entry['end_rev'] == [[9.0, 0.1, False]]
    mean = torch.zeros(2, 4)
    out, _ = apply_pair_offsets(mean, mean, state, 'sys', LABELS, EDGES)
    assert out[0, 3].item() == pytest.approx(3.0, abs=0.05)
    assert out[1, 3].item() == pytest.approx(9.0, abs=0.1)


def test_sampled_offsets_shared_by_both_edges():
    mean = torch.zeros(2, 4)
    widened = {OFFSET_KEY: {KEY: {'explore': {s: 2.0 for s in (
        'linear', 'quadratic', 'skew_fwd', 'skew_rev', 'end_fwd', 'end_rev')}}}}
    out, n = apply_pair_offsets(mean, mean, widened, 'sys', LABELS, EDGES, sample=True,
                                rng=random.Random(1))
    assert n == 2
    assert out[0, 0].item() == pytest.approx(-out[1, 0].item())   # linear antisymmetric
    assert out[0, 1].item() == pytest.approx(out[1, 1].item())    # quadratic shared
    assert out[0, 3].item() != pytest.approx(out[1, 3].item())    # end: per direction


def _spread(state, n=400):
    rng = random.Random(0)
    mean = torch.zeros(2, 4)
    vals = [apply_pair_offsets(mean, mean, state, 'sys', LABELS, EDGES,
                               sample=True, rng=rng)[0][0, 3].item() for _ in range(n)]
    m = sum(vals) / n
    return m, (sum((x - m) ** 2 for x in vals) / n) ** 0.5


def test_sampling_spread_base_widened_and_shrunk_by_data():
    _, sd_new = _spread({})
    assert sd_new == 0.0                       # new pair: only the model's Thompson noise
    _, sd_once = _spread({OFFSET_KEY: {KEY: {'explore': {'end_fwd': 2.0}}}})
    assert sd_once == pytest.approx(1.0, rel=0.15)                # base 0.5 x factor 2
    _, sd_wide = _spread({OFFSET_KEY: {KEY: {'explore': {'end_fwd': 8.0}}}})
    assert sd_wide == pytest.approx(4.0, rel=0.15)
    m, sd_data = _spread({OFFSET_KEY: {KEY: {'explore': {'end_fwd': 8.0},
                                            'end_fwd': [[6.0, 0.04]]}}})
    assert m == pytest.approx(6.0, abs=0.1) and sd_data < 0.3     # data sets the spread


def _saved(pops, trans):
    return {'populations': pops, 'transition_pairs': {'2_3': trans}}


def test_exploration_widens_on_directionless_failure_and_recovers():
    state = {}
    no_info = torch.full((2, 4), math.inf, dtype=torch.float64)
    for _ in range(5):
        update_exploration(state, 'sys', LABELS, EDGES, no_info, _saved([1000, 0], 0))
    f = state[OFFSET_KEY][KEY]['explore']
    assert f['end_fwd'] == EXPLORE_FACTOR_MAX and f['linear'] == EXPLORE_FACTOR_MAX
    info = torch.ones(2, 4, dtype=torch.float64)
    update_exploration(state, 'sys', LABELS, EDGES, info, _saved([500, 500], 30))
    assert f['end_fwd'] == EXPLORE_FACTOR_MAX / 2


def test_failure_with_a_direction_does_not_widen():
    info = torch.ones(2, 4, dtype=torch.float64)
    widened, _ = update_exploration({}, 'sys', LABELS, EDGES, info, _saved([1000, 0], 0))
    assert widened == 0


def test_record_pair_outcomes_counts_consecutive_failures():
    state = {}
    for _ in range(3):
        record_pair_outcomes(state, 'sys', LABELS, EDGES, _saved([1000, 0], 0))
    assert state[OFFSET_KEY][KEY]['fail_streak'] == 3
    record_pair_outcomes(state, 'sys', LABELS, EDGES, _saved([500, 500], 30))
    assert state[OFFSET_KEY][KEY]['fail_streak'] == 0


def test_record_pair_outcomes_counts_epochs_once_per_epoch():
    state = {}
    for ep in (0, 0, 1, 3):           # e.g. both orientations in epoch 0
        record_pair_outcomes(state, 'sys', LABELS, EDGES, _saved([500, 500], 30), stage='s', epoch=ep)
    st = state[OFFSET_KEY][KEY]['stages']['s']
    assert st['visits'] == 4 and st['epochs'] == 3 and st['last_epoch'] == 3


@pytest.mark.parametrize('pops,trans,stuck', [([1000, 0], 0, True),     # stuck at an endpoint
                                               ([0, 0], 0, True),       # stranded
                                               ([500, 500], 30, False)])
def test_stuck_or_stranded_run_sets_run_both(pops, trans, stuck):
    state = {}
    record_pair_outcomes(state, 'sys', LABELS, EDGES, _saved(pops, trans))
    assert bool(state[OFFSET_KEY][KEY].get('run_both')) is stuck


def test_run_both_stays_set_after_a_crossing():
    state = {}
    record_pair_outcomes(state, 'sys', LABELS, EDGES, _saved([1000, 0], 0))
    record_pair_outcomes(state, 'sys', LABELS, EDGES, _saved([500, 500], 30))
    assert state[OFFSET_KEY][KEY]['run_both']


def test_persistent_explained_failure_widens_every_slot():
    # A b target on every run (a direction toward the unsampled sub) explains
    # each failure, so widening waits for the third failure in a row and then
    # includes the measured b slot.
    state = {OFFSET_KEY: {KEY: {'linear': [[3.0, 2.25, True]]}}}
    info = torch.full((2, 4), math.inf, dtype=torch.float64)
    info[:, 0] = 2.25
    saved = _saved([1000, 0], 0)
    for n in range(1, 4):
        record_pair_outcomes(state, 'sys', LABELS, EDGES, saved)
        widened, _ = update_exploration(state, 'sys', LABELS, EDGES, info, saved)
        assert (widened > 0) == (n >= 3)
    f = state[OFFSET_KEY][KEY]['explore']
    assert f['linear'] == 2.0 and f['quadratic'] == 2.0
    # A success halves the factors again and resets the streak.
    ok = _saved([500, 500], 30)
    record_pair_outcomes(state, 'sys', LABELS, EDGES, ok)
    update_exploration(state, 'sys', LABELS, EDGES, info, ok)
    assert f['linear'] == 1.0 and state[OFFSET_KEY][KEY]['fail_streak'] == 0


def test_persistent_widening_can_be_disabled():
    state = {OFFSET_KEY: {KEY: {'fail_streak': 10}}}
    info = torch.ones(2, 4, dtype=torch.float64)
    widened, _ = update_exploration(state, 'sys', LABELS, EDGES, info, _saved([1000, 0], 0),
                                    persistent_failures=0)
    assert widened == 0


def test_pair_attention_prefers_failed_and_widened_pairs():
    state = {}
    assert pair_attention(state, 'sys', LABELS, EDGES) == 0.0
    record_pair_outcomes(state, 'sys', LABELS, EDGES, _saved([1000, 0], 0))
    failed = pair_attention(state, 'sys', LABELS, EDGES)
    assert failed == pytest.approx(1.0)
    state[OFFSET_KEY][KEY]['explore'] = {'end_fwd': EXPLORE_FACTOR_MAX}
    assert pair_attention(state, 'sys', LABELS, EDGES) == pytest.approx(2.0)
    record_pair_outcomes(state, 'sys', LABELS, EDGES, _saved([500, 500], 30))
    assert pair_attention(state, 'sys', LABELS, EDGES) == pytest.approx(1.0)


def test_explained_failure_does_not_widen_other_coefficients():
    # Only the end bias has a direction (held-off endpoint): c/x/b stay put.
    var = torch.full((2, 4), math.inf, dtype=torch.float64)
    var[1, 3] = 4.0
    widened, _ = update_exploration({}, 'sys', LABELS, EDGES, var, _saved([0, 0], 0))
    assert widened == 0


def test_data_backed_offset_dominates_prior():
    # A held-off end target (sd 3.5) moves the offset most of the way.
    state = {OFFSET_KEY: {KEY: {'end_rev': [[8.55, 12.4]]}}}
    mean = torch.tensor([[0.0, 0.0, 0.0, 7.6], [0.0, 0.0, 0.0, 15.6]])
    out, _ = apply_pair_offsets(mean, mean, state, 'sys', LABELS, EDGES)
    assert out[1, 3].item() < 15.6 - 0.8 * (15.6 - 8.55)


def test_measured_slot_drops_model_thompson_noise():
    state = {OFFSET_KEY: {KEY: {'quadratic': [[30.0, 0.0001]]}}}
    mean = torch.tensor([[0.0, 20.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0]])
    noisy = mean.clone()
    noisy[:, 1] += 5.0                       # the model's own Thompson draw
    out, _ = apply_pair_offsets(noisy, mean, state, 'sys', LABELS, EDGES, sample=True,
                                rng=random.Random(0))
    assert out[0, 1].item() == pytest.approx(30.0, abs=0.1)   # pair data, no +5
    assert out[0, 3].item() == 0.0                            # unmeasured, not widened: model only


def test_directional_targets_replace_each_other_measured_ones_accumulate():
    state = {}
    t = torch.zeros(2, 4, dtype=torch.float64)
    v = torch.full((2, 4), math.inf, dtype=torch.float64)
    v[:, 0] = 2.25
    d = torch.zeros(2, 4, dtype=torch.bool)
    d[:, 0] = True
    for b in (8.0, 10.0, 12.0):                     # trapped: three successive steps
        t[0, 0], t[1, 0] = b, -b
        record_pair_targets(state, 'sys', LABELS, EDGES, t, v, directional=d)
    assert state[OFFSET_KEY][KEY]['linear'] == [[12.0, 2.25, True]]
    v[:, 0] = 0.1
    for b in (12.5, 12.7):                          # now measured two-sided
        t[0, 0], t[1, 0] = b, -b
        record_pair_targets(state, 'sys', LABELS, EDGES, t, v)
    assert [o[0] for o in state[OFFSET_KEY][KEY]['linear']] == [12.0, 12.5, 12.7]


def test_hindsight_targets_directional_mask(tmp_path):
    _write_variables(tmp_path, s01=7.6, s10=15.6)
    saved = {'populations': [500, 0], 'transitions': [0], 'barrier_pairs': {},
             'endpoint_pairs': {'2_3': {'edge': 1000, 'at_lo': 0, 'near_lo': 800,
                                        'near_hi': 0, 'at_hi': 0}}}
    _, v, d = hindsight_targets(EDGES, read_applied_biases(tmp_path), saved,
                                barrier_target=1.2, return_directional=True)
    assert d[:, 0].all()                       # b: toward the unsampled substituent
    assert d[1, 3] and not d[0, 3]             # s at the held-off lo end only
    saved_ok = {'populations': [500, 400], 'transition_pairs': {'2_3': 30}}
    _, _, d2 = hindsight_targets(EDGES, read_applied_biases(tmp_path), saved_ok,
                                 barrier_target=1.2, return_directional=True)
    assert not d2.any()


def _trapped(state, b1, steps, base=3.0, max_step=12.0, labels=LABELS):
    """Run `steps` consecutive trapped runs (sub 2 never sampled) and return step sizes."""
    from mllf.cb.hindsight import grow_directional_b_steps
    sizes = []
    for _ in range(steps):
        applied = {'b': [0.0, b1]}
        t = torch.zeros(2, 4, dtype=torch.float64)
        t[0, 0], t[1, 0] = b1 + base, -(b1 + base)      # base step toward sub 2
        d = torch.zeros(2, 4, dtype=torch.bool)
        d[:, 0] = True
        grow_directional_b_steps(state, 'sys', labels, EDGES, t, d, applied,
                                 base_step=base, max_step=max_step)
        assert t[1, 0].item() == pytest.approx(-t[0, 0].item())   # antisymmetric kept
        sizes.append(t[0, 0].item() - b1)
        b1 = t[0, 0].item()
    return sizes


def test_b_steps_double_while_trapped_and_cap():
    assert _trapped({}, 5.0, 4) == pytest.approx([3.0, 6.0, 12.0, 12.0])


def test_grown_b_steps_get_smaller_variance():
    from mllf.cb.hindsight import grow_directional_b_steps
    state, b1, vars_ = {}, 5.0, []
    for _ in range(3):
        t = torch.zeros(2, 4, dtype=torch.float64)
        t[0, 0], t[1, 0] = b1 + 3.0, -(b1 + 3.0)
        d = torch.zeros(2, 4, dtype=torch.bool)
        d[:, 0] = True
        v = torch.full((2, 4), 1.5 ** 2, dtype=torch.float64)
        grow_directional_b_steps(state, 'sys', LABELS, EDGES, t, d, {'b': [0.0, b1]},
                                 base_step=3.0, max_step=12.0, var=v)
        vars_.append(v[0, 0].item())
        b1 = t[0, 0].item()
    # steps 3, 6, 12: variance (1.5)^2 x (3/size)^2
    assert vars_ == pytest.approx([2.25, 2.25 / 4, 2.25 / 16])


def test_b_streak_resets_when_both_sampled():
    from mllf.cb.hindsight import grow_directional_b_steps
    state = {}
    _trapped(state, 5.0, 2)
    d = torch.zeros(2, 4, dtype=torch.bool)                     # measured: not directional
    grow_directional_b_steps(state, 'sys', LABELS, EDGES, torch.zeros(2, 4, dtype=torch.float64),
                             d, {'b': [0.0, 14.0]})
    assert _trapped(state, 14.0, 1) == pytest.approx([3.0])


def test_b_streak_follows_original_sub_order():
    # same physical pair with nodes reversed: the streak still continues
    state = {}
    _trapped(state, 5.0, 2)                                      # nodes (1, 2): b2 - b1 rising
    rev = [(1, 2), (1, 1)]
    from mllf.cb.hindsight import grow_directional_b_steps
    t = torch.zeros(2, 4, dtype=torch.float64)
    t[0, 0], t[1, 0] = -14.0 - 3.0, 14.0 + 3.0                   # edge 0: sub2 -> sub1 falling
    d = torch.zeros(2, 4, dtype=torch.bool)
    d[:, 0] = True
    grow_directional_b_steps(state, 'sys', rev, EDGES, t, d, {'b': [0.0, -14.0]})
    assert t[0, 0].item() == pytest.approx(-14.0 - 12.0)         # third step: 3 x 2^2


def _directional_run(state, b1, up, var=None, labels=LABELS):
    """One trapped run at b1 (lower->higher orientation); up = needs a higher b."""
    from mllf.cb.hindsight import grow_directional_b_steps
    t = torch.zeros(2, 4, dtype=torch.float64)
    step = 3.0 if up else -3.0
    t[0, 0], t[1, 0] = b1 + step, -(b1 + step)
    d = torch.zeros(2, 4, dtype=torch.bool)
    d[:, 0] = True
    v = var if var is not None else torch.full((2, 4), 2.25, dtype=torch.float64)
    g, br = grow_directional_b_steps(state, 'sys', labels, EDGES, t, d, {'b': [0.0, b1]},
                                     base_step=3.0, max_step=12.0, var=v)
    return t[0, 0].item(), v[0, 0].item(), br


def test_bracket_midpoint_after_reversal():
    state = {}
    _directional_run(state, 5.0, up=True)            # stuck, needs higher: lo = 5
    t, v, br = _directional_run(state, 13.0, up=False)   # overshot, needs lower: hi = 13
    assert br == 1
    assert t == pytest.approx(9.0)                   # midpoint, not 13 - 3
    assert v == pytest.approx(2.25)                  # width 8 -> sd 2, capped at the base step sd 1.5
    t, v, _ = _directional_run(state, 9.0, up=True)  # still low at 9: lo = 9
    assert t == pytest.approx(11.0) and v == pytest.approx((4.0 / 4.0) ** 2)


def test_inverted_bracket_is_unlocked():
    # indole-like: stuck needing lower at 1.4, needing higher at 3.4 (hysteresis)
    state = {}
    _directional_run(state, 1.4, up=False)
    t, v, br = _directional_run(state, 3.4, up=True)
    entry = state[OFFSET_KEY][KEY]
    assert br == 0 and t == pytest.approx(2.4)       # midpoint is still the best b guess
    assert v == pytest.approx(2.25)                  # base-step variance, not locked at sd 0.25
    assert 'b_bracket' not in entry and 'b_streak' not in entry
    assert entry['barrier_trap'] and entry['b_unlocks'] == 1
    # the bracket can form again; the trap flag lasts one run
    _directional_run(state, 2.4, up=True)
    assert 'barrier_trap' not in entry and entry['b_bracket']['lo'] == pytest.approx(2.4)


def test_unlocked_b_bracket_steps_c_up():
    state = {}
    _directional_run(state, 1.4, up=False)
    _directional_run(state, 3.4, up=True)            # inverted -> unlocked, barrier trap
    t, _, d, n = _c_run(state, 20.0)
    assert n == 1 and d and t == pytest.approx(32.0)


def test_inverted_c_bracket_is_unlocked():
    state = {}
    _c_run(state, 30.0, trapped=False, stranded=True)           # too high at 30: hi = 30
    _directional_run(state, 1.4, up=False)
    _directional_run(state, 3.4, up=True)                       # barrier trap
    t, v, d, n = _c_run(state, 40.0)                            # too low at 40: lo 40 > hi 30
    entry = state[OFFSET_KEY][KEY]
    assert n == 1 and d and t == pytest.approx(52.0) and v == pytest.approx(36.0)
    assert 'c_bracket' not in entry and entry['c_unlocks'] == 1


def test_bracket_in_reversed_node_order():
    state = {}
    _directional_run(state, 5.0, up=True)            # nodes (1, 2): lower -> higher
    rev = [(1, 2), (1, 1)]
    from mllf.cb.hindsight import grow_directional_b_steps
    t = torch.zeros(2, 4, dtype=torch.float64)       # edge 0 is 2 -> 1: db = -13, needs b(2-1) up
    t[0, 0], t[1, 0] = -13.0 + 3.0, 13.0 - 3.0
    d = torch.zeros(2, 4, dtype=torch.bool)
    d[:, 0] = True
    _, br = grow_directional_b_steps(state, 'sys', rev, EDGES, t, d, {'b': [0.0, -13.0]})
    assert br == 1
    assert t[1, 0].item() == pytest.approx(9.0) and t[0, 0].item() == pytest.approx(-9.0)


def test_bracket_cleared_when_both_sampled():
    from mllf.cb.hindsight import grow_directional_b_steps
    state = {}
    _directional_run(state, 5.0, up=True)
    _directional_run(state, 13.0, up=False)
    grow_directional_b_steps(state, 'sys', LABELS, EDGES, torch.zeros(2, 4, dtype=torch.float64),
                             torch.zeros(2, 4, dtype=torch.bool), {'b': [0.0, 9.0]})
    assert 'b_bracket' not in state[OFFSET_KEY][KEY]
    t, _, br = _directional_run(state, 9.0, up=True)
    assert br == 0 and t == pytest.approx(12.0)


def _c_run(state, c, trapped=True, grow=True, pinned=False, stranded=False):
    """c targets for one run at c; returns (c target, var, directional, n pairs stepped)."""
    from mllf.cb.hindsight import barrier_trapped_c_targets
    t = torch.zeros(2, 4, dtype=torch.float64)
    t[:, 1] = c
    v = torch.full((2, 4), math.inf, dtype=torch.float64)       # no barrier measured
    d = torch.zeros(2, 4, dtype=torch.bool)
    d[:, 0] = trapped and not stranded
    applied = {'b': [0.0, 4.0], 'c': [[0.0, c], [0.0, 0.0]]}
    up, down = barrier_trapped_c_targets(
        state, 'sys', LABELS, EDGES, t, v, d, applied, base_step=12.0, max_step=48.0,
        trap_width=2.0, grow=grow, pinned=torch.tensor([pinned, pinned]),
        stranded=torch.tensor([stranded, stranded]))
    assert t[0, 1].item() == t[1, 1].item()                     # shared by both edges
    return t[0, 1].item(), v[0, 1].item(), bool(d[0, 1]), up + down


def test_no_c_step_until_b_bracket_closes():
    state = {}
    _directional_run(state, 0.0, up=True)
    _directional_run(state, 10.0, up=False)                     # bracket [0, 10]: b may be off
    assert _c_run(state, 20.0) == (20.0, math.inf, False, 0)


def test_c_steps_up_and_grows_once_trapped_at_working_b():
    # p38-like: stuck on opposite sides at nearly the same b
    state = {}
    _directional_run(state, 4.9, up=True)
    _directional_run(state, 4.4, up=False)                      # overlapping bounds
    t, v, d, n = _c_run(state, 20.0)
    assert n == 1 and d and t == pytest.approx(32.0) and v == pytest.approx(36.0)
    assert _c_run(state, 32.0)[0] == pytest.approx(56.0)
    t, v, _, _ = _c_run(state, 56.0)
    assert t == pytest.approx(104.0) and v == pytest.approx(36.0)   # base-step variance
    assert _c_run(state, 104.0)[0] == pytest.approx(152.0)     # capped at 48


def test_grown_c_step_moves_the_offset_most_of_the_way():
    from mllf.cb.hindsight import pair_offset_posterior
    state = {}
    _directional_run(state, 4.9, up=True)
    _directional_run(state, 4.4, up=False)
    for c in (20.0, 32.0):
        _c_run(state, c)
    t, v, _, _ = _c_run(state, 56.0)                            # step 48
    off, _ = pair_offset_posterior(56.0, [[t, v, True]], 8.0)   # quadratic prior sd 8
    assert off > 0.6 * 48.0


def test_shared_model_gets_base_c_step_without_touching_state():
    state = {}
    _directional_run(state, 4.9, up=True)
    _directional_run(state, 4.4, up=False)
    _c_run(state, 20.0)
    before = dict(state[OFFSET_KEY][KEY]['c_streak'])
    assert _c_run(state, 32.0, grow=False)[0] == pytest.approx(44.0)
    assert state[OFFSET_KEY][KEY]['c_streak'] == before


def test_c_streak_cleared_by_a_crossing_run():
    state = {}
    _directional_run(state, 4.9, up=True)
    _directional_run(state, 4.4, up=False)
    _c_run(state, 20.0)
    _c_run(state, 32.0)
    assert _c_run(state, 56.0, trapped=False)[3] == 0
    assert 'c_streak' not in state[OFFSET_KEY][KEY]


# ---------------------------------------------------------------------------
# Revisits
# ---------------------------------------------------------------------------

def _outcome(state, combo, ok, labels=LABELS):
    record_pair_outcomes(state, 'sys', labels, EDGES,
                         _saved([5000, 5000], 20) if ok else _saved([10000, 0], 0), combo=combo)


def test_outcomes_count_visits_successes_and_combos():
    state = {}
    _outcome(state, '/o/sys/comb_1', ok=False)
    _outcome(state, '/o/sys/comb_2', ok=True, labels=[(1, 2), (1, 1)])   # reverse orientation
    _outcome(state, '/o/sys/comb_1', ok=False)
    e = state[OFFSET_KEY][KEY]
    assert (e['visits'], e['successes']) == (3, 1)
    assert e['combos'] == ['/o/sys/comb_2', '/o/sys/comb_1']              # most recent last


def test_revisits_until_satisfied_or_exhausted():
    from mllf.cb.hindsight import select_revisits
    state = {}
    _outcome(state, '/o/sys/comb_1', ok=False)
    picks, counts = select_revisits(state, {'sys': 2}, lambda c: True, successes_needed=2, max_visits=3)
    assert picks == ['/o/sys/comb_1'] and counts['active'] == 1
    _outcome(state, '/o/sys/comb_1', ok=True)
    _outcome(state, '/o/sys/comb_1', ok=True)
    picks, counts = select_revisits(state, {'sys': 2}, lambda c: True, successes_needed=2, max_visits=3)
    assert picks == [] and counts['satisfied'] == 1
    assert state[OFFSET_KEY][KEY]['retired'] == 'satisfied'

    other = {}
    for _ in range(3):
        _outcome(other, '/o/sys/comb_1', ok=False)
    picks, counts = select_revisits(other, {'sys': 2}, lambda c: True, successes_needed=2, max_visits=3)
    assert picks == [] and counts['exhausted'] == 1


def test_revisits_respect_quota_order_and_allowed():
    from mllf.cb.hindsight import select_revisits
    state = {}
    labels = {'a': [(1, 1), (1, 2)], 'b': [(1, 1), (1, 3)], 'c': [(1, 2), (1, 3)]}
    for name, n in (('a', 1), ('b', 3), ('c', 2)):
        for _ in range(n):
            _outcome(state, f'/o/sys/comb_{name}', ok=False, labels=labels[name])
    picks, _ = select_revisits(state, {'sys': 2}, lambda c: True, max_visits=8)
    assert picks == ['/o/sys/comb_b', '/o/sys/comb_c']                    # most visited first
    picks, _ = select_revisits(state, {'sys': 2}, lambda c: not c.endswith('_b'), max_visits=8)
    assert picks == ['/o/sys/comb_c', '/o/sys/comb_a']


def test_merge_revisits_dedupes_and_keeps_system_minimum():
    from mllf.cb.hindsight import merge_revisits
    system = {'/out/A/comb_1': 'A', '/man/A/comb_1': 'A', '/man/A/comb_2': 'A', '/man/A/comb_3': 'A',
              '/man/B/comb_7': 'B', '/man/B/comb_8': 'B'}
    out = merge_revisits(['/out/A/comb_1'],
                         ['/man/A/comb_1', '/man/A/comb_2', '/man/A/comb_3', '/man/B/comb_7', '/man/B/comb_8'],
                         system.get, n_total=4, min_per_system=2)
    assert out == ['/out/A/comb_1', '/man/A/comb_2', '/man/B/comb_7', '/man/B/comb_8']


# ---------------------------------------------------------------------------
# Pinned / stranded runs
# ---------------------------------------------------------------------------

def _pair_run(pops, at_lo, at_hi, edge=37500):
    return {'populations': pops,
            'endpoint_pairs': {'2_3': {'edge': edge, 'at_lo': at_lo, 'near_lo': edge - at_lo - at_hi,
                                       'near_hi': 0, 'at_hi': at_hi}}}


def test_lambda_states_pinned_and_stranded():
    from mllf.cb.hindsight import lambda_states
    pinned, stranded = lambda_states(EDGES, _pair_run([37500, 0], 37500, 0))
    assert pinned.tolist() == [True, True] and not stranded.any()
    pinned, _ = lambda_states(EDGES, _pair_run([30000, 0], 30000, 0))   # left the start: moving
    assert not pinned.any()
    pinned, _ = lambda_states(EDGES, _pair_run([0, 37000], 0, 37000))   # went to the other side
    assert not pinned.any()
    pinned, stranded = lambda_states(EDGES, _pair_run([0, 0], 0, 0))
    assert stranded.tolist() == [True, True] and not pinned.any()


def test_pinned_runs_leave_b_bracket_and_streak_alone():
    from mllf.cb.hindsight import grow_directional_b_steps
    state = {}
    _directional_run(state, 5.0, up=True)
    before = json.loads(json.dumps(state[OFFSET_KEY][KEY]))
    t = torch.zeros(2, 4, dtype=torch.float64)
    t[0, 0], t[1, 0] = 2.0, -2.0                                # a "needs lower b" direction
    d = torch.zeros(2, 4, dtype=torch.bool)
    d[:, 0] = True
    grow_directional_b_steps(state, 'sys', LABELS, EDGES, t, d, {'b': [0.0, 5.0]},
                             skip=torch.tensor([True, True]))
    assert state[OFFSET_KEY][KEY] == before
    grow_directional_b_steps(state, 'sys', LABELS, EDGES, torch.zeros(2, 4, dtype=torch.float64),
                             torch.zeros(2, 4, dtype=torch.bool), {'b': [0.0, 5.0]},
                             skip=torch.tensor([True, True]))       # stranded: not "both sampled"
    assert state[OFFSET_KEY][KEY] == before


def test_pinned_run_steps_c_up_without_a_closed_b_bracket():
    state = {}
    t, v, d, n = _c_run(state, 30.0, pinned=True)
    assert n == 1 and d and t == pytest.approx(42.0) and v == pytest.approx(36.0)
    assert _c_run(state, 42.0, pinned=True)[0] == pytest.approx(66.0)    # grows


def test_stranded_run_steps_c_down_and_brackets():
    state = {}
    _c_run(state, 30.0, pinned=True)                            # too low at 30
    _c_run(state, 42.0, pinned=True)                            # too low at 42
    t, v, d, n = _c_run(state, 90.0, stranded=True)             # too high at 90
    assert n == 1 and d
    assert t == pytest.approx(66.0)                             # midpoint of [42, 90]
    assert v == pytest.approx(min(36.0, (48.0 / 4) ** 2))
    t, _, _, _ = _c_run(state, 66.0, stranded=True)             # still too high: [42, 66]
    assert t == pytest.approx(54.0)


def test_stranded_steps_grow_down_and_stop_at_zero():
    state = {}
    assert _c_run(state, 100.0, stranded=True)[0] == pytest.approx(88.0)
    assert _c_run(state, 88.0, stranded=True)[0] == pytest.approx(64.0)
    assert _c_run(state, 20.0, stranded=True)[0] == pytest.approx(0.0)  # 20 - 48 floored


def test_stranded_run_keeps_a_measured_barrier_target():
    from mllf.cb.hindsight import barrier_trapped_c_targets
    t = torch.zeros(2, 4, dtype=torch.float64)
    t[:, 1] = 70.0                                              # measured-barrier target
    v = torch.full((2, 4), math.inf, dtype=torch.float64)
    v[:, 1] = 4.0
    d = torch.zeros(2, 4, dtype=torch.bool)
    up, down = barrier_trapped_c_targets({}, 'sys', LABELS, EDGES, t, v, d,
                                         {'c': [[0.0, 80.0], [0.0, 0.0]]},
                                         stranded=torch.tensor([True, True]))
    assert (up, down) == (0, 0) and t[0, 1].item() == 70.0


def test_crossing_run_clears_c_bracket():
    state = {}
    _c_run(state, 30.0, pinned=True)
    _c_run(state, 90.0, stranded=True)
    _c_run(state, 60.0, trapped=False)                          # crossed
    assert 'c_bracket' not in state[OFFSET_KEY][KEY] and 'c_streak' not in state[OFFSET_KEY][KEY]


def test_uncrossed_pairs():
    from mllf.cb.hindsight import uncrossed_pairs
    moving = {'populations': [30000, 0], 'transition_pairs': {'2_3': 0}}
    assert uncrossed_pairs(EDGES, moving).tolist() == [True, True]
    crossed = {'populations': [20000, 15000], 'transition_pairs': {'2_3': 4}}
    assert not uncrossed_pairs(EDGES, crossed).any()
    stranded = {'populations': [0, 0], 'transition_pairs': {'2_3': 0}}
    assert not uncrossed_pairs(EDGES, stranded).any()          # lambda_states handles it
    unknown = {'populations': [30000, 0]}
    assert not uncrossed_pairs(EDGES, unknown).any()           # no count: not flagged


def test_uncrossed_run_steps_c_up_and_leaves_b_alone():
    # A run that moved off its start but never crossed: treated like a pinned
    # run (the trainer ORs uncrossed_pairs into the pinned mask).
    from mllf.cb.hindsight import grow_directional_b_steps, uncrossed_pairs
    state = {}
    _directional_run(state, 5.0, up=True)
    before = json.loads(json.dumps(state[OFFSET_KEY][KEY]))
    run = {'populations': [30000, 0], 'transition_pairs': {'2_3': 0}}
    mask = uncrossed_pairs(EDGES, run)
    t = torch.zeros(2, 4, dtype=torch.float64)
    t[0, 0], t[1, 0] = 8.0, -8.0
    d = torch.zeros(2, 4, dtype=torch.bool)
    d[:, 0] = True
    grow_directional_b_steps(state, 'sys', LABELS, EDGES, t, d, {'b': [0.0, 5.0]}, skip=mask)
    assert state[OFFSET_KEY][KEY].get('b_bracket') == before.get('b_bracket')
    assert state[OFFSET_KEY][KEY].get('b_streak') == before.get('b_streak')
    assert _c_run(state, 30.0, pinned=bool(mask[0]))[0] == pytest.approx(42.0)
