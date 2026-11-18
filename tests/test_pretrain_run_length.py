"""Run-length handling in pretraining's AWR weights (short ALF runs vs 1 ns runs)."""

import pytest
import torch

from mllf.pretrain.rewards import (
    compute_reward_from_sim_results,
    pair_advantages,
    run_length,
    run_pair_success,
)


def _sim(ns, pops, transitions, pair_trans):
    """One-site simulation_results dict; pops are the highest-lambda counts of blocks 2.."""
    return {
        'simulated_ns': ns,
        'populations': {str(b + 2): {'counts': {'0.99': p}, 'site': 1} for b,
                        p in enumerate(pops)},
        'transitions': {'1': {'0.99': transitions}},
        'transition_pairs': pair_trans,
        'ddg_pairs': {k: 0.0 for k in pair_trans},
    }


PAIR_EDGES = torch.tensor([[0, 1], [1, 0]])


def test_run_length_classifies_against_half_the_full_length():
    assert run_length({'simulated_ns': 0.104}) == pytest.approx((0.104, True))
    assert run_length({'simulated_ns': 1.0}) == (1.0, False)
    assert run_length({'simulated_ns': 0.6}, full_length_ns=1.0)[1] is False
    # Unknown length counts as full.
    assert run_length({}) == (1.0, False)
    assert run_length({'simulated_ns': None}) == (1.0, False)


def test_scalar_reward_scales_counts_with_length():
    # The same sampling per unit time scores the same at either length (once
    # both clear their transition thresholds, 10 full / 3 short).
    full = _sim(1.0, [400, 400], 70, {'2_3': 70})
    short = _sim(0.1, [40, 40], 7, {'2_3': 7})
    r_full = compute_reward_from_sim_results(full, 1, [2])
    r_short = compute_reward_from_sim_results(short, 1, [2])
    assert r_short == pytest.approx(r_full)
    # Without a length, the short run is judged as a full one and penalised.
    del short['simulated_ns']
    assert compute_reward_from_sim_results(short, 1, [2]) < r_full - 5


def test_short_run_needs_only_the_explore_transitions():
    short = _sim(0.104, [40, 10], 3, {'2_3': 3})
    assert compute_reward_from_sim_results(short, 1, [2], explore_min_transitions=3) > \
        compute_reward_from_sim_results(short, 1, [2], explore_min_transitions=10)


def test_pair_success_uses_the_online_test_for_each_length():
    cfg = {'success_test': 'pair', 'explore_min_transitions': 3, 'explore_min_fraction': 0.0}
    # 3 crossings, imbalanced: passes the short-run test only.
    short = {'sim_results': _sim(0.104, [90, 10], 3, {'2_3': 3})}
    full = {'sim_results': _sim(1.0, [90, 10], 3, {'2_3': 3})}
    assert run_pair_success(short, PAIR_EDGES, cfg).tolist() == [True, True]
    assert run_pair_success(full, PAIR_EDGES, cfg).tolist() == [False, False]
    # A full run passes with >= 10 crossings and minority >= 0.2.
    good = {'sim_results': _sim(1.0, [60, 40], 12, {'2_3': 12})}
    assert run_pair_success(good, PAIR_EDGES, cfg).tolist() == [True, True]
    # Too few crossings fails even a short run.
    stuck = {'sim_results': _sim(0.104, [99, 1], 1, {'2_3': 1})}
    assert run_pair_success(stuck, PAIR_EDGES, cfg).tolist() == [False, False]
    # No per-pair transitions: unknown.
    bare = {'sim_results': {**_sim(0.104, [50, 50], 3, {}), 'transition_pairs': None}}
    assert run_pair_success(bare, PAIR_EDGES, cfg) is None


def test_end_state_gate_judges_each_substituent():
    # 1 ns = 37500 frames; floor 0.02 / 2 = 1% (375 frames) at a 2-sub site.
    # 10 crossings of the pair = 5 entries into each endpoint.
    ok = {'sim_results': _sim(1.0, [30000, 400], 10, {'2_3': 10})}
    assert run_pair_success(ok, PAIR_EDGES, {}).tolist() == [True, True]
    # imbalanced but sampled: the pair test fails it, the end-state test passes it
    assert run_pair_success(ok, PAIR_EDGES, {'success_test': 'pair'}).tolist() == [False, False]
    rare = {'sim_results': _sim(1.0, [30000, 300], 10, {'2_3': 10})}    # 0.8% < 1%
    assert run_pair_success(rare, PAIR_EDGES, {}).tolist() == [False, False]
    # rate checked against the run's length: 1.5 entries / ns
    slow = {'sim_results': _sim(4.0, [80000, 70000], 12, {'2_3': 12})}
    assert run_pair_success(slow, PAIR_EDGES, {}).tolist() == [False, False]
    # thresholds from the reward config
    assert run_pair_success(ok, PAIR_EDGES, {'min_visits_per_ns': 6}).tolist() == [False, False]
    # unknown without per-pair transitions or 0.99 counts
    bare = {'sim_results': {**_sim(1.0, [50, 50], 3, {}), 'transition_pairs': None}}
    assert run_pair_success(bare, PAIR_EDGES, {}) is None
    no99 = _sim(1.0, [50, 50], 3, {'2_3': 3})
    no99['populations'] = {'2': {'counts': {'0.95': 50}}, '3': {'counts': {'0.95': 50}}}
    assert run_pair_success({'sim_results': no99}, PAIR_EDGES, {}) is None
    with pytest.raises(ValueError):
        run_pair_success(ok, PAIR_EDGES, {'success_test': 'other'})


def test_end_state_gate_needs_entries_in_every_run_length():
    # 0.1 ns: the rate alone (0.2 entries) passes one entry; min_end_entries (5) does not
    one = {'sim_results': _sim(0.1, [3000, 400], 2, {'2_3': 2})}
    assert run_pair_success(one, PAIR_EDGES, {}).tolist() == [False, False]
    assert run_pair_success(one, PAIR_EDGES, {'min_end_entries': 0}).tolist() == [True, True]
    five = {'sim_results': _sim(0.1, [3000, 400], 10, {'2_3': 10})}
    assert run_pair_success(five, PAIR_EDGES, {}).tolist() == [True, True]
    # and in full-length runs: 4 entries in 1 ns meets the rate, not the floor
    four = {'sim_results': _sim(1.0, [30000, 4000], 8, {'2_3': 8})}
    assert run_pair_success(four, PAIR_EDGES, {}).tolist() == [False, False]
    assert run_pair_success(four, PAIR_EDGES, {'min_end_entries': 4}).tolist() == [True, True]


def test_end_state_gate_passes_edges_between_sampled_substituents():
    # one 3-sub site, nodes 0-2; node 2 never reaches its endpoint
    sim = {'simulated_ns': 1.0,
           'populations': {'2': {'counts': {'0.99': 20000}}, '3': {'counts': {'0.99': 10000}},
                           '4': {'counts': {'0.99': 0}}},
           'transition_pairs': {'2_3': 12, '2_4': 0, '3_4': 0}}
    edges = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    ok = run_pair_success({'sim_results': sim}, edges, {})
    assert ok.tolist() == [True, True, False, False, False, False]
    # nsubs from the metadata gives the same answer
    run = {'sim_results': sim, 'metadata': {'nsubs_per_site': [3]}}
    assert run_pair_success(run, edges, {}).tolist() == ok.tolist()


def test_success_gate_sets_the_sign_of_the_advantage():
    rewards = [torch.full((2, 4), 0.1), torch.full((2, 4), 0.9)]
    success = [torch.tensor([True, True]), torch.tensor([False, False])]
    adv, _, _ = pair_advantages(rewards, pair_success=success)
    # Low reward but passed: not below average. High reward but failed: not above.
    assert (adv[0] == 0).all() and (adv[1] == 0).all()
    adv, _, _ = pair_advantages(rewards, pair_success=[None, None])
    assert (adv[0] < 0).all() and (adv[1] > 0).all()
    # Floor still applies to failed pairs.
    adv, _, _ = pair_advantages([torch.full((2, 4), -10.0)] + [torch.full((2, 4), 1.0)] * 9,
                                success_floor=-2.0)
    assert (adv[0] >= -2.0).all()
    assert pair_advantages([None])[0] == [None]
