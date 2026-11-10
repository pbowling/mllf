"""Group-level end-term correction (mllf.cb.group_correction)."""
import math

import torch

from mllf.cb import group_correction as gc

# Site 1: three substituents (nodes 0-2); site 2: two (nodes 3-4).
LABELS = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
SITE1 = [(i, j) for i in range(3) for j in range(3) if i != j]
CFG = gc.correction_config({'group_correction': {}})


def _applied(x=2.0, s=4.0, n=5):
    return {'x': [[0.0 if i == j else x for j in range(n)] for i in range(n)],
            's': [[0.0 if i == j else s for j in range(n)] for i in range(n)]}


def _site_stuck():
    """Site 1 never samples and barely transitions; site 2 samples normally."""
    return {'populations': [0, 0, 0, 500, 500], 'transitions': [1, 30]}


def _edge_index(edges):
    return torch.tensor([[i for i, _ in edges], [j for _, j in edges]])


def test_config_off_unless_set_and_merges_defaults():
    assert gc.correction_config({}) is None
    assert gc.correction_config({'group_correction': False}) is None
    cfg = gc.correction_config({'group_correction': {'step': 0.1}})     # former key: both terms
    assert cfg['step_x'] == cfg['step_s'] == 0.1 and cfg['floor_x'] == gc.DEFAULTS['floor_x']


def test_whole_site_flagged_only_when_another_site_samples():
    f = gc.flagged_sites(_site_stuck(), LABELS, CFG)
    assert [(x['kind'], x['site']) for x in f] == [('site', 1)]
    assert set(f[0]['edges']) == set(SITE1) and set(f[0]['edges'].values()) == {1.0}
    # no site sampled normally: nothing to compare against, so no flag
    assert gc.flagged_sites({**_site_stuck(), 'transitions': [1, 5]}, LABELS, CFG) == []


def test_two_substituent_sites_are_left_to_the_pair_rules():
    stuck_site2 = {'populations': [300, 300, 300, 0, 0], 'transitions': [30, 0]}
    assert gc.flagged_sites(stuck_site2, LABELS, CFG) == []


def test_held_off_endpoint_flags_partner_edges_by_lambda_share():
    saved = {'populations': [800, 100, 50, 500, 500], 'transitions': [20, 20],
             'site_holdoff': [{'blocks': [2, 3, 4], 'dom': 2, 'dom_frac': 0.9, 'at_frac': 0.01,
                               'mean_dom': 0.8, 'partners': {3: 0.10, 4: 0.05}}]}
    (f,) = gc.flagged_sites(saved, LABELS, CFG)
    assert f['kind'] == 'endpoint' and f['dom'] == 0
    assert f['edges'] == {(1, 0): 1.0, (2, 0): 0.5}       # partner -> held-off sub, top share = 1


def test_ratio_steps_down_to_the_floor_then_caps():
    state = {}
    ratios = []
    for run in range(3):
        res = gc.update_group(state, 's', 'group01', _site_stuck(), LABELS, _applied(), CFG, run, f'run_{run}')
        ratios.append(res['edges'][(0, 1)][2:])
    assert ratios == [(0.65, 0.85), (0.4, 0.7), (0.4, 0.7)]
    assert res['capped'] and not res['stepped']
    # each run is recorded once, and run_record finds it by name
    rec = gc.run_record(state, 's', 'group01', '/a/b/run_1')
    assert rec['ratio'] == {'1': 0.4} and rec['ratio_s'] == {'1': 0.7}


def test_group_values_replace_x_and_s_for_corrected_pairs_only():
    state = {}
    gc.update_group(state, 's', 'group01', _site_stuck(), LABELS, _applied(), CFG, 0, 'run_0')
    edges = SITE1 + [(3, 4)]
    actions = torch.full((len(edges), 4), 9.0)
    out, n = gc.apply_group_values(actions, state, 's', 'group01', LABELS, _edge_index(edges))
    assert n == len(SITE1)
    xs, ss = gc.END_COLS
    assert torch.allclose(out[:len(SITE1), xs], torch.full((len(SITE1),), 2.0 * 0.65))
    assert torch.allclose(out[:len(SITE1), ss], torch.full((len(SITE1),), 4.0 * 0.85))
    assert torch.equal(out[-1], actions[-1])                    # site 2 untouched
    assert torch.equal(out[:, :2], actions[:, :2])              # b and c never touched
    # another group of the same system has its own (empty) corrections
    same, n0 = gc.apply_group_values(actions, state, 's', 'group02', LABELS, _edge_index(edges))
    assert n0 == 0 and same is actions


def test_overcorrected_site_steps_back_half_the_last_step():
    state = {}
    gc.update_group(state, 's', 'group01', _site_stuck(), LABELS, _applied(), CFG, 0, 'run_0')   # r 0.65 / 0.85
    trapped = {'populations': [300, 0, 200, 500, 500], 'transitions': [2, 30]}   # some unsampled, few crossings
    res = gc.update_group(state, 's', 'group01', trapped, LABELS, _applied(), CFG, 1, 'run_1')
    assert not res['guarded']                       # no endpoint trapped: left alone
    trapped['site_holdoff'] = [{'blocks': [2, 3, 4], 'dom': 2, 'dom_frac': 0.8, 'at_frac': 0.5,
                                'mean_dom': 0.9, 'partners': {}}]
    res = gc.update_group(state, 's', 'group01', trapped, LABELS, _applied(), CFG, 2, 'run_2')
    assert res['guarded'] == [1] and not res['flagged']
    assert res['edges'][(1, 0)][2:] == (0.825, 0.925)       # pairs acting on the trapped sub
    assert (0, 1) not in res['edges']


def test_correction_targets_cover_the_site_and_leave_b_c_free():
    state = {}
    res = gc.update_group(state, 's', 'group01', _site_stuck(), LABELS, _applied(), CFG, 0, 'run_0')
    edges = SITE1 + [(3, 4)]
    t, v, mask = gc.correction_targets(_edge_index(edges), res, CFG)
    assert mask.tolist() == [True] * len(SITE1) + [False]
    xs, ss = gc.END_COLS
    assert math.isclose(t[0, xs], 2.0 * 0.65) and math.isclose(t[0, ss], 4.0 * 0.85)
    assert math.isclose(v[0, ss], (0.1 * 3.4) ** 2)             # sd = 10% of the value
    assert math.isclose(v[0, xs], (0.13) ** 2)
    assert torch.isinf(v[:, :2]).all() and torch.isinf(v[-1]).all()
