"""Multi-substituent hindsight, the per-site b fit, the cost ledger, per-system
stages and the final-bias export's labelling."""
import math
from types import SimpleNamespace

import pytest
import torch
import yaml

from mllf.cb import system_stages as ss
from mllf.cb.cost_ledger import alf_cost, record_runs, simulated_ns, summarize_ledger
from mllf.cb.export_biases import input_dir_labels
from mllf.cb.hindsight import (
    KB_KCAL,
    OFFSET_KEY,
    hindsight_targets,
    lambda_states,
    linear_edge_variance,
    linear_hindsight,
    pair_key,
    pair_succeeded,
    record_pair_outcomes,
    uncrossed_pairs,
)
from mllf.cli.workflow import fit_site_b, write_variables_from_actions

KT = KB_KCAL * 298.15
# One site, three substituents: every directed edge.
EI3 = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])


def _applied(n, b=None):
    z = [[0.0] * n for _ in range(n)]
    return {'b': b or [0.0] * n, 'c': z, 'x': z, 's': z}


# ── Item 1: pairs that exchange through other substituents ──────────────────

def test_linear_measured_through_other_substituents():
    # Both sampled, no direct transitions, but each exchanged with the site.
    t, v = linear_hindsight(0.0, 100, 50, 0, KT, node_transitions=(10, 10))
    assert t == pytest.approx(-KT * math.log(50 / 100))
    assert v < (3.0 / 2.0) ** 2          # tighter than a directional step


def test_linear_still_directional_when_a_side_unsampled():
    t, v = linear_hindsight(0.0, 100, 0, 0, KT, node_transitions=(10, 0))
    assert t == pytest.approx(3.0)          # step toward the unsampled substituent


def test_hindsight_targets_three_sub_pair_without_direct_crossings():
    saved = {'populations': [100, 80, 60],
             'transition_pairs': {'2_3': 10, '2_4': 10, '3_4': 0}}
    t, v, d = hindsight_targets(EI3, _applied(3), saved, barrier_target=1.2,
                                return_directional=True)
    k = 3                                 # edge 1 -> 2
    assert not bool(d[k, 0])
    assert float(t[k, 0]) == pytest.approx(-KT * math.log(60 / 80))
    # Each of nodes 1 and 2 exchanged 10 times (with node 0).
    step = -KT * math.log(60 / 80)
    assert float(v[k, 0]) == pytest.approx(KT ** 2 * (2 / 10 + 2 / 10) + 0.35 ** 2
                                           + (0.25 * step) ** 2)


def test_uncrossed_only_when_whole_site_stuck():
    moving = {'populations': [100, 80, 0],
              'transition_pairs': {'2_3': 10, '2_4': 0, '3_4': 0}}
    assert not uncrossed_pairs(EI3, moving).any()
    stuck = {'populations': [100, 0, 0],
             'transition_pairs': {'2_3': 0, '2_4': 0, '3_4': 0}}
    flags = uncrossed_pairs(EI3, stuck)
    assert flags.tolist() == [True, True, True, False, True, False]


def test_two_sub_uncrossed_unchanged():
    ei = torch.tensor([[0, 1], [1, 0]])
    saved = {'populations': [100, 0], 'transition_pairs': {'2_3': 0}}
    assert uncrossed_pairs(ei, saved).all()


def test_pair_succeeded_uses_node_transitions_on_larger_site():
    saved = {'populations': [100, 80, 60],
             'transition_pairs': {'2_3': 12, '2_4': 12, '3_4': 0}}
    assert pair_succeeded(saved, 1, 2, site=[0, 1, 2])
    assert not pair_succeeded(saved, 1, 2)          # direct count alone: 0


# ── Item 2: pinned / stranded on larger sites ───────────────────────────────

def test_pinned_only_for_pairs_with_the_starting_substituent():
    ep = {'edge': 1000, 'at_lo': 1000, 'near_lo': 0, 'near_hi': 0, 'at_hi': 0}
    saved = {'populations': [100, 0, 0], 'endpoint_pairs': {'2_3': ep, '2_4': ep, '3_4': {}}}
    pinned, stranded = lambda_states(EI3, saved)
    assert pinned.tolist() == [True, True, True, False, True, False]
    assert not stranded.any()               # 1-2 was simply never reached


def test_stranded_only_when_no_substituent_sampled():
    pinned, stranded = lambda_states(EI3, {'populations': [0, 0, 0]})
    assert stranded.all() and not pinned.any()
    _, stranded = lambda_states(EI3, {'populations': [0, 50, 0]})
    assert not stranded.any()


def test_two_sub_pinned_and_stranded_unchanged():
    ei = torch.tensor([[0, 1], [1, 0]])
    ep = {'edge': 1000, 'at_lo': 995, 'near_lo': 5, 'near_hi': 0, 'at_hi': 0}
    pinned, _ = lambda_states(ei, {'populations': [100, 0], 'endpoint_pairs': {'2_3': ep}})
    assert pinned.all()
    _, stranded = lambda_states(ei, {'populations': [0, 0]})
    assert stranded.all()


# ── Item 3: b from every within-site edge ───────────────────────────────────

def test_fit_site_b_consistent_differences_exact():
    lin = {(0, 1): 1.0, (1, 0): -1.0, (0, 2): 3.0, (2, 0): -3.0, (1, 2): 2.0, (2, 1): -2.0}
    b, fit = fit_site_b([0, 1, 2], lin)
    assert b[0] == 0.0 and b[1] == pytest.approx(1.0) and b[2] == pytest.approx(3.0)
    assert fit['rms'] == pytest.approx(0.0, abs=1e-9)


def test_fit_site_b_uses_non_reference_edges_and_weights():
    # The model says 0->2 is 3, but 0->1->2 says 1 + 4 = 5; the well-measured
    # 1->2 pair (small variance) pulls b[2] toward the path.
    lin = {(0, 1): 1.0, (1, 0): -1.0, (0, 2): 3.0, (2, 0): -3.0, (1, 2): 4.0, (2, 1): -4.0}
    var = {e: 9.0 for e in lin}
    var[(1, 2)] = var[(2, 1)] = 0.01
    b_eq, fit = fit_site_b([0, 1, 2], lin)
    b_w, _ = fit_site_b([0, 1, 2], lin, var)
    assert 3.0 < b_eq[2] < 5.0 and fit['rms'] > 0
    assert b_w[2] - b_w[1] == pytest.approx(4.0, abs=0.05)


def test_fit_site_b_pair_is_direction_mean():
    b, _ = fit_site_b([0, 1], {(0, 1): 2.0, (1, 0): -1.0})
    assert b[1] == pytest.approx(1.5)


def test_write_variables_uses_all_edges(tmp_path):
    data = SimpleNamespace(x=torch.zeros(3, 4), edge_index=EI3, edge_type=None)
    extras = {'relation_names': ['linear', 'quadratic', 'skew', 'end'],
              'base_relation_map': {}, 'nsubs_per_site': [3]}
    actions = torch.zeros(6, 4)
    actions[:, 0] = torch.tensor([1.0, 3.0, -1.0, 4.0, -3.0, -4.0])
    write_variables_from_actions(str(tmp_path), data, extras, actions)
    ns = {}
    exec((tmp_path / 'variables.py').read_text(), ns)
    b = yaml.safe_load(ns['bias_string'])['b'][0]
    assert b[0] == 0.0 and 3.0 < b[2] < 5.0     # not the reference edge's 3.0 alone
    assert (tmp_path / 'b_fit.json').exists()


def test_linear_edge_variance_uses_pair_observations():
    labels = [(1, 1), (1, 2), (1, 3)]
    key, _ = pair_key('sys', (1, 2), (1, 3))
    state = {OFFSET_KEY: {key: {'linear': [[1.0, 0.25, False]]}}}
    v = linear_edge_variance(state, 'sys', labels, EI3)
    assert float(v[3]) < 0.25 and float(v[0]) == pytest.approx(9.0)


# ── Item 12: cost ledger ────────────────────────────────────────────────────

_OUT = """ header
   NSTEP =   125000    NSAVC =     1000    NSAVV =       10
 ...
  TIME STEP =  4.09097E-02 AKMA       2.00000E-03 PS
   NSTEP =   375000    NSAVC =     1000    NSAVV =       10
  TIME STEP =  4.09097E-02 AKMA       2.00000E-03 PS
"""


def test_simulated_ns_sums_dynamics_calls(tmp_path):
    (tmp_path / 'output.out').write_text(_OUT)
    assert simulated_ns(tmp_path / 'output.out') == pytest.approx(1.0)
    assert simulated_ns(tmp_path / 'missing') is None


def test_ledger_records_once_and_summarizes(tmp_path):
    run = tmp_path / 'combo' / 'run_000'
    run.mkdir(parents=True)
    (run / 'output.out').write_text(_OUT)
    ledger = tmp_path / 'cost_ledger.json'
    assert record_runs(ledger, [('sysA', str(run), True)], 'train', 0, {'sysA': 'pairs'}) == 1
    assert record_runs(ledger, [('sysA', str(run), True)], 'train', 0, {'sysA': 'pairs'}) == 0
    import json
    summ = summarize_ledger(json.loads(ledger.read_text()))
    assert summ['sysA']['stages']['pairs'] == {'runs': 1, 'ns': pytest.approx(1.0)}


def test_alf_cost_follows_metadata_source(tmp_path):
    src = tmp_path / 'alf' / 'run1'
    src.mkdir(parents=True)
    (src / 'output').write_text(_OUT)
    ds = tmp_path / 'pretraining' / 'sys_group1'
    (ds / 'run1').mkdir(parents=True)
    (ds / 'run1' / 'metadata.json').write_text('{"source_run_dir": "%s"}' % src)
    (ds / 'run2').mkdir()
    cost = alf_cost([ds])['sys_group1']
    assert cost == {'runs': 1, 'ns': pytest.approx(1.0), 'missing': 1}


# ── Items 4-5: per-system stages ────────────────────────────────────────────

STAGES = [
    {'name': 'pairs', 'epochs': 5, 'systems': ['a'],
     'advance': {'type': 'pairs_resolved', 'fraction': 0.5}},
    {'name': 'site', 'epochs': 3, 'advance': {'type': 'combos_balanced', 'fraction': 1.0}},
]


def test_init_state_skips_stages_a_system_is_not_in():
    st = ss.init_state(['a', 'b'], STAGES)
    assert st['a']['stage'] == 0 and st['b']['stage'] == 1


def test_combo_pair_keys_from_name():
    keys = ss.combo_pair_keys('s', '/x/comb_0012_site1_subs_3_1_5__site2_subs_2_4')
    assert set(keys) == {'s|s1|1|3', 's|s1|3|5', 's|s1|1|5', 's|s2|2|4'}


def test_pair_resolved_needs_stage_successes_and_narrow_offset():
    entry = {'stages': {'pairs': {'visits': 3, 'successes': 2}}}
    assert ss.pair_resolved(entry, 'pairs', successes=2)
    assert not ss.pair_resolved(entry, 'site', successes=2)
    assert not ss.pair_resolved(entry, 'pairs', 2, max_offset_sd={'linear': 1.0})
    entry['linear'] = [[0.5, 0.1, False]]
    assert ss.pair_resolved(entry, 'pairs', 2, max_offset_sd={'linear': 1.0})


def test_record_pair_outcomes_counts_per_stage():
    ei = torch.tensor([[0, 1], [1, 0]])
    saved = {'populations': [100, 100], 'transition_pairs': {'2_3': 20}}
    state = {}
    record_pair_outcomes(state, 's', [(1, 1), (1, 2)], ei, saved, stage='pairs')
    entry = state[OFFSET_KEY]['s|s1|1|2']
    assert entry['stages']['pairs'] == {'visits': 1, 'successes': 1}


def test_pairs_progress_pool_and_visited():
    state = {OFFSET_KEY: {'s|s1|1|2': {'stages': {'pairs': {'visits': 2, 'successes': 2}}},
                          's|s1|1|3': {'stages': {'pairs': {'visits': 1, 'successes': 0}}}}}
    keys = ['s|s1|1|2', 's|s1|1|3', 's|s1|2|3']
    assert ss.pairs_progress(state, keys, 'pairs', {}) == (1, 3)
    assert ss.pairs_progress(state, keys, 'pairs', {'scope': 'visited'}) == (1, 2)


def test_site_balanced():
    good = {'populations': [100, 90, 80, 50, 60], 'transitions': [20, 15]}
    assert ss.site_balanced(good, [3, 2])
    assert not ss.site_balanced({**good, 'populations': [100, 90, 5, 50, 60]}, [3, 2])
    assert not ss.site_balanced({**good, 'transitions': [20, 3]}, [3, 2])


def test_advance_through_stages_to_done():
    st = ss.init_state(['a'], STAGES)['a']
    st['stage_epochs'] = 2
    assert ss.advance(st, STAGES, 'a', 4, 'criterion', (3, 4)) == 1
    assert st['stage_epochs'] == 0 and st['history'][0]['reason'] == 'criterion'
    assert ss.advance(st, STAGES, 'a', 7, 'max_epochs') is None
    assert st['done'] and st['converged'] is False
    st2 = ss.init_state(['b'], STAGES)['b']
    ss.advance(st2, STAGES, 'b', 2, 'epochs')       # a stage without a criterion
    assert st2['done'] and st2['converged'] is None


def test_criterion_met_respects_min_combos():
    assert ss.criterion_met({'fraction': 0.5}, 1, 2)
    assert not ss.criterion_met({'type': 'combos_balanced', 'min_combos': 3}, 2, 2)
    assert not ss.criterion_met({}, 0, 0)


# ── Item 13: export labels for a system input dir ───────────────────────────

def test_input_dir_labels_sorted_numerically(tmp_path):
    prep = tmp_path / 'prep'
    prep.mkdir()
    for name in ['site1_sub10_frag.pdb', 'site1_sub2_frag.pdb', 'site2_sub1_frag.pdb',
                 'site1_sub1_frag.pdb']:
        (prep / name).write_text('')
    assert input_dir_labels(tmp_path) == [(1, 1), (1, 2), (1, 10), (2, 1)]


# ── Item 7: explicit combo sets from ALF groups ─────────────────────────────

from mllf.file_handling import combo_sets as cs
from mllf.file_handling.manifests import EXPLICIT_COMBOS, build_combo_dir_lookup


def _rtf(path, source, atoms):
    lines = ['* stream', f'* (fragment from {source})', '*', 'PRES S1S1 0.0']
    lines += [f'ATOM {n} {t} {q:.6f}' for n, t, q in atoms]
    path.write_text('\n'.join(lines) + '\n')


def _alf_group(root, name, nodes):
    """An ALF group whose fragments are written with LOCAL numbering
    (site1_sub1..n), as some groups were prepared."""
    src = root / 'alf_src' / name
    src.mkdir(parents=True)
    run = root / 'pretraining' / name / 'run1'
    run.mkdir(parents=True)
    sites = {}
    for n, (source, atoms) in enumerate(nodes, start=1):
        f = src / f'site1_sub{n}_pres.rtf'
        _rtf(f, source, atoms)
        sites[f'site1_sub{n}'] = {'site': 1, 'sub': n, 'filepath': str(f)}
    (run / 'graph_info.json').write_text(__import__('json').dumps({'sites': sites}))


A = ('1', [('C1', 'CG2R61', -0.1)])
B = ('2', [('C1', 'CG2R61', -0.2)])
C = ('3', [('C1', 'CG2R61', -0.3)])
D = ('4', [('N1', 'NG2R50', -0.4)])


def _system(root):
    prep = root / 'sys' / 'prep'
    prep.mkdir(parents=True)
    for n, (source, atoms) in enumerate([A, B, C, D], start=1):
        _rtf(prep / f'site1_sub{n}_pres.rtf', source, atoms)
    return root / 'sys'


def test_alf_groups_matched_by_content_not_name(tmp_path):
    sysdir = _system(tmp_path)
    _alf_group(tmp_path, 'sysA_group1', [A, B, C])
    _alf_group(tmp_path, 'sysA_group2', [A, D, ('9', [('X', 'Y', 0.0)])])   # local numbering
    groups, warns = cs.read_alf_groups('sysA', tmp_path / 'pretraining', sysdir)
    assert groups == [{1: [1, 2, 3]}, {1: [1, 4]}]
    assert any('matches no substituent' in w for w in warns)


def test_group_pairs_reference_only():
    g = [{1: [3, 1, 2]}, {1: [3, 4]}]
    assert cs.group_pairs(g) == [(1, 3, 1), (1, 3, 2), (1, 1, 2), (1, 3, 4)]
    assert cs.group_pairs(g, reference_only=True) == [(1, 3, 1), (1, 3, 2), (1, 3, 4)]


def test_group_combo_info_names_parse_back():
    info = cs.group_combo_info('group', 1, {1: [3, 2], 2: [15, 6]}, '/out')
    assert info['subs'] == [3, 2, 15, 6] and info['subs_per_site_counts'] == (2, 2)
    assert ss.combo_sites(info['path']) == [(1, [3, 2]), (2, [15, 6])]


def test_build_combo_sets_registers_for_creation(tmp_path):
    sysdir = _system(tmp_path)
    _alf_group(tmp_path, 'sysB_group1', [A, B, C])
    entry = {'name': 'sysB', 'input_dir': str(sysdir), 'out_dir': str(tmp_path / 'out'),
             'combo_sets': {'refs': 'alf_group_ref_pairs', 'groups': 'alf_groups'}}
    sets, _ = cs.build_combo_sets(entry, tmp_path / 'pretraining')
    assert len(sets['refs']) == 2 and len(sets['groups']) == 1
    lookup = build_combo_dir_lookup(sets['groups'], {})
    assert lookup[sets['groups'][0]] is EXPLICIT_COMBOS[sets['groups'][0]]


def test_combos_progress_pool_scope():
    st = {'combos': {'g1': True}}
    assert ss.combos_progress(st, {}, ['g1', 'g2']) == (1, 2)
    assert ss.combos_progress(st, {'scope': 'visited'}, ['g1', 'g2']) == (1, 1)


def test_explicit_groups_and_their_pairs(tmp_path):
    entry = {'name': 'sysC', 'input_dir': str(tmp_path), 'out_dir': str(tmp_path / 'out'),
             'combo_sets': {'groups': [{1: [1, 2, 3, 4]}],
                            'ref_pairs': {'pairs_of': 'groups', 'reference_only': True}}}
    sets, _ = cs.build_combo_sets(entry)
    assert [p.split('/')[-1] for p in sets['ref_pairs']] == [
        'pair01_site1_subs_1_2', 'pair02_site1_subs_1_3', 'pair03_site1_subs_1_4']
    assert len(sets['groups']) == 1


# ── Short exploratory runs + full-length confirmation ───────────────────────

from mllf.cb.workflow_utils import set_msld_steps, template_steps

_MSLD = """timestep = 0.002            # ps
esteps = 125000             # number of time steps for equilibration
nsteps = 375000             # number of time steps for production
"""


def test_set_msld_steps_and_template(tmp_path):
    (tmp_path / 'msld_flat.py').write_text(_MSLD)
    assert template_steps(tmp_path) == {'equil': 125000, 'prod': 375000, 'timestep': 0.002}
    assert set_msld_steps(tmp_path, 13000, 39000)
    assert template_steps(tmp_path)['equil'] == 13000 and template_steps(tmp_path)['prod'] == 39000
    assert '# number of time steps for production' in (tmp_path / 'msld_flat.py').read_text()
    assert not set_msld_steps(tmp_path, 13000, 39000)          # unchanged


def test_short_run_flags_confirm_not_success():
    ei = torch.tensor([[0, 1], [1, 0]])
    state = {}
    promising = {'populations': [100, 80], 'transition_pairs': {'2_3': 3}}
    record_pair_outcomes(state, 's', [(1, 1), (1, 2)], ei, promising, stage='p', short=True)
    e = state[OFFSET_KEY]['s|s1|1|2']
    assert e['awaiting_confirm'] and e['stages']['p']['successes'] == 0
    assert not e['last_failed']
    # the confirming full-length run succeeds -> counts, flag cleared
    full_ok = {'populations': [100, 80], 'transition_pairs': {'2_3': 30}}
    record_pair_outcomes(state, 's', [(1, 1), (1, 2)], ei, full_ok, stage='p')
    assert not e['awaiting_confirm'] and e['stages']['p']['successes'] == 1
    # a short run with too few crossings is not promising
    record_pair_outcomes(state, 's', [(1, 1), (1, 2)], ei,
                         {'populations': [100, 80], 'transition_pairs': {'2_3': 1}}, stage='p', short=True)
    assert not e['awaiting_confirm'] and e['last_failed']


def test_combo_pair_flags_and_resolved():
    state = {OFFSET_KEY: {'s|s1|1|2': {'awaiting_confirm': True, 'stages': {'p': {'visits': 1, 'successes': 0}}},
                          's|s1|1|3': {'stages': {'p': {'visits': 2, 'successes': 1}}}}}
    assert ss.combo_pair_flags(state, 's', '/x/pair01_site1_subs_1_2', 'p') == (True, False)
    assert ss.combo_pair_flags(state, 's', '/x/pair02_site1_subs_1_4', 'p') == (False, True)
    assert ss.combo_resolved(state, 's', '/x/pair03_site1_subs_1_3', 'p', {'successes': 1})
    assert not ss.combo_resolved(state, 's', '/x/pair01_site1_subs_1_2', 'p', {'successes': 1})


# ── Groups run when ready, retire once every substituent is sampled ─────────

def test_group_test_every_sub_sampled():
    sampled = {'populations': [5, 900, 1, 40, 2000], 'transitions': [0, 1]}
    assert ss.site_balanced(sampled, [3, 2], 0.0, 0)          # uneven, no transitions: still passes
    assert not ss.site_balanced({**sampled, 'populations': [5, 900, 0, 40, 2000]}, [3, 2], 0.0, 0)
    assert not ss.site_balanced(sampled, [3, 2])               # the strict default still applies


def test_group_ready_needs_all_its_reference_pairs_resolved():
    ref = {'s|s1|1|2', 's|s1|1|3', 's|s1|1|4'}
    ok = {'stages': {'p': {'visits': 1, 'successes': 1}}}
    state = {OFFSET_KEY: {'s|s1|1|2': ok, 's|s1|1|3': ok,
                          's|s1|1|4': {'stages': {'p': {'visits': 3, 'successes': 0}}}}}
    adv = {'successes': 1}
    assert ss.group_ready(state, 's', '/x/group01_site1_subs_1_2_3', ref, 'p', adv)
    assert not ss.group_ready(state, 's', '/x/group02_site1_subs_1_2_4', ref, 'p', adv)


def test_short_test_without_balance_requirement():
    ei = torch.tensor([[0, 1], [1, 0]])
    state = {}
    lopsided = {'populations': [1000, 20], 'transition_pairs': {'2_3': 3}}   # each visited twice
    record_pair_outcomes(state, 's', [(1, 1), (1, 2)], ei, lopsided, stage='p', short=True,
                         short_min_transitions=3, short_min_fraction=0.0)
    assert state[OFFSET_KEY]['s|s1|1|2']['awaiting_confirm']
    record_pair_outcomes(state, 's', [(1, 1), (1, 2)], ei,
                         {'populations': [1000, 20], 'transition_pairs': {'2_3': 2}}, stage='p',
                         short=True, short_min_transitions=3, short_min_fraction=0.0)
    assert not state[OFFSET_KEY]['s|s1|1|2']['awaiting_confirm']


def test_group_ready_fraction():
    ref = {'s|s1|1|2', 's|s1|1|3', 's|s1|1|4', 's|s1|1|5'}
    ok = {'stages': {'p': {'visits': 2, 'successes': 2}}}
    state = {OFFSET_KEY: {'s|s1|1|2': ok, 's|s1|1|3': ok, 's|s1|1|4': ok}}
    g = '/x/group01_site1_subs_1_2_3_4_5'
    assert ss.group_ready(state, 's', g, ref, 'p', {'successes': 2}, fraction=0.75)      # 3 of 4
    assert not ss.group_ready(state, 's', g, ref, 'p', {'successes': 2}, fraction=1.0)


def test_short_runs_count_without_confirmation():
    ei = torch.tensor([[0, 1], [1, 0]])
    state = {}
    good = {'populations': [1000, 20], 'transition_pairs': {'2_3': 3}}
    for _ in range(2):
        record_pair_outcomes(state, 's', [(1, 1), (1, 2)], ei, good, stage='p', short=True,
                             short_min_transitions=3, short_min_fraction=0.0, confirm=False)
    e = state[OFFSET_KEY]['s|s1|1|2']
    assert e['stages']['p']['successes'] == 2 and not e['awaiting_confirm']


def test_label_group_export_renames_unconverged(tmp_path):
    from mllf.cb.export_biases import label_group_export, UNCONVERGED_SUFFIX
    g = tmp_path / 'sys' / 'group01_site1_subs_1_2_3'
    g.mkdir(parents=True)
    (g / 'variables.py').write_text('x')
    rows = [{'pair': 's1 1-2', 'reference': True, 'status': 'resolved', 'visits': 3, 'good_runs': 2,
             'b_offset_sd': 0.4, 'c_offset_sd': 0.9},
            {'pair': 's1 2-3', 'reference': False, 'status': 'never run', 'visits': 0, 'good_runs': 0,
             'b_offset_sd': 3.0, 'c_offset_sd': 8.0}]
    out = label_group_export(g, 'sys', g, None, rows)
    assert out.name == 'group01_site1_subs_1_2_3' + UNCONVERGED_SUFFIX and (out / 'variables.py').exists()
    assert 'did NOT pass' in (out / 'CONFIDENCE.txt').read_text()
    out2 = label_group_export(out, 'sys', g, '/runs/run_007', rows)        # later passed: renamed back
    assert out2.name == 'group01_site1_subs_1_2_3' and (out2 / 'confidence.json').exists()


# ── Retirement, group holds, reversed partners ──────────────────────────────

def _st(successes=0, epochs=0):
    return {'stages': {'p': {'visits': epochs, 'successes': successes, 'epochs': epochs}}}


def test_pair_retired_after_epochs_unless_resolved():
    adv = {'successes': 2}
    assert not ss.pair_retired(_st(epochs=9), 'p', 10)
    assert ss.pair_retired(_st(epochs=10), 'p', 10)
    assert not ss.pair_retired(_st(epochs=50), 'p', None)
    assert ss.pair_settled(_st(epochs=10), 'p', adv, retire_after=10)
    assert ss.pair_settled(_st(successes=2), 'p', adv, retire_after=10)
    state = {OFFSET_KEY: {'s|s1|1|2': _st(epochs=10)}}
    assert not ss.combo_resolved(state, 's', '/x/pair01_site1_subs_1_2', 'p', adv)
    assert ss.combo_resolved(state, 's', '/x/pair01_site1_subs_1_2', 'p', adv, retire_after=10)


def test_group_ready_leaves_out_retired_reference_pairs():
    ref = {'s|s1|1|2', 's|s1|1|3', 's|s1|1|4'}
    state = {OFFSET_KEY: {'s|s1|1|2': _st(successes=2), 's|s1|1|3': _st(epochs=10),
                          's|s1|1|4': _st(epochs=10)}}
    g = '/x/group01_site1_subs_1_2_3_4'
    adv = {'successes': 2}
    assert not ss.group_ready(state, 's', g, ref, 'p', adv, fraction=0.67)
    assert ss.group_ready(state, 's', g, ref, 'p', adv, fraction=0.67, retire_after=10)   # 1 of 1


def test_group_ready_waits_for_held_pairs():
    ref = {'s|s1|1|2', 's|s1|1|3'}
    state = {OFFSET_KEY: {'s|s1|1|2': _st(successes=2), 's|s1|1|3': _st(successes=2),
                          's|s1|2|3': _st()}}
    g = '/x/group01_site1_subs_1_2_3'
    adv = {'successes': 2}
    assert ss.group_ready(state, 's', g, ref, 'p', adv)
    assert not ss.group_ready(state, 's', g, ref, 'p', adv, held=['s|s1|2|3'])
    state[OFFSET_KEY]['s|s1|2|3'] = _st(epochs=10)                     # retired: no longer waits
    assert ss.group_ready(state, 's', g, ref, 'p', adv, retire_after=10, held=['s|s1|2|3'])


def test_group_failed_pairs_are_those_with_an_unsampled_substituent(tmp_path):
    import json
    g = tmp_path / 'group01_site1_subs_1_5_9'
    g.mkdir()
    (g / 'mapping.json').write_text(json.dumps([
        {'original_site': 1, 'original_sub': s, 'new_site': 1, 'new_sub': n, 'new_name': f'site1_sub{n}_pres.rtf'}
        for n, s in ((1, 1), (2, 5), (3, 9))]))
    keys = ss.group_failed_pairs('s', g, {'populations': [800, 0, 300]})
    assert sorted(keys) == ['s|s1|1|5', 's|s1|5|9']


def test_reopen_pairs_restarts_stage_count():
    state = {OFFSET_KEY: {'s|s1|1|2': {**_st(successes=2, epochs=4), 'successes': 2, 'visits': 4,
                                        'retired': 'satisfied'}}}
    assert ss.reopen_pairs(state, ['s|s1|1|2'], 'p') == 1
    e = state[OFFSET_KEY]['s|s1|1|2']
    assert e['stages']['p']['successes'] == 0 and e['stages']['p']['epochs'] == 0
    assert 'retired' not in e and e['reopened'] == 1
    assert not ss.pair_resolved(e, 'p', 2)


def test_reversed_pair_partner(tmp_path):
    p = cs.reversed_pair(tmp_path / 'pair07_site1_subs_1_9')
    assert p == str(tmp_path / 'pair07r_site1_subs_9_1')
    info = EXPLICIT_COMBOS[p]
    assert info['subs'] == [9, 1] and info['sites'] == [1]
    assert ss.combo_pair_keys('s', p) == ss.combo_pair_keys('s', tmp_path / 'pair07_site1_subs_1_9')
    assert cs.reversed_pair(p) is None                                 # a partner has no partner
    assert cs.reversed_pair(tmp_path / 'group01_site1_subs_1_2_3') is None
