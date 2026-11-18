"""mllf.analysis: log parsing, run outcomes, convergence tables, names and figures."""
import json

import numpy as np
import pytest


pd = pytest.importorskip('pandas')
pytest.importorskip('matplotlib')

from mllf.analysis import online_data as od  # noqa: E402
from mllf.analysis.common import Output, environment_of, table_text  # noqa: E402
from mllf.analysis.pretrain_data import family_of, system_name  # noqa: E402
from mllf.analysis.pretrain_plots import metrics, nearest_systems  # noqa: E402


LOG = """Scouting before epoch 0 (s): 1 groups scouted
  [SCOUT] sysA: 0/1 groups passed on their scouting run, 0 pairs resolved, 0.1 ns

Epoch 0 Summary: 4 runs = 0 groups + 2 reference pairs + 2 other pairs
  Pair runs good: short 2/4 (50.0%)
  Validation: 0/1 (0.0%) held-out pairs good; model accepted, new best saved
  System               Stage          Epoch  Progress     Ref pairs  Retired  Train ns
  sysA                 pairs_groups   1/2    0/1 groups   0/5        -             0.5

Epoch 1 Summary: 3 runs = 1 groups + 2 reference pairs + 0 other pairs
  Pair runs good: short 1/2 (50.0%)
  Groups: 1 run, 1 passed: sysA/group01
  System               Stage          Epoch  Progress     Ref pairs  Retired  Train ns
  sysA                 pairs_groups   2/2    1/1 groups   2/5        -             1.0
  Stage changes: sysA pairs_groups -> done (converged)
"""


def test_epoch_log_gives_pair_group_and_validation_success(tmp_path):
    path = tmp_path / 'epoch_summary.log'
    path.write_text(LOG)
    epochs, stages, scout_ns = od.parse_epoch_log(path)
    assert list(epochs['pairs_pct']) == [50.0, 50.0]
    assert np.isnan(epochs['groups_pct'][0]) and epochs['groups_pct'][1] == 100.0
    assert epochs['validation_pct'][0] == 0.0 and np.isnan(epochs['validation_pct'][1])
    assert list(stages['stage_epoch']) == [1, 2]
    assert stages['next'].iloc[1] == 'done' and scout_ns == pytest.approx(0.1)


def test_pair_and_group_outcomes():
    assert od.classify_pair(0, 0, 0) == 'neither sampled'
    assert od.classify_pair(10, 0, 0) == 'one-sided'
    assert od.classify_pair(95, 5, 20) == 'unbalanced'
    assert od.classify_pair(60, 40, 3) == 'few crossings'
    assert od.classify_pair(60, 40, 3, test=(2, 0.2)) == 'success'      # short-run test
    assert od.classify_group([30, 30, 40], [12], [3]) == 'passed'
    assert od.classify_group([30, 0, 70], [12], [3]) == 'unsampled substituent'
    assert od.classify_group([5, 45, 50], [12], [3]) == 'uneven'
    assert od.classify_group([30, 30, 40], [4], [3]) == 'few transitions'
    sites = od.combo_subs('group01_site1_subs_1_2_3__site2_subs_1_2')
    assert sites == [(1, [1, 2, 3]), (2, [1, 2])]


def _data(tmp_path, ledger, stage_state, alf_group=True):
    return od.OnlineData(
        config={'systems': [{'name': 'sysA', 'solvent_state': 'solv'}]}, run_dir=tmp_path,
        stages=[], env={'sysA': 'solvent'}, epochs=pd.DataFrame(), stage_rows=pd.DataFrame(),
        ledger=ledger, stage_state=stage_state,
        group_combos={'sysA': ['group01_site1_subs_1_2_3']},
        alf_group_systems={'sysA'} if alf_group else set())


def test_convergence_counts_ns_until_each_side_passes(tmp_path):
    g = 'group01_site1_subs_1_2_3'
    ledger = {
        'sim/sysA/pair01_site1_subs_1_2/run_000': {
            'system': 'sysA', 'kind': 'train', 'epoch': 0, 'ns': 0.1},
        f'sim/sysA/{g}/run_001': {'system': 'sysA', 'kind': 'train', 'epoch': 1, 'ns': 1.0},
        f'sim/sysA/{g}/run_002': {'system': 'sysA', 'kind': 'train', 'epoch': 2, 'ns': 1.0},
        'sim/sysA/pair09_site1_subs_1_3/run_000': {'system': 'sysA', 'kind': 'validation',
                                                   'epoch': 0, 'ns': 5.0}}
    state = {'sysA': {'converged': True, 'groups_done': {g: f'sim/sysA/{g}/run_001'},
                      'history': [{'ended_epoch': 1}]}}
    alf = {'sysA': {'sysA_group1': [('run1', 2.0, False), ('run2', 3.0, True)]}}
    sys_df, grp_df = od.convergence_tables(_data(tmp_path, ledger, state), alf)
    s, gr = sys_df.iloc[0], grp_df.iloc[0]
    assert s['mllf_converged'] and s['mllf_ns'] == pytest.approx(1.1)   # validation left out
    assert s['alf_converged'] and s['alf_ns'] == pytest.approx(5.0)
    assert gr['mllf_passed'] and gr['mllf_ns'] == pytest.approx(1.1)    # through its passing run
    assert gr['alf_passed'] and gr['alf_ns'] == pytest.approx(5.0)
    # hand-written groups are not ALF's: no group-level ALF comparison
    _, grp_df = od.convergence_tables(_data(tmp_path, ledger, state, alf_group=False), alf)
    assert grp_df.iloc[0]['alf_dataset'] is None


def test_alf_runs_judged_by_the_production_test():
    from mllf.cb.system_stages import FRAMES_PER_NS
    frames = FRAMES_PER_NS          # 1 ns
    res = {'populations': {'2': {'counts': {'0.99': 0.30 * frames}},
                           '3': {'counts': {'0.99': 0.02 * frames}}},
           'transition_pairs': {'2_3': 6}}
    assert od.alf_production(res, [2], 1.0, 0.01, 2.0)        # 3 entries / ns each
    res['populations']['3']['counts']['0.99'] = 0.005 * frames
    assert not od.alf_production(res, [2], 1.0, 0.01, 2.0)    # sub 2 under 1% at its endpoint
    res['populations']['3']['counts']['0.99'] = 0.02 * frames
    res['transition_pairs']['2_3'] = 2
    assert not od.alf_production(res, [2], 1.0, 0.01, 2.0)    # 1 entry / ns


def test_alf_test_follows_the_group_stage(tmp_path):
    data = _data(tmp_path, {}, {})
    data.stages = [{'name': 'g', 'advance': {'type': 'groups_done', 'min_population_ratio': 0.0,
                                             'min_site_transitions': 0,
                                             'production': {'min_end_fraction': 0.01,
                                                            'min_visits_per_ns': 2.0}}}]
    assert data.alf_test() == ('production', 0.01, 2.0, None)
    data.stages[0]['advance']['production']['min_end_share'] = 0.02
    assert data.alf_test() == ('production', 0.01, 2.0, 0.02)
    data.group_test_mode, data.balance_test = 'balance', (0.5, 10)
    assert data.alf_test() == ('balance', 0.5, 10)
    assert data.stage_group_test('g')['production'] is None


def test_balance_mode_recomputes_mllf_convergence(tmp_path):
    g = 'group01_site1_subs_1_2_3'
    ledger = {f'sim/sysA/{g}/run_00{e}': {'system': 'sysA', 'kind': 'train', 'epoch': e,
                                          'ns': 1.0} for e in range(3)}
    # training (end-state test) passed the group at epoch 2; balance passes it at epoch 0
    state = {'sysA': {'converged': True, 'groups_done': {g: f'sim/sysA/{g}/run_002'},
                      'history': [{'ended_epoch': 2}]}}
    runs = pd.DataFrame([{'system': 'sysA', 'combo': g, 'role': 'group', 'test': 'group',
                          'epoch': e, 'outcome': o}
                         for e, o in enumerate(['passed', 'short of end-state test',
                                                'passed'])])
    data = _data(tmp_path, ledger, state)
    sys_df, _ = od.convergence_tables(data, {}, runs)
    assert sys_df.iloc[0]['mllf_ns'] == pytest.approx(3.0)
    data.group_test_mode = 'balance'
    sys_df, grp_df = od.convergence_tables(data, {}, runs)
    assert sys_df.iloc[0]['mllf_converged'] and sys_df.iloc[0]['mllf_ns'] == pytest.approx(1.0)
    assert grp_df.iloc[0]['mllf_ns'] == pytest.approx(1.0)


def test_balance_mode_with_alf_runs_for_several_systems(tmp_path):
    """ALF's runs of one system must not replace the scanned runs the next
    system's balance test reads."""
    g = 'group01_site1_subs_1_2_3'
    ledger = {f'sim/{s}/{g}/run_000': {'system': s, 'kind': 'train', 'epoch': 0, 'ns': 1.0}
              for s in ('sysA', 'sysB')}
    state = {s: {'converged': False, 'groups_done': {}, 'history': [{}]} for s in ('sysA', 'sysB')}
    runs = pd.DataFrame([{'system': s, 'combo': g, 'role': 'group', 'test': 'group',
                          'epoch': 0, 'outcome': 'passed'} for s in ('sysA', 'sysB')])
    data = od.OnlineData(
        config={'systems': [{'name': s, 'solvent_state': 'solv'} for s in ('sysA', 'sysB')]},
        run_dir=tmp_path, stages=[], env={'sysA': 'solvent', 'sysB': 'solvent'},
        epochs=pd.DataFrame(), stage_rows=pd.DataFrame(), ledger=ledger, stage_state=state,
        group_combos={'sysA': [g], 'sysB': [g]}, alf_group_systems={'sysA', 'sysB'})
    data.group_test_mode = 'balance'
    alf = {s: {f'{s}_group1': [('run1', 1.0, True)]} for s in ('sysA', 'sysB')}
    sys_df, _ = od.convergence_tables(data, alf, runs)
    assert list(sys_df['mllf_converged']) == [True, True]


def test_sampled_mode_passes_a_run_with_every_substituent_sampled_once(tmp_path):
    """The old criterion: any time at every endpoint passes, uneven or not, with
    no transitions needed and no pooling; ALF's runs get the same test."""
    data = od.OnlineData(config={'systems': []}, run_dir=tmp_path,
                         stages=[{'name': 'g', 'group_pass_window': 3,
                                  'advance': {'type': 'groups_done',
                                              'production': {'min_end_fraction': 0.01}}}],
                         env={}, epochs=pd.DataFrame(), stage_rows=pd.DataFrame(), ledger={},
                         stage_state={}, group_test_mode='sampling', balance_test=(0.0, 0))
    t = data.stage_group_test('g')
    assert t == {'ratio': 0.0, 'trans': 0, 'production': None, 'window': 1}
    assert od.classify_group([500, 1, 2], [0], [3], t['ratio'], t['trans']) == 'passed'
    assert od.classify_group([500, 0, 2], [40], [3], t['ratio'], t['trans']) == \
        'unsampled substituent'
    assert data.alf_test() == ('balance', 0.0, 0)
    with pytest.raises(ValueError):
        od.load_online({'systems': []}, tmp_path, group_test='bogus')


def test_alf_runs_read_again_when_a_dataset_is_replaced(tmp_path, monkeypatch):
    import os
    import shutil
    root = tmp_path / 'pre'
    (root / 'sysA_group1' / 'run1').mkdir(parents=True)
    (root / 'sysA_group1.new' / 'run1').mkdir(parents=True)     # being rebuilt: ignored
    calls = []
    monkeypatch.setattr(od, 'alf_group_runs', lambda d, key: calls.append(d.name) or [])
    data = od.OnlineData(config={'systems': [{'name': 'sysA'}]}, run_dir=tmp_path, stages=[],
                         env={}, epochs=pd.DataFrame(), stage_rows=pd.DataFrame(), ledger={},
                         stage_state={})
    cache = tmp_path / 'alf.pkl'
    od.alf_runs(data, root, cache)
    od.alf_runs(data, root, cache)
    assert calls == ['sysA_group1']                   # cached the second time
    shutil.rmtree(root / 'sysA_group1')
    (root / 'sysA_group1' / 'run1').mkdir(parents=True)
    st = (root / 'sysA_group1').stat()
    os.utime(root / 'sysA_group1', ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    od.alf_runs(data, root, cache)
    assert calls == ['sysA_group1', 'sysA_group1']    # replaced: read again


def test_caches_saved_at_once_do_not_collide(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from mllf.analysis.common import load_cache, save_cache
    path = tmp_path / 'c.pkl'
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda i: save_cache(path, {'i': i}), range(40)))
    assert load_cache(path)['i'] in range(40)
    assert not list(tmp_path.glob('*.tmp'))


def test_names_environments_and_tables(tmp_path):
    assert system_name('luis_bace_solvent_group3') == 'luis_bace_solvent'
    assert system_name('14benz_solv_pair_combos') == '14benz_solvent'
    assert family_of('14benz_vac') == '14benz'
    assert environment_of('x_vac') == 'vacuum' and environment_of('x', 'protein') == 'protein'
    text = table_text([['a', 1.234, True], ['bb', None, False]], ['Name', 'Value', 'Ok'])
    assert '1.23' in text and '-' in text and 'yes' in text
    out = Output(tmp_path, only=['parity_systems'])
    assert out.wanted('parity_systems/sysA') and not out.wanted('parity_all')


def test_metrics_and_nearest_systems():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert m['mue'] == pytest.approx(1 / 3) and m['r2'] == pytest.approx(0.5)
    D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], float)
    near = nearest_systems(['a', 'b', 'c'], D, {'a': 'f', 'b': 'f', 'c': 'g'})
    assert near['a'] == [('c', 2.0)]           # same-family b left out


def test_figures_draw(tmp_path):
    from mllf.analysis import online_plots as op
    from mllf.analysis import style as bs
    bs.apply_style()
    epochs = pd.DataFrame({'epoch': [0, 1], 'pairs_pct': [50.0, 75.0],
                           'groups_pct': [np.nan, 100.0], 'validation_pct': [0.0, np.nan]})
    runs = pd.DataFrame({'system': ['sysA'] * 3, 'epoch': [0, 0, 1],
                         'role': ['reference pair', 'other pair', 'group'],
                         'test': ['pair', 'pair', 'group'],
                         'outcome': ['success', 'one-sided', 'passed']})
    sys_df = pd.DataFrame({'system': ['sysA'], 'mllf_converged': [True], 'mllf_ns': [1.1],
                           'alf_converged': [False], 'alf_ns': [5.0]})
    out = Output(tmp_path, fmt='pdf', grayscale=True)
    out.figure('success', lambda: op.success_over_epochs(epochs))
    out.figure('outcomes', lambda: op.outcomes_by_epoch(runs))
    out.figure('convergence', lambda: op.convergence_systems(sys_df, {'sysA': 'solvent'}))
    assert out.written == ['success', 'outcomes', 'convergence']
    for name in out.written:
        assert (tmp_path / 'figures' / f'{name}.pdf').exists()
        assert (tmp_path / 'figures' / f'{name}_gray.png').exists()
    assert json.dumps(out.skipped) == '{}'
    png = Output(tmp_path / 'png')                       # the default: PNG only
    png.figure('success', lambda: op.success_over_epochs(epochs))
    assert [p.name for p in (tmp_path / 'png' / 'figures').iterdir()] == ['success.png']


def test_free_energy_panels_share_a_range_covering_every_curve():
    pytest.importorskip('matplotlib')
    from mllf.analysis import free_energy as fe
    x = np.linspace(0.0125, 0.9875, 40)
    bump = np.sin(np.pi * x)

    def prof(h):
        return (x, h * bump, None)
    one_d = {1: prof(2.0), 2: prof(4.6), 3: prof(1.0)}
    trans = {(1, 2): prof(2.0), (1, 3): prof(5.3), (2, 3): prof(1.2)}   # tallest not last
    alf = ({1: prof(2.5)}, {(1, 2): prof(2.4)})
    fig = fe.profile_figure({}, 1, (one_d, trans), alf)
    axes = [ax for ax in fig.axes if ax.lines]
    pairs = [ax for ax in axes if ax.get_title() in ('1 / 2', '1 / 3', '2 / 3')]
    assert len(pairs) == 3
    for ax in pairs:
        lo, hi = ax.get_ylim()
        assert lo == 0 and hi >= 5.3 and hi < 7     # covers 1 / 3, ends on a tick
        assert ax._bowling_no_snap
    top = [ax for ax in axes if ax not in pairs]
    assert all(ax.get_ylim()[1] >= 4.6 for ax in top)
