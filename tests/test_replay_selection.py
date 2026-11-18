"""Tests for mllf.cb.replay_selection: coverage choice of the regression replay."""
import pytest

from mllf.cb.replay_selection import (
    greedy_cover,
    owners,
    pair_quality,
    system_budgets,
    system_of_dataset,
)


KNOWN = [
    '12benz_solvent',
    '123benz_vac',
    '14benz_solv',
    'thrombin_solvent',
    'luis_p38_protein',
    '1benz_vac']


@pytest.mark.parametrize('name,system', [
    ('12benz_solvent_group2', '12benz_solvent'),
    ('123benz_vacuum_group1', '123benz_vac'),
    ('14benz_solv_triplet_combos_v2', '14benz_solv'),
    ('thrombin_solvent_group_combos', 'thrombin_solvent'),
    ('luis_p38_protein_group3', 'luis_p38_protein'),
    ('1benz_vac_group10', '1benz_vac'),
    ('p38_water_groupA', 'p38_water'),
])
def test_system_of_dataset(name, system):
    assert system_of_dataset(name, KNOWN) == system


def _sim(pops, pair_trans=0, site_trans=0, ns=1.0, ddg=0.5):
    return {'populations': {str(n + 2): {'counts': {'0.99': p}} for n, p in enumerate(pops)},
            'transition_pairs': {'2_3': pair_trans},
            'transitions': {'1': {'0.99': site_trans}},
            'ddg_pairs': {'2_3': ddg}, 'simulated_ns': ns}


def test_pair_quality_balance_times_crossings():
    assert pair_quality(_sim([500, 500], pair_trans=20), 0, 1, 1, 2) == pytest.approx(1.0)
    assert pair_quality(_sim([1000, 250], pair_trans=5), 0, 1, 1, 2) == pytest.approx(0.25 * 0.5)
    assert pair_quality(_sim([1000, 0], pair_trans=5), 0, 1, 1, 2) == 0.0           # unsampled
    assert pair_quality(_sim([500, 500], pair_trans=5, ddg=float('nan')), 0, 1, 1, 2) == 0.0


def test_pair_quality_short_run_test_and_site_crossings():
    # a short run needs 3 crossings, not 10
    assert pair_quality(_sim([40, 40], pair_trans=3, ns=0.104), 0, 1, 1, 2) == pytest.approx(1.0)
    # on a larger site, lambda crosses through the others: the site's count is used
    assert pair_quality(_sim([400, 400, 400], site_trans=12), 0, 1, 1, 3) == pytest.approx(1.0)


def test_greedy_prefers_one_group_run_over_slightly_better_pair_runs():
    cands = {'group': {'ab': 0.8, 'ac': 0.8, 'bc': 0.8},
             'pair_ab': {'ab': 0.95}, 'pair_ac': {'ac': 0.95}, 'pair_bc': {'bc': 0.95}}
    assert greedy_cover(cands, budget=10) == ['group']      # nothing else adds coverage
    assert owners(cands, ['group']) == {'ab': 'group', 'ac': 'group', 'bc': 'group'}


def test_greedy_upgrades_poorly_covered_pairs_and_respects_budget():
    cands = {'g1': {'ab': 0.9, 'ac': 0.2}, 'p_ac': {'ac': 0.7}, 'p_ad': {'ad': 0.3}}
    assert greedy_cover(cands, budget=1) == ['g1']
    chosen = greedy_cover(cands, budget=5)
    assert set(chosen) == {'g1', 'p_ac', 'p_ad'}
    assert owners(cands, chosen)['ac'] == 'p_ac'


def test_budgets_proportional_to_substituents_with_floor():
    b = system_budgets({'big': 53, 'small': 4, 'tiny': 1}, total=116, min_per_system=2)
    assert b == {'big': 106, 'small': 8, 'tiny': 2}


def test_greedy_second_level_adds_a_second_measurement():
    from mllf.cb.replay_selection import greedy_cover
    cands = {'g1': {'ab': 0.9, 'ac': 0.9}, 'g2': {'ab': 0.8, 'ac': 0.7}, 'p': {'ab': 0.95}}
    assert greedy_cover(cands, 5) == ['g1']                       # covered: nothing else adds
    assert greedy_cover(cands, 2, chosen=['g1'], level=2) == ['g1', 'g2']


def test_select_runs_redistributes_leftover_budget():
    from mllf.cb.replay_selection import select_runs
    cand = {
        'small': {'s1': {'ab': 0.9, 'ac': 0.9, 'bc': 0.9}, 's2': {'ab': 0.6}, 's3': {'ac': 0.6}},
        'big': {f'b{i}': {f'p{i}': 0.9} for i in range(10)},
    }
    chosen, extra = select_runs(cand, {'small': 5, 'big': 5}, total=10, min_graphs=1)
    assert chosen['small'] == ['s1']                                # stops: all covered
    assert len(chosen['big']) == 9                                  # 5 + the 4 'small' left


def test_select_runs_minimum_graphs_per_system():
    from mllf.cb.replay_selection import select_runs
    cand = {'one': {'g': {'ab': 0.9, 'ac': 0.9}, 'h': {'ab': 0.7, 'ac': 0.6}, 'k': {'ab': 0.2}}}
    chosen, extra = select_runs(cand, {'one': 10}, total=10, min_graphs=3)
    assert chosen['one'] == ['g']
    assert extra['one'][0] == 'h' and len(extra['one']) == 2       # second measurement, then any


def test_select_runs_minimum_limited_by_candidates():
    from mllf.cb.replay_selection import select_runs
    chosen, extra = select_runs({'tiny': {'g': {'ab': 0.9}}}, {'tiny': 3}, total=3, min_graphs=3)
    assert chosen['tiny'] == ['g'] and extra['tiny'] == []


# Scan: manifest paths, per-directory cache

def _write_run(run, done=True):
    import json
    run.mkdir(parents=True)
    (run / 'graph_info.json').write_text(json.dumps({
        'sites': {'a': {'site': 1, 'sub': 1, 'filename': 'site1_sub1_pres.rtf'},
                  'b': {'site': 1, 'sub': 2, 'filename': 'site1_sub2_pres.rtf'}},
        'active_subs_ordered': {'site1': ['a', 'b']}}))
    (run / 'metadata.json').write_text(json.dumps({'terminated_normally': True,
                                                   'nsubs_per_site': [2]}))
    if done:
        _finish(run)


def _finish(run):
    import json
    (run / 'simulation_results.json').write_text(json.dumps({
        'populations': {'2': {'counts': {'1.0': 90}}, '3': {'counts': {'1.0': 100}}},
        'ddg_pairs': {'2_3': 1.0}, 'transition_pairs': {'2_3': 20}, 'simulated_ns': 1.0}))


@pytest.fixture
def pretraining(tmp_path):
    """models/manifest.json listing ../pretraining/sysA_solvent/comb_1 (two
    finished runs, one unfinished) as pretraining writes it."""
    import json
    combo = tmp_path / 'pretraining' / 'sysA_solvent' / 'comb_1'
    (combo / 'prep').mkdir(parents=True)
    for sub, atom in ((1, 'C1 CG2R61 -0.1150'), (2, 'N1 NG2R50 -0.3000')):
        (combo / 'prep' / f'site1_sub{sub}_pres.rtf').write_text(f'ATOM {atom}\n')
    _write_run(combo / 'run_001')
    _write_run(combo / 'run_002')
    _write_run(combo / 'run_003', done=False)
    (tmp_path / 'models').mkdir()
    manifest = tmp_path / 'models' / 'manifest.json'
    manifest.write_text(json.dumps({'directories': ['../pretraining/sysA_solvent/comb_1']}))
    (tmp_path / 'systems').mkdir()
    return tmp_path, manifest, combo


def test_manifest_resolver_normalizes_as_text(pretraining, monkeypatch):
    from mllf.cb.replay_selection import manifest_resolver
    tmp, manifest, combo = pretraining
    monkeypatch.chdir(tmp / 'models')
    resolve = manifest_resolver(manifest)
    assert resolve('../pretraining/sysA_solvent/comb_1') == combo
    # the base is found once; later entries are not checked
    assert resolve('../pretraining/sysA_solvent/gone') == combo.parent / 'gone'
    assert resolve('../elsewhere/x') is None


def test_plan_replay_reuses_the_cached_scan(pretraining, monkeypatch):
    from mllf.cb import replay_selection as rs
    tmp, manifest, combo = pretraining
    monkeypatch.chdir(tmp / 'models')
    cache = tmp / 'cache' / 'scan.pkl'
    plan, _ = rs.plan_replay(manifest, tmp / 'systems', 2, 1, workers=2, verbose=False,
                             scan_cache=cache)
    assert cache.exists()
    first = sorted(p['run_dir'] for p in plan)
    assert first and all('run_003' not in r for r in first)

    labelled = []
    real = rs.node_labels
    monkeypatch.setattr(rs, 'node_labels', lambda *a: labelled.append(a[0]) or real(*a))
    plan, _ = rs.plan_replay(manifest, tmp / 'systems', 2, 1, workers=2, verbose=False,
                             scan_cache=cache)
    assert sorted(p['run_dir'] for p in plan) == first and not labelled

    _finish(combo / 'run_003')      # an unfinished run finishing: the directory is rescanned
    rs.plan_replay(manifest, tmp / 'systems', 2, 1, workers=2, verbose=False, scan_cache=cache)
    assert any('run_003' in str(r) for r in labelled)


def test_node_labels_checks_each_prep_file_once_per_combo(pretraining, monkeypatch):
    from mllf.cb import replay_selection as rs
    _, _, combo = pretraining
    index = rs.FragmentIndex(combo / 'prep')
    calls = []
    real = rs._match_fragment
    monkeypatch.setattr(rs, '_match_fragment', lambda *a: calls.append(a[1]) or real(*a))
    cache = {}
    assert rs.node_labels(combo / 'run_001', index, cache) == [(1, 1), (1, 2)]
    assert rs.node_labels(combo / 'run_002', index, cache) == [(1, 1), (1, 2)]
    assert len(calls) == 2
