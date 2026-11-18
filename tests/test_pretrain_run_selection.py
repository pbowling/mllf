"""Run selection in pretraining: the per-combo cap and loading each run once."""
import json

from mllf.pretrain.config import PretrainOptions
from mllf.pretrain.filters import cap_runs_per_combo, select_runs
from mllf.pretrain.rewards import run_end_state
from mllf.pretrain.runs import load_runs_from_dirs


def _sim(ns, p2, p3, trans):
    """One 2-sub site; p2, p3: the frames at lambda >= 0.99 of blocks 2 and 3."""
    return {'simulated_ns': ns,
            'populations': {str(b + 2): {'counts': {'0.95': p, '0.99': p}, 'site': 1}
                            for b, p in enumerate((p2, p3))},
            'transitions': {'1': {'0.99': trans}},
            'transition_pairs': {'2_3': trans}}


GOOD = (30000, 5000, 20)      # passes the end-state test (and has a positive reward)
POOR = (36000, 100, 4)        # second substituent under 1% of frames


def _run(tmp_path, combo, name, sim):
    d = tmp_path / 'ds' / combo / name
    return {'run_dir': d, 'metadata': {'nsubs_per_site': [2]}, 'sim_results': sim}


def test_end_state_of_a_whole_run():
    assert run_end_state({'metadata': {'nsubs_per_site': [2]}, 'sim_results': _sim(1.0, *GOOD[:2],
                          GOOD[2])}, {}) is True
    assert run_end_state({'metadata': {'nsubs_per_site': [2]}, 'sim_results': _sim(1.0, *POOR[:2],
                          POOR[2])}, {}) is False
    assert run_end_state({'sim_results': _sim(1.0, *GOOD[:2], GOOD[2])}, {}) is None


def test_cap_keeps_passing_runs_first_and_exempts_long_runs(tmp_path):
    runs = [_run(tmp_path, 'pair01', f'run_{i:03d}', _sim(1.0, *POOR[:2], POOR[2]))
            for i in range(4)]
    runs.append(_run(tmp_path, 'pair01', 'run_good', _sim(1.0, *GOOD[:2], GOOD[2])))
    runs += [_run(tmp_path, 'pair01', f'prod_{i}', _sim(7.0, *POOR[:2], POOR[2]))
             for i in range(3)]
    runs.append(_run(tmp_path, 'pair02', 'run_000', _sim(1.0, *POOR[:2], POOR[2])))
    kept = cap_runs_per_combo(runs, 2, exempt_ns=5.0)
    names = [(r['run_dir'].parent.name, r['run_dir'].name) for r in kept]
    pair01 = [n for c, n in names if c == 'pair01' and not n.startswith('prod')]
    assert len(pair01) == 2 and 'run_good' in pair01
    assert sum(n.startswith('prod') for _, n in names) == 3         # exempt, not counted
    assert ('pair02', 'run_000') in names                             # under the cap
    # input order kept
    assert [r for r in runs if r in kept] == kept
    # without the exemption the long runs count toward the cap
    assert len(cap_runs_per_combo(runs, 2, exempt_ns=None)) == 3


def test_select_runs_applies_the_cap(tmp_path):
    runs = [_run(tmp_path, 'pair01', f'run_{i:03d}', _sim(1.0, *GOOD[:2], GOOD[2]))
            for i in range(5)]
    opts = PretrainOptions(filter_outliers=False, max_runs_per_combo=3)
    assert len(select_runs(runs, opts, {})) == 3
    assert len(select_runs(runs, PretrainOptions(filter_outliers=False), {})) == 5


def _write_run(d, sim):
    d.mkdir(parents=True)
    (d / 'metadata.json').write_text(json.dumps({'terminated_normally': True,
                                                 'nsubs_per_site': [2]}))
    (d / 'simulation_results.json').write_text(json.dumps(sim))


def test_runs_listed_twice_are_loaded_once(tmp_path):
    ds = tmp_path / 'sys_solv'
    sim = _sim(1.0, *GOOD[:2], GOOD[2])
    _write_run(ds / 'run1', sim)                                   # the dataset's own run
    _write_run(ds / 'pair01_site1_subs_1_2' / 'run_000', sim)      # a combo's run
    # the manifest scan lists the combo and the dataset (which reads into the combo)
    runs = load_runs_from_dirs([ds / 'pair01_site1_subs_1_2', ds])
    assert sorted(str(r['run_dir'].relative_to(ds)) for r in runs) == [
        'pair01_site1_subs_1_2/run_000', 'run1']
    assert {r['system_label'] for r in runs} == {'sys_solv'}
