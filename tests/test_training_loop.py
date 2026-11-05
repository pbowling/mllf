"""train_epoch and run_training end to end, on the fake cluster (conftest.py)."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from conftest import make_system
from mllf.cb.bandit_config import with_hindsight_defaults
from mllf.cb.epoch_runs import EpochOptions
from mllf.cb.policy import UnimolPolicy
from mllf.cb.reinforce_trainer import train_epoch
from mllf.cb.training_loop import run_training


def _config(tmp_path, systems, **extra):
    cfg = {'seed': 7, 'phase1_workers': 1, 'max_concurrent_jobs': 2, 'max_retries': 1,
           'poll_interval': 0, 'output': {'base_dir': str(tmp_path / 'ck'), 'checkpoint_freq': 1},
           'training': {'policy': {}}, 'reward': {'T_baseline': 20.0}, 'systems': systems}
    cfg.update(extra)
    return cfg


def _policy(bayesian):
    torch.manual_seed(0)
    return UnimolPolicy(unimol_dim=512, mlp_hidden=64, mlp_out_dim=4, use_dual_embeddings=True,
                        use_bayesian_heads=bayesian)


def _hindsight(cfg):
    cfg['bandit'] = {'algorithm': 'hindsight', 'regression': {'replay_weight': 0.0, 'steps': 2}}
    return with_hindsight_defaults(cfg)


def test_train_epoch_hindsight_records_runs_and_pair_state(tmp_path, fake_cluster):
    sys_a, combos = make_system(tmp_path, 'sysA', [(1, [1, 2, 3])])
    cfg = _hindsight(_config(tmp_path, [sys_a]))
    cmap = {c: sys_a for c in combos}
    policy = _policy(bayesian=True)
    opt = torch.optim.Adam(policy.edge_mlp.parameters(), lr=1e-3)
    fake_cluster.fail_first = ('subs_2_3',)
    state = {}
    stats = train_epoch(policy, opt, combos, cmap, 0, cfg, bandit_state=state,
                        stage_of_system={'sysA': 'pairs'})

    assert stats['num_combos'] == 3 and stats['permanently_failed'] == []
    assert len(stats['run_dirs']) == 3 and all(ok for _, _, ok in stats['run_dirs'])
    assert stats['pair_success']['total'] == 3
    # the failed first attempt was resubmitted with sbatch and then succeeded
    assert max(fake_cluster.attempts.values()) == 2
    for c in combos:
        rec = torch.load(Path(sys_a['out_dir']) / Path(c).name / 'run_000' / 'epoch_results.pt',
                         weights_only=False)
        assert rec['run_id'] == cfg['output']['base_dir'] and rec['actions'].shape == (2, 4)
    # per-pair offsets and outcomes were recorded for all three pairs
    assert len(state['_pair_offsets']) == 3
    assert 'hindsight' in stats and 'regression' in stats


def test_train_epoch_takes_options_or_keywords_not_both(tmp_path, fake_cluster):
    sys_a, combos = make_system(tmp_path, 'sysA', [(1, [1, 2])])
    with pytest.raises(TypeError):
        train_epoch(_policy(False), None, combos, {}, 0, _config(tmp_path, [sys_a]),
                    options=EpochOptions(), update_policy=False)


def test_train_epoch_reinforce_updates_only_when_training(tmp_path, fake_cluster):
    sys_a, combos = make_system(tmp_path, 'sysA', [(1, [1, 2, 3])])
    cfg = _config(tmp_path, [sys_a])
    cmap = {c: sys_a for c in combos}
    policy = _policy(bayesian=False)
    opt = torch.optim.Adam(policy.edge_mlp.parameters(), lr=1e-2)

    def weights():
        return torch.cat([p.detach().flatten().clone() for p in policy.edge_mlp.parameters()])
    before = weights()
    train_epoch(policy, opt, combos, cmap, 0, cfg, update_policy=False, deterministic_actions=True)
    assert torch.equal(before, weights())
    stats = train_epoch(policy, opt, combos, cmap, 1, cfg, ema_baseline={})
    assert stats['n_success'] == 3 and not torch.equal(before, weights())


def test_failed_runs_are_replaced_and_an_epoch_without_data_aborts(tmp_path, fake_cluster):
    sys_a, combos = make_system(tmp_path, 'sysA', [(1, [1, 2, 3])])
    cfg = _config(tmp_path, [sys_a], max_retries=0)
    cmap = {c: sys_a for c in combos}
    fake_cluster.fail_always = ('subs_1_2',)
    stats = train_epoch(_policy(False), None, combos[:1], cmap, 0, cfg, update_policy=False,
                        reserve_pool=combos[1:2])
    assert stats['permanently_failed'] == ['sysA/pair_site1_subs_1_2']
    assert [Path(d).parent.name for _, d, ok in stats['run_dirs'] if ok] == ['pair_site1_subs_1_3']

    fake_cluster.fail_always = ('subs_1_2', 'subs_1_3', 'subs_2_3')
    with pytest.raises(SystemExit):
        train_epoch(_policy(False), torch.optim.Adam(_policy(False).parameters()), combos, cmap,
                    1, cfg)


def _write_config(tmp_path, cfg):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_run_training_per_system_stages(tmp_path, fake_cluster):
    sys_a, _ = make_system(tmp_path, 'sysA', [(1, [1, 2, 3])])
    sys_b, _ = make_system(tmp_path, 'sysB', [(1, [1, 2]), (2, [1, 2])])
    for s in (sys_a, sys_b):
        s['combo_sets'] = {'groups': [{1: [1, 2]}], 'refpairs': {'pairs_of': 'groups'}}
        s['export_combo_set'] = 'groups'
    cfg = _hindsight(_config(tmp_path, [sys_a, sys_b], validation_per_system=1))
    cfg['curriculum'] = {
        'enabled': True, 'per_system': True, 'max_train_combos_per_stage': 4,
        'min_combos_per_system': 1,
        'stages': [
            {'name': 'pairs', 'epochs': 1, 'min_sites': 1, 'max_sites': 1, 'min_subs_per_site': 2,
             'max_subs_per_site': 2, 'run_steps': {'equil': 100, 'prod': 400},
             'advance': {'type': 'pairs_resolved', 'successes': 1}},
            {'name': 'groups', 'epochs': 1, 'combo_set': 'refpairs', 'group_set': 'groups',
             'reference_set': 'refpairs', 'group_ready_fraction': 0.0,
             'advance': {'type': 'groups_done'}},
        ]}
    torch.manual_seed(0)
    run_training(_write_config(tmp_path, cfg))

    ck = tmp_path / 'ck'
    final = torch.load(ck / 'final_model.pt', weights_only=False)
    assert len(final['all_stats']) == 2
    stages = json.loads((ck / 'system_stages.json').read_text())
    assert all(st['done'] for st in stages.values())
    assert len(json.loads((ck / 'validation_combos.json').read_text())) == 2   # 1 per system
    assert (ck / 'cost_ledger.json').exists()
    assert 'Epoch 1' in (ck / 'epoch_summary.log').read_text()
    # both systems finished their last stage: final biases exported per group
    for name in ('sysA', 'sysB'):
        assert list((ck / 'final_biases' / name).rglob('variables.py'))


def test_run_training_resumes_and_finishes_with_no_epochs_left(tmp_path, fake_cluster):
    sys_a, _ = make_system(tmp_path, 'sysA', [(1, [1, 2, 3])])
    cfg = _config(tmp_path, [sys_a], validation_per_system=0)
    cfg['bandit'] = {'algorithm': 'reinforce'}
    cfg['training']['context_baseline'] = {'enabled': True}
    cfg['curriculum'] = {'enabled': True, 'max_train_combos_per_stage': 2, 'stages': [
        {'name': 's1', 'epochs': 2, 'min_sites': 1, 'max_sites': 1, 'min_subs_per_site': 2,
         'max_subs_per_site': 2}]}
    path = _write_config(tmp_path, cfg)
    torch.manual_seed(0)
    run_training(path)
    ck = tmp_path / 'ck'
    names = sorted(p.name for p in ck.glob('checkpoint_*.pt'))
    assert names == ['checkpoint_000.pt', 'checkpoint_001.pt']
    # Resuming a finished run trains no epoch and still saves the final model
    # (with the context baseline) -- this used to raise NameError.
    (ck / 'final_model.pt').unlink()
    run_training(path)
    final = torch.load(ck / 'final_model.pt', weights_only=False)
    assert final['all_stats'] == [] and 'baseline_state_dict' in final
