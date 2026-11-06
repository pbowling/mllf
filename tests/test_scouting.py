"""mllf.cb.scouting: tiers, stage choice, model-settings check, CLI."""
import json

import pytest

from mllf.cb import scouting


def test_scout_tier_orders_hardest_pairs_first():
    assert scouting.scout_tier('one empty', 0.0, 0) == 0
    assert scouting.scout_tier('both empty', 0.0, 0) == 0
    assert scouting.scout_tier('crossing', 0.2, 4) == 0
    assert scouting.scout_tier('crossing', 0.2, 7) == 1
    assert scouting.scout_tier('crossing', 0.2, 10) == 2
    assert scouting.scout_tier('sampled', 0.3, 0) == 2


def test_scouting_stage_is_named_or_first_with_groups():
    cfg = {'curriculum': {'stages': [{'name': 'a'}, {'name': 'b', 'group_set': 'g'},
                                     {'name': 'c', 'group_set': 'h'}]}}
    assert scouting.scouting_stage(cfg)['name'] == 'b'
    assert scouting.scouting_stage(cfg, 'c')['name'] == 'c'
    with pytest.raises(ValueError):
        scouting.scouting_stage(cfg, 'missing')
    with pytest.raises(ValueError):
        scouting.scouting_stage({'curriculum': {'stages': [{'name': 'a'}]}})


def test_model_path_prefers_deploy():
    assert scouting.model_path({'pretrain': {'model_path': 'p'}}) == 'p'
    assert scouting.model_path({'pretrain': {'model_path': 'p'},
                                'deploy': {'model_path': 'd'}}) == 'd'


def test_model_settings_must_match_the_config(tmp_path):
    """Scouting and training check the config (with the unimol defaults filled
    in) against the model's pretrain_metadata.json."""
    from mllf.cb.training_setup import check_pretrain_inputs
    (tmp_path / 'pretrain_metadata.json').write_text(json.dumps(
        {'env_input': 'phase', 'site_size_input': True, 'ligand_other_sites': True}))
    model = tmp_path / 'best_policy_bayesian.pt'
    check_pretrain_inputs({}, model)                          # the defaults match
    with pytest.raises(ValueError, match='site_size_input'):
        check_pretrain_inputs({'unimol': {'site_size_input': False}}, model)
    with pytest.raises(ValueError, match='env_input'):
        check_pretrain_inputs({'unimol': {'env_input': 'dual'}}, model)
    check_pretrain_inputs({'unimol': {'env_input': 'dual'}}, tmp_path / 'other' / 'm.pt')


def test_scout_command_is_registered():
    from mllf.cli.main import COMMANDS
    assert COMMANDS['scout'][0] == 'mllf.cli.scout'


def test_wait_needs_a_terminal_state_when_sacct_answers(monkeypatch):
    """A job missing from squeue but not ended in sacct (e.g. requeued) is
    still waited for."""
    polls = iter([(set(), set()), (set(), {'7'})])
    seen = []

    def fake_queued(ids):
        seen.append(set(ids))
        return fake_queued.now[0]

    def fake_ended(ids):
        return fake_queued.now[1]

    def advance():
        fake_queued.now = next(polls)
    advance()
    monkeypatch.setattr(scouting, '_queued', fake_queued)
    monkeypatch.setattr(scouting, '_ended', fake_ended)
    monkeypatch.setattr(scouting.time, 'sleep', lambda s: advance())
    scouting.wait_for_jobs(['7'], poll_seconds=0)
    assert len(seen) == 2


def test_wait_falls_back_to_squeue_without_sacct(monkeypatch):
    monkeypatch.setattr(scouting, '_queued', lambda ids: set())
    monkeypatch.setattr(scouting, '_ended', lambda ids: None)
    scouting.wait_for_jobs(['7'], poll_seconds=0)
