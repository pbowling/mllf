"""bandit.algorithm names and the settings hindsight runs default to (mllf.cb.bandit_config)."""
from mllf.cb.bandit_config import HINDSIGHT_DEFAULTS, bandit_algorithm, with_hindsight_defaults


def test_bandit_algorithm_maps_the_former_name():
    assert bandit_algorithm({'algorithm': 'neurallinear_ts'}) == 'hindsight'
    assert bandit_algorithm({'algorithm': 'hindsight'}) == 'hindsight'
    assert bandit_algorithm({}) == 'reinforce' and bandit_algorithm(None) == 'reinforce'
    assert bandit_algorithm({'algorithm': 'typo'}) == 'typo'     # rejected by run_training


def _config(**bandit):
    return {'bandit': {'algorithm': 'hindsight', **bandit},
            'curriculum': {'stages': [{'name': 'a'}, {'name': 'b', 'group_pass_window': 1}]}}


def test_hindsight_defaults_fill_missing_settings():
    cfg = with_hindsight_defaults(_config())
    b = cfg['bandit']
    assert b['update_target'] == 'hindsight' and b['shared_update'] == 'regression'
    assert b['c_cap'] == HINDSIGHT_DEFAULTS['c_cap']
    assert cfg['training']['policy']['use_bayesian_heads'] is True
    assert [s['group_pass_window'] for s in cfg['curriculum']['stages']] == [3, 1]   # a stage's own value wins
    assert cfg['curriculum']['stages'][0]['partial_runs_to_resolve'] == 3


def test_config_values_win_and_null_turns_a_default_off():
    b = with_hindsight_defaults(_config(c_cap=None, exploration_scale=1.0, group_correction=False))['bandit']
    assert b['c_cap'] is None and b['group_correction'] is False and b['exploration_scale'] == 1.0


def test_revisit_and_regression_merge_key_by_key():
    b = with_hindsight_defaults(_config(revisit={'max_visits': 4}, regression={'lr': 1e-4}))['bandit']
    assert b['revisit'] == {'max_visits': 4, 'enabled': True}
    assert b['regression']['lr'] == 1e-4 and b['regression']['replay_selection'] == 'coverage'


def test_defaults_are_copies_and_other_algorithms_are_untouched():
    a = with_hindsight_defaults(_config())
    a['bandit']['c_cap']['floor'] = 0.0
    assert HINDSIGHT_DEFAULTS['c_cap']['floor'] == 15.0
    reinforce = {'bandit': {'algorithm': 'reinforce'}}
    assert with_hindsight_defaults(reinforce) == {'bandit': {'algorithm': 'reinforce'}}
    assert with_hindsight_defaults({}) == {}
