"""bandit.algorithm names and the settings hindsight runs default to (mllf.cb.bandit_config)."""
from mllf.cb.bandit_config import (
    HINDSIGHT_DEFAULTS,
    BanditConfig,
    bandit_algorithm,
    with_hindsight_defaults,
)
from mllf.cb.hindsight import BIAS_DIMS


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
    assert [s['group_pass_window']
            for s in cfg['curriculum']['stages']] == [3, 1]   # a stage's own value wins
    assert cfg['curriculum']['stages'][0]['partial_runs_to_resolve'] == 3


def test_config_values_win_and_null_turns_a_default_off():
    b = with_hindsight_defaults(
        _config(
            c_cap=None,
            exploration_scale=1.0,
            group_correction=False))['bandit']
    assert b['c_cap'] is None and b['group_correction'] is False and b['exploration_scale'] == 1.0


def test_revisit_and_regression_merge_key_by_key():
    b = with_hindsight_defaults(
        _config(
            revisit={
                'max_visits': 4}, regression={
                'lr': 1e-4}))['bandit']
    assert b['revisit'] == {'max_visits': 4, 'enabled': True}
    assert b['regression']['lr'] == 1e-4 and b['regression']['replay_selection'] == 'coverage'


def test_defaults_are_copies_and_other_algorithms_are_untouched():
    a = with_hindsight_defaults(_config())
    a['bandit']['c_cap']['floor'] = 0.0
    assert HINDSIGHT_DEFAULTS['c_cap']['floor'] == 15.0
    reinforce = {'bandit': {'algorithm': 'reinforce'}}
    assert with_hindsight_defaults(reinforce) == {'bandit': {'algorithm': 'reinforce'}}
    assert with_hindsight_defaults({}) == {}


# BanditConfig

def test_bandit_config_defaults_without_a_bandit_block():
    for cfg in ({}, {'bandit': None}):
        b = BanditConfig.from_config(cfg)
        assert b.algorithm == 'reinforce' and b.update_target == 'awr'
        assert not b.use_hindsight and not b.shared_regression
        assert b.selection_mode == 'random' and b.hindsight_dims == BIAS_DIMS


def test_bandit_config_reads_hindsight_defaults_and_derived_settings():
    b = BanditConfig.from_config(with_hindsight_defaults(_config()))
    assert b.use_hindsight and b.shared_regression and b.validation_gate
    assert b.selection_mode == 'uncertainty' and b.exploration_scale == 0.4
    assert b.weight_scale(['linear', 'quadratic']) == {'linear': 0.1, 'quadratic': 0.01}
    assert b.step_caps['linear_step_cap'] == 3.0


def test_bandit_config_null_turns_off_or_keeps_the_default():
    b = BanditConfig.from_dict({'c_cap': None, 'exploration_scale': None,
                                'hindsight_dims': ['linear']})
    assert b.c_cap is None and b.exploration_scale == 1.0
    assert b.hindsight_dims == ('linear',) and b.offset_dims == ['linear']


def test_unknown_bandit_keys_are_reported():
    block = {'algoritm': 'hindsight', 'c_cap': {}, 'group_correction': {}}
    assert BanditConfig.unknown_keys(block) == ['algoritm']


def test_unimol_defaults_are_the_current_model_settings():
    """A config without a unimol: block builds the 515-D phase + site-size policy
    of the current pretrained model; keys it sets override the defaults."""
    from mllf.cb.policy import unimol_options
    from mllf.cb.training_setup import build_policy
    from mllf.pretrain.config import UnimolSettings

    u = unimol_options({})
    assert (u['env_input'], u['site_size_input'], u['ligand_other_sites']) == ('phase', True, True)
    assert unimol_options({'unimol': {'env_input': 'dual', 'site_size_input': False}})[
        'env_input'] == 'dual'
    policy = build_policy({}, 'cpu')
    first = next(m for m in policy.edge_mlp.mlps[0].trunk if hasattr(m, 'in_features'))
    assert first.in_features == 515
    s = UnimolSettings.from_config({})
    assert (s.env_input, s.site_size_input, s.ligand_other_sites,
            s.structure_selection) == ('phase', True, True, 'unrelaxed')
