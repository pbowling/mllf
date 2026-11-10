"""bandit.algorithm and the settings a hindsight run uses unless its config says otherwise."""
import copy
from typing import Dict, Optional

# 'reinforce' (default) or 'hindsight'; 'neurallinear_ts' is hindsight's
# former name, still accepted.
BANDIT_ALGORITHMS = ('reinforce', 'hindsight')
_ALGORITHM_ALIASES = {'neurallinear_ts': 'hindsight'}

# bandit.* for algorithm: hindsight. A key present in the config wins, even
# when null (null turns off c_cap / group_holdoff; group_correction: false).
HINDSIGHT_DEFAULTS = {
    'update_target': 'hindsight',
    'shared_update': 'regression',
    'exploration_scale': 0.4,
    'offset_sampling': True,
    'selection_attention_weight': 0.5,
    # one-sided and pinned runs step b; c steps up only once the b bracket
    # closes without a crossing (barrier trap)
    'b_direction_requires_crossing': False,
    'c_up_requires_b_bracket': True,
    'linear_step_max': 12.0,
    'hindsight_weight_scale': {'linear': 0.1, 'quadratic': 0.01, 'skew': 0.1, 'end': 0.1},
    'c_cap': {'floor': 15.0, 'fraction': 0.5},
    'group_holdoff': {'dominant_min': 0.7, 'at_max': 0.05, 'partner_min': 0.02, 'min_partners': 2},
    'group_correction': {'step_x': 0.35, 'floor_x': 0.4, 'step_s': 0.15, 'floor_s': 0.7},
    'revisit': {'enabled': True, 'max_visits': 12},
    'regression': {'replay_selection': 'coverage', 'replay_pretraining_only': False},
}
# Merged key by key into a config's own block rather than replaced by it.
_MERGED = ('revisit', 'regression')

# curriculum.stages[*] keys for algorithm: hindsight.
HINDSIGHT_STAGE_DEFAULTS = {
    'bracketed_runs_sticky': True,
    'partial_runs_to_resolve': 3,
    'group_pass_window': 3,
}


def bandit_algorithm(bandit_cfg: Optional[Dict]) -> str:
    """The configured bandit algorithm, with former names mapped to current ones."""
    name = (bandit_cfg or {}).get('algorithm', 'reinforce')
    return _ALGORITHM_ALIASES.get(name, name)


def with_hindsight_defaults(config: Dict) -> Dict:
    """Fill a hindsight config's missing bandit, stage and policy settings in
    place (HINDSIGHT_DEFAULTS, HINDSIGHT_STAGE_DEFAULTS, Bayesian heads).
    Other configs are returned unchanged."""
    bandit = config.get('bandit')
    if not bandit or bandit_algorithm(bandit) != 'hindsight':
        return config
    for key, value in HINDSIGHT_DEFAULTS.items():
        if key in _MERGED and isinstance(bandit.get(key), dict):
            for k, v in value.items():
                bandit[key].setdefault(k, copy.deepcopy(v))
        else:
            bandit.setdefault(key, copy.deepcopy(value))
    if not isinstance(config.get('training'), dict):
        config['training'] = {}
    if not isinstance(config['training'].get('policy'), dict):
        config['training']['policy'] = {}
    config['training']['policy'].setdefault('use_bayesian_heads', True)
    for stage in (config.get('curriculum') or {}).get('stages') or []:
        for key, value in HINDSIGHT_STAGE_DEFAULTS.items():
            stage.setdefault(key, value)
    return config
