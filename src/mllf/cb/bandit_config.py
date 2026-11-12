"""The ``bandit:`` block of a workflow config.

``with_hindsight_defaults`` fills in the settings a hindsight run uses unless
its config says otherwise; ``BanditConfig.from_config`` then reads the block
once, with every remaining default in one place.
"""
import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from mllf.cb.hindsight import (
    BARRIER_TRAP_WIDTH,
    BIAS_DIMS,
    DEFAULT_TEMPERATURE,
    EXPLORE_FACTOR_MAX,
    OFFSET_DIMS,
    PERSISTENT_FAILURES,
    QUADRATIC_STEP_MAX,
)


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


@dataclass(frozen=True)
class BanditConfig:
    """Typed view of ``config['bandit']`` with the defaults the training code
    uses when a key is absent. Build it with ``from_config`` after
    ``with_hindsight_defaults``; ``raw`` keeps the block itself for the
    sub-blocks other modules merge with their own defaults
    (``group_correction.correction_config``, ``hindsight_regression.regression_config``).
    """
    algorithm: str = 'reinforce'
    update_target: str = 'awr'                 # 'awr' or 'hindsight'
    shared_update: str = 'posterior'           # 'posterior' or 'regression'
    selection: Optional[str] = None            # None: uncertainty for hindsight, else random
    selection_attention_weight: float = 0.0
    candidate_pool_size: Optional[int] = None  # None: 5 x the epoch's combo budget
    min_observations_before_confident: int = 3
    prior_precision: float = 1.0
    exploration_scale: float = 1.0
    failure_retry_cap: int = 3
    awr_temperature: float = 0.5
    # per-pair offsets
    pair_offsets: bool = True
    offset_sampling: bool = False
    offset_prior_sd: Optional[Dict[str, float]] = None
    offset_base_sd: Optional[Dict[str, float]] = None
    offset_history: int = 5
    # the pair offsets group runs use (None: all of hindsight_dims; []: none,
    # every group run starts from the model's prediction plus the group correction)
    group_offset_dims: Optional[Tuple[str, ...]] = None
    c_cap: Optional[Dict[str, float]] = None
    # hindsight targets
    hindsight_dims: Tuple[str, ...] = BIAS_DIMS
    hindsight_weight_scale: Union[float, Dict[str, float]] = 1.0
    temperature: float = DEFAULT_TEMPERATURE
    hindsight_linear_step_cap: float = 3.0
    hindsight_quadratic_step_cap: float = 12.0
    hindsight_skew_step_cap: float = 3.0
    hindsight_end_step_cap: float = 1.5
    end_reference_s: float = 1.5
    # endpoint_depth (hindsight.depth_config): s targets each substituent's
    # endpoint depth; false / None off, true defaults, or a mapping
    endpoint_depth: Any = None
    linear_step_max: Optional[float] = None    # None: hindsight_linear_step_cap
    quadratic_step_max: float = QUADRATIC_STEP_MAX
    barrier_trap_width: float = BARRIER_TRAP_WIDTH
    b_direction_requires_crossing: bool = True
    c_up_requires_b_bracket: bool = False
    # A stranded run's c step down also trains the shared model (True), or
    # only the pairs' own offsets (False): a group stranded by something other
    # than c otherwise lowers every system's c.
    stranded_c_shared: bool = True
    explore_factor_max: float = EXPLORE_FACTOR_MAX
    explore_after_failures: int = PERSISTENT_FAILURES
    # group runs and revisits (sub-blocks)
    group_holdoff: Optional[Dict[str, float]] = None
    revisit: Dict[str, Any] = field(default_factory=dict)
    regression: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_config(cls, config: Dict) -> 'BanditConfig':
        """Read ``config['bandit']`` (absent or null: all defaults)."""
        return cls.from_dict(config.get('bandit') or {})

    @classmethod
    def from_dict(cls, bandit: Dict) -> 'BanditConfig':
        names = {f.name for f in dataclasses.fields(cls)} - {'raw', 'algorithm'}
        # An explicit null keeps the default, except for keys where null means "off".
        kwargs = {k: bandit[k] for k in names
                  if k in bandit and (bandit[k] is not None or k in _NULLABLE)}
        for k in ('hindsight_dims', 'group_offset_dims'):
            if kwargs.get(k) is not None:
                kwargs[k] = tuple(kwargs[k])
        for k in ('revisit', 'regression'):
            kwargs[k] = dict(kwargs.get(k) or {})
        return cls(algorithm=bandit_algorithm(bandit), raw=bandit, **kwargs)

    @staticmethod
    def unknown_keys(bandit: Optional[Dict]) -> List[str]:
        """Keys of a bandit block that nothing reads (likely typos)."""
        known = {f.name for f in dataclasses.fields(BanditConfig)} | {'group_correction'}
        return sorted(k for k in (bandit or {}) if k not in known)

    # -- derived settings -------------------------------------------------
    @property
    def use_hindsight(self) -> bool:
        """Heads regress toward hindsight targets (``update_target: hindsight``)."""
        return self.update_target == 'hindsight'

    @property
    def shared_regression(self) -> bool:
        """Hindsight targets fit by gradient steps on the full network."""
        return self.use_hindsight and self.shared_update == 'regression'

    @property
    def validation_gate(self) -> bool:
        """Keep the shared model only while held-out pairs hold up."""
        return self.shared_update == 'regression' and self.regression.get('validation_gate', True)

    @property
    def selection_mode(self) -> str:
        return self.selection or ('uncertainty' if self.algorithm == 'hindsight' else 'random')

    @property
    def offset_dims(self) -> List[str]:
        """``hindsight_dims`` that have per-pair offsets, in config order."""
        return [d for d in self.hindsight_dims if d in OFFSET_DIMS]

    @property
    def group_dims(self) -> List[str]:
        """The offset dims group runs use (``group_offset_dims``, default all)."""
        if self.group_offset_dims is None:
            return self.offset_dims
        return [d for d in self.offset_dims if d in set(self.group_offset_dims)]

    def weight_scale(self, dims) -> Dict[str, float]:
        """Shared-model update strength per head (one number, or per head, others 1.0)."""
        ws = self.hindsight_weight_scale
        if isinstance(ws, dict):
            return {d: float(ws.get(d, 1.0)) for d in dims}
        return {d: float(ws) for d in dims}

    @property
    def step_caps(self) -> Dict[str, float]:
        """``hindsight_targets`` step caps (and the endpoint-depth target)."""
        from mllf.cb.hindsight import depth_config
        return dict(linear_step_cap=float(self.hindsight_linear_step_cap),
                    quadratic_step_cap=float(self.hindsight_quadratic_step_cap),
                    skew_step_cap=float(self.hindsight_skew_step_cap),
                    end_step_cap=float(self.hindsight_end_step_cap),
                    end_reference=float(self.end_reference_s),
                    endpoint_depth_cfg=depth_config(self.endpoint_depth))


# Keys whose explicit null means "off" rather than "use the default".
_NULLABLE = ('c_cap', 'group_holdoff', 'offset_prior_sd', 'offset_base_sd', 'selection',
             'candidate_pool_size', 'linear_step_max')
