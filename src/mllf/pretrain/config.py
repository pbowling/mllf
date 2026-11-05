"""Pretraining settings: the command-line options (``PretrainOptions``) and
the ``unimol:`` block of the pretraining config (``UnimolSettings``)."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PretrainOptions:
    """How one pretraining run selects data, trains and what it writes.

    Run selection (applied in this order; see ``filters.select_runs``):
        use_best_only: keep only the best run per system.
        min_transitions: keep runs whose every site has >= N transitions at the
            highest lambda; takes precedence over the two below.
        stratified_negative_fraction: keep every positive-reward run and sample
            negative-reward buckets with a quadratic ramp up to this fraction;
            takes precedence over min_reward_threshold.
        min_reward_threshold: keep runs with reward >= this.
        filter_outliers / outlier_std_threshold: drop runs with any coefficient
            outside mean +- N std.

    Training:
        epochs, learning_rate (cosine annealed to lr/100), device (None: auto),
        patience: early stopping after this many epochs without a lower loss.
        reward_weighted: weight each run's loss by its (clamped) reward.
        awr_temperature: beta in exp(r / beta).
        use_per_pair_awr: per-edge, per-bias-type advantages from the online
            pair reward instead of one scalar weight per run.
        data_hygiene: a target counts only where the run measured what sets it.
        system_emphasis: {system-name substring: weight} up-weighting in the loss.
        init_checkpoint: start from this checkpoint's policy_state.
        resume_checkpoint: continue an interrupted run (same data and output dir).

    Bayesian heads (``bayesian_init``):
        bayesian_heads: also write best/final_policy_bayesian.pt.
        bayesian_prior_precision, exploration_fraction.
    """
    epochs: int = 20
    learning_rate: float = 1e-3
    device: Optional[str] = None
    use_best_only: bool = False
    filter_outliers: bool = True
    outlier_std_threshold: float = 3.0
    min_reward_threshold: Optional[float] = None
    min_transitions: Optional[int] = None
    stratified_negative_fraction: Optional[float] = None
    patience: int = 10
    reward_weighted: bool = False
    awr_temperature: float = 0.5
    use_per_pair_awr: bool = True
    bayesian_heads: bool = False
    bayesian_prior_precision: float = 1.0
    exploration_fraction: float = 0.3
    system_emphasis: Optional[Dict[str, float]] = None
    data_hygiene: bool = True
    init_checkpoint: Optional[Path] = None
    resume_checkpoint: Optional[Path] = None


@dataclass(frozen=True)
class UnimolSettings:
    """The config's ``unimol:`` block (the representation of each substituent).

    structure_selection: 'minimized' (default), 'unrelaxed' or 'random'; random
        picks a structure per run (``features._select_structure_type_for_run``)
        and records the choice in pretraining_representations_used.csv.
    env_input: 'dual' (mean full Uni-Mol embedding), 'phase' (2 phase flags) or
        'phase_ligmean' (mean ligand embedding + phase flags); see UnimolPolicy.
    ligand_other_sites: the ligand half is core + substituent + every other
        site's reference substituent (as each combo holds them fixed).
    site_size_input: the edge input also carries 1 / (n - 1) for the n
        substituents at the edge's site (policy.fill_site_size).
    ligand_from_minimized: runs given the minimized structure embed the ligand
        in its minimized.pdb geometry, not the unrelaxed fragment files.
    use_environment_consensus: filter each site's environment to the atoms
        shared by all its substituents (``environment_consensus``).
    cache_embeddings / cache_dir: persistent on-disk embedding cache (default
        <output_dir>/embeddings_cache).
    """
    structure_selection: str = 'minimized'
    environment_cutoff: float = 8.0
    use_environment_difference: bool = True
    env_input: str = 'dual'
    ligand_other_sites: bool = False
    site_size_input: bool = False
    ligand_from_minimized: bool = False
    use_environment_consensus: bool = True
    skip_unrelaxed_for_systems: List[str] = field(default_factory=list)
    cache_embeddings: bool = False
    cache_dir: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict) -> 'UnimolSettings':
        u = config.get('unimol', {}) or {}
        return cls(
            structure_selection=u.get('structure_selection', 'minimized').lower(),
            environment_cutoff=u.get('environment_cutoff', 8.0),
            use_environment_difference=u.get('use_environment_difference', True),
            env_input=u.get('env_input', 'dual'),
            ligand_other_sites=bool(u.get('ligand_other_sites', False)),
            site_size_input=bool(u.get('site_size_input', False)),
            ligand_from_minimized=bool(u.get('ligand_from_minimized', False)),
            use_environment_consensus=u.get('use_environment_consensus', True),
            skip_unrelaxed_for_systems=u.get('skip_unrelaxed_for_systems', []),
            cache_embeddings=bool(u.get('cache_embeddings', False)),
            cache_dir=u.get('cache_dir'),
        )

    @property
    def per_system(self) -> bool:
        """Embeddings are the same for every run of a system (built once)."""
        return self.structure_selection in ('minimized', 'unrelaxed')

    def embedding_kwargs(self) -> Dict:
        """Keyword arguments of ``compute_unimol_embeddings_and_edges``."""
        return dict(env_cutoff=self.environment_cutoff,
                    use_environment_difference=self.use_environment_difference,
                    env_input=self.env_input, ligand_other_sites=self.ligand_other_sites,
                    ligand_minimized=self.ligand_from_minimized,
                    site_size_input=self.site_size_input)

    def epoch_config(self) -> Dict:
        """The ``unimol_config`` dict ``pretrain_epoch`` takes."""
        return {'environment_cutoff': self.environment_cutoff,
                'use_environment_difference': self.use_environment_difference,
                'env_input': self.env_input, 'ligand_other_sites': self.ligand_other_sites,
                'site_size_input': self.site_size_input}
