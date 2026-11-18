"""Compatibility alias: pretraining now lives in ``mllf.pretrain`` and its
command line in ``mllf.cli.pretrain``. ``python -m mllf.cb.pretrain_policy``
still runs it."""
from mllf.cli.pretrain import main
from mllf.pretrain.bayesian_init import (
    estimate_reward_noise_variance,
    initialize_bayesian_heads_from_pretrained,
)
from mllf.pretrain.features import (
    compute_unimol_embeddings_and_edges,
    extract_targets_from_variables,
)
from mllf.pretrain.filters import filter_best_runs_per_system
from mllf.pretrain.rewards import compute_reward_from_sim_results
from mllf.pretrain.runs import load_pretraining_runs
from mllf.pretrain.trainer import pretrain_with_runs


__all__ = ['compute_reward_from_sim_results', 'compute_unimol_embeddings_and_edges',
           'estimate_reward_noise_variance', 'extract_targets_from_variables',
           'filter_best_runs_per_system', 'initialize_bayesian_heads_from_pretrained',
           'load_pretraining_runs', 'main', 'pretrain_with_runs']

if __name__ == "__main__":
    main()
