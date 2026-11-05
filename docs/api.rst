mllf API
=========

File Handling
-------------

For file formats and usage, see :doc:`file_handling`.

.. autosummary::
   :toctree: api

   mllf.file_handling.read_bias_coeff
   mllf.file_handling.read_rtf
   mllf.file_handling.read_pdb
   mllf.file_handling.write_bias_coeff
   mllf.file_handling.read_output
   mllf.file_handling.generate_combinations

Uni-Mol Representation
-----------------------

For Uni-Mol embeddings and environment context, see :doc:`unimol_representation`.

.. autosummary::
   :toctree: api

   mllf.cb.unimol_representation
   mllf.cb.environment_consensus

Policy Modules
--------------

For the policy architecture, see :doc:`cb_setup`.

.. autosummary::
   :toctree: api

   mllf.cb.pairs
   mllf.cb.policy
   mllf.cb.bayesian_head
   mllf.cb.msld_reward
   mllf.cb.pair_weights

Pretraining (Behavior Cloning)
-------------------------------

For behavior-cloning pretraining, see :doc:`cb_pretraining`.

.. autosummary::
   :toctree: api

   mllf.pretrain.config
   mllf.pretrain.runs
   mllf.pretrain.filters
   mllf.pretrain.features
   mllf.pretrain.rewards
   mllf.pretrain.trainer
   mllf.pretrain.bayesian_init

Workflow Utilities
------------------

For the workflow system, see :doc:`workflow`.

.. autosummary::
   :toctree: api

   mllf.cb.training_loop
   mllf.cb.training_setup
   mllf.cb.bandit_config
   mllf.cb.reinforce_trainer
   mllf.cb.epoch_runs
   mllf.cb.epoch_updates
   mllf.cb.workflow_utils
   mllf.cb.graph_build
   mllf.cb.hindsight
   mllf.cb.hindsight_regression
   mllf.cb.replay_selection
   mllf.cb.group_correction
   mllf.cb.curriculum
   mllf.cb.system_stages
   mllf.cb.export_biases
   mllf.cb.cost_ledger
   mllf.cb.epoch_log
   mllf.cb.slurm
   mllf.file_handling.manifests
   mllf.file_handling.combo_sets

Command Line and Logging
------------------------

For the ``mllf`` command, see :doc:`cli`.

.. autosummary::
   :toctree: api

   mllf.cli.main
   mllf.cli.common
   mllf.cli.combos
   mllf.cli.collect_pretraining_data
   mllf.cli.pretrain
   mllf.cli.replay
   mllf.cli.train
   mllf.cli.export
   mllf.cli.cost
   mllf.log
   mllf.slurm
