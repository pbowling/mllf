Running Examples
================

Overview
--------

The repository includes a complete contextual bandit training workflow for
multisite λ-dynamics simulations. The main example demonstrates:

1. Generating combinations from site/substituent fragments
2. Training an Actor-Critic model (policy + value network) to predict bias coefficients
3. Running CHARMM simulations with predicted biases
4. Computing rewards from simulation metrics
5. Updating policy with REINFORCE and value network for variance reduction

Main Training Workflow
----------------------

The primary example is ``examples/run_workflow.py``, which implements the
complete training pipeline.

Quick Start
~~~~~~~~~~~

.. code-block:: bash

   cd examples
   python run_workflow.py workflow_sample.yaml

This will:

- Generate combinations from the 14benz system
- Split into train/val/test sets (70/15/15)
- Train for 50 epochs with SLURM job submission
- Save checkpoints every 5 epochs
- Write outputs to ``training_output/``

Configuration
~~~~~~~~~~~~~

Edit ``examples/workflow_sample.yaml`` to customize:

.. code-block:: yaml

   # System configuration
   system:
     solvent_state: solv  # Environment: 'solv' (solvated), 'gas' (vacuum), or 'protein'
   
   # Combination generation
   create_combos:
     input_dir: /path/to/site_sub_fragments
     out_dir: /path/to/generated_combos
     include_patterns:
       - msld_flat.py  # Copy simulation scripts
   
   # Training/validation split
   split:
     train_frac: 0.70
     val_frac: 0.15
     seed: 42
   
   # Model architecture (Actor-Critic with value network)
   training:
     num_epochs: 50
     encoder:
       hidden_dims: [64, 64]
       out_dim: 32
     policy:
       mlp_hidden: 64
     value_network:
       hidden_dims: [64, 32]  # Value network for variance reduction
       lr: 0.001              # 10x policy LR (learns scalar prediction faster)
     optimizer:
       lr: 0.0001             # Policy LR (reduced 10x for stability)
   
   # Reward configuration
   reward:
     lambda_entropy: 0.5      # Entropy regularization (increased 50x for exploration)
   
   # Checkpointing
   output:
     base_dir: /path/to/training_output
     save_checkpoints: true
     checkpoint_freq: 5
   
   # Optional: Pretrain from existing simulations
   pretrain:
     data_dir: /path/to/pretraining_data
     num_epochs: 1
     model_path: models/pretrained_policy.pt
   
   # Load pretrained model before training
   training:
     load_pretrained: models/pretrained_policy.pt
   
   # Archive combinations after training (optional)
   archive:
     enabled: true
     remove_after: false  # Keep originals for verification

Pretraining Example
-------------------

Before running main training, you can pretrain the policy on existing
simulation data to warm-start the model with meaningful bias coefficients.

Setup Pretraining Data
~~~~~~~~~~~~~~~~~~~~~~~

Collect completed simulation runs into a pretraining directory:

.. code-block:: bash

   mkdir -p pretraining
   
   # Copy runs from various sources
   cp -r previous_training/epoch_*/comb_* pretraining/
   cp -r manual_tuning/good_runs/* pretraining/
   cp -r systematic_sweep/successful_configs/* pretraining/

Each run directory should contain:

- ``variables.py``: Bias coefficients used in the simulation
- ``info.py``: System configuration (nsubs, nblocks, temp)
- ``res/*_flat.lmd``: Lambda dynamics trajectory for reward computation

Run Pretraining
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run pretraining (1 epoch is usually sufficient)
   python run_pretraining.py --config workflow_sample.yaml
   
   # This saves: models/pretrained_policy.pt

Then update ``workflow_sample.yaml`` to use the pretrained model:

.. code-block:: yaml

   training:
     load_pretrained: models/pretrained_policy.pt
     num_epochs: 50  # Main training epochs

Finally run main training:

.. code-block:: bash

   python run_workflow.py workflow_sample.yaml

Benefits
~~~~~~~~

- **Faster convergence**: Start with reasonable bias values
- **Better sample efficiency**: Fewer training epochs needed
- **Transfer learning**: Leverage knowledge from related systems
- **Multi-system learning**: Combine data from different ligands/environments

The pretrained policy learns to map graph structure (system size, atom types,
environment) to successful bias coefficients, providing a strong initialization
for the main training phase.

.. note::
   The value network does not require pretraining. It starts from random initialization
   and learns to predict rewards during reinforcement learning training by observing
   actual simulation outcomes.

Example System: 14benz
----------------------

The ``examples/14benz/`` directory contains:

- ``site1_sub1_pres.rtf`` through ``site1_sub5_pres.rtf``: Site 1 substituents
- ``site2_sub1_pres.rtf`` through ``site2_sub6_pres.rtf``: Site 2 substituents
- ``msld_flat.py``: CHARMM/pyCHARMM simulation script
- ``prep/``: Pre-equilibrated structures

With 5 substituents at site 1 and 6 at site 2, the rotating anchor strategy
generates:

- 75 within-site combinations for site 1 (5 anchors × 15 each)
- 186 within-site combinations for site 2 (6 anchors × 31 each)
- 13,950 cross-site combinations (75 × 186)
- **Total: 14,211 unique combinations**

SLURM Job Submission
--------------------

For production training on a cluster:

.. code-block:: bash

   cd examples/
   sbatch training_test.sh

The training script:

- Activates the conda environment
- Runs ``python -u run_workflow.py workflow_sample.yaml``
- Submits MSLD simulations as separate SLURM jobs
- Manages up to 30 concurrent simulation jobs
- Writes progress to ``training_status.out``

The ``-u`` flag enables unbuffered output for real-time monitoring:

.. code-block:: bash

   tail -f training_status.out

Resume Capability
-----------------

If training is interrupted, simply resubmit the job:

.. code-block:: bash

   sbatch training_test.sh

The workflow automatically:

- Detects the latest checkpoint (``checkpoint_epoch_XXX.pt``)
- Loads model and optimizer state
- Resumes from the next epoch
- Skips combinations with existing ``epoch_results.pt`` files

See :doc:`workflow` for details on checkpoint/resume functionality.

Customizing the Workflow
------------------------

To adapt the workflow for your system:

1. **Prepare fragments**: Create ``siteN_subM_pres.rtf`` files for each
   site/substituent combination

2. **Update config**: Edit ``workflow_sample.yaml`` with your paths and system settings:
   
   - Set ``system.solvent_state`` to match your environment ('solv', 'gas', or 'protein')
   - Adjust ``value_network.hidden_dims`` if needed (default [64, 32] works well)
   - Tune ``reward.lambda_entropy`` for exploration (0.5 is recommended)

3. **Modify simulation script**: Adapt ``msld_flat.py`` for your system
   (force field, topology, equilibration protocol)

4. **Adjust reward function**: Edit ``src/mllf/cb/train_improved.py::compute_msld_reward_improved``
   to weight different simulation metrics

5. **Run training**: Execute ``python run_workflow.py your_config.yaml``

Tips
----

- Start with fewer epochs (e.g., 5) to test the full pipeline
- Use ``max_concurrent_jobs`` to control cluster load
- Monitor ``training_status.out`` for real-time progress
- Check ``training_output/epoch_NNN/`` for per-epoch results
- Verify checkpoint files are saved at ``checkpoint_freq`` intervals

Reward Function Tuning
-----------------------

Epoch checkpoints save raw simulation metrics, enabling reward function
experimentation without re-running simulations:

.. code-block:: bash

   # Test different reward configurations on pretraining data
   cd tests/tools
   python test_reward_improved.py ../../pretraining/indolizine_solv \
       --good-threshold 215 \
       --bad-threshold 15 \
       --configs reward_configs_improved.yaml

This compares multiple reward configurations and identifies which yields
the best separation between good and bad runs. The improved reward function
includes:

* **Confidence Factor**: Scales population rewards by data reliability
* **Tiered Penalties**: Continuous gradient feedback instead of binary thresholds

You can then update ``workflow_sample.yaml`` with the best configuration
(e.g., ``higher_rewards_v1``) and resume training—cached results will be
automatically recomputed with the new reward parameters.

See Also
--------

* :doc:`workflow` - Complete workflow system documentation
* :doc:`cb_setup` - Contextual bandit architecture details
* :doc:`api` - API reference for workflow modules
