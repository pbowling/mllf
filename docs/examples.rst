Running Examples
================

Overview
--------

The repository includes a complete contextual bandit training workflow for
multisite λ-dynamics simulations. The main example demonstrates:

1. Generating combinations from site/substituent fragments
2. Training a graph neural network policy to predict bias coefficients
3. Running CHARMM simulations with predicted biases
4. Computing rewards from simulation metrics
5. Updating the policy with REINFORCE

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

- Generate combinations from the 14benz_solv_5.5 system
- Split into train/val/test sets (70/15/15)
- Train for 50 epochs with SLURM job submission
- Save checkpoints every 5 epochs
- Write outputs to ``training_output/``

Configuration
~~~~~~~~~~~~~

Edit ``examples/workflow_sample.yaml`` to customize:

.. code-block:: yaml

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
   
   # Model architecture
   training:
     num_epochs: 50
     encoder:
       hidden_dims: [64, 64]
       out_dim: 32
     policy:
       mlp_hidden: 64
     optimizer:
       lr: 0.001
   
   # Checkpointing
   output:
     base_dir: /path/to/training_output
     save_checkpoints: true
     checkpoint_freq: 5
   
   # Archive combinations after training (optional)
   archive:
     enabled: true
     remove_after: false  # Keep originals for verification

Example System: 14benz_solv_5.5
--------------------------------

The ``examples/cb/14benz_solv_5.5/`` directory contains:

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

   cd examples
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

2. **Update config**: Edit ``workflow_sample.yaml`` with your paths

3. **Modify simulation script**: Adapt ``msld_flat.py`` for your system
   (force field, topology, equilibration protocol)

4. **Adjust reward function**: Edit ``src/mllf/cb/train.py::compute_msld_reward``
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

   # Test different reward configurations on epoch 5 data
   cd examples
   python test_reward_configs.py training_output/epoch_005 \\
       --configs reward_configs_example.yaml

This compares multiple reward configurations and identifies which yields
the highest mean reward. You can then update ``workflow_sample.yaml``
and resume training with the best configuration—cached results will be
automatically recomputed.

**Example workflow**:

1. Run 5 epochs with baseline configuration
2. Test various reward configs on epoch 5 data
3. Identify best configuration
4. Update ``workflow_sample.yaml`` with best config
5. Resume training—rewards recompute automatically
6. Continue for remaining 45 epochs with optimized reward

See Also
--------

* :doc:`workflow` - Complete workflow system documentation
* :doc:`cb_setup` - Contextual bandit architecture details
* :doc:`api` - API reference for workflow modules
