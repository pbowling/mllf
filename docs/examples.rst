Running Examples
================

Overview
--------

The repository includes a complete training workflow for multisite λ-dynamics 
bias coefficient optimization. The main example demonstrates the end-to-end pipeline 
from combination generation through training and simulation execution.

For architectural details, see :doc:`cb_setup`. For workflow configuration options, 
see :doc:`workflow`.

Main Training Workflow
----------------------

The primary example is ``examples/run_workflow_deepset.py``, which implements the
complete training pipeline using pretrained DeepSet embeddings.

Quick Start
~~~~~~~~~~~

.. code-block:: bash

   cd examples
   python run_workflow_deepset.py workflow_14benz.yaml

- Generate combinations from the 14benz system
- Split into train/val/test sets (70/15/15)
- Train for 50 epochs with SLURM job submission
- Save checkpoints every 5 epochs
- Write outputs to ``training_output/``

Configuration
~~~~~~~~~~~~~

Edit ``examples/workflow_14benz.yaml`` to customize paths and basic settings:

.. code-block:: yaml

   system:
     solvent_state: solv
   
   create_combos:
     input_dir: /path/to/site_sub_fragments
     out_dir: /path/to/generated_combos
   
   split:
     train_frac: 0.70
     val_frac: 0.15
   
   training:
     num_epochs: 50
   
   output:
     base_dir: /path/to/training_output
     save_checkpoints: true
     checkpoint_freq: 5

See :doc:`workflow` for complete configuration options including curriculum learning, 
pretraining, archiving, and reward function tuning.

Pretraining Example
-------------------

To pretrain from existing simulations:

.. code-block:: bash

   # Organize existing simulation data
   mkdir -p pretraining
   cp -r previous_runs/good_combos/* pretraining/
   
   # Run pretraining
   python run_deepset_pretraining.py --pretraining-dir pretraining --output-dir models --steps all
   
   # Use pretrained model in main training
   python run_workflow_deepset.py workflow_14benz.yaml

See :doc:`workflow` for detailed pretraining configuration, data organization, 
and benefits.

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
- Runs ``python -u run_workflow_deepset.py workflow_14benz.yaml``
- Submits MSLD simulations as separate SLURM jobs
- Manages up to 30 concurrent simulation jobs
- Writes progress to ``training_status.out``

The ``-u`` flag enables unbuffered output for real-time monitoring:

.. code-block:: bash

   tail -f training_status.out

Resume Capability
-----------------

Training automatically resumes from the latest checkpoint if interrupted:

.. code-block:: bash

   sbatch training_test.sh  # Automatically detects and resumes

See :doc:`workflow` for checkpoint configuration and resume details.

Customizing the Workflow
------------------------

To adapt for your system:

1. **Prepare fragments**: Create ``siteN_subM_pres.rtf`` files for your sites/substituents
2. **Update config**: Edit ``workflow_14benz.yaml`` with your paths and ``system.solvent_state``
   (see ``workflow_deepset.yaml`` for an alternate template)
3. **Modify simulation script**: Adapt ``msld_flat.py`` for your force field and protocol
4. **Run training**: Execute ``python run_workflow_deepset.py your_config.yaml``

See :doc:`workflow` for configuration options and :doc:`cb_setup` for architecture details.

Tips
----

- Start with fewer epochs (e.g., 5) to test the full pipeline
- Use ``max_concurrent_jobs`` to control cluster load
- Monitor ``training_status.out`` for real-time progress
- Check ``training_output/epoch_NNN/`` for per-epoch results
- Verify checkpoint files are saved at ``checkpoint_freq`` intervals

Reward Function Tuning
-----------------------

Epoch checkpoints enable testing different reward configurations without re-running simulations:

.. code-block:: bash

   python examples/test_reward_configs.py training_output/epoch_005 \
       --configs reward_configs_example.yaml

See :doc:`workflow` for reward function details and experimentation workflow.

See Also
--------

* :doc:`file_handling` - File format documentation and parsers
* :doc:`workflow` - Complete workflow system documentation
* :doc:`cb_setup` - Contextual bandit architecture details
* :doc:`deepset_pretraining` - DeepSet pretraining for node embeddings
* :doc:`cb_pretraining` - Behavior cloning from expert coefficients
* :doc:`api` - API reference for workflow modules
