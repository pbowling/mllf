Running Examples
================

Overview
--------

The repository includes a complete training workflow for multisite λ-dynamics
bias coefficient optimization. This page walks through the end-to-end pipeline —
combination generation, behavior-cloning pretraining, and online REINFORCE training —
using the 1,4-benzene (``examples/14benz``) system as a concrete, runnable example.

For architectural details, see :doc:`cb_setup`. For the full workflow configuration
reference, see :doc:`workflow`.

Example System: 14benz
-----------------------

``examples/14benz/`` contains:

* ``prep/``: ``site1_sub1_pres.rtf`` .. ``site1_sub6_pres.rtf`` (site 1, 6 substituents),
  ``site2_sub1_pres.rtf`` .. ``site2_sub5_pres.rtf`` (site 2, 5 substituents), each with a
  matching ``_frag.pdb``, plus shared support files (``core.pdb``, ``core.rtf``,
  ``toppar.str``, ``solvent.pdb``, ``cubic.xtl``, ``full_ligand.prm``)
* ``msld_flat.py``: CHARMM/pyCHARMM simulation script
* ``archive_combos.sh`` / ``tar_triples.slurm``: helper scripts for archiving completed runs

Combination generation reads site/sub files from **both** ``examples/14benz`` and
``examples/14benz/prep`` (``find_site_sub_files`` searches an ``input_dir`` and its
``prep/`` subdirectory), while support files and ``msld_flat.py`` are copied from
``examples/14benz`` itself — so ``input_dir: examples/14benz`` is correct, not
``examples/14benz/prep``. With 6 substituents at site 1 and 5 at site 2, the default
generator (``max_subs_per_site=10``) would produce thousands of within- and cross-site
combinations; the example config below caps this at ``max_subs_per_site: 3`` and further
restricts training to single-site pairs via curriculum learning so the example stays fast.

Quick Start: Online Training
------------------------------

The primary example is ``examples/run_workflow.py`` (a tracked copy of the multi-system
UniMol CB training driver) paired with ``examples/workflow_14benz.yaml``:

.. code-block:: bash

   cd mllf   # repository root
   python examples/run_workflow.py examples/workflow_14benz.yaml

This will:

1. Generate 14benz combinations (capped at 3 substituents/site) into
   ``examples/14benz/generated_combos/``
2. Restrict training, via a one-stage curriculum, to single-site pairs only
   (``min_sites: 1, max_sites: 1, min/max_subs_per_site: 2``), capped at 20 combos/epoch
3. Run 5 epochs of on-policy REINFORCE, submitting each combo's ``msld_flat.py`` simulation
   as a SLURM job (adjust the ``slurm:`` block in the config for your cluster)
4. Write checkpoints to ``examples/14benz/training_output/``

``examples/workflow_14benz.yaml`` trains ``UnimolPolicy`` from a random initialization by
default (no ``pretrain:`` block) so it has no dependency on any pretrained checkpoint. To
warm-start instead, add a ``pretrain.model_path`` pointing at a checkpoint produced by the
pretraining pipeline below (see :doc:`cb_pretraining`) — the config file has a commented-out
example.

Configuration
~~~~~~~~~~~~~

``examples/workflow_14benz.yaml`` is heavily commented; edit it directly to adjust
``systems[0].input_dir``/``out_dir`` for your own copy of 14benz, curriculum stages, reward
weights, or SLURM settings. See :doc:`workflow` for the complete configuration reference,
including multi-system training, NeuralLinear + Thompson Sampling
(``bandit.algorithm: neurallinear_ts``), and per-stage archiving.

Pretraining Example
--------------------

Behavior-cloning pretraining warm-starts the policy from existing simulation data before
online training. See :doc:`cb_pretraining` for the full loss function, filtering options,
and per-pair AWR weighting details.

.. code-block:: bash

   # Collect existing MSLD run directories (run1/, run2/, ...) into pretraining-ready records
   python -m mllf.cli.collect_pretraining_data \
       previous_runs/14benz_solv \
       --output-dir pretraining/14benz_solv

   # Pretrain UnimolPolicy on the collected data
   python -m mllf.cb.pretrain_policy \
       --pretraining-dir pretraining/14benz_solv \
       --output-dir models/14benz_pretrain \
       --config examples/workflow_pretrain.yaml \
       --epochs 50

   # Point the online workflow at the resulting checkpoint
   #   pretrain:
   #     model_path: models/14benz_pretrain/best_policy.pt
   python examples/run_workflow.py examples/workflow_14benz.yaml

``examples/pretrain_wUnimol.sh`` (sourcing ``examples/pretrain_with_filtering.sh``) is a
complete SLURM submission example for pretraining across multiple systems at once, with all
the filtering/AWR flags set via environment variables.

SLURM Job Submission
---------------------

Online training itself submits one SLURM job per combination per epoch (via the ``slurm:``
config block — partition, gres, time, module). To run the driver script itself under SLURM
as a long-lived job (recommended for anything beyond a handful of epochs), wrap it in a
submission script, e.g.:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=14benz_train
   #SBATCH --output=training_status.out
   #SBATCH --cpus-per-task=4
   #SBATCH --time=24:00:00

   python -u examples/run_workflow.py examples/workflow_14benz.yaml

The ``-u`` flag enables unbuffered output for real-time monitoring:

.. code-block:: bash

   tail -f training_status.out
   # or, for a smaller, epoch-summary-only view:
   tail -f examples/14benz/training_output/epoch_summary.log

Resume Capability
------------------

Training automatically resumes from the latest ``checkpoint_*.pt`` in ``output.base_dir``
if interrupted — just re-run the same command:

.. code-block:: bash

   python examples/run_workflow.py examples/workflow_14benz.yaml  # detects and resumes

See :doc:`workflow` for checkpoint contents and resume details.

Customizing the Workflow
--------------------------

To adapt for your own system:

1. **Prepare fragments**: create ``siteN_subM_pres.rtf``/``_frag.pdb`` files for your
   sites/substituents (typically via `msld-py-prep
   <https://github.com/Vilseck-Lab/msld-py-prep>`_), alongside ``core.rtf``/``core.pdb`` and
   a working ``msld_flat.py``
2. **Add a systems entry**: copy the single ``systems:`` list entry in
   ``examples/workflow_14benz.yaml``, pointing ``input_dir``/``out_dir`` at your system and
   setting the correct ``solvent_state``
3. **Adjust curriculum/reward**: widen the curriculum stages (or disable curriculum
   entirely and set ``training.num_epochs``) once the minimal example runs cleanly
4. **Run training**: ``python examples/run_workflow.py your_config.yaml``

See :doc:`workflow` for configuration options and :doc:`cb_setup` for architecture details.

Tips
----

* Start with a restrictive curriculum stage (single-site pairs, as in
  ``workflow_14benz.yaml``) to test the full pipeline before widening it
* Use ``max_concurrent_jobs`` to control cluster load
* Monitor ``epoch_summary.log`` in ``output.base_dir`` for a compact per-epoch view, or the
  full SLURM/stdout log for everything
* Verify checkpoint files are saved at ``checkpoint_freq`` intervals

See Also
--------

* :doc:`file_handling` - File format documentation and parsers
* :doc:`workflow` - Complete workflow system documentation
* :doc:`cb_setup` - Contextual bandit architecture details
* :doc:`unimol_representation` - Uni-Mol embeddings for node features
* :doc:`cb_pretraining` - Behavior cloning from expert coefficients
* :doc:`api` - API reference for workflow modules
