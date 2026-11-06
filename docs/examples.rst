Running Examples
================

This page runs the pipeline end to end: first on a single small system, then with the
staged setup used for production biases. Configuration keys are explained in
:doc:`workflow`.

Minimal Example: 14benz
-----------------------

``examples/minimal/`` runs the whole pipeline on one system, 1,4-benzene in water
(``examples/14benz/``: 6 substituents at site 1, 5 at site 2), and is small enough to check
an installation end to end. From the repository root:

.. code-block:: bash

   sbatch examples/minimal/run_pipeline.sh
   # or with another install and dataset location:
   sbatch --export=ALL,PRETRAINING_ROOT=/path/to/pretraining,MLLF=/path/to/mllf \
       examples/minimal/run_pipeline.sh

The driver runs on a CPU node and submits each simulation as its own SLURM job (the
``slurm:`` block of ``workflow.yaml``; adjust it for your cluster). Its steps are:

1. **Pretraining** (``mllf pretrain``): 10 epochs with Bayesian heads on 39 collected
   ``14benz_solv`` runs (``pretraining_combos.txt``: the 31 one-site pair and triplet
   combos and 8 two-site combos) under ``$PRETRAINING_ROOT/14benz_solv``. Pretraining also
   writes ``pretraining_manifest.json``, the list of runs the model saw.
2. **Replay** (``mllf replay``): 20 of those runs, prebuilt into ``output/replay.pt`` for
   the regression update (``bandit.regression``).
3. **Scouting** (``mllf scout``): one run of the system's group (site 1 substituents 1-3,
   site 2 substituents 1-2) with the new model's biases, then ``scout_summary.json`` and
   ``pair_priority.json`` (:ref:`Scouting`).
4. **Online training** (``mllf train``): one epoch of the ``pairs_groups`` stage. The
   training pairs cover site 1 substituents 1-4 and site 2 substituents 1-3, so 5 of the
   9 pairs lie outside the scouted group and run even when scouting passes the group;
   ``advance.min_epochs: 1`` also keeps the system in the stage for that epoch. The
   system's export and the archive of its combo directories follow the epoch.
5. **Cost** (``mllf cost``): simulated time of the online run, scouting included.

Every unimol setting is left at its default, so the online config matches the model it
pretrains (:doc:`unimol_representation`). The simulations are shortened to 13,000
equilibration and 39,000 production steps (``run_steps``), so the example tests the
pipeline, not the quality of the biases. Outputs go to ``examples/minimal/output/``, with
a log file per step (``pretrain/pretrain.log``, ``replay.log``, ``scouting/scouting.log``,
``training/training.log``). A rerun skips the steps whose outputs exist and resumes
training from its checkpoint, so delete that directory to start over.

Set ``input_dir: examples/14benz``, not ``.../prep``, for this system. Fragment files are
found in both the input directory and its ``prep/``, but ``msld_flat.py`` and the support
files are copied from the input directory itself.

Pretraining
-----------

Pretraining warm-starts the policy from completed simulations (:doc:`cb_pretraining`):

.. code-block:: bash

   # collect run directories into pretraining records
   mllf collect previous_runs/14benz_solv \
       --output-dir pretraining/14benz_solv

   # behavior cloning
   mllf pretrain --pretraining-dir pretraining/14benz_solv \
       --output-dir models/14benz_pretrain --config examples/workflow_pretrain.yaml \
       --epochs 50

Then set ``pretrain.model_path: models/14benz_pretrain/best_policy.pt`` in the workflow
config (``best_policy_bayesian.pt``, written with ``--bayesian-heads``, for
``algorithm: hindsight``). ``examples/pretrain_with_filtering.sh`` is the SLURM launcher
for pretraining on many systems at once.

Staged Online Training
----------------------

The configs ``training/online_1.yaml`` through ``online_4_hne_pr3.yaml`` use the
production setup, with per-system stages over ALF-derived combo sets, hindsight targets
with per-pair offsets, and a pretrained Bayesian model. The driver runs as one long SLURM
job that submits the simulations:

.. code-block:: bash

   cd training
   sbatch --time=4-00:00:00 --export=ALL,CONFIG=online_1.yaml start_test_training.sh
   tail -f checkpoints_online_1/epoch_summary.log   # one block per epoch

   # simulated time so far, against the iterative-flattening runs of the same systems
   mllf cost checkpoints_online_1 --pretraining-root ../pretraining

Each stage starts from the previous stage's final model, and stages 3 and 4 test the
trained model on systems it has not seen (``mode: deploy``, :ref:`Testing on New
Systems`). The system lists illustrate the setup; which systems enter a run should follow
from what is prepared and from how well the pretrained model predicts each one.

Adding a System
---------------

1. Prepare ``siteN_subM_pres.rtf`` / ``_frag.pdb`` files, ``core.rtf`` / ``core.pdb`` and a
   working ``msld_flat.py``, typically with
   `msld-py-prep <https://github.com/Vilseck-Lab/msld-py-prep>`_ (:doc:`file_handling`).
2. Add an entry to ``systems:`` with its ``input_dir``, ``out_dir`` and ``solvent_state``.
3. Run the minimal configuration first. A single stage of single-site pairs tests the
   whole pipeline before it is widened.
