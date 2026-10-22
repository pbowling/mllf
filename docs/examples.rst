Running Examples
================

This page runs the pipeline end to end: first on the smallest working system, then with
the staged setup used for production biases. Configuration keys are explained in
:doc:`workflow`.

Minimal Example: 14benz
-----------------------

``examples/14benz/`` holds a two-site benzene system (6 substituents at site 1, 5 at
site 2):

* ``prep/``: ``site{1,2}_sub*_pres.rtf`` and matching ``_frag.pdb`` files, with the shared
  support files (``core.pdb``, ``core.rtf``, ``toppar.str``, ``solvent.pdb``,
  ``cubic.xtl``, ``full_ligand.prm``)
* ``msld_flat.py``: the CHARMM/pyCHARMM simulation script
* ``archive_combos.sh`` / ``tar_triples.slurm``: archiving helpers

Set ``input_dir: examples/14benz``, not ``.../prep``. Fragment files are found in both
the input directory and its ``prep/``, but ``msld_flat.py`` and the support files are copied
from the input directory itself.

.. code-block:: bash

   cd mllf   # repository root
   python examples/run_workflow.py examples/workflow_14benz.yaml

``examples/workflow_14benz.yaml`` is kept small so that it runs quickly and without a
pretrained model:

* ``max_subs_per_site: 3`` and a single global-curriculum stage of single-site pairs,
  at most 20 per epoch, for 5 epochs
* REINFORCE from a random initialization (a commented ``pretrain:`` block shows how to
  warm-start)
* ``validation_fraction: 0.0``, since there are too few pairs to hold any out
* one SLURM job per combination per epoch; adjust the ``slurm:`` block for your cluster

Checkpoints go to ``examples/14benz/training_output/``. Rerunning the same command resumes
from the latest checkpoint.

Pretraining
-----------

Pretraining warm-starts the policy from completed simulations (:doc:`cb_pretraining`):

.. code-block:: bash

   # collect run directories into pretraining records
   python -m mllf.cli.collect_pretraining_data previous_runs/14benz_solv \
       --output-dir pretraining/14benz_solv

   # behavior cloning
   python -m mllf.cb.pretrain_policy --pretraining-dir pretraining/14benz_solv \
       --output-dir models/14benz_pretrain --config examples/workflow_pretrain.yaml \
       --epochs 50

Then set ``pretrain.model_path: models/14benz_pretrain/best_policy.pt`` in the workflow
config. ``examples/pretrain_wUnimol.sh`` is the SLURM script for pretraining on many
systems at once.

Staged Online Training
----------------------

The configs in ``training/online_*.yaml`` use the production setup: per-system stages over
ALF-derived combo sets, hindsight targets with per-pair offsets, and a pretrained Bayesian
model. The driver runs as one long SLURM job that submits the simulations:

.. code-block:: bash

   cd training
   sbatch --time=4-00:00:00 --export=ALL,CONFIG=online_A_well_solv_vac.yaml \
       start_test_training.sh
   tail -f checkpoints_online_A/epoch_summary.log   # one block per epoch

   # simulated time so far, against the iterative-flattening runs of the same systems
   python -m mllf.cb.cost_ledger checkpoints_online_A --pretraining-root ../pretraining

Each config's system list only illustrates the setup. Which systems go into a run should
follow from what is prepared and from how well the pretrained model predicts each one.
To test a trained model on systems it has not seen, use ``mode: deploy``
(:ref:`Testing on New Systems`).

Adding a System
---------------

1. Prepare ``siteN_subM_pres.rtf`` / ``_frag.pdb`` files, ``core.rtf`` / ``core.pdb`` and a
   working ``msld_flat.py``, typically with
   `msld-py-prep <https://github.com/Vilseck-Lab/msld-py-prep>`_ (:doc:`file_handling`).
2. Add an entry to ``systems:`` with its ``input_dir``, ``out_dir`` and ``solvent_state``.
3. Run the minimal configuration first. A single stage of single-site pairs tests the
   whole pipeline before you widen it.
