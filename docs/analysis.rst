Analysis
========

``mllf analyze`` draws the figures and writes the report of a pretrained model or an
online-training run. Each analysis writes one directory:

* ``figures/``: each figure as a PNG (``--format pdf`` for a vector PDF with editable
  text; ``--grayscale`` adds a grayscale PNG, to check print legibility), in a single
  house style (``mllf.analysis.style``) at ACS column widths
* ``tables/``: the data behind each figure and table as CSV
* ``report.txt``: the statistics of the run in plain text
* ``analysis.log``: every message, and ``cache/``, which makes a rerun fast

The figures need matplotlib, pandas and pillow:

.. code-block:: bash

   pip install "mllf[analysis]"

``--only FIGURE ...`` draws only the named figures (the report and tables are always
written), and ``--rescan`` ignores the cache.

Pretrained Model
----------------

.. code-block:: bash

   mllf analyze pretrain models/pretraining_Nov02b      # writes models/.../analysis

The model predicts the runs of its own ``pretraining_manifest.json``, embedded exactly
as pretraining embedded them (the unimol settings of its ``pretrain_metadata.json``). A
run's coefficient is scored only where the run's biases were good for it: the pair passed
online training's success test for a run of its length and, for c, x and s, the run
measured that coefficient's signal. Each dataset contributes the
``--max-runs-per-dataset`` runs (default 10) with the most such targets, of 50 of its
runs read at random (0 reads and analyzes every run, which takes hours for a full
pretraining set). ``--config``
names the pretraining config when the model directory holds none; its
``unimol.cache_dir`` reuses pretraining's embeddings.

``parity_all``
   Predicted against actual b, c, x and s for every scored target, with the systems of
   the largest errors labelled; the gray band is ±1 (kcal/mol for b).
``parity_systems/<system>``
   The same for each system.
``embedding_pca``
   The first principal components of the substituents' Uni-Mol ligand embeddings, the
   five most common families highlighted, and the variance each component explains.
``similarity``
   Distance between systems in embedding space: for each substituent of one system, the
   distance to the nearest substituent of the other (each dimension standardized),
   averaged and symmetrized.
``dataset_stats``
   Runs per dataset, substituents and sites per run, and the distribution of each scored
   coefficient.

The report gives N, R², Pearson r, MUE, RMSE and the median posterior predictive sd per
coefficient, the same per family (``tables/metrics_by_system.csv`` per system), and each
system's three closest systems of other families. ``tables/predictions.csv`` holds every
directed pair and coefficient of every analyzed run.

Online Training
---------------

.. code-block:: bash

   mllf analyze online online_1.yaml                    # writes <output.base_dir>/analysis
   mllf analyze online online_1.yaml --pretraining-root ../pretraining

The run is read from what training wrote to ``output.base_dir`` (``epoch_summary.log``,
``cost_ledger.json``, ``system_stages.json``, ``validation_combos.json`` and each run's
``epoch_results.pt``); ``--run-dir`` points at another directory. ALF's runs of the same
systems come from the pretraining datasets (``--pretraining-root``, by default the
config's ``alf_root``; ``none`` leaves ALF out). Reading them takes minutes, so they are
cached once per pretraining root in ``~/.cache/mllf`` and shared by every analysis.

Runs are judged by online training's tests. A pair run succeeds with both substituents
sampled, a minority fraction of at least 0.2 and at least 10 crossings, or the stage's
short-run test (``explore_min_transitions`` / ``explore_min_fraction``) for runs under
0.5 ns. A group run passes the stage's group test (``advance.min_population_ratio`` /
``min_site_transitions``).

``success_over_epochs``
   Percent of pair runs, group runs and held-out pairs that passed their test per epoch
   (``--no-validation`` leaves out the held-out pairs).
``outcomes_by_epoch`` / ``outcomes_by_system``
   Training runs by outcome, as counts: success, unbalanced, few crossings, one-sided or
   neither sampled for pairs; passed, uneven, few transitions or an unsampled substituent
   for groups.
``stage_timeline``
   The epochs each system spent in each stage: solid where it met the stage's criterion,
   lighter and hatched where it did not (stopped at the epoch limit, or still running), a
   diamond for a system that converged on its scouting runs, and its groups passed.
``convergence_systems`` / ``convergence_groups`` (pages of 12 systems)
   Simulated time to convergence for MLLF and ALF, per system and per group. Solid bars
   converged (every group passed the group test); hatched bars never did and show every
   ns that method ran. MLLF's time includes scouting and leaves out validation; a group's
   time is its own runs plus its share of its pair runs. ALF's groups are compared one to
   one only when the stage's ``group_set`` is ALF's groups (``alf_groups``).
``free_energy/<system>_<group>_site<N>``
   With ``--free-energy [SYSTEM ...]`` only, since it reads every run's trajectory
   (minutes per system): free-energy profiles of each passed group's run from its λ
   trajectory, next to ALF's best run of the same group (its full-length run with the most
   pairs passing), with G(λ) for each substituent and the profile along each pair's edge.

The report gives the systems and groups converged by each method, the simulated time to
convergence and in all, run outcomes, and the last epoch's success.

Running the Analysis Automatically
----------------------------------

``analysis: true`` in a workflow config analyzes the run when training finishes, before
any archiving (``archive.remove`` deletes the run directories it reads). A mapping sets
the options:

.. code-block:: yaml

   analysis:
     out: checkpoints_online_1/analysis   # default: <output.base_dir>/analysis
     pretraining_root: ../pretraining     # default: alf_root
     validation: true
     free_energy: [luis_bace_solvent]     # or true for every system; default: none
     format: png                          # or pdf

For pretraining, ``mllf pretrain --analyze`` (or ``analysis: true`` in the pretraining
config) writes ``<output-dir>/analysis``. In both cases a failed analysis is a warning,
and the trained model stands.
