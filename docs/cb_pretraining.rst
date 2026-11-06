CB Behavior Cloning
===================

Before online training, the policy is **pretrained** by behavior cloning on the bias
coefficients of completed simulations. Online training then starts near working biases,
so its first runs correct the model rather than search for a region where λ moves at all.
Pretraining also supplies two things online training relies on, namely the prior of the
Bayesian heads and the replay set that keeps online updates from overwriting what the
model has learned (:doc:`cb_setup`). How much it helps a new system depends on how similar
that system is to the pretraining data, and the online stages correct the rest.

Pretraining Data
----------------

``mllf.cli.collect_pretraining_data`` turns completed run directories (``variables.py``,
output and ``prep/``) into records under ``pretraining/``. Three layouts are recognized:

.. code-block:: text

   pretraining/
   ├── 14benz_solv/run1/{prep/, variables.py}           # per-run prep
   ├── 123benz_solvent_group1/{prep/, run1/, run2/...}  # shared prep (takes priority)
   └── 14benz_pair_combos/comb_0063.../run_001/         # one directory per combination

Each run needs ``variables.py`` and a ``prep/`` with the RTF and PDB files, and
``metadata.json`` with the run's metrics is optional. The data come from
iterative-flattening runs and from online runs collected later (below).

**Which substituent is which.** A ``variables.py`` lists biases by position
(``lams1s1``, ``lams1s2``, ...), so every coefficient is only as correct as the mapping
from position to substituent. That mapping is read from the BLOCK setup the run actually
executed, as echoed in its output. The prep ``.inp`` files are used only when the output
does not echo it, and sequential numbering is the last resort. The ``.inp`` files alone are
not reliable, because a prep can hold several or define blocks out of read order, and
either case silently mislabels substituents. ``--relabel-subs`` reapplies the mapping to
existing datasets in place, keeping the original as ``graph_info.orig.json``:

.. code-block:: bash

   mllf collect --relabel-subs pretraining \
       [--datasets luis_bace_solvent ...] [--dry-run]

Adding Online Runs
~~~~~~~~~~~~~~~~~~

Online runs can be collected so that a later model starts from what they learned:

.. code-block:: bash

   python training/collect_online_runs.py --simulations-dir training/simulations_online_1 \
       --output-dir pretraining \
       --exclude-combos-file checkpoints_online_1/validation_combos.json

Only runs with more than ``--min-transitions`` total transitions (default 3) are
collected. Each is named ``<run>__<tag>`` (``--tag``, by default the simulations
directory name), so runs from different online trainings that share combo and run names
do not overwrite each other, and its ``metadata.json`` records the source run, online run,
epoch and ``collection_tag``. ``--exclude-combos-file`` skips the online run's held-out
pairs, so a later test on the same held-out set still tests unseen pairs. To keep them
anyway, collect them alone (``--only-combos-file``) into a directory that pretraining does
not read. A model pretrained without the collected runs should not replay them online
(``regression.replay_pretraining_only: true``, :doc:`cb_setup`).

Running Pretraining
-------------------

.. code-block:: bash

   mllf pretrain --pretraining-dir pretraining/ \
       --output-dir models/my_pretrain_run --config examples/workflow_pretrain.yaml \
       --epochs 50 --bayesian-heads

``--pretraining-dir`` can be repeated. The ``--config`` file sets the reward weights used
to weight runs (below) and the Uni-Mol options, ``unimol.env_input``,
``ligand_other_sites``, ``site_size_input``, ``structure_selection``,
``environment_cutoff`` and ``cache_embeddings``. The first three are recorded in
``pretrain_metadata.json``, and an online config must use the same values; online
training and ``mllf scout`` check them (:doc:`unimol_representation`). Filters and optimizer settings are command-line flags.

* ``--no-data-hygiene``, ``--no-per-pair-awr`` and ``--awr-temperature`` (default 1.0) set
  the objective (below).
* ``--use-best-only``, ``--min-transitions``, ``--min-reward-threshold``,
  ``--stratified-negative-fraction`` and ``--no-filter-outliers`` /
  ``--outlier-std-threshold`` filter runs (below).
* ``--reward-weighted`` and ``--emphasize-systems`` add per-run weights.
* ``--bayesian-heads`` (with ``--bayesian-prior-precision``) also saves
  ``best_policy_bayesian.pt``, the checkpoint for ``algorithm: hindsight``. Each posterior
  mean starts from the trained readout, and its precision is calibrated on the pretraining
  data, so the posterior's spread reflects how much real biases vary.

``examples/pretrain_with_filtering.sh`` is the SLURM launcher, driven by a per-run script
such as ``examples/pretrain_wUnimol_Nov02b.sh`` that sets the config, output directory and
flags. The learning rate follows a cosine schedule down to ``lr / 100``. The output
directory holds:

* ``best_policy.pt`` and ``best_policy_bayesian.pt``, either of which can be
  ``pretrain.model_path`` in a workflow config, and ``final_policy*.pt``
* ``checkpoint_epoch_NNN.pt``, from which ``--resume-checkpoint`` continues
* ``pretrain_metadata.json``: the Uni-Mol settings the model was trained with
* ``pretraining_manifest.json``: every directory pretraining read, as absolute paths, the
  ``bandit.regression.replay_manifest`` of the regression replay
* ``pretraining_systems.json``: the systems pretraining used, read by ``mllf replay``
* ``pretrain.log``: every message with time stamps, including the per-run detail and the
  filter statistics the console leaves out (:doc:`cli`)
* ``pretraining_representations_used.csv``: the structure each run was embedded from
* ``environment_consensus.json``, with ``env_input: dual`` only
  (:doc:`unimol_representation`)

Objective
---------

The loss is an advantage-weighted regression (AWR), the Gaussian negative log-likelihood
of each target coefficient weighted by how well the run sampled:

.. math::

   \mathcal{L}_{\text{AWR}} = -\frac{1}{|\mathcal{G}|}
   \sum_{r \in \mathcal{G}}
   \frac{1}{|\mathcal{A}_r|}
   \sum_{(i,j,k) \in \mathcal{A}_r}
   w_{ij}^{(k)} \log \mathcal{N}\!\left(a_{ij}^{(k)};
     \mu_{ij}^{(k)}, (\sigma_{ij}^{(k)})^2\right)

where :math:`\mathcal{G}` is a group of runs with the same graph, :math:`\mathcal{A}_r` is
the run's set of active targets, :math:`k` is the bias type, and
:math:`w = \exp(R/\beta)` (capped at 20). The NLL trains σ as well as μ, and σ later sets
the starting exploration scale.

* **Per-pair weighting** (default). :math:`R` is computed per edge and per coefficient with
  the per-edge reward used online (:doc:`cb_setup`), so well-resolved pairs drive more
  gradient than poorly resolved pairs in the same run. A pair's weight is zero if a
  substituent was never sampled or its ΔΔG is missing (a missing direction is filled from
  the reverse one). ``--no-per-pair-awr`` uses one weight per run.
* **Run length.** The data mix short runs (∼0.1 ns) with 1 ns runs, and a short run has a
  tenth of the time to cross and balance. Pretraining therefore judges each run the way
  online training judges its short and full-length runs, and a run under half of
  ``full_length_ns`` (default 1.0) is short.

  - *Success gate.* A pair counts as a success when it passes the online test for a run of
    its length. A full-length run needs at least 10 crossings and a minority fraction of at
    least 0.2. A short run needs ``explore_min_transitions`` crossings and
    ``explore_min_fraction`` (defaults 3 and 0.0), which should match the online pair
    stage. A pair that passed gets an advantage of at least 0, so its biases are imitated
    at least as much as average, and a pair that failed gets at most 0. The reward then
    ranks pairs within each side. As online, length decides which test a pair must pass
    rather than how much passing is worth. ``success_gate: false`` turns the gate off.
  - *Counts scale with length.* In the whole-run reward used for stratified sampling,
    reward filtering and the fallback weight, the population and transition baselines
    scale with run length, and a short run needs only ``explore_min_transitions``
    crossings per site.

  These keys go in the config's ``reward`` section. The length comes from
  ``simulated_ns`` in ``simulation_results.json``, recorded at collection
  (``collect_pretraining_data --backfill-barrier PRETRAINING_ROOT --workers N`` adds it to
  existing datasets). Runs without it count as full length.
* **Data hygiene** (default). A run's coefficients are evidence for a coefficient only when
  the run sampled the part of λ space that coefficient shapes. A trapped run leaves the
  other coefficients wherever its optimizer stopped, and flattening runs that never
  reached their endpoints drove end biases to 10–18, far outside converged values. Each
  target is therefore kept only when the run measured it: **linear** when both
  substituents were sampled with a finite ΔΔG, **quadratic** when the midpoint barrier was
  measured, and **skew / end** when the asymmetry was measured (frames at both endpoint
  cores). Runs collected without profile signals keep only their linear targets. Each
  run's loss is divided by its number of active targets, so a run with few measured
  targets counts less, and the Bayesian heads' noise and prior use the same mask.

Run Filters
-----------

Without data hygiene, every run counts equally and the weights only rank targets within a
run, so poor runs must be filtered out. With hygiene, a run that never sampled a pair adds
no weight for it, so the filters are optional. They are:

* **Outliers** (default ±3σ), which drops runs with coefficients far from the dataset mean,
  usually a diverging run rather than a sampling strategy;
* **Minimum reward** or **minimum transitions**, which drops runs that sampled poorly;
* **Best only**, which keeps the highest-reward run per system and is clean but narrow,
  working poorly with few systems;
* **Stratified negatives**, which keeps every positive-reward run and a share of each
  negative-reward bucket that rises quadratically toward zero (up to 55% for the best
  bucket), so the model sees near misses as well as successes.

Graphs are built once and kept in memory for all epochs. Many runs share a prep
directory, so embedding computation scales with the number of distinct preps rather than
runs, and ``unimol.cache_embeddings`` keeps the embeddings on disk between invocations.

See Also
--------

* :doc:`cb_setup`: the policy and the online signals pretraining prepares for
* :doc:`unimol_representation`: the frozen embeddings pretraining builds on
* :doc:`file_handling`: ``variables.py`` and output formats
