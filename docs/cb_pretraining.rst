CB Behavior Cloning
===================

Before online training, the policy is **pretrained** by behavior cloning on bias
coefficients from completed simulations. Online training then starts near working biases,
so its first runs correct the model instead of searching for a region where λ moves at
all. Pretraining also supplies what online training relies on: the prior of the Bayesian
heads, and the replay set that keeps online updates from overwriting what the model
learned (:doc:`cb_setup`). How much it helps a new system depends on how similar that
system is to the pretraining data; the online stages handle the rest.

Pretraining Data
----------------

``mllf.cli.collect_pretraining_data`` turns completed run directories (``variables.py``,
output, ``prep/``) into records under ``pretraining/``. Three layouts are recognized:

.. code-block:: text

   pretraining/
   ├── 14benz_solv/run1/{prep/, variables.py}           # per-run prep
   ├── 123benz_solvent_group1/{prep/, run1/, run2/...}  # shared prep (takes priority)
   └── 14benz_pair_combos/comb_0063.../run_001/         # one directory per combination

Each run needs ``variables.py`` and a ``prep/`` with the RTF/PDB files; ``metadata.json``
with the run's metrics is optional. The data come from iterative-flattening runs and from
online runs added later (below).

**Which substituent is which.** A ``variables.py`` lists biases by position
(``lams1s1``, ``lams1s2``, ...), so every coefficient is only as correct as the mapping
from position to substituent. That mapping is read from the BLOCK setup the run actually
executed (in its output). The prep ``.inp`` files are used only when the output does not
echo it, and sequential numbering is the last resort. The ``.inp`` files are not reliable
on their own: a prep can hold several, or define blocks out of read order, and either
case silently mislabels substituents. Datasets collected before this change can be
corrected in place (the original is kept as ``graph_info.orig.json``):

.. code-block:: bash

   python -m mllf.cli.collect_pretraining_data --relabel-subs pretraining \
       [--datasets luis_bace_solvent ...] [--dry-run]

Adding Online Runs
~~~~~~~~~~~~~~~~~~

Online runs can be added so that a later model starts from what they learned:

.. code-block:: bash

   python training/collect_online_runs.py --simulations-dir training/simulations_online_A \
       --output-dir pretraining \
       --exclude-combos-file checkpoints_online_A/validation_combos.json

Only runs with more than ``--min-transitions`` total transitions (default 3) are
collected. Each is named ``<run>__<tag>`` (``--tag``, by default the simulations
directory name), so runs from different online trainings that share combo and run names
do not overwrite each other. Its ``metadata.json`` records the source run, online run,
epoch and ``collection_tag``. Two rules keep later comparisons fair:

* **Held-out pairs stay held out.** ``--exclude-combos-file`` skips the online run's
  validation pairs, so a later test on the same held-out set still tests unseen pairs. To
  keep them anyway, collect them alone (``--only-combos-file``) into a directory that
  pretraining does not read.
* **Replay uses only the original data.** ``regression.replay_pretraining_only`` (default
  ``true``) skips tagged runs during online regression, so a new test does not replay
  earlier tests.

Running Pretraining
-------------------

.. code-block:: bash

   python -m mllf.cb.pretrain_policy --pretraining-dir pretraining/ \
       --output-dir models/my_pretrain_run --config examples/workflow_pretrain.yaml \
       --epochs 50

``--pretraining-dir`` can be repeated. The ``--config`` file sets the Uni-Mol options
(``unimol.structure_selection``, ``unimol.environment_cutoff``,
``unimol.cache_embeddings``) and the reward weights used for run weighting. Filters and
optimizer settings are command-line flags:

* ``--no-data-hygiene``, ``--no-per-pair-awr``, ``--awr-temperature`` (default 0.5):
  the objective (below)
* ``--use-best-only``, ``--min-transitions``, ``--min-reward-threshold``,
  ``--stratified-negative-fraction``, ``--no-filter-outliers`` /
  ``--outlier-std-threshold``: run filters (below)
* ``--reward-weighted``, ``--emphasize-systems``: extra per-run weights
* ``--bayesian-heads`` / ``--bayesian-prior-precision``: also save a Bayesian-head copy
  (``best_policy_bayesian.pt``) for NeuralLinear + Thompson sampling, with each posterior
  mean started from the trained readout

``examples/pretrain_wUnimol.sh`` is a complete SLURM example. The outputs are
``best_policy.pt`` (set it as ``pretrain.model_path`` in a workflow config),
``training_log.txt``, ``filtering_stats.json`` (which runs were excluded and why) and
checkpoints. The learning rate follows a cosine schedule down to ``lr / 100``.

Objective
---------

The loss is an advantage-weighted regression (AWR): the Gaussian negative log-likelihood
of each target coefficient, weighted by how well the run sampled:

.. math::

   \mathcal{L}_{\text{AWR}} = -\frac{1}{|\mathcal{G}|}
   \sum_{r \in \mathcal{G}}
   \frac{1}{|\mathcal{A}_r|}
   \sum_{(i,j,k) \in \mathcal{A}_r}
   w_{ij}^{(k)} \log \mathcal{N}\!\left(a_{ij}^{(k)};
     \mu_{ij}^{(k)}, (\sigma_{ij}^{(k)})^2\right)

where :math:`\mathcal{G}` is a group of runs with the same graph, :math:`\mathcal{A}_r`
the run's active targets, :math:`k` the bias type, and :math:`w = \exp(R/\beta)` (capped
at 20). The NLL trains σ as well as μ, and σ later sets the starting exploration scale.

* **Per-pair weighting** (default). :math:`R` is computed per edge and per coefficient with
  the same per-edge reward used online (:doc:`cb_setup`), so well-resolved pairs drive
  more gradient than poorly resolved pairs in the same run. A pair's weight is zero if a
  substituent was never sampled or its ΔΔG is missing (a missing direction is filled from
  the reverse one). ``--no-per-pair-awr`` uses one weight per run.
* **Run length.** The data mix short runs (~0.1 ns) with 1 ns runs, and a short run has a
  tenth of the time to cross and balance. Pretraining judges each run the way online
  training judges its own short and full runs. A run under half of ``full_length_ns``
  (default 1.0) is short.

  - *Success gate.* A pair counts as a success when it passes online training's test for
    a run of its length. A full run needs at least 10 crossings and a minority fraction of
    at least 0.2. A short run needs only ``explore_min_transitions`` crossings and
    ``explore_min_fraction`` (defaults 3 and 0.0, the same values as the online pair stage).
    A pair that passed gets an advantage of at least 0, so its biases are imitated at least
    as much as average. A pair that failed gets at most 0. The reward then ranks pairs
    within each side. As in online training, a pass in a short run counts as much as a
    pass in a full one: length decides which test a pair must pass, not how much passing
    is worth. ``success_gate: false`` turns the gate off.
  - *Counts scale with length.* In the whole-run reward used for stratified sampling,
    reward filtering and the fallback weight, the population and transition baselines
    scale with run length. A short run there needs only ``explore_min_transitions``
    crossings per site.

  All of these keys go in the config's ``reward`` section. The length comes from
  ``simulated_ns`` in ``simulation_results.json``. It is recorded at collection, and
  ``collect_pretraining_data --backfill-barrier PRETRAINING_ROOT --workers N`` adds it to
  older data. Runs without it count as full length.
* **Data hygiene** (default). A run's coefficients are evidence for a coefficient only
  when the run sampled the part of λ space that coefficient shapes. A trapped run leaves
  the others wherever its optimizer stopped: flattening runs that never reached their
  endpoints drove end biases to 10–18, far outside converged values. Each target is
  therefore kept only when the run measured it: **linear** when both substituents were
  sampled with a finite ΔΔG, **quadratic** when the midpoint barrier was measured,
  **skew / end** when the asymmetry was measured (frames at both endpoint cores). Runs
  collected without profile signals keep only their linear targets. Each run's loss is
  divided by its number of active targets, so a run with few measured targets counts
  less. The Bayesian heads' noise and prior use the same mask.

Run Filters
-----------

Without hygiene, every run counts equally and the weights only rank targets within a run,
so poor runs must be filtered out. With hygiene, a run that never sampled a pair adds no
weight for it, so the filters are optional; ``examples/pretrain_wUnimol.sh`` uses none.
They remain available:

* **Outliers** (default ±3σ): drops runs with coefficients far from the dataset mean,
  which usually reflect a diverging run rather than a sampling strategy.
* **Minimum reward** or **minimum transitions**: drops runs that sampled poorly.
* **Best only**: keeps the highest-reward run per system. This is clean but narrow, and
  it works poorly with few systems.
* **Stratified negatives**: keeps every positive-reward run and a share of each
  negative-reward bucket that rises quadratically toward zero (up to 55% for the best
  bucket). The model then sees near-misses as well as successes.

Graphs are built once and kept in memory for all epochs. Many runs share a prep
directory, so embedding computation scales with the number of distinct preps, not runs.
``unimol.cache_embeddings`` also keeps them on disk between invocations.

See Also
--------

* :doc:`cb_setup` - the policy and the online signals pretraining prepares for
* :doc:`unimol_representation` - the frozen embeddings pretraining builds on
* :doc:`file_handling` - ``variables.py`` and output formats
