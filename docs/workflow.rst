Workflow System
===============

Overview
--------

The online workflow takes a set of systems from fragment files to production biases. The
shared model proposes biases, and every simulation corrects both the model and the pair it
sampled (see :doc:`index` for the motivation and :doc:`cb_setup` for the model). The
pipeline has five parts:

1. **Combination generation**: enumerate the site/substituent combinations of each
   system (:ref:`Combination Generation`).
2. **Held-out split**: set aside pairs the training never runs, so validation measures
   the shared model alone (:ref:`Held-Out Validation`).
3. **Staged online training**: each system moves on its own from pairs of substituents
   to the substituent groups used in production (:ref:`Staged Online Training`).
4. **Production biases**: fixed biases, per group, for each finished system
   (:ref:`Production Biases`).
5. **Cost accounting**: simulated nanoseconds per system and stage, measured the same
   way for the online run and for the iterative-flattening runs it is compared against
   (:ref:`Simulation Cost`).

Running the Workflow
--------------------

Basic Usage
~~~~~~~~~~~

Online training is driven by a YAML configuration file. The driver is
``mllf.cli.train_workflow``, an importable entry point rather than a repo-local script,
so it can be installed and used outside this repository:

.. code-block:: bash

   cd mllf   # repository root
   python -m mllf.cli.train_workflow examples/workflow_14benz.yaml
   python -m mllf.cli.train_workflow my_config.yaml

``examples/run_workflow.py`` runs the same driver. See :doc:`examples` for a walkthrough of
the 1,4-benzene (14benz) example. Combination generation and manifest splitting can also be
run on their own from ``mllf.file_handling.generate_combinations`` and
``mllf.cli.workflow``. The lighter ``mllf.cli.workflow.run_from_config`` entry point covers
combination generation, the split, a single-combo sanity check, an optional simulation
batch and archiving, without the training loop.

Configuration Format
~~~~~~~~~~~~~~~~~~~~

The top-level key of a workflow config is a **list of systems** (``systems:``), even for a
single system, because the driver trains one shared model across all of them:

* **systems**: one entry per system, each with its own ``name``, ``input_dir``,
  ``out_dir`` and ``solvent_state``, plus optional per-system settings
  (``max_subs_per_site``, ``protein_pdbs``, ``solvent_pdb``, ``combo_sets``,
  ``export_combo_set``)
* **mode**: ``train`` (default) or ``deploy`` (:ref:`Testing on New Systems`)
* **pretrain**: warm start from a behavior-cloned checkpoint (see :doc:`cb_pretraining`)
* **curriculum**: the stages each system passes through (:ref:`Staged Online Training`)
* **validation_per_system** / **eval_schedule**: the held-out set and when it is run
* **alf_root**: where the iterative-flattening datasets live, for ALF-derived combo sets
  and the cost comparison
* **training**: policy architecture and optimizer settings
* **bandit**: NeuralLinear + Thompson sampling, hindsight targets and per-pair offsets
  (see :doc:`cb_setup`)
* **reward**: reward weights and thresholds (:ref:`Reward Function`)
* **output**: checkpoint directory and frequency
* **slurm**: cluster submission settings (:ref:`Launching Simulations`)

Working configs are listed in :ref:`Example Configurations`.

.. _Combination Generation:

Combination Generation
----------------------

Principles
~~~~~~~~~~

A combination is the set of substituents simulated together in one MSLD run. Combinations
are generated from the fragment files ``site{N}_sub{M}_{label}.{rtf,pdb}`` in a system's
input directory (and its ``prep/``), where N is the site and M the substituent.

.. warning::
   **At least two substituents per site.** MSLD does not run correctly with a single
   substituent at a site, so generation stops with an error if any site has only one.
   Either add substituents or move that site into the core files (``core.pdb``,
   ``core.rtf``).

* **Within-site combinations** use a rotating anchor: each substituent in turn is the
  first one (the site's reference in that run), followed by subsets of the others. With
  substituents [1, 2, 3], anchor 1 gives [1,2], [1,3], [1,2,3]; anchor 2 gives [2,1],
  [2,3], [2,1,3]; and so on.
* **Cross-site combinations** are the Cartesian product of the within-site selections,
  with at least two substituents per site.
* **Size limit.** At most ``max_subs_per_site`` substituents per site (default 10;
  ``--max-subs``). Every substituent still appears in some combination, but the count
  stays finite.
* **Single-site combinations of a multi-site system** keep the molecule whole: the core
  files are augmented with the first substituent of each site left out.

Each combination's files are renumbered (original substituents [2, 5] become [1, 2]), and
the PRES tokens in the RTF files are rewritten to match so CHARMM reads the topology
correctly. ``mapping.json`` records the original numbers.

For a large site, the cross-site product is usually too large to run, and the pairs
alone are what training needs; :ref:`Combo Sets` select explicit combinations instead.

Lazy Directory Creation
^^^^^^^^^^^^^^^^^^^^^^^

A large system can have tens of thousands of combinations, of which training runs only a
small fraction. Generation therefore only lists them, in ``combo_metadata.json`` (name,
path, sites and substituents) and the manifest files. A directory is created the first
time its combination is needed for training, validation or uncertainty scoring. The split
then takes seconds rather than hours, and disk use scales with the combinations actually
run.

Directory Structure
~~~~~~~~~~~~~~~~~~~

Each combination directory (created on demand) has a standard layout:

.. code-block:: text

   generated_combos/
   ├── combo_metadata.json                  # Metadata for all combinations
   ├── manifest.txt                         # List of all combination names
   ├── train_manifest.txt                   # Training combination names
   ├── val_manifest.txt                     # Validation combination names
   ├── test_manifest.txt                    # Test combination names
   ├── comb_0001_site2_1__site2_2/          # Created on demand
   │   ├── info.py                          # System configuration
   │   ├── mapping.json                     # File renumbering mapping
   │   ├── msld_flat.py                     # Simulation script (copied)
   │   └── prep/
   │       ├── site2_sub1_pres.rtf
   │       ├── site2_sub1_frag.pdb
   │       ├── site2_sub2_pres.rtf
   │       ├── site2_sub2_frag.pdb
   │       ├── core.rtf
   │       ├── core.prm
   │       └── other_support_files...
   ├── comb_0262_site1_1__site1_2__site2_1__site2_2/  # Cross-site
   │   ├── info.py
   │   ├── mapping.json
   │   ├── msld_flat.py
   │   └── prep/
   │       ├── site1_sub1_pres.rtf          # Preserves site numbering
   │       ├── site1_sub2_pres.rtf
   │       ├── site2_sub1_pres.rtf          # Site 2 keeps site2_ prefix
   │       ├── site2_sub2_pres.rtf
   │       └── ...
   └── ...

Combination Metadata Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

**info.py** holds the system configuration read by the simulation script:

.. code-block:: python

   import numpy as np
   import os

   info = {}
   info['name'] = 'comb_0262_site1_1__site1_2__site2_1__site2_2'
   info['nsubs'] = [2, 2]              # Substituents per site [site1, site2]
   info['nblocks'] = np.sum(info['nsubs'])  # Total substituents (4)
   info['ncentral'] = 0                # Central replica for replica exchange
   info['nreps'] = 1                   # Number of replicas
   info['nnodes'] = 1                  # MPI nodes
   info['enginepath'] = os.environ.get('CHARMMEXEC', '')
   info['temp'] = 298.15               # Temperature in Kelvin

**mapping.json** records each file's original and new site/sub numbers:

.. code-block:: json

   [
     {
       "original": "/path/to/site1_sub2_pres.rtf",
       "new_name": "site1_sub1_pres.rtf",
       "original_site": 1, "original_sub": 2,
       "new_site": 1, "new_sub": 1
     }
   ]

Per-pair state (offsets, visits, outcomes) is stored under the *original* numbers, so a
pair keeps its history across every combination it appears in.

Manifest files list combination names, one per line (``comb_0001_site2_1__site2_2``,
...); the full path is ``{out_dir}/{combo_name}``. Because splits are stored as names,
they are reproducible and can be reused by later runs.

Each combination becomes a graph of its substituents for the policy (:doc:`cb_setup`).
The system's ``solvent_state`` (``solv``, ``protein``, or ``gas``/``vacuum``) sets which
environment atoms are part of each substituent's embedding (:doc:`unimol_representation`).

.. _Staged Online Training:

Staged Online Training
----------------------

Why Stages
~~~~~~~~~~

A production run simulates a *group* of substituents at once, but a group run is a poor
place to learn biases from. When it fails, the trajectory rarely shows which pair's
biases were wrong, and every run costs the full simulation length. A *pair* run is the
opposite: λ moves between two substituents, so the run measures that pair's linear (b)
and quadratic (c) error directly (see the hindsight targets in :doc:`cb_setup`), and a
short run is often enough to see it.

The workflow therefore resolves pairs first and only then runs the groups built from
them. With ``curriculum.per_system: true``, each system does this on its own schedule: a
system the model already predicts well can finish its pairs in a few epochs, while a
poorly predicted one keeps running pairs, and every epoch's batch mixes systems at
different points. (The older mode, in which all systems move through the stages
together, is described in :ref:`Global Curriculum`.)

Terms
~~~~~

The following terms are used throughout this section and in the code
(``mllf.cb.system_stages``, ``mllf.cb.hindsight``):

* **Group**: a set of substituents simulated together in production, usually one of
  the iterative-flattening groups of that system (:ref:`Combo Sets`). Its first
  substituent at each site is the group's **reference**.
* **Reference pair**: a group's reference paired with one of its other substituents.
  Every substituent of a group is connected to the reference by one such pair, so these
  pairs fix the group's b values.
* **Good run** of a pair: both substituents were sampled, λ crossed between them at
  least a minimum number of times, and the minority substituent held at least a minimum
  fraction of the pair's population. Full-length runs need 10 crossings and 0.2; short
  runs use the stage's ``explore_min_transitions`` and ``explore_min_fraction`` (defaults 2
  and 0.2).
* **Resolved pair**: a pair with ``advance.successes`` good runs in the current stage
  *and* a per-pair offset whose posterior sd is at most ``advance.max_offset_sd`` for each
  listed coefficient. Both conditions are needed: good runs show that the current biases
  work, and the offset sd shows that the pair's correction is known, not just lucky.
  Resolved pairs leave the stage's pool.
* **Group test**: a group run in which, at every site, each substituent has at least
  ``min_population_ratio / n_subs`` of the site's population and the site has at least
  ``min_site_transitions`` transitions. A group that passes once is retired.

Example
~~~~~~~

The configuration below runs one stage that holds both pairs and groups. It is taken from
the staged configs in ``training/online_*.yaml``, with the system list shortened:

.. code-block:: yaml

   alf_root: /path/to/pretraining

   systems:
     - name: luis_bace_solvent
       input_dir: training/systems/luis_bace_solvent
       out_dir: training/simulations_online_A/luis_bace_solvent/
       solvent_state: solv
       include_patterns: [msld_flat.py]
       combo_sets: {ref_pairs: alf_group_ref_pairs, group_pairs: alf_group_pairs,
                    groups: alf_groups}
       export_combo_set: groups

   curriculum:
     enabled: true
     per_system: true
     max_train_combos_per_stage: 300      # pairs + groups per epoch, all systems
     min_combos_per_system: 12
     combos_per_system: reference_pairs   # shares proportional to reference pairs
     export_final_biases: true
     stages:
       - name: pairs_groups
         combo_set: group_pairs           # every within-group pair
         reference_set: ref_pairs
         reference_pair_bonus: 1.0
         reference_priority: strict       # reference pairs fill each share first
         retire_after_epochs: 10
         prioritize_unvisited: false
         run_steps: {equil: 13000, prod: 39000}
         confirm_full_length: false
         explore_min_transitions: 3
         explore_min_fraction: 0.0
         group_set: groups
         group_ready_fraction: 0.67
         group_max_failures: 3
         epochs: 20
         advance:
           type: groups_done
           fraction: 1.0
           successes: 2
           max_offset_sd: {linear: 1.0, quadratic: 3.0}
           min_population_ratio: 0.0      # group test: every substituent sampled
           min_site_transitions: 0

   validation_per_system: 4
   eval_schedule: stage

A system here runs short pair runs, reference pairs first, until two thirds of a group's
reference pairs are resolved. That group then runs at full length, and the system is done
once every group has passed.

What Happens Each Epoch
~~~~~~~~~~~~~~~~~~~~~~~

For each system that has not finished, the driver:

1. **Builds the pool**: the stage's combos for that system (from ``combo_set``, or the
   manifest filtered by the stage's site and substituent limits), minus held-out pairs,
   resolved pairs and, with ``retire_after_epochs``, retired pairs (run in that many of
   the stage's epochs without resolving). A stage that leaves a system no combos is
   skipped for that system.
2. **Adds ready groups**: with ``group_set``, a group joins the pool once
   ``group_ready_fraction`` of its reference pairs are resolved (retired ones are left
   out of the count), and stays until it passes the group test. With
   ``group_max_failures``, a group that fails that many runs goes back to the pair phase:
   its pairs that include a substituent the last run never sampled are reopened (they
   must resolve again, with a fresh retirement allowance), and the group waits until they
   are resolved or retired.
3. **Orders the batch**: ready groups first; with ``reference_priority: strict``, each
   system's unresolved reference pairs next (not yet run first, then by uncertainty), up to
   the system's share. A reference pair whose run never crossed (stuck at one end, or
   stranded between them) also runs *reversed* from then on, with λ starting on its other
   substituent, so the end a barrier holds λ at says something about b instead of the
   starting state. The reversed partner (``pair07r_site1_subs_9_1`` for
   ``pair07_site1_subs_1_9``) is the same pair, shares its offsets and takes a slot of the
   share. Then (with short runs) pairs waiting for a
   full-length confirming run, then pairs not yet run in this stage (up to
   ``min_combos_per_system``, unless ``prioritize_unvisited: false``), then pairs still
   in progress (``bandit.revisit``, :doc:`cb_setup`), and finally the rest by the shared
   model's uncertainty plus each pair's own state. ``reference_pair_bonus`` is added to a
   reference pair's ranking, so it is chosen ahead of other pairs of similar uncertainty.
4. **Sets each system's share**: ``combos_per_system: reference_pairs`` makes each
   system's share of ``max_train_combos_per_stage`` proportional to its number of
   reference pairs, with ``min_combos_per_system`` as the floor. Without it, a system with
   50 substituents would cover its reference pairs no faster than one with four. (The
   epoch grows if the floors add up to more than the budget.)
5. **Runs and learns**: each combo is simulated, its λ trajectory gives hindsight targets,
   and the per-pair offsets and the shared model are updated (:doc:`cb_setup`).
6. **Checks the stage**: a system leaves the stage when its ``advance`` criterion holds,
   no sooner than ``advance.min_epochs`` (default 1), or once it has spent ``epochs`` in
   the stage. In the second case it is recorded as *not converged*, and that label follows
   its exported biases.

Progress is kept in ``<output.base_dir>/system_stages.json``, so a resumed run continues
each system where it stopped.

Short Runs
~~~~~~~~~~

A stage's ``run_steps: {equil: 13000, prod: 39000}`` (MD steps) sets the length of its
pair runs. It is written into each combination's copy of ``msld_flat.py`` (the ``esteps``
and ``nsteps`` lines) before every run, so the system's template is never changed. Group
runs are always full length.

Short runs are cheaper, but they have fewer frames and crossings, so a good short run is
weaker evidence than a good full-length one. How much weight a short run carries is the
stage's choice:

* With ``confirm_full_length: true`` (default), a short run counts only as a signal: a
  pair whose short run is good gets one full-length *confirming* run in the next epoch,
  and only full-length runs count toward ``successes``.
* With ``confirm_full_length: false``, good short runs count directly. This is
  reasonable when the stage's thresholds already ask for enough crossings, and when a
  later group run checks the same substituents again.

Stage Settings
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Meaning
   * - ``epochs``
     - Most epochs a system spends in the stage.
   * - ``systems``
     - Systems that enter this stage; others skip it. Omit for all systems.
   * - ``min_sites`` / ``max_sites`` / ``min_subs_per_site`` / ``max_subs_per_site``
     - Manifest filters, ignored when ``combo_set`` is given.
   * - ``combo_set``
     - Run this named combo set instead of the filtered manifest (:ref:`Combo Sets`).
   * - ``reference_set``
     - Combo set whose pairs are the reference pairs. Without it, every pair in the pool
       counts as a reference pair.
   * - ``reference_pair_bonus``
     - Added to a reference pair's selection score (default 0).
   * - ``reference_priority``
     - ``strict``: unresolved reference pairs (and reopened pairs) fill each system's share
       before other pairs, with reversed partners for pairs that got stuck. Omit to rank
       them with the others.
   * - ``retire_after_epochs``
     - Retire a pair once it has run in this many of the stage's epochs without
       resolving; it leaves the pool and no longer holds back its groups. Omit to never
       retire.
   * - ``group_set``
     - Combo set of groups that join the pool once ready.
   * - ``group_ready_fraction``
     - Fraction of a group's reference pairs that must be resolved before the group
       runs (default 1.0).
   * - ``group_max_failures``
     - Failed runs after which a group goes back to the pair phase (omit to keep running
       it).
   * - ``prioritize_unvisited``
     - Put pairs not yet run in this stage ahead of selection (default true).
   * - ``run_steps``
     - ``{equil, prod}`` MD steps for this stage's pair runs.
   * - ``confirm_full_length``
     - Require a full-length confirming run before a short run counts (default true).
   * - ``explore_min_transitions`` / ``explore_min_fraction``
     - Good-run thresholds for short runs (defaults 2 and 0.2).
   * - ``advance``
     - When a system may leave the stage early (below).

Keys that set the whole epoch's training in the global curriculum (``update_policy``,
``reward_override``, ``policy_lr``, ``bc_mode`` ...) are ignored in per-system mode,
because one epoch now holds systems at different stages.

Advance Criteria
~~~~~~~~~~~~~~~~

``advance.type`` selects the criterion; ``fraction`` is the share of the pairs, combos or
groups that must meet it, and ``min_epochs`` the earliest epoch at which it is checked.

* ``pairs_resolved``: ``fraction`` of the system's pairs are resolved (``successes``,
  ``max_offset_sd``). ``scope: pool`` (default) counts every pair in the stage's pool;
  ``scope: visited`` counts only pairs run in this stage, which lets a stage end without
  running every pair of a large site.
* ``groups_done``: ``fraction`` of the system's groups have passed the group test
  (``min_population_ratio``, ``min_site_transitions``). ``successes`` and
  ``max_offset_sd`` still define resolved pairs, which decide when a group is ready.
* ``combos_balanced``: the latest run of ``fraction`` of the stage's combos (at least
  ``min_combos``) passes the group test. Use this for a stage of multi-substituent
  combos without explicit groups.
* ``epochs`` (default): the system stays for the full ``epochs``.

.. _Combo Sets:

Combo Sets
~~~~~~~~~~

The manifest of a large site holds mostly pairs, and enumerating every larger subset is
impractical. Named **combo sets** let a stage run exactly the combinations that matter
instead (``mllf.file_handling.combo_sets``). The built-in sets come from the
iterative-flattening groups of each system (``<alf_root>/<system>_group*``), so the online
run and the baseline simulate the same substituent sets and their costs can be compared
directly:

.. code-block:: yaml

   systems:
     - name: luis_bace_solvent
       ...
       combo_sets:
         groups: alf_groups               # one combo per group
         group_pairs: alf_group_pairs     # every within-site pair of those groups
         ref_pairs: alf_group_ref_pairs   # each group's reference vs its other subs
         # explicit groups ({site: [subs]}, first sub = reference), for a system
         # without iterative-flattening runs, and pairs built from them:
         # mine: [{1: [3, 2], 2: [15, 6, 7]}]
         # mine_ref_pairs: {pairs_of: mine, reference_only: true}
       export_combo_set: groups           # production biases per group

Some groups were prepared with the system's numbering and some with their own
(``site1_sub1..n``), so each group's fragments are matched to the system's by content
(source ligand, atom names, types and charges), not by file name. A site with a single
substituent in a group is a fixed substituent, not a perturbation, and is dropped. A
system with no ``_group*`` datasets uses its single dataset as one group.

.. _Held-Out Validation:

Held-Out Validation
~~~~~~~~~~~~~~~~~~~

Validation answers one question: does the *shared model* predict biases for pairs it has
never been trained on? To keep the answer clean, held-out pairs are never trained on, and
validation runs never apply per-pair offsets.

``validation_per_system: N`` holds out ``N`` pairs per system. When the first stage runs a
combo set, the held-out pairs are drawn from manifest pairs in none of the system's sets,
so no stage ever runs them. (A system whose groups cover every pair has nothing to hold
out.) ``validation_per_system: 0`` holds out nothing. To reuse another run's held-out set
exactly, set ``validation_combos_file`` to its ``validation_combos.json``; otherwise the
seeded draw picks by index and changes whenever the combo pool changes.

With ``eval_schedule: stage``, each system's held-out pairs are run at the start, middle
and end of each of its stages (epochs 1, ceil(n/2) and n of an n-epoch stage) and in the
epoch it meets a stage's criterion, instead of every ``eval_freq`` epochs. Because the set
of systems being validated then differs from pass to pass, a validation score is
comparable only to earlier passes of the same system.

.. _Production Biases:

Production Biases
~~~~~~~~~~~~~~~~~

When a system finishes its last stage, its production biases are written to
``<output.base_dir>/final_biases/<system>/``. They are the shared model's prediction plus
each pair's offset posterior mean, with no Thompson or offset sampling; b is fitted per
site from every within-site edge, weighted by each pair's linear-offset variance.

* With ``export_combo_set``, one ``variables.py`` is written per combo of that set
  (``final_biases/<system>/<group>/``), matching how production is run.
* Otherwise, one export covers every substituent in the system's ``input_dir``, or the
  subset in ``export_combo_dir``.

Each exported group also gets ``CONFIDENCE.txt`` and ``confidence.json``. These record
whether the group passed its group test and, for every pair in it, whether the pair was
resolved, run but not resolved, or never run (with visits, good runs and the b/c offset
uncertainty). A group that never passed is renamed ``<group>_unconverged``. Its biases are
still the best available estimate, but they have not been shown to sample, so they should
be checked before production.

Cross-site couplings are not predicted and are exported as 0. Exported biases are
therefore complete for groups whose perturbed substituents share one site, but for
multi-site groups they are only a starting point.

Set ``curriculum.export_final_biases: false`` to skip the export. To export or label from a
finished run:

.. code-block:: bash

   python -m mllf.cb.export_biases config.yaml --model final_model.pt \
       --bandit-state <output.base_dir>/bandit_state.json --out final_biases
   python -m mllf.cb.export_biases config.yaml --label <output.base_dir>

.. _Testing on New Systems:

Testing on New Systems
~~~~~~~~~~~~~~~~~~~~~~

``mode: deploy`` runs a trained model on systems it was not trained on, through the same
per-system stages (always on in this mode):

.. code-block:: yaml

   mode: deploy
   deploy:
     model_path: checkpoints_run/final_model.pt
     update_shared_model: false   # per-pair offsets still learn
   output:
     base_dir: deploy_test        # must not be the model's directory

Nothing is held out for validation, because every pair of a test system is part of the
test, and the model file is never overwritten. With ``update_shared_model: false``, the
cost recorded for a test system is what a user of the trained model would pay: the shared
model's first guess plus the per-pair corrections that system needed.

.. _Simulation Cost:

Simulation Cost
~~~~~~~~~~~~~~~

The comparison that matters is simulated time to converged biases, so both sides are
measured in nanoseconds the same way. Each finished run's simulated time is added to
``<output.base_dir>/cost_ledger.json`` by system, stage, and train or validation
(``mllf.cb.cost_ledger``). The time is read from the dynamics calls in the CHARMM output
and includes equilibration, since that is compute spent. The iterative-flattening cost for
the same systems is read the same way from the original run outputs named in the
pretraining datasets:

.. code-block:: bash

   python -m mllf.cb.cost_ledger <output.base_dir> --pretraining-root pretraining

The comparison is like for like only under three conditions: both sides ran the same
substituent groups (use the ALF-derived combo sets), validation runs are reported
separately (they measure the model and are not needed to produce biases), and the
online cost is that of the run that produced the exported biases. A system trained on
earlier in a ``train`` run has also contributed to the shared model, so a cost measured
in ``deploy`` mode is the cleaner estimate for a new system.

Simulation Execution
--------------------

.. _Launching Simulations:

Launching Simulations
~~~~~~~~~~~~~~~~~~~~~

Each combo's biases are written to ``variables.py`` and its simulation is submitted as one
SLURM job per combo per epoch (``mllf.cli.slurm.submit_epoch_simulation``). Every
cluster-specific detail comes from the config's ``slurm:`` block; nothing is hardcoded:

* ``partition``: passed as ``#SBATCH -p``. Omit it to use the account's default partition.
* ``gres``: passed as ``#SBATCH --gres``. Omit it on a CPU-only cluster.
* ``time``: passed as ``#SBATCH --time``. Defaults to ``01:00:00`` if the whole ``slurm:``
  block is omitted.
* ``module``: passed to ``module load``. Omit it if the simulator is already on ``PATH``.

An empty or absent ``slurm:`` block still produces a valid script; it just omits the
directives that were not set.

When a job finishes, its output is parsed for populations, transitions, pair profiles and
simulated time (:doc:`file_handling`).

.. _Reward Function:

Reward Function
---------------

The ``reward:`` block configures a *run reward*: one number per simulation that summarizes
how well it sampled. It is reported as the epoch's average reward, decides
``progression: reward`` in the global curriculum, and filters or weights runs in
pretraining (:doc:`cb_pretraining`). Online updates do not use it: they use the per-edge,
per-coefficient reward or the hindsight targets (:doc:`cb_setup`); the ``barrier_*``,
``skew_width``, ``end_width`` and ``no_transition_weight`` keys below belong to that
per-edge reward.

It is built so that a degenerate run, such as one that settles into a single
substituent, scores below any run that samples:

.. math::

   R = \left(\frac{N_{\text{visited}}}{N_{\text{subs}}}\right)^2 (R_P + R_T + R_{\text{entropy}}) - \min(60,\; P_{\text{trans}} + P_{\text{conc}})

* :math:`R_P = w_P \, C_F \sum_{k\,\text{visited}} p_k / P_{\text{baseline}}`: population,
  discounted by :math:`C_F = \min(1, T_{\min}/2N_{\text{req}})` because populations from
  runs with few transitions are not yet reliable.
* :math:`R_T = w_T \sum_s T_s / T_{\text{baseline}}` if every site reaches
  :math:`N_{\text{req}}` = ``min_transitions_per_site`` transitions (×1.5 above twice that
  on average), else 0.
* :math:`R_{\text{entropy}}`: ``entropy_bonus`` × the normalized Shannon entropy of the
  populations.
* :math:`P_{\text{trans}}` is set by the worst site, so one failing site cannot hide
  behind good ones: 40, 32 or 24 for :math:`T_{\min}` = 0, 1 or 2, then
  :math:`2 + 2(N_{\text{req}} - T_{\min})` up to :math:`N_{\text{req}}`, plus 4 per
  additional failing site.
* :math:`P_{\text{conc}}`: for each site where one substituent holds more than 0.8 of the
  population, :math:`5\gamma` × the excess.

.. code-block:: yaml

   reward:                                   # defaults
     w_P: 0.5
     w_T: 0.75
     gamma: 4.0
     P_baseline: 500.0
     T_baseline: 50.0
     min_transitions_per_site: 10
     entropy_bonus: 8.0
     concentration_penalty_threshold: 0.8
     # per-edge reward (cb_setup)
     barrier_target: 1.2                     # kcal/mol
     barrier_width: 1.29                     # kcal/mol
     skew_width: 0.78
     end_width: 0.58
     no_transition_weight: 0.15
     # accepted but unused: w_U, min_coverage_ratio

.. _Global Curriculum:

Global Curriculum
-----------------

Without ``per_system``, all systems move through the curriculum stages together: every
epoch draws from one stage, and the stage changes for everyone at once. This was the
original mode and remains useful for REINFORCE training and for small single-system runs
such as the 14benz example. Its weakness is that the stage length is set by the slowest
system, so systems that are already predicted well keep spending simulations.

Stages run from single-site pairs, through larger single-site combinations, to
multi-site combinations, for the reason given in :ref:`Staged Online Training`: smaller
combinations show their own errors more directly. For example:

.. code-block:: yaml

   curriculum:
     enabled: true
     max_train_combos_per_stage: 100
     stages:
       - {name: pairs_single_site, min_sites: 1, max_sites: 1,
          min_subs_per_site: 2, max_subs_per_site: 2, epochs: 50}
       - {name: triplets_single_site, min_sites: 1, max_sites: 1,
          min_subs_per_site: 3, max_subs_per_site: 3, epochs: 50}
       - {name: pairs_two_sites, min_sites: 2, max_sites: 2,
          min_subs_per_site: 2, max_subs_per_site: 2, epochs: 50}
     progression:
       type: epoch

Each stage takes:

* ``min_subs_per_site``, ``max_subs_per_site``: bounds on the substituents at any one
  site of a combination (checked against the per-site counts, not the combination's
  total)
* ``min_sites``, ``max_sites``: number of sites represented
* ``combo_whitelist``: an explicit list of combo names, bypassing the size filters
* ``min_transitions_for_stage`` / ``transition_filter_lookback``: on entering a later
  stage, keep only combos whose recent runs reached this many transitions per site
* ``epochs``: epochs in the stage
* ``max_train_combos``: per-stage override of ``max_train_combos_per_stage``
* ``reward_override``: per-stage reward weights, e.g. a lower
  ``min_transitions_per_site`` while the policy is still learning pairs

Each epoch draws up to ``max_train_combos_per_stage`` of the stage's combos: stratified
across systems by default (``curriculum.stratified_sampling``,
``curriculum.min_combos_per_system``), or by uncertainty with ``bandit.selection:
uncertainty``. Draws are seeded by ``seed`` plus the epoch. The draw is uniform over the
combos that pass the filter, so a stage allowing pairs and triplets draws mostly triplets,
in proportion to how many of each exist.

Progression
~~~~~~~~~~~

* ``type: epoch`` (default): advance after the stage's epochs.
* ``type: reward``: advance only if the average reward over the last 5 epochs reaches
  ``reward_threshold``.
* ``type: both``: complete the epochs *and* meet the threshold.

Checkpointing and Resume
------------------------

Long runs are interrupted by SLURM time limits, maintenance or cancellation, so the driver
saves enough state to continue from the last completed epoch.

.. code-block:: yaml

   output:
     base_dir: /path/to/training_output
     save_checkpoints: true
     checkpoint_freq: 5         # Save every N epochs

``base_dir`` then holds:

* ``checkpoint_{epoch:03d}.pt``: the policy and optimizer state, the epoch, the global
  curriculum stage index, and that epoch's statistics
* ``bandit_state.json``: per-combo bookkeeping and the per-pair state (offset targets,
  visits, outcomes, exploration factors, b and c brackets). This is the state the
  per-pair corrections depend on, so a run resumed without it starts every pair over.
* ``system_stages.json``: each system's stage, epochs spent and progress
  (per-system mode)
* ``validation_combos.json``: the held-out set, so a resumed run validates on the same
  pairs
* ``cost_ledger.json``: simulated time per run
* ``final_model.pt``: written at the end, with the full per-epoch statistics

On restart, the driver loads the latest ``checkpoint_*.pt`` and continues from the next
epoch. If that checkpoint cannot be loaded, it falls back to ``pretrain.model_path``
(partial load, ``strict=False``).

The Uni-Mol embeddings and graph tensors of each combo are saved to ``graph_data.pt`` in
its run directory before the simulation is submitted. The update after the SLURM wait
reloads them rather than recomputing the embeddings or holding every combo in memory
during the wait.

Archiving Combinations
----------------------

The online training driver does not archive anything itself (an ``archive:`` block in
its config is ignored), because the per-pair state and the cost ledger point back into
the run directories. Archive a run's combination directories once its biases are exported
and its costs summarized, with ``archive_combo_dirs(out_dir, pattern='comb_*', remove=False)``
(``mllf.file_handling.generate_combinations``) or ``examples/14benz/archive_combos.sh``.

.. _Example Configurations:

Example Configurations
----------------------

* ``examples/workflow_14benz.yaml``: the smallest working config (one system, REINFORCE,
  global curriculum, no pretrained model), walked through in :doc:`examples`.
* ``training/online_*.yaml``: the staged setup used for production biases, with many
  systems, per-system stages over ALF-derived combo sets, a pretrained Bayesian model,
  hindsight targets with per-pair offsets, and full-network regression. Each header
  explains which systems the run covers and why.

See Also
--------

* :doc:`file_handling` - file formats and parsers
* :doc:`cb_setup` - policy architecture, hindsight targets and per-pair offsets
* :doc:`unimol_representation` - Uni-Mol embeddings for node features
* :doc:`cb_pretraining` - behavior cloning from completed simulations
* :doc:`examples` - example workflows and usage
* :doc:`api` - API reference
