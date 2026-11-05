Workflow System
===============

Overview
--------

The online workflow takes a set of systems from fragment files to production biases. A
shared model proposes biases, and every simulation corrects both the model and the pair
of substituents it sampled (:doc:`cb_setup`). The pipeline has five parts:

1. **Combination generation** enumerates the site and substituent combinations of each
   system (:ref:`Combination Generation`).
2. **A held-out split** sets aside pairs that training never runs, so validation measures
   the shared model alone (:ref:`Held-Out Validation`).
3. **Staged online training** moves each system on its own schedule from pairs of
   substituents to the substituent groups used in production
   (:ref:`Staged Online Training`).
4. **Production biases** are exported per group for each finished system
   (:ref:`Production Biases`).
5. **Cost accounting** records simulated nanoseconds per system and stage, measured the
   same way for the online run and for the iterative-flattening runs it is compared with
   (:ref:`Simulation Cost`).

Running the Workflow
--------------------

Online training is driven by a YAML configuration file:

.. code-block:: bash

   cd mllf   # repository root
   mllf train examples/workflow_14benz.yaml

:doc:`examples` walks through the 1,4-benzene (14benz) example. The driver itself is
``mllf.cb.training_loop.run_training``: setup in ``mllf.cb.training_setup``, one epoch's
simulations and updates in ``mllf.cb.reinforce_trainer`` (``epoch_runs`` /
``epoch_updates``).

Configuration Format
~~~~~~~~~~~~~~~~~~~~

A config lists its systems under ``systems:``, even when there is only one, because one
shared model is trained across all of them. The top-level sections are:

* **systems**: one entry per system with its ``name``, ``input_dir``, ``out_dir`` and
  ``solvent_state``, plus optional ``max_subs_per_site``, ``protein_pdbs``,
  ``solvent_pdb``, ``combo_sets`` and ``export_combo_set``
* **mode**: ``train`` (default) or ``deploy`` (:ref:`Testing on New Systems`)
* **pretrain**: the behavior-cloned checkpoint to start from (:doc:`cb_pretraining`)
* **unimol**: how the node features are built (``env_input``, ``ligand_other_sites``,
  ``site_size_input``, ``cache_embeddings``, ``cache_dir``), which must match the
  pretrained model (:doc:`unimol_representation`)
* **curriculum**: the stages each system passes through (:ref:`Staged Online Training`)
* **validation_per_system** / **eval_schedule**: the held-out set and when it is run
* **alf_root**: the iterative-flattening datasets, used for ALF-derived combo sets and
  the cost comparison
* **training**: policy and optimizer settings
* **bandit**: the learning algorithm (``algorithm: hindsight``) and its settings
  (:doc:`cb_setup`)
* **reward**: run-reward weights and thresholds (:ref:`Reward Function`)
* **output**: the checkpoint directory and frequency
* **slurm**: cluster submission settings (:ref:`Launching Simulations`)

With ``algorithm: hindsight``, most bandit and stage settings have defaults chosen for
staged training (:ref:`Bandit Defaults`), so a config sets only what differs from them.
Working configs are listed in :ref:`Example Configurations`.

.. _Combination Generation:

Combination Generation
----------------------

A combination is the set of substituents simulated together in one MSLD run.
Combinations are generated from the fragment files ``site{N}_sub{M}_{label}.{rtf,pdb}``
in a system's input directory (and its ``prep/``), where N is the site and M the
substituent.

.. warning::
   MSLD does not run correctly with a single substituent at a site, so generation stops
   with an error if any site has only one. Either add substituents or move that site into
   the core files (``core.pdb``, ``core.rtf``).

* **Within-site combinations** use a rotating anchor, in which each substituent in turn
  is the site's reference, followed by subsets of the others. With substituents
  [1, 2, 3], anchor 1 gives [1,2], [1,3] and [1,2,3], anchor 2 gives [2,1], [2,3] and
  [2,1,3], and so on.
* **Cross-site combinations** are the Cartesian product of the within-site selections,
  with at least two substituents per site.
* **Size limit.** At most ``max_subs_per_site`` substituents per site (default 10) keeps
  the count finite while every substituent still appears in some combination.
* **Single-site combinations of a multi-site system** keep the molecule whole by adding
  each other site's first substituent to the core files.

Each combination's files are renumbered (original substituents [2, 5] become [1, 2]), and
the PRES tokens in the RTF files are rewritten to match, so CHARMM reads the topology
correctly. ``mapping.json`` records the original numbers, and all per-pair state
(offsets, visits, outcomes) is stored under those original numbers, so a pair keeps its
history across every combination it appears in.

For a large site, the cross-site product is far larger than what training needs, which is
mostly pairs. :ref:`Combo Sets` select explicit combinations instead.

Lazy Directory Creation
~~~~~~~~~~~~~~~~~~~~~~~

A large system can have tens of thousands of combinations, of which training runs a small
fraction. Generation therefore only lists them, in ``combo_metadata.json`` (name, path,
sites and substituents) and the manifest files, and a combination's directory is created
the first time it is needed for training, validation or uncertainty scoring. Disk use then
scales with the combinations actually run. Manifests store names rather than paths
(``{out_dir}/{combo_name}``), so a split can be reused exactly by later runs.

Directory Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   generated_combos/
   ├── combo_metadata.json                  # metadata for all combinations
   ├── manifest.txt                         # all combination names
   ├── train_manifest.txt / val_manifest.txt / test_manifest.txt
   ├── comb_0001_site2_1__site2_2/          # created on demand
   │   ├── info.py                          # system configuration
   │   ├── mapping.json                     # file renumbering
   │   ├── msld_flat.py                     # simulation script (copied)
   │   └── prep/
   │       ├── site2_sub1_pres.rtf, site2_sub1_frag.pdb
   │       ├── site2_sub2_pres.rtf, site2_sub2_frag.pdb
   │       └── core.rtf, core.prm, ...
   └── comb_0262_site1_1__site1_2__site2_1__site2_2/   # cross-site; keeps site numbering

``info.py`` holds the system configuration read by the simulation script:

.. code-block:: python

   info = {}
   info['name'] = 'comb_0262_site1_1__site1_2__site2_1__site2_2'
   info['nsubs'] = [2, 2]                    # substituents per site
   info['nblocks'] = np.sum(info['nsubs'])
   info['ncentral'] = 0                      # central replica
   info['nreps'] = 1
   info['nnodes'] = 1
   info['enginepath'] = os.environ.get('CHARMMEXEC', '')
   info['temp'] = 298.15                     # K

``mapping.json`` records each file's original and new numbers:

.. code-block:: json

   [{"original": "/path/to/site1_sub2_pres.rtf", "new_name": "site1_sub1_pres.rtf",
     "original_site": 1, "original_sub": 2, "new_site": 1, "new_sub": 1}]

.. _Staged Online Training:

Staged Online Training
----------------------

Why Pairs Come First
~~~~~~~~~~~~~~~~~~~~

Production simulates a *group* of substituents at once, but a group run is a poor source
of corrections. When it fails, the trajectory rarely shows which pair's biases were wrong,
and each attempt costs the full simulation length. A *pair* run isolates one pair, so its
λ trajectory measures that pair's linear (b) and quadratic (c) error directly
(:doc:`cb_setup`), and a run of ∼100 ps is often enough to see it. The workflow therefore
resolves pairs first and runs each group only once the pairs it depends on are known.

With ``curriculum.per_system: true``, each system follows this schedule on its own. A
system the model already predicts well finishes its pairs in a few epochs, while a poorly
predicted one keeps running pairs, and every epoch's batch mixes systems at different
points. Progress is kept in ``<output.base_dir>/system_stages.json``, so a resumed run
continues each system where it stopped.

Terms
~~~~~

The following terms are used throughout this section and in the code
(``mllf.cb.system_stages``, ``mllf.cb.hindsight``).

* **Group**: a set of substituents simulated together in production, usually one of the
  system's iterative-flattening groups (:ref:`Combo Sets`). Its first substituent at each
  site is the group's **reference**.
* **Reference pair**: the group's reference paired with one of its other substituents.
  Every substituent is connected to the reference by one such pair, so these pairs fix
  the group's b values.
* **Good run** of a pair: both substituents sampled, at least a minimum number of
  crossings between them, and at least a minimum fraction of the pair's population on
  the minority substituent. Full-length runs need 10 crossings and 0.2. Short runs use the
  stage's ``explore_min_transitions`` and ``explore_min_fraction``.
* **Resolved pair**: a pair with ``advance.successes`` good runs in the stage, *and* an
  offset posterior sd of at most ``advance.max_offset_sd`` for each listed coefficient.
  Good runs show that the current biases sample, and the narrow offset shows that the
  pair's correction is known rather than lucky. A pair whose scouting run passed the
  full-length test is also resolved. Resolved pairs leave the pool.
* **Retired pair**: a pair that has run in a set number of the stage's epochs without
  resolving. It leaves the pool and no longer holds back its groups.
* **Group test**: every substituent at each site holds at least
  ``min_population_ratio / n_subs`` of the site's population, and each site has at least
  ``min_site_transitions`` transitions, pooled over the group's last
  ``group_pass_window`` runs. A group that passes is retired.

Example
~~~~~~~

The stage below holds both pairs and groups. It is the stage of
``training/online_1.yaml``, with the system list shortened; keys it leaves out take the
hindsight defaults.

.. code-block:: yaml

   alf_root: /path/to/pretraining

   systems:
     - name: luis_bace_solvent
       input_dir: training/systems/luis_bace_solvent
       out_dir: training/simulations_online_1/luis_bace_solvent/
       solvent_state: solv
       include_patterns: [msld_flat.py]
       combo_sets: {ref_pairs: alf_group_ref_pairs, group_pairs: alf_group_pairs,
                    groups: alf_groups}
       export_combo_set: groups

   curriculum:
     enabled: true
     per_system: true
     max_train_combos_per_stage: 500      # pairs + groups per epoch, all systems
     min_combos_per_system: 12
     combos_per_system: reference_pairs
     stages:
       - name: pairs_groups
         combo_set: group_pairs           # every within-group pair
         reference_set: ref_pairs
         reference_pair_bonus: 1.0
         reference_priority: strict
         retire_after_epochs: 5
         retire_requires_success: reference
         retire_min_successes: 0
         reference_retire_after_epochs: 10
         scouting: training/scouting/oct27
         prioritize_unvisited: false
         run_steps: {equil: 13000, prod: 39000}
         bracketed_run_steps: {equil: 13000, prod: 112000}
         confirm_full_length: false
         explore_min_transitions: 5
         explore_min_fraction: 0.1
         group_set: groups
         group_ready_fraction: 0.67
         group_min_resolved_per_site: 1
         group_max_failures: 3
         epochs: 25
         advance:
           type: groups_done
           fraction: 1.0
           successes: 2
           max_offset_sd: {linear: 1.0, quadratic: 3.0}
           min_population_ratio: 0.0      # group test: every substituent sampled
           min_site_transitions: 0

   validation_per_system: 4
   eval_schedule: stage

A system in this stage runs ∼104 ps pair runs, reference pairs first, until two thirds
of a group's reference pairs are resolved. That group then runs at full length, and the
system is done once every group has passed.

What Happens Each Epoch
~~~~~~~~~~~~~~~~~~~~~~~

For each system that has not finished, the driver carries out six steps.

1. **Builds the pool** from the stage's combos (``combo_set``, or the manifest filtered by
   the stage's site and substituent limits), less held-out, resolved and retired pairs.
   A stage that leaves a system no combos is skipped for that system.
2. **Adds ready groups.** A group joins the pool once ``group_ready_fraction`` of its
   reference pairs are resolved (retired ones are left out of the count) with at least
   ``group_min_resolved_per_site`` resolved at every site, and stays until it passes the
   group test.
3. **Sets each system's share** of ``max_train_combos_per_stage``. With
   ``combos_per_system: reference_pairs``, the share is proportional to the system's
   number of reference pairs, with ``min_combos_per_system`` as the floor, so a system
   with 50 substituents covers its reference pairs as fast as one with four. The epoch
   grows if the floors add up to more than the budget.
4. **Orders the batch.** Ready groups go first. With ``reference_priority: strict``,
   unresolved reference pairs and pairs reopened by a failed group fill each system's
   share next, ordered by pairs not yet run, then by scouting tier (hardest first), then
   by selection rank. A reference pair whose run never crossed also runs *reversed* from
   then on, with λ starting on its other substituent, so the endpoint at which a barrier
   holds λ reflects b rather than the starting state. The reversed partner
   (``pair07r_site1_subs_9_1`` for ``pair07_site1_subs_1_9``) is the same pair, shares
   its offsets and takes a slot of the share. Pairs awaiting a full-length confirming run
   come next, then pairs not yet run (with ``prioritize_unvisited``), then pairs in
   progress (``bandit.revisit``, :doc:`cb_setup`), and finally the rest by the shared
   model's uncertainty plus each pair's own state. ``reference_pair_bonus`` raises a
   reference pair's selection score.
5. **Runs and learns.** Each combo is simulated, its λ trajectory gives hindsight
   targets, and the per-pair offsets and the shared model are updated (:doc:`cb_setup`).
6. **Checks the stage.** A system leaves the stage when its ``advance`` criterion holds,
   no sooner than ``advance.min_epochs`` (default 1), or after ``epochs`` epochs in the
   stage. In the second case it is recorded as *not converged*, and that label follows its
   exported biases.

Run Length
~~~~~~~~~~

``run_steps: {equil, prod}`` (MD steps) sets the length of a stage's pair runs. It is
written into each combination's copy of ``msld_flat.py`` (the ``esteps`` and ``nsteps``
lines), so the system's template is never changed. Group runs are always full length.

A short run has fewer frames and crossings than a full-length one, so a good short run is
weaker evidence. With ``confirm_full_length: true`` (default), a good short run earns the
pair one full-length confirming run in the next epoch, and only full-length runs count
toward ``successes``. With ``false``, good short runs count directly, which is reasonable
when the short-run thresholds already require several crossings and a later group run
tests the same substituents again.

Two settings handle pairs that cross slowly rather than not at all. A pair whose b has
been pushed both ways (its b bracket has closed, :doc:`cb_setup`) is near its target, so a
failure is more likely a crossing time too short for the barrier than a wrong bias. Such
a pair runs ``bracketed_run_steps`` instead of ``run_steps``, and with
``bracketed_runs_sticky`` it keeps the longer runs until it resolves or retires.
``partial_runs_to_resolve`` short runs with at least ``partial_min_transitions`` crossings
(and the short-run minority fraction) also resolve a pair, in place of
``advance.successes`` good runs. The offset sd limits still apply.

.. _Retiring Pairs:

Retiring Pairs
~~~~~~~~~~~~~~

A pair that never crosses would hold its groups back for the whole stage, although a
group run can still sample its substituents through their other pairs. With
``retire_after_epochs``, a pair retires once it has run in that many of the stage's
epochs without resolving. ``retire_requires_success`` sets how many good runs a pair
needs before it may retire. With ``reference``, reference pairs retire after
``reference_retire_after_epochs`` once they have ``retire_min_successes`` good runs
(default 1), and other pairs retire after ``retire_after_epochs`` with none. ``true``
requires one good run of every pair, and ``false`` (default) none.

A group that fails ``group_max_failures`` runs goes back to the pair phase. Its pairs with
a substituent its failed runs left unsampled are reopened (they must resolve again, with a
fresh retirement allowance), and the group waits for them. A run in which the group
correction lowered the group's end terms (:doc:`cb_setup`) does not count as a failure,
since it tested values that have since changed.

Scouting
~~~~~~~~

A stage's ``scouting:`` directory holds one ∼1 ns run per ALF group with the starting
model's biases (``training/scouting/scout_groups.py``), and is applied once, before the
system's first epoch in the stage. A group whose scouting run sampled every substituent
has passed (its scouting run is the exported run), a pair whose scouting run passed the
full-length test is resolved, and the scouting runs' simulated time enters the cost
ledger. The directory's ``pair_priority.json`` (or the stage's ``scout_priority`` file)
ranks the reference pairs by how hard scouting found them, from tier 0 (a substituent
never sampled, or fewer than 5 crossings) to tier 2 (10 or more crossings). Pairs that
scouting did not cover count as tier 1. Scouting runs must come from the same starting
model as the stage, so each stage that uses scouting needs its own.

Stage Settings
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Meaning
   * - ``epochs``
     - Most epochs a system spends in the stage.
   * - ``systems``
     - Systems that enter this stage; others skip it. Omit for all systems.
   * - ``min_sites`` / ``max_sites`` / ``min_subs_per_site`` / ``max_subs_per_site``
     - Manifest filters, ignored when ``combo_set`` is given.
   * - ``combo_set``
     - The named combo set to run in place of the filtered manifest (:ref:`Combo Sets`).
   * - ``reference_set``
     - The combo set whose pairs are the reference pairs. Without it, every pair in the
       pool counts as a reference pair.
   * - ``reference_pair_bonus``
     - Added to a reference pair's selection score (default 0).
   * - ``reference_priority``
     - ``strict`` fills each system's share with unresolved reference pairs and reopened
       pairs first, with reversed partners for pairs that got stuck.
   * - ``retire_after_epochs`` / ``reference_retire_after_epochs``
     - Epochs run without resolving after which a pair (a reference pair) retires. Omit to
       never retire.
   * - ``retire_requires_success`` / ``retire_min_successes``
     - Good runs a pair needs before it may retire (:ref:`Retiring Pairs`).
   * - ``scouting`` / ``scout_priority``
     - Scouting results applied at stage entry, and an optional separate tier file.
   * - ``prioritize_unvisited``
     - Run pairs not yet run in this stage ahead of selection (default true).
   * - ``run_steps`` / ``bracketed_run_steps``
     - ``{equil, prod}`` MD steps for pair runs, and for pairs whose b bracket has closed.
   * - ``bracketed_runs_sticky``
     - Keep a pair on longer runs until it resolves or retires.
   * - ``confirm_full_length``
     - Require a full-length confirming run before a good short run counts (default true).
   * - ``explore_min_transitions`` / ``explore_min_fraction``
     - Good-run thresholds for short runs (defaults 2 and 0.2).
   * - ``partial_runs_to_resolve`` / ``partial_min_transitions``
     - Slow-crossing runs that resolve a pair, and the crossings each needs (default 3).
       0 turns this off.
   * - ``group_set``
     - The combo set of groups that join the pool once ready.
   * - ``group_ready_fraction`` / ``group_min_resolved_per_site``
     - Share of a group's reference pairs that must be resolved (default 1.0), and the
       fewest resolved at each site (default 0).
   * - ``group_pass_window``
     - Recent group runs pooled in the group test; the run that sampled the most
       substituents is exported.
   * - ``group_max_failures``
     - Failed runs after which a group goes back to the pair phase. Omit to keep running
       it.
   * - ``advance``
     - When a system may leave the stage early (below).

Keys that set an entire epoch's training in the global curriculum (``update_policy``,
``reward_override``, ``policy_lr``, ``bc_mode``) do not apply in per-system mode, where
one epoch holds systems at different stages.

Advance Criteria
~~~~~~~~~~~~~~~~

``advance.type`` selects the criterion, ``fraction`` (default 1.0) is the share of pairs,
groups or combos that must meet it, and ``min_epochs`` is the earliest epoch at which it
is checked.

* ``pairs_resolved``: ``fraction`` of the system's pairs are resolved (``successes``,
  default 2, and ``max_offset_sd``). ``scope: pool`` (default) counts every pair in the
  pool, and ``scope: visited`` counts only pairs run in this stage, which lets a stage end
  without running every pair of a large site.
* ``groups_done``: ``fraction`` of the system's groups have passed the group test
  (``min_population_ratio``, default 0.5, and ``min_site_transitions``, default 10).
  ``successes`` and ``max_offset_sd`` still define resolved pairs, which decide when a
  group is ready.
* ``combos_balanced``: the latest run of ``fraction`` of the stage's combos (at least
  ``min_combos``) passes the group test, for a stage of multi-substituent combos without
  explicit groups.
* ``epochs`` (default): the system stays for the full ``epochs``.

.. _Combo Sets:

Combo Sets
~~~~~~~~~~

Named **combo sets** let a stage run exactly the combinations that matter
(``mllf.file_handling.combo_sets``). The built-in sets come from each system's
iterative-flattening groups (``<alf_root>/<system>_group*``), so the online run and the
baseline simulate the same substituent sets and their costs compare directly:

.. code-block:: yaml

   systems:
     - name: luis_bace_solvent
       ...
       combo_sets:
         groups: alf_groups               # one combo per group
         group_pairs: alf_group_pairs     # every within-site pair of those groups
         ref_pairs: alf_group_ref_pairs   # each group's reference vs its other subs
         # explicit groups ({site: [subs]}, first sub = reference) and pairs built
         # from them, for a system without iterative-flattening runs:
         # mine: [{1: [3, 2], 2: [15, 6, 7]}]
         # mine_ref_pairs: {pairs_of: mine, reference_only: true}
       export_combo_set: groups           # production biases per group

Group fragments are matched to the system's by content (source ligand, atom names, types
and charges) rather than by file name, because a group may be numbered locally
(``site1_sub1..n``). A site with a single substituent in a group is a fixed substituent
rather than a perturbation and is dropped. A system with no ``_group*`` datasets uses its
single dataset as one group.

.. _Held-Out Validation:

Held-Out Validation
~~~~~~~~~~~~~~~~~~~

Validation tests whether the *shared model* predicts biases for pairs it has never been
trained on. Held-out pairs are therefore never trained on, and validation runs use the
model's posterior mean with neither Thompson sampling nor per-pair offsets.

``validation_per_system: N`` holds out N pairs per system. When the first stage runs a
combo set, they are drawn from manifest pairs in none of the system's sets, so no stage
ever runs them (a system whose groups cover every pair has none), and
``validation_per_system: 0`` holds out nothing. Without ``validation_per_system``,
``validation_fraction`` (default 0.1) of a pool of more than 20 combos is held out.
``validation_combos_file`` reuses another run's
``validation_combos.json`` exactly; otherwise the seeded draw changes whenever the combo
pool does.

Validation runs every ``eval_freq`` epochs (default 1) and after the last epoch. With
``eval_schedule: stage``, each system is validated instead at the first, middle and last
epoch of each of its stages and in the epoch it meets a stage's criterion. The set of
systems validated then differs from pass to pass, so a score is comparable only to
earlier passes of the same system.

.. _Production Biases:

Production Biases
~~~~~~~~~~~~~~~~~

When a system finishes its last stage, its production biases are written to
``<output.base_dir>/final_biases/<system>/``. They are the shared model's prediction plus
each pair's offset posterior mean, with no sampling, the c cap and the group's corrected
end terms applied. b is fitted per site from every within-site edge, weighted by each
pair's linear-offset variance.

* With ``export_combo_set``, one ``variables.py`` is written per combo of that set
  (``final_biases/<system>/<group>/``), matching how production is run.
* Otherwise, one export covers every substituent in the system's ``input_dir``, or the
  subset in ``export_combo_dir``.

Each exported group also gets ``CONFIDENCE.txt`` and ``confidence.json``, which record
whether the group passed its group test and, for each of its pairs, whether the pair was
resolved, run but not resolved, or never run (with visits, good runs and the b and c
offset uncertainty). A group that never passed is renamed ``<group>_unconverged``. Its
biases are the best available estimate but have not been shown to sample, so they should
be checked before production.

Cross-site couplings are not predicted and are exported as 0. The exported biases are
therefore complete for groups whose perturbed substituents share one site, and a starting
point for multi-site groups.

``curriculum.export_final_biases: false`` skips the export. To export or label a finished
run:

.. code-block:: bash

   mllf export config.yaml --model final_model.pt \
       --bandit-state <output.base_dir>/bandit_state.json --out final_biases
   mllf export config.yaml --label <output.base_dir>

.. _Testing on New Systems:

Testing on New Systems
~~~~~~~~~~~~~~~~~~~~~~

``mode: deploy`` runs a trained model through the same per-system stages on systems it
was not trained on:

.. code-block:: yaml

   mode: deploy
   deploy:
     model_path: checkpoints_online_2/final_model.pt
     update_shared_model: false   # per-pair offsets still learn
   output:
     base_dir: checkpoints_online_3   # must not be the model's directory

Nothing is held out, because every pair of a test system is part of the test, and the
model file is never overwritten. With ``update_shared_model: false``, the cost recorded
for a test system is what a user of the trained model would pay, namely the model's first
guess plus the per-pair corrections that system needed.

.. _Simulation Cost:

Simulation Cost
~~~~~~~~~~~~~~~

The comparison that matters is simulated time to converged biases, so both methods are
measured in nanoseconds in the same way. Each finished run's simulated time is added to
``<output.base_dir>/cost_ledger.json`` by system, stage, and training or validation
(``mllf.cb.cost_ledger``). The time is read from the dynamics calls in the CHARMM output
and includes equilibration, since that is compute spent. The iterative-flattening cost of
the same systems is read the same way from the run outputs named in the pretraining
datasets:

.. code-block:: bash

   mllf cost <output.base_dir> --pretraining-root pretraining

The comparison is like for like when both methods ran the same substituent groups (the
ALF-derived combo sets), validation runs are reported separately (they test the model and
are not needed to produce biases), and the online cost is that of the run that produced
the exported biases. A system trained on in ``train`` mode has also shaped the shared
model, so a cost measured in ``deploy`` mode is the cleaner estimate for a new system.

Simulation Execution
--------------------

.. _Launching Simulations:

Launching Simulations
~~~~~~~~~~~~~~~~~~~~~

Each combo's biases are written to ``variables.py``, and its simulation is submitted as one
SLURM job per combo per epoch (``mllf.cb.slurm.submit_epoch_simulation``). The job script
is built from the config's ``slurm:`` block (``mllf.slurm``), where every key becomes an
``#SBATCH`` option, so any sbatch setting can be used without code changes:

.. code-block:: yaml

   slurm:
     partition: gpu2080
     gres: gpu:1
     time: '02:00:00'
     exclude: [gollum003, gollum007]     # or 'gollum[003,007]'
     exclude_file: bad_nodes.txt         # re-read at every submission
     qos: normal
     mem: 8G
     mail_type: FAIL                     # underscores become dashes: --mail-type=FAIL
     requeue: true                       # true writes a bare flag
     module: charmm/charmm/c51a1         # one or a list: 'module load' lines
     setup:                              # shell lines run before the simulation
       - source ~/miniconda3/etc/profile.d/conda.sh
       - conda activate charmm
     extra: ['--signal=B:USR1@60']       # raw directives, written as given

* A list value is joined with commas; ``null`` or ``false`` leaves an option out.
* ``ntasks: 1``, ``cpus_per_task: 1``, ``export: ALL`` and ``time: '01:00:00'`` are the
  defaults; set a key to change one, or to ``null`` to drop it. Nothing else is assumed,
  so with no ``slurm:`` block the script asks for no partition, GPU or module.
* ``job_name`` and ``output`` are set for each job and cannot be configured.
* ``exclude_file`` names a file of node names (whitespace, comma or line separated, ``#``
  comments). It is read every time a job is submitted or resubmitted, so adding a node to
  it keeps the next jobs of a running training off that node without restarting it.

The same block sets the ``run.sh`` written into each combination directory for submitting
it by hand (``mllf combos --slurm-config workflow.yaml``).

At most ``max_concurrent_jobs`` (default 10) run at once. A failed run is resubmitted up
to ``max_retries`` times (default 2), and a combo that still fails is replaced from a
reserve of unused combos. ``max_failed_combos_per_epoch`` (default 50) permanent failures
stop training, since that many usually means a broken setup. A finished job's output is
parsed for populations, transitions, pair profiles and simulated time
(:doc:`file_handling`).

.. _Reward Function:

Reward Function
---------------

The ``reward:`` block sets a *run reward*, one number per simulation summarizing how well
it sampled. It is reported as the epoch's average reward, decides ``progression: reward``
in the global curriculum, and weights runs in pretraining (:doc:`cb_pretraining`).
Hindsight updates do not use it (:doc:`cb_setup`). The ``barrier_*``, ``skew_width``,
``end_width`` and ``no_transition_weight`` keys belong to the per-edge reward described
there.

The run reward is built so that a degenerate run, such as one that settles into a single
substituent, scores below any run that samples:

.. math::

   R = \left(\frac{N_{\text{visited}}}{N_{\text{subs}}}\right)^2 (R_P + R_T + R_{\text{entropy}}) - \min(60,\; P_{\text{trans}} + P_{\text{conc}})

* :math:`R_P = w_P \, C_F \sum_{k\,\text{visited}} p_k / P_{\text{baseline}}` rewards
  population, discounted by :math:`C_F = \min(1, T_{\min}/2N_{\text{req}})` because
  populations from runs with few transitions are not yet reliable.
* :math:`R_T = w_T \sum_s T_s / T_{\text{baseline}}` if every site reaches
  :math:`N_{\text{req}}` = ``min_transitions_per_site`` transitions (×1.5 above twice that
  on average), and 0 otherwise.
* :math:`R_{\text{entropy}}` is ``entropy_bonus`` times the normalized Shannon entropy of
  the populations.
* :math:`P_{\text{trans}}` is set by the worst site, so one failing site cannot hide
  behind good ones: 40, 32 or 24 for :math:`T_{\min}` = 0, 1 or 2, then
  :math:`2 + 2(N_{\text{req}} - T_{\min})` up to :math:`N_{\text{req}}`, plus 4 per
  additional failing site.
* :math:`P_{\text{conc}}` adds :math:`5\gamma` times the excess for each site where one
  substituent holds more than 0.8 of the population.

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
     no_transition_weight: 0.2

.. _Global Curriculum:

Global Curriculum
-----------------

Without ``per_system``, all systems move through the stages together: every epoch draws
from one stage, and the stage changes for every system at once. This suits REINFORCE
training and small single-system runs such as the 14benz example. Because the slowest
system sets each stage's length, systems the model already predicts well keep spending
simulations, which is why staged training uses per-system mode.

Stages typically run from single-site pairs, through larger single-site combinations, to
multi-site combinations, since smaller combinations expose their own errors more directly:

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

* ``min_subs_per_site`` / ``max_subs_per_site``: bounds on the substituents at any one
  site (checked per site, not against the combination's total)
* ``min_sites`` / ``max_sites``: the number of sites represented
* ``combo_whitelist``: explicit combo names, bypassing the size filters
* ``min_transitions_for_stage`` / ``transition_filter_lookback``: on entering a later
  stage, keep only combos whose recent runs reached this many transitions per site
* ``epochs``, and ``max_train_combos`` to override ``max_train_combos_per_stage``
* ``reward_override``: per-stage reward weights, e.g., a lower
  ``min_transitions_per_site`` while the policy is still learning pairs

Each epoch draws up to ``max_train_combos_per_stage`` of the stage's combos, stratified
across systems (``curriculum.min_combos_per_system``) or by uncertainty with
``bandit.selection: uncertainty``, seeded by ``seed`` plus the epoch. The draw is uniform
over the combos that pass the filter, so a stage allowing pairs and triplets draws mostly
triplets, in proportion to how many of each exist.

``progression.type`` sets when the stage changes: ``epoch`` (default) after the stage's
epochs, ``reward`` once the average reward over the last 5 epochs reaches
``reward_threshold``, and ``both`` when both hold.

Checkpointing and Resume
------------------------

Long runs are interrupted by SLURM time limits, maintenance or cancellation, so the driver
saves enough state to continue from the last completed epoch.

.. code-block:: yaml

   output:
     base_dir: /path/to/training_output
     checkpoint_freq: 1         # every N epochs (default 5)

``base_dir`` then holds:

* ``checkpoint_{epoch:03d}.pt``: the policy and optimizer state, the epoch and that
  epoch's statistics
* ``bandit_state.json``: the per-pair state (offset targets, visits, outcomes, exploration
  factors, b and c brackets, group corrections) and per-combo failure counts. The per-pair
  corrections depend on it, so a run resumed without it starts every pair over.
* ``system_stages.json``: each system's stage, epochs spent and progress
* ``validation_combos.json``: the held-out set, so a resumed run validates on the same
  pairs
* ``regression_online_buffer.pt``: the recent hindsight targets the regression trains on
* ``cost_ledger.json``: simulated time per run
* ``epoch_summary.log``: one block per epoch
* ``training.log``: every message of the run with time stamps, including the per-combo
  detail the console leaves out (:doc:`cli`)
* ``best_model.pt``: the best validated model (with the validation gate)
* ``final_model.pt``: written at the end, from the best validated model when the last
  validation scored lower

On restart, the driver loads the latest ``checkpoint_*.pt`` and continues from the next
epoch, falling back to ``pretrain.model_path`` if that checkpoint cannot be loaded. Each
combo's Uni-Mol embeddings and graph tensors are saved to ``graph_data.pt`` in its run
directory before submission, so the update after the SLURM wait reloads them rather than
recomputing them or holding every combo in memory.

Archiving Combinations
----------------------

A long run leaves one directory per combination and one subdirectory per run in each
system's ``out_dir``, often tens of thousands of files per system. The ``archive:`` block
packs each system's ``out_dir`` into one ``.tar.gz``:

.. code-block:: yaml

   archive:
     when: end          # end: after training; system: as each system finishes its last
                        # stage (per-system curricula), the rest at the end
     dir: archive       # relative to output.base_dir (default), or absolute
     remove: false      # delete the combination directories once archived

Each archive is written under a temporary name and renamed once it is complete, and with
``remove: true`` a system's directories are deleted only after its archive has been read
back and found to hold every file. ``when: system`` is safe in a per-system curriculum
because a finished system's directories are no longer read: its held-out pairs are
validated in the epoch it finishes, before its biases are exported, and its pairs leave the
pool. Analysis that reads run directories directly (for example the ``variables.py`` of
each run) needs the archive extracted when ``remove`` is on.

``mllf archive CONFIG`` does the same for a finished or stopped run (``--systems`` to limit
it, ``--dir`` and ``--remove`` to override the block), and ``mllf combos --archive`` packs
each generated combination directory separately (``archive_combo_dirs``).

.. _Example Configurations:

Example Configurations
----------------------

* ``examples/workflow_14benz.yaml`` is the smallest working config (one system,
  REINFORCE, global curriculum, no pretrained model), walked through in :doc:`examples`.
* ``training/online_1.yaml`` through ``online_4_hne_pr3.yaml`` are the staged curriculum
  for production biases: known systems in solvent and protein (stage 1), the hardest
  systems (1b), vacuum legs (2), and deployment on chemistry and proteins the model was
  not trained on (3 and 4). Each header states its systems and starting model.

See Also
--------

* :doc:`cb_setup`: the policy, hindsight targets and per-pair offsets
* :doc:`unimol_representation`: the Uni-Mol node features
* :doc:`cb_pretraining`: behavior cloning from completed simulations
* :doc:`file_handling`: file formats and parsers
* :doc:`examples`: running the examples
