Workflow System
===============

Overview
--------

The workflow system automates the complete pipeline for multisite λ-dynamics
simulations with contextual bandit training:

1. **Combination Generation**: Create all valid site/substituent combinations
2. **Splitting**: Divide combinations into training/validation/test sets
3. **Training**: Train CB policy on graph structures
4. **Simulation**: Run MD simulations with optimized bias coefficients
5. **Compression**: Archive simulation outputs for storage

Running the Workflow
--------------------

Basic Usage
~~~~~~~~~~~

Online (REINFORCE) training is driven by a YAML configuration file and
``examples/run_workflow.py`` — a tracked copy of the multi-system UniMol CB training driver:

.. code-block:: bash

   cd mllf   # repository root
   python examples/run_workflow.py examples/workflow_14benz.yaml
   python examples/run_workflow.py my_config.yaml  # Use a custom config

See :doc:`examples` for a full walkthrough of the 1,4-benzene (14benz) example. Combination
generation and manifest splitting can also be driven directly from
``mllf.file_handling.generate_combinations`` / ``mllf.cli.workflow`` for scripting outside
the full training loop; the low-level ``mllf.cli.workflow.run_from_config`` entry point
handles a lighter-weight subset (combo generation, split, a single-combo sanity check,
optional simulation batch, archiving) without the policy training loop.

Configuration Format
~~~~~~~~~~~~~~~~~~~~

A workflow config is a YAML file specifying which operations to run and their parameters.
Unlike a single ``create_combos:`` block, the top-level key is a **list of systems** —
``systems:`` — even when training on just one system, since the driver is designed to train
one shared policy across any number of systems simultaneously:

* **systems**: List of systems to train on, each with its own ``name``, ``input_dir``,
  ``out_dir``, ``solvent_state``, and optional per-system overrides (e.g.
  ``max_subs_per_site``, ``protein_pdbs``, ``solvent_pdb``)
* **pretrain**: Optional warm-start from a behavior-cloned checkpoint (see :doc:`cb_pretraining`)
* **curriculum**: Progressive training stages (see :ref:`Curriculum Learning`)
* **training**: Policy architecture and optimizer hyperparameters
* **bandit**: Optional NeuralLinear + Thompson Sampling configuration (see :doc:`cb_setup`)
* **reward**: Reward function weights and thresholds
* **output**: Checkpointing and output organization
* **slurm**: Cluster submission settings (partition, gres, time, module)
* **archive**: Automatic compression of completed runs

See the :ref:`Complete Configuration Example` for a full annotated YAML file.

Combination Generation
----------------------

Principles
~~~~~~~~~~

Combinations are generated from site/substituent fragment files:

* **Input files**: ``site{N}_sub{M}_{label}.{rtf,pdb}`` files in the input directory
* **Sites**: Identified by the site number (N)
* **Substituents**: Identified by the sub number (M) within each site

.. warning::
   **Minimum Substituents Required**: Each site must have at least 2 substituents. MSLD simulations 
   will not run correctly with only a single substituent at a site. If any site has only 1 substituent, 
   combination generation will fail with an error. To resolve this, either add more substituents to the site or 
   add the site information to your core structure files (e.g., ``core.pdb`` and ``core.rtf`` if using 
   msld-py-prep).

The generator creates two types of combinations:

1. **Within-site combinations**: Multiple substituents from a single site
2. **Cross-site combinations**: Substituents from multiple sites simultaneously

.. note::
   **Combination Size Limit**: By default, each combination is limited to at most 10 substituents 
   per site (``max_subs_per_site=10``). This prevents combinatorial explosion while still allowing 
   all substituents to participate across different combinations. For example, with 50 substituents 
   at a site, the generator will create combinations like ``[1,2,...,10]``, ``[1,2,...,9,11]``, etc., 
   but not ``[1,2,...,11]``. This limit can be increased via the ``--max-subs`` command-line option 
   or the ``max_subs_per_site`` parameter in the API.

Lazy Directory Creation
^^^^^^^^^^^^^^^^^^^^^^^

For systems with large combination spaces (e.g., 14,211 total combinations),
creating all directories upfront is inefficient—most will never be used in
training. The workflow implements **lazy (on-demand) directory creation**:

**Metadata Generation**: During the combination generation phase, the system:

1. Lists all possible combinations without creating directories
2. Saves metadata to ``combo_metadata.json`` with:
   
   - Combination name (e.g., ``comb_0001_site2_1__site2_2``)
   - Path where directory will be created
   - Sites and substituents included
   - Counter for ordering

3. Writes manifest files listing all possible combinations

**On-Demand Creation**: Directories are created only when needed:

* During training/validation splits, combinations are selected but not created
* When a combination is accessed for training, the workflow:
  
  1. Checks if the directory exists
  2. If not, loads metadata from ``combo_metadata.json``
  3. Creates the directory with all required files
  4. Continues with training

**Benefits**:

* **Disk space efficiency**: Only create ~1-2% of possible combinations (e.g., 
  142 training + 142 validation out of 14,211 total)
* **Faster initialization**: Split generation completes in seconds instead of hours
* **Filesystem efficiency**: Avoid creating thousands of unused directories
* **Scalability**: Handle massive combination spaces (100K+ combinations)

Directory Structure
~~~~~~~~~~~~~~~~~~~

Each combination directory (created on-demand) has a standardized structure:

.. code-block:: text

   generated_combos/
   ├── combo_metadata.json                  # Metadata for all combinations
   ├── manifest.txt                         # List of all combination names
   ├── train_manifest.txt                   # Training combination names
   ├── val_manifest.txt                     # Validation combination names
   ├── test_manifest.txt                    # Test combination names
   ├── comb_0001_site2_1__site2_2/          # Created on-demand
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

**File Naming Convention**: The renaming preserves site identity:

* Files maintain their site number (``site1_*``, ``site2_*``, etc.)
* Substituents are renumbered sequentially within each site
* Original site/sub mapping is preserved in ``mapping.json``

Combination Metadata Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each combination directory contains standardized metadata files:

**info.py**: System configuration loaded by simulation scripts

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

**mapping.json**: File renumbering information

.. code-block:: json

   [
     {
       "original": "/path/to/site1_sub2_pres.rtf",
       "new_name": "site1_sub1_pres.rtf",
       "original_site": 1,
       "original_sub": 2,
       "new_site": 1,
       "new_sub": 1
     },
     {
       "original": "/path/to/site2_sub5_pres.rtf",
       "new_name": "site2_sub1_pres.rtf",
       "original_site": 2,
       "original_sub": 5,
       "new_site": 2,
       "new_sub": 1
     }
   ]

This tracks how original fragment files were renumbered during combination
creation, enabling traceability back to source files.

Manifest Files
~~~~~~~~~~~~~~

Manifest files list combination names (one per line):

.. code-block:: text

   comb_0001_site2_1__site2_2
   comb_0002_site2_1__site2_3
   comb_0003_site2_1__site2_4
   comb_0075_site1_5__site1_1__site1_2
   comb_0262_site1_1__site1_2__site2_1__site2_2
   ...

Manifest files enable reproducible splits and batch operations. The full paths
are constructed by prepending the ``out_dir`` from the configuration:
``{out_dir}/{combo_name}``.

Graph Construction
------------------

During training, molecular graphs are constructed from combination directories to provide
input for the policy network. Nodes are pretrained, frozen Uni-Mol embeddings (see
:doc:`unimol_representation`) representing each substituent's 3D structure, chemistry, and
environment as learned 512D (or 1024D dual) vectors; edges connect every ordered pair of
substituents within the same λ-site.

For complete details on graph construction, node features, edge expansion, and the
``UnimolPolicy`` architecture, see :doc:`cb_setup`.

Training Pipeline
-----------------

System Configuration
~~~~~~~~~~~~~~~~~~~~

Each entry in the top-level ``systems:`` list carries its own ``solvent_state``, since a
single training run can combine systems in different environments:

.. code-block:: yaml

   systems:
     - name: 14benz_solv
       input_dir: examples/14benz
       out_dir: examples/14benz/generated_combos
       solvent_state: solv   # Environment type

**Solvent State**:

Specifies the simulation environment to determine which atoms are included as environment
context when computing each substituent's Uni-Mol embedding:

* ``solv`` or ``solvent``: Water molecules within the environment cutoff (8.0 Å) of the core scaffold
* ``gas`` or ``vacuum``: No environment atoms — core + substituent(+ reference subs from other sites) only
* ``protein``: Protein atoms (and typically crystallographic waters) within the environment cutoff

The environment type determines what molecular context Uni-Mol "sees" when computing the
embedding. For protein systems, including nearby protein atoms naturally encodes
protein-specific interactions into the learned embedding. See :doc:`unimol_representation`
for technical details on environment context, the environment-consensus filtering used to
keep this signal consistent across substituents at a site, and the periodic-boundary
handling used for solvated/crystal systems.

The solvent state is also preserved in ``graph_info.json`` for metadata tracking.

**Auto-Detection** (legacy):

Previously, the system attempted to auto-detect solvent state from directory names
(e.g., ``14benz_solv`` → ``solv``). This is now deprecated in favor of explicit 
configuration for clarity and reliability.

Reward Function
~~~~~~~~~~~~~~~

**Pretraining**

Before training begins, the policy can be pretrained using behavior cloning
(AWR NLL loss) to imitate successful bias coefficients from completed simulations.
For complete details on pretraining loss, data organization, and transfer learning
strategies, see :doc:`cb_pretraining`.

**Training Reward**

During training, the policy is optimized using REINFORCE with rewards computed from simulation trajectories.
The reward function prevents degenerate solutions (e.g., convergence to single-substituent states)
through multiple components:

.. math::

   R_{\text{total}} = coverage\_factor \times (R_P + R_T + R_{\text{entropy}}) + R_{\text{penalties}}

where :math:`coverage\_factor = \left(\frac{N_{\text{visited}}}{N_{\text{subs}}}\right)^2` is a smooth
quadratic multiplier that scales all positive reward components by coverage. At 100% coverage it is 1.0;
at 50% it is 0.25; at 0% it is 0.0. This replaces the earlier hard completeness gate
(which clipped all positive reward to −0.01 when any substituent was unvisited) with a
smooth gradient signal that rewards partial progress.

This eliminates :math:`R_U` (the explicit uniformity term) and the adaptive
coverage penalty :math:`P_{\text{cov}}` — both are now subsumed by
:math:`coverage\_factor`.

**Population Balance Reward** :math:`R_P`:

Encourages equal sampling across all substituents with balanced populations:

.. math::

   R_P = w_P \cdot \frac{\sum_{k \in \text{visited}} p_k}{P_{\text{baseline}}} \cdot C_F

where:

* :math:`w_P` is the population weight (default: 0.5)
* :math:`p_k` is the population count for visited substituent :math:`k`
* :math:`P_{\text{baseline}}` is the normalization constant (default: 500.0)
* :math:`C_F = \min(1.0, T_{\min} / (2 \times N_{\text{req}}))` is the confidence factor
* :math:`T_{\min}` is the minimum transitions across all sites
* :math:`N_{\text{req}}` is the minimum required transitions per site (default: 10)

The confidence factor scales population rewards based on data reliability, reducing
false rewards from low-transition runs with unreliable population distributions.
Within-visited uniformity is now captured entirely by :math:`R_{\text{entropy}}`
(see below) rather than by the balance factor :math:`e^{-CV}` which has been removed.

**Transition Reward** :math:`R_T`:

Rewards frequent transitions between substituents, with bonus for high transition counts:

.. math::

   R_T = \begin{cases}
   w_T \cdot \frac{\sum_{s=1}^{N_{\text{sites}}} T_s}{T_{\text{baseline}}} & \text{if all sites have } \geq \text{min_transitions_per_site} \\
   w_T \cdot \frac{\sum_{s=1}^{N_{\text{sites}}} T_s}{T_{\text{baseline}}} \times 1.5 & \text{if avg. trans/site} > 2 \times \text{min_transitions_per_site} \\
   0 & \text{otherwise (sites below threshold)}
   \end{cases}

where:

* :math:`w_T` is the transition weight (default: 0.75)
* :math:`T_s` is the transition count for site :math:`s`
* :math:`T_{\text{baseline}}` is the normalization constant (default: 50.0)
* The 1.5× bonus applies when average transitions per site exceeds 20 (2× the default minimum)

**Entropy Bonus** :math:`R_{\text{entropy}}`:

Rewards uniform population distributions using normalized Shannon entropy:

.. math::

   R_{\text{entropy}} = \alpha_{\text{entropy}} \cdot \frac{H(\mathbf{p})}{H_{\max}}

where :math:`H(\mathbf{p}) = -\sum_k \frac{p_k}{P_{\text{total}}} \log \frac{p_k}{P_{\text{total}}}` 
is Shannon entropy and :math:`H_{\max} = \log(N_{\text{subs}})` is maximum possible entropy.

**Tiered Transition Penalties** :math:`R_{\text{penalties}}`:

The penalty system uses three tiers based on the worst-performing site, with 
multi-site awareness to fairly handle systems with multiple λ-sites:

**Base Penalty** (determined by :math:`T_{\min}`, the minimum transitions across all sites):

.. math::

   P_{\text{base}} = \begin{cases}
   40.0 & \text{if } T_{\min} = 0 \quad \text{(Tier 1: "Death Floor")} \\
   32.0 & \text{if } T_{\min} = 1 \\
   24.0 & \text{if } T_{\min} = 2 \\
   2.0 + 2.0(N_{\text{req}} - T_{\min}) & \text{if } 3 \leq T_{\min} < N_{\text{req}} \quad \text{(Tier 2: "Climbing Ramp")} \\
   0.0 & \text{if } T_{\min} \geq N_{\text{req}} \quad \text{(Tier 3: "Success Zone")}
   \end{cases}

**Multi-Site Degradation** (incremental penalty for multiple failing sites):

.. math::

   P_{\text{trans}} = \begin{cases}
   P_{\text{base}} + 4.0(n_{\text{bad}} - 1) & \text{if } n_{\text{bad}} > 1 \\
   P_{\text{base}} & \text{if } n_{\text{bad}} = 1 \\
   0 & \text{if } n_{\text{bad}} = 0
   \end{cases}

where :math:`n_{\text{bad}} = |\{s : T_s < N_{\text{req}}\}|` counts sites below threshold.


**Concentration Penalty** (per-site check for single-substituent dominance):

.. math::

   P_{\text{conc}} = \sum_{s=1}^{N_{\text{sites}}} \mathbb{1}\left[\frac{\max_k p_{s,k}}{\sum_k p_{s,k}} > 0.8\right] \cdot \gamma \cdot 5.0 \cdot \left(\frac{\max_k p_{s,k}}{\sum_k p_{s,k}} - 0.8\right)

Total penalties are summed and clamped: :math:`R_{\text{penalties}} = -\min(60.0, P_{\text{trans}} + P_{\text{conc}})`

**Default Hyperparameters**:

.. code-block:: yaml

   reward:
     w_P: 0.5                                # Population weight
     w_T: 0.75                               # Transition weight
     w_U: 0.3                                # Accepted for API compatibility; coverage handled by coverage_factor
     gamma: 4.0                              # Base penalty coefficient
     P_baseline: 500.0                       # Population normalization
     T_baseline: 50.0                        # Transition normalization
     min_transitions_per_site: 10            # Tier 3 threshold
     min_coverage_ratio: 0.5                 # Accepted for API compatibility; coverage handled by coverage_factor
     entropy_bonus: 8.0                      # Entropy bonus coefficient
     concentration_penalty_threshold: 0.8    # Single-substituent dominance threshold

**Policy Gradient Training**:

The policy is optimized using **REINFORCE** with per-edge per-dimension reward signals
(or, with ``bandit.algorithm: neurallinear_ts`` in the config, by folding rewards directly
into each head's Bayesian posterior instead — see :doc:`cb_setup`). ``UnimolPolicy`` samples
bias coefficients from independent Gaussian distributions (one per bias type per edge), runs
the MSLD simulation, and then updates weights by maximising the log-probability of actions
that produced positive rewards. Each ``BiasHeadMLP`` receives gradient only from the reward
dimension that corresponds to its bias type, providing clean per-type credit assignment
without a value network or Q-critic.

For architectural details on the policy network and per-dimension reward signals, see :doc:`cb_setup`.

Simulation Execution
--------------------

Launching Simulations
~~~~~~~~~~~~~~~~~~~~~

Simulations are launched via subprocess, running CHARMM with bias coefficients 
written to ``variables.py`` from the policy's sampled actions. The simulator 
outputs transition counts and population distributions for reward computation.

Output Parsing
~~~~~~~~~~~~~~

After simulation completes, the framework parses ``output.txt`` from the output 
directory to extract:

* Total transitions per site :math:`T_s` for each λ-site
* Per-substituent populations :math:`p_{s,k}` at each site
* Coverage ratio (fraction of substituents visited)
* Per-site concentration (maximum population fraction at each site)

These metrics feed directly into the reward function components described 
in the Reward Function section above.

.. _Curriculum Learning:

Curriculum Learning
-------------------


**Curriculum learning** progressively trains the policy on increasingly complex
combinations, similar to how students learn from simple to complex problems.
Instead of training on all possible combinations at once, the policy masters
simpler tasks before advancing to harder ones.

Why Curriculum Learning for MSLD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MSLD bias coefficient optimization has a natural difficulty hierarchy:

**Easy**: Single-site pairs (2 substituents, 1 site)

* Simplest edge interactions to learn
* Clear cause-and-effect relationships
* Provides foundation for pairwise biases

**Medium**: Single-site triplets (3 substituents, 1 site)

* Introduces crowding/density effects
* More complex interaction patterns
* Tests generalization from pairs

**Hard**: Multi-site combinations (2+ sites with multiple substituents each)

* Cross-site interaction effects
* Exponentially larger search space
* Requires composition of learned patterns

Training directly on hard combinations often fails because:

* Reward signals are noisy and unclear
* Policy has no foundation to build upon
* Pretrained weights get overwhelmed by complex gradients

Curriculum learning solves this by building skills incrementally.

Configuration
~~~~~~~~~~~~~

Enable curriculum learning in your workflow YAML:

.. code-block:: yaml

   curriculum:
     enabled: true
     max_train_combos_per_stage: 100  # Optional: limit combinations per stage
     
     stages:
       # Stage 1: Pairs at single sites
       - name: pairs_single_site_easy
         min_subs_per_site: 2
         max_subs_per_site: 2
         min_sites: 1
         max_sites: 1
         epochs: 50
         
       # Stage 2: Triplets at single sites
       - name: triplets_single_site
         min_subs_per_site: 3
         max_subs_per_site: 3
         min_sites: 1
         max_sites: 1
         epochs: 50
         
       # Stage 3: Cross-site combinations
       - name: pairs_two_sites
         min_subs_per_site: 4  # 2 per site
         max_subs_per_site: 4
         min_sites: 2
         max_sites: 2
         epochs: 50
     
     # Progression criteria
     progression:
       type: epoch  # Advance after completing stage epochs

Stage Configuration
~~~~~~~~~~~~~~~~~~~

Each stage specifies:

**Combination Filters**:

* ``min_subs_per_site``, ``max_subs_per_site``: Bounds on substituents at any single site in
  the combination (checked against the min/max across that combination's per-site counts,
  not the combo's total substituent count)
* ``min_sites``, ``max_sites``: Number of sites represented
* ``combo_whitelist``: Optional explicit list of combo names to restrict this stage to,
  bypassing the size filters above
* ``min_transitions_for_stage`` / ``transition_filter_lookback``: On stage advancement
  (not the first stage), additionally require a combo's most recent runs to have hit this
  many transitions per site before it's eligible for the new stage

**Training Duration**:

* ``epochs``: Number of training epochs for this stage

**Optional Settings**:

* ``max_train_combos``: Stage-specific limit on training combinations (overrides global setting)
* ``reward_override``: Modify reward weights for this stage (e.g., emphasize transitions early)


Combination Selection
~~~~~~~~~~~~~~~~~~~~~

**Filtering Process**:

For each stage, the workflow:

1. Filters all training combinations by stage criteria (min/max subs-per-site/sites)
2. If filtered count exceeds ``max_train_combos_per_stage``, randomly selects a subset each
   epoch (stratified across systems by default — see ``curriculum.stratified_sampling`` /
   ``curriculum.min_combos_per_system``)
3. Uses reproducible random selection (seeded by the top-level ``seed`` plus the epoch index)

**Important**: Random selection is uniform across all matching combinations.
If a stage allows both pairs (2 subs) and triplets (3 subs) via ``min_subs_per_site: 2,
max_subs_per_site: 3``, the 100 selected combinations will be a random mix with no
preference for either size.

**Reproducibility**: Same seed produces same combination selection across runs.

Progression Criteria
~~~~~~~~~~~~~~~~~~~~

Stages advance based on progression criteria:

**Epoch-based** (default):

.. code-block:: yaml

   progression:
     type: epoch

Advances after completing the specified number of epochs for current stage.

**Reward-based** (experimental):

.. code-block:: yaml

   progression:
     type: reward
     reward_threshold: 10.0  # Minimum average reward to advance

Advances only if average reward over last 5 epochs exceeds threshold.

**Combined**:

.. code-block:: yaml

   progression:
     type: both
     reward_threshold: 10.0

Must complete all epochs AND meet reward threshold.

Training Flow Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   === Training with Curriculum ===
   
   Stage 1: pairs_single_site_easy (epochs 1-50)
   ├── Filtered: 41 combinations (2 subs, 1 site)
   ├── Training on all 41 combinations
   └── Epoch 50 completes → Advance to Stage 2
   
   Stage 2: triplets_single_site (epochs 51-100)
   ├── Filtered: 186 combinations (3 subs, 1 site)
   ├── Limited to 100 random combinations
   └── Epoch 100 completes → Advance to Stage 3
   
   Stage 3: pairs_two_sites (epochs 101-150)
   ├── Filtered: 1,681 combinations (4 subs, 2 sites)
   ├── Limited to 100 random combinations
   └── Epoch 150 completes → Training complete

**Training Output**:

.. code-block:: text

   === Starting Stage 1/3: pairs_single_site_easy ===
   Filtered to 41 training combinations for this stage
   
   --- Epoch 1/150 - Stage 1/3: pairs_single_site_easy (epoch 1/50) ---
   Epoch 1 Stats:
     Loss: 12.3456
     Value Loss: 45.6789
     Avg Reward: -28.5432
   
   [... epochs 2-50 ...]
   
   ============================================================
   === Advancing to Stage 2/3: triplets_single_site ===
   ============================================================
   Filtered to 186 training combinations for this stage
   Limiting to 100 random training combos (from 186 available)
   
   --- Epoch 51/150 - Stage 2/3: triplets_single_site (epoch 1/50) ---

Stage-Specific Reward Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced users can override reward parameters per stage:

.. code-block:: yaml

   stages:
     - name: pairs_single_site_easy
       min_subs_per_site: 2
       max_subs_per_site: 2
       min_sites: 1
       max_sites: 1
       epochs: 50
       reward_override:
         w_T: 0.9              # Emphasize transitions early
         min_transitions_per_site: 5  # Lower threshold for easier combinations

This allows fine-tuning the reward function to match stage difficulty.


Checkpointing and Resume
-------------------------

Long-running training jobs (e.g., 50 epochs) can be interrupted by SLURM time
limits, system maintenance, or manual cancellation. The workflow implements
two-level checkpointing to enable automatic resume without losing progress.

Configuration
~~~~~~~~~~~~~

Enable checkpointing in your workflow YAML:

.. code-block:: yaml

   output:
     base_dir: /path/to/training_output
     save_checkpoints: true    # Enable checkpoint saving
     checkpoint_freq: 5         # Save every N epochs

Training-Level Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``{base_dir}/checkpoint_{epoch:03d}.pt`` (e.g. ``checkpoint_005.pt``)

Saved every ``checkpoint_freq`` epochs, containing:

* ``epoch``: Completed epoch number
* ``policy_state_dict``: Full ``UnimolPolicy`` state dict
* ``rl_optimizer_state_dict``: Optimizer state (momentum, learning rates, etc.)
* ``current_stage_idx``: Curriculum stage index, for resuming mid-curriculum
* ``stats``: Training statistics for that epoch (loss, average reward)

A separate ``bandit_state.json`` in ``base_dir`` tracks per-combo failure counts and
(in NeuralLinear + TS mode) uncertainty bookkeeping across epochs; it is loaded/saved
alongside checkpoints but is not itself a model checkpoint. When training finishes, a final
``{base_dir}/final_model.pt`` is written with the last policy/optimizer state plus the full
per-epoch ``all_stats`` history.

Automatic Resume
~~~~~~~~~~~~~~~~

When training restarts, the workflow:

1. Scans for ``checkpoint_*.pt`` files in ``base_dir``
2. Loads the latest checkpoint (highest epoch number)
3. Restores policy and optimizer state, and the curriculum stage (if any)
4. Continues from the next epoch — falling back to the ``pretrain.model_path`` checkpoint
   (partial load, ``strict=False``) if the latest checkpoint fails to load

For each combination in each epoch, the Uni-Mol embeddings and graph tensors computed
before submitting the simulation are persisted to a small ``graph_data.pt`` in that epoch's
run directory, so the later REINFORCE update (after the SLURM wait) reloads them instead of
recomputing embeddings — without holding every epoch's combos resident in memory for the
full SLURM wait.

Archiving Combinations
-----------------------

Combination directories can be automatically archived to save disk space using
two strategies: **per-stage archiving** (during curriculum training) or
**post-training archiving** (after all training completes). Each combination
directory is compressed into a ``.tar.gz`` file, optionally removing the original.

Configuration
~~~~~~~~~~~~~

Enable archiving in your workflow YAML:

.. code-block:: yaml

   archive:
     enabled: true               # Enable archiving
     per_stage: true             # Archive after each curriculum stage (or false for post-training)
     pattern: 'comb_*'           # Glob pattern for directories to archive (post-training only)
     remove_after: false         # Remove originals after successful archiving
     archive_dir: /path/to/archives  # Where to store .tar.gz files

Per-Stage Archiving (Curriculum Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for**: Long curriculum training runs where disk space is limited.

When ``per_stage: true``, the workflow archives combinations at the end of each
curriculum stage **in the background** while the next stage's simulations begin.
This provides:

* **Immediate space recovery**: Free up disk as soon as each stage completes
* **No training delays**: Archiving runs concurrently with next stage setup
* **Stage-specific organization**: Each stage gets its own archive directory

**Behavior**:

1. After a curriculum stage completes (e.g., after epoch 50 of stage 1)
2. Archive job launches in background (bash script with tar commands)
3. Next stage begins immediately (simulations submit while archiving runs)
4. After training completes, workflow waits for any remaining archive jobs

**Configuration Example**:

.. code-block:: yaml

   curriculum:
     enabled: true
     stages:
       - name: pairs_single_site_easy
         min_subs_per_site: 2
         max_subs_per_site: 2
         epochs: 50
       - name: pairs_single_site_full
         min_subs_per_site: 2
         max_subs_per_site: 2
         epochs: 50
   
   archive:
     enabled: true
     per_stage: true              # Archive after each stage
     remove_after: false
     archive_dir: /path/to/archives

**Timeline**:

.. code-block:: text

   Epoch 1-50 (Stage 1) → Stage 1 completes → Archive job starts in background
                                             ↓
   Epoch 51 begins (Stage 2) ← Simulations submit while Stage 1 archives
   
   Epoch 51-100 (Stage 2) → Stage 2 completes → Archive job starts in background
                                               ↓
   Epoch 101 begins (Stage 3) ← Stage 2 continues archiving in background

Post-Training Archiving
~~~~~~~~~~~~~~~~~~~~~~~~

**Best for**: Non-curriculum training or when you want to keep all data until
the end.

When ``per_stage: false`` (or not specified), the workflow archives combinations
once after all training completes.

**Behavior**:

1. After training completes successfully, all directories matching ``pattern``
   are compressed into individual ``.tar.gz`` archives
2. Archives are moved to ``archive_dir`` (if different from source)
3. Original directories are removed if ``remove_after`` is ``true``


**Configuration Example**:

.. code-block:: yaml

   archive:
     enabled: true
     per_stage: false             # Archive once at the end (default)
     pattern: 'comb_*'            # Directories to archive
     remove_after: false
     archive_dir: /path/to/archives


Manual Archiving
~~~~~~~~~~~~~~~~

You can also archive combinations manually:

.. code-block:: python

   from mllf.file_handling.generate_combinations import archive_combo_dirs
   from pathlib import Path
   
   # Archive all comb_* directories
   archived = archive_combo_dirs(
       out_dir=Path('generated_combos'),
       pattern='comb_*',
       remove=False  # Keep originals
   )
   
   print(f"Created {len(archived)} archive files")


Complete Workflow Example
--------------------------

Full Pipeline Script
~~~~~~~~~~~~~~~~~~~~

The main online-training workflow is implemented in ``examples/run_workflow.py``:

.. code-block:: bash

   cd mllf   # repository root
   python examples/run_workflow.py examples/workflow_14benz.yaml

This executes:

1. Combination manifest generation for every system in ``systems:`` (parallelized across
   systems; large combo pools are handled separately from small ones to avoid OOM)
2. Curriculum stage setup and held-out validation split (if ``validation_fraction > 0`` and
   enough combos are available)
3. Policy initialization (``UnimolPolicy``, optionally with Bayesian heads)
4. Checkpoint detection and resume, or warm start from ``pretrain.model_path``
5. Training loop: per epoch, sample/filter combos, compute Uni-Mol embeddings, sample
   actions, submit SLURM simulations, compute per-edge rewards, and update the policy
6. Checkpoint saving at ``checkpoint_freq`` intervals, plus a final ``final_model.pt``

.. _Complete Configuration Example:

Configuration File
~~~~~~~~~~~~~~~~~~

A minimal single-system workflow configuration (``examples/workflow_14benz.yaml``, trimmed
here — see the file itself for full inline documentation):

.. code-block:: yaml

   seed: 42

   # One system is still a list -- see "Configuration Format" above
   systems:
     - name: 14benz_solv
       input_dir: examples/14benz
       out_dir: examples/14benz/generated_combos
       solvent_state: solv
       include_patterns: [msld_flat.py]
       max_subs_per_site: 3

   # Warm start from a behavior-cloned checkpoint (optional -- see :doc:`cb_pretraining`).
   # Omitted here; UnimolPolicy trains from a random initialization by default.
   # pretrain:
   #   model_path: models/<your_pretrain_run>/best_policy.pt

   # Curriculum: restrict to the easiest stage so this example runs quickly
   curriculum:
     enabled: true
     max_train_combos_per_stage: 20
     stages:
       - name: pairs_single_site
         min_sites: 1
         max_sites: 1
         min_subs_per_site: 2
         max_subs_per_site: 2
         epochs: 5
     progression:
       type: epoch

   validation_fraction: 0.0   # too few combos for a held-out split in this tiny example

   # Policy architecture (UnimolPolicy -- see :doc:`cb_setup`)
   training:
     policy:
       unimol_dim: 512
       mlp_hidden: 64
     optimizer:
       lr: 0.0001
     bias_clip: 1000.0

   output:
     base_dir: examples/14benz/training_output
     save_checkpoints: true
     checkpoint_freq: 1

   run_sims: true
   wait_for_jobs: true
   max_concurrent_jobs: 10
   timeout: 1200

   slurm:
     partition: gpu
     gres: gpu:1
     time: '01:00:00'
     module: charmm/charmm/c51a1

   reward:
     w_P: 0.5
     w_T: 0.75
     w_U: 0.3
     gamma: 4.0
     entropy_bonus: 8.0
     P_baseline: 500.0
     T_baseline: 50.0
     min_transitions_per_site: 10
     no_transition_weight: 0.15

For a large-scale, multi-system production config (30+ systems, curriculum stages through
cross-site combinations, per-system overrides, NeuralLinear + Thompson Sampling) see the
``bandit:``, ``curriculum.stratified_sampling``, and per-system override options referenced
throughout this page — ``examples/workflow_14benz.yaml`` intentionally keeps most of these
at their defaults.

See Also
~~~~~~~~

* :doc:`file_handling` - File format documentation and parsers
* :doc:`cb_setup` - CB infrastructure and policy architecture
* :doc:`unimol_representation` - Uni-Mol embeddings for node features
* :doc:`cb_pretraining` - Behavior cloning from expert coefficients
* :doc:`examples` - Example workflows and usage patterns
* :doc:`api` - API reference for workflow modules
