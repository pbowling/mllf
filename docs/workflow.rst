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

The workflow is driven by a YAML configuration file:

.. code-block:: bash

   python -m mllf.cli.workflow --config examples/workflow_sample.yaml

Or using the convenience wrapper:

.. code-block:: bash

   cd examples
   python run_workflow.py  # Uses workflow_sample.yaml by default
   python run_workflow.py my_config.yaml  # Use custom config

Configuration Format
~~~~~~~~~~~~~~~~~~~~

A workflow config is a YAML file specifying which operations to run and their parameters.
Key sections include:

* **system**: Environment type (solvent, gas, protein)
* **create_combos**: Generate combinations from fragment files
* **split**: Divide combinations into train/val/test sets
* **pretrain**: Optional pretraining from existing simulations
* **curriculum**: Progressive training stages (see :ref:`Curriculum Learning`)
* **training**: Model architecture and hyperparameters
* **reward**: Reward function weights and thresholds
* **output**: Checkpointing and output organization
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

Rotating Anchor Strategy
^^^^^^^^^^^^^^^^^^^^^^^^

For within-site combinations, each substituent can serve as the "anchor":

* **Anchor first**: The anchor substituent is always listed first
* **Tail sorted**: Remaining substituents are sorted numerically
* **Minimum size**: At least 2 substituents per combination

This ensures unique, ordered combinations without duplicates.

For a site with 5 substituents, each acting as anchor generates combinations:

* Anchor 1: ``[1,2]``, ``[1,3]``, ``[1,4]``, ``[1,5]``, ``[1,2,3]``, ``[1,2,4]``, ..., ``[1,2,3,4,5]``
* Anchor 2: ``[2,1]``, ``[2,3]``, ``[2,4]``, ...
* Total: 5 anchors × 15 combinations each = **75 within-site combinations**

.. note::
   **Combination Size Limit**: By default, each combination is limited to at most 10 substituents 
   per site (``max_subs_per_site=10``). This prevents combinatorial explosion while still allowing 
   all substituents to participate across different combinations. For example, with 50 substituents 
   at a site, the generator will create combinations like ``[1,2,...,10]``, ``[1,2,...,9,11]``, etc., 
   but not ``[1,2,...,11]``. This limit can be increased via the ``--max-subs`` command-line option 
   or the ``max_subs_per_site`` parameter in the API.

Cross-Site Combinations
^^^^^^^^^^^^^^^^^^^^^^^

When multiple sites are available, the generator also creates cross-site combinations:

* Takes the cartesian product of within-site selections across all sites
* Each site must contribute at least 2 substituents
* Results in a much larger combination space

Example: With 5 subs at site1 (75 selections) and 6 subs at site2 (186 selections):

* Site 1 within-site: 75 combinations
* Site 2 within-site: 186 combinations  
* Cross-site: 75 × 186 = **13,950 combinations**
* Total: **14,211 combinations**

Single-Site Core Augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When generating **single-site pair combinations** (e.g., testing only site 1 pairs while
site 2 is excluded), the core structure must include atoms from the excluded site's first
substituent to maintain the complete molecular structure.

The combination generator automatically augments ``core.rtf`` and ``core.pdb`` for
single-site combinations:

.. code-block:: text

   # Example: comb_0001_site1_1__site1_2 (site2 excluded)
   
   # Original core.rtf:
   RESI  LIG    -0.012
   ATOM C001 CG2R61  -0.120335 
   ATOM H002 HGR61    0.114301
   BOND C001 H002
   
   # Augmented core.rtf (adds site2_sub1 atoms):
   * Core augmented with atoms from excluded site's first substituent for single-site combination
   * 
   RESI  LIG    -0.002000   # Charge updated: -0.012 + 0.010 from site2_sub1
   ATOM C001 CG2R61  -0.120335 
   ATOM H002 HGR61    0.114301
   ATOM C062 CG2R61  -0.110800  # From site2_sub1
   ATOM H063 HGR61    0.121200   # From site2_sub1
   BOND C001 H002
   BOND C001 C062  # From site2_sub1
   BOND C003 C062  # From site2_sub1
   BOND C062 H063  # From site2_sub1

This ensures that:

* CHARMM simulations run successfully with proper energy landscapes
* Single-site combinations have the correct molecular topology
* Core structure includes all necessary atoms for simulation

The augmentation is performed automatically during combination directory creation and
only applies to single-site combinations. Cross-site combinations (involving multiple
sites) already have complete core structures and are not modified.

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

Graph Building
--------------

From RTF Fragments
~~~~~~~~~~~~~~~~~~

The preferred method extracts connectivity from CHARMM topology fragments:

.. code-block:: python

   from mllf.file_handling.read_rtf import parse_rtf_dir
   from mllf.cb.graph import Graph
   
   rtf_results = parse_rtf_dir('examples/cb/14benz_solv_5.5')
   graph = Graph.from_rtf_results(rtf_results)

RTF files (``site*_sub*_*_pres.rtf``) contain CHARMM topology patches
defining atom connectivity for each substituent.

The ``Graph`` object stores nodes (λ-sites) and edges with associated bias
coefficients. See :doc:`cb_setup` for detailed information on graph structure,
edge types (linear, quadratic, skew, end), and their physical meanings.

From Bias Matrices
~~~~~~~~~~~~~~~~~~

Alternatively, build from existing ``variables.py``:

.. code-block:: python

   from mllf.cli.workflow import load_bias_from_variables, graph_from_bias
   
   bias = load_bias_from_variables('examples/14benz/variables.py')
   graph = graph_from_bias(bias)

This parses the YAML ``bias_string`` to extract bias matrices.

PyTorch Geometric Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For neural network training, convert graphs to PyTorch Geometric format:

.. code-block:: python

   from mllf.cb import graph_utils
   
   data, extras = graph_utils.build_pyg_graph_from_mllf_graph(graph)

This creates a ``Data`` object with node features, edge indices, and edge types
suitable for GNN training. See :doc:`cb_setup` for details on node features,
directed edge expansion, and the RGCN/policy architecture.

Training Pipeline
-----------------

Reward and Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The training system uses different objective functions for pretraining (behavior cloning) and reinforcement learning (policy gradients).

**Pretraining Loss (Behavior Cloning)**

Pretraining uses supervised learning with Mean Squared Error (MSE) loss:

.. math::

   \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{a}_i^{\text{pred}} - \mathbf{a}_i^{\text{target}}\|^2

where:

* :math:`\mathbf{a}_i^{\text{pred}}` is the policy's predicted bias coefficients for edge :math:`i`
* :math:`\mathbf{a}_i^{\text{target}}` is the known successful bias coefficients from completed simulations
* :math:`N` is the total number of edges in the graph

The policy learns to **imitate successful bias coefficients** from existing simulation data.
This provides a warm start before reinforcement learning begins.

**Training Reward**

During training, the policy is optimized using REINFORCE with rewards computed from simulation trajectories.
The reward function prevents degenerate solutions (e.g., convergence to single-substituent states)
through multiple components:

.. math::

   R_{\text{total}} = R_P + R_T + R_U + R_{\text{entropy}} + R_{\text{penalties}}

**Population Balance Reward** :math:`R_P`:

Encourages equal sampling across all substituents:

.. math::

   R_P = w_P \cdot \frac{\sum_{k=1}^{N_{\text{subs}}} p_k}{P_{\text{baseline}}}

where:

* :math:`w_P` is the population weight (default: 0.5)
* :math:`p_k` is the population count for substituent :math:`k`
* :math:`P_{\text{baseline}}` is the normalization constant (default: 500.0)

**Transition Reward** :math:`R_T`:

Rewards frequent transitions between substituents, with bonus for high transition counts:

.. math::

   R_T = \begin{cases}
   w_T \cdot \frac{\sum_{s=1}^{N_{\text{sites}}} T_s}{T_{\text{baseline}}} & \text{if all sites have } \geq 10 \text{ transitions} \\
   w_T \cdot \frac{\sum_{s=1}^{N_{\text{sites}}} T_s}{T_{\text{baseline}}} \times 1.5 & \text{if avg. trans/site} > 20 \\
   0 & \text{otherwise (sites below threshold)}
   \end{cases}

where:

* :math:`w_T` is the transition weight (default: 0.75)
* :math:`T_s` is the transition count for site :math:`s`
* :math:`T_{\text{baseline}}` is the normalization constant (default: 50.0)
* The 1.5× bonus applies when average transitions per site exceeds 20

**Uniformity Reward** :math:`R_U`:

Rewards visiting a minimum fraction of substituents:

.. math::

   R_U = w_U \cdot \frac{\text{coverage\_ratio}}{\text{min\_coverage\_ratio}}

where coverage_ratio is the fraction of substituents with non-zero population.

**Entropy Bonus** :math:`R_{\text{entropy}}`:

Rewards uniform population distributions using Shannon entropy:

.. math::

   R_{\text{entropy}} = \beta_{\text{entropy}} \cdot H(\mathbf{p})

where :math:`H(\mathbf{p}) = -\sum_k \frac{p_k}{P_{\text{total}}} \log \frac{p_k}{P_{\text{total}}}` is the normalized entropy.

**Tiered Transition Penalties** :math:`R_{\text{penalties}}`:

The system uses a three-tier penalty structure to provide continuous feedback:

**Tier 1: "Death Floor"** (0-2 transitions):

.. math::

   \text{penalty} = \begin{cases}
   -40.0 & \text{if } T_s = 0 \\
   -32.0 & \text{if } T_s = 1 \\
   -24.0 & \text{if } T_s = 2
   \end{cases}

Worst possible state, signaling total inactivity is unacceptable.

**Tier 2: "Climbing Ramp"** (3-9 transitions):

.. math::

   \text{penalty} = -\left(2.0 + 2.0 \times \text{deficit}\right)

where :math:`\text{deficit} = 10 - T_s` for site :math:`s`. This creates a linear gradient:

* 3 transitions: :math:`-2.0 - 2.0 \times 7 = -16.0`
* 5 transitions: :math:`-2.0 - 2.0 \times 5 = -12.0`
* 9 transitions: :math:`-2.0 - 2.0 \times 1 = -4.0`

Each additional transition improves the reward by 2.0 points, providing continuous feedback.

**Tier 3: "Success Zone"** (≥10 transitions):

.. math::

   \text{penalty} = 0.0

Site is "unlocked" and eligible for positive :math:`R_T` rewards.

**Additional Penalties**:

* **Coverage penalty**: If fewer than :math:`\text{min\_coverage\_ratio}` of substituents are visited:

  .. math::

     \text{penalty} = -\gamma \times 20.0 \times \left(\text{min\_coverage\_ratio} - \text{coverage\_ratio}\right)

* **Concentration penalty**: Per-site penalty if any substituent exceeds 80% of that site's population:

  .. math::

     \text{penalty} = -\gamma \times 5.0 \times (\text{concentration} - 0.8) \quad \text{(per concentrated site)}

* **Simulation failure**: :math:`R = -100 \times \gamma` if simulation does not terminate normally

**Default Hyperparameters**:

.. code-block:: yaml

   reward:
     w_P: 0.5                                # Population weight
     w_T: 0.75                                # Transition weight
     w_U: 0.3                                # Uniformity weight
     gamma: 4.0                              # Base penalty coefficient
     P_baseline: 500.0                       # Population normalization
     T_baseline: 50.0                        # Transition normalization
     min_transitions_per_site: 10            # Tier 3 threshold
     min_coverage_ratio: 0.5                 # Minimum fraction of substituents to visit
     entropy_bonus: 8.0                      # Entropy bonus coefficient
     concentration_penalty_threshold: 0.8    # Single-substituent dominance threshold

**Policy Gradient Update with Value Network**:

The policy is updated using an **Actor-Critic** architecture that combines REINFORCE 
with a learned value network for variance reduction.

**Actor-Critic Architecture**:

* **Actor (Policy Network)**: Predicts bias coefficients from graph structure
* **Critic (Value Network)**: Predicts expected reward for a given graph

The value network provides a state-dependent baseline :math:`V(s)` that adapts to 
each combination's difficulty, rather than using a fixed or moving average baseline.

**Value Network**:

.. code-block:: python

   # Architecture: Node embeddings → Global pooling → MLP → Scalar value
   value_network = ValueNetwork(
       emb_dim=32,           # Node embedding dimension from encoder
       hidden_dims=[64, 32]  # MLP layers [input→64→32→1]
   )
   
   # Predict expected reward
   node_embeddings = encoder(graph)
   predicted_value = value_network(node_embeddings)

**Training Loop**:

For each combination:

1. Encode graph structure to get node embeddings
2. Predict bias coefficients with policy (stochastic sampling)
3. Run simulation and compute actual reward :math:`R`
4. Predict expected value :math:`V(s)` with value network
5. Compute advantage: :math:`A = R - V(s)`
6. Update value network: minimize :math:`(V(s) - R)^2`
7. Update policy: maximize :math:`\log \pi(a|s) \cdot A`

**Policy Gradient with Advantage**:

.. math::

   \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t) \cdot A_t \right]

where:

* :math:`\mathbf{a}_t` is the action (bias coefficients) at time :math:`t`
* :math:`\mathbf{s}_t` is the state (graph representation) at time :math:`t`
* :math:`A_t = R - V(\mathbf{s}_t)` is the advantage (state-dependent baseline)

**Benefits of Value Network**:

* **Lower variance**: Advantages are centered around state-dependent expectations rather than global average
* **Faster convergence**: Stable gradients enable higher learning rates
* **Better credit assignment**: Easy vs hard combinations get different baselines
* **Catastrophic forgetting prevention**: Reduces gradient noise that destroys pretrained weights

**Hyperparameters**:

.. code-block:: yaml

   training:
     value_network:
       hidden_dims: [64, 32]  # MLP architecture
       lr: 0.001              # 10x policy LR for faster baseline learning
     optimizer:
       lr: 0.0001             # Policy learning rate (reduced for stability)
     reward:
       lambda_entropy: 0.5    # Entropy regularization (exploration bonus)

Pretraining from Existing Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before starting main training, you can pretrain the policy on existing MSLD
simulation data. This provides a warm start with meaningful bias coefficients
rather than random initialization.

**Pretraining Data Sources**:

Pretraining data consists of combination directories that have already been
simulated with known bias coefficients. These can come from:

* Previous training runs
* Manual expert tuning
* Systematic parameter sweeps
* Multi-system datasets (different ligands, environments, etc.)

**Benefits**:

* **Faster convergence**: Start with reasonable bias values
* **Improved sample efficiency**: Fewer epochs needed for good performance
* **Transfer learning**: Leverage knowledge from related systems
* **Robustness**: More stable early training with informed initialization

**Data Collection**:

Organize pretraining data by grouping completed runs by system/environment:

.. code-block:: bash

   pretraining/
   ├── system1_solv/         # System name with environment
   │   ├── run1/
   │   │   ├── graph_info.json           # Graph metadata (sites, substituents, edges)
   │   │   ├── metadata.json             # Run metadata (transitions, populations, solvent_state)
   │   │   ├── simulation_results.json   # Detailed populations and transitions by lambda
   │   │   └── variables.py              # Bias coefficients that were used
   │   ├── run2/
   │   └── ...
   ├── system2_solv/
   │   └── run*/
   ├── system3_prot/
   │   └── run*/
   └── ...

Each run directory should contain:

* ``graph_info.json``: Graph structure with sites, substituents, and connectivity information
* ``variables.py``: Bias coefficients that were used for the simulation
* ``metadata.json``: High-level run information (total transitions, num sites/subs, solvent state, termination status)
* ``simulation_results.json``: Detailed simulation outputs with populations and transitions organized by lambda value

This structure enables multi-system pretraining by organizing runs hierarchically by system/environment.

**Pretraining Configuration**:

Enable pretraining in your workflow YAML:

.. code-block:: yaml

   pretrain:
     data_dir: /path/to/pretraining       # Directory with existing runs
     num_epochs: 1                         # Usually 1 epoch is sufficient
     model_path: models/pretrained_policy.pt  # Where to save pretrained model
   
   training:
     num_epochs: 50
     load_pretrained: models/pretrained_policy.pt  # Load before training

**Pretraining Process**:

1. **Load existing data**: Read bias coefficients from ``variables.py`` and
   compute rewards from lambda trajectories
2. **Build graphs**: Construct graph representations from RTF files or info.py
3. **Supervised learning**: Train policy to reproduce the bias coefficients
   that led to good rewards
4. **Save model**: Store pretrained encoder and policy weights

**Running Pretraining**:

.. code-block:: bash

   # Step 1: Organize pretraining data
   mkdir -p pretraining
   cp -r previous_runs/good_combos/* pretraining/
   
   # Step 2: Run pretraining
   python run_pretraining.py --config workflow_sample.yaml
   
   # Step 3: Use pretrained model in training
   python run_workflow.py workflow_sample.yaml

**Deterministic Rewards**:

Since pretraining uses completed simulations, rewards are deterministic (not
resampled). This means:

* Multiple epochs don't improve training (data is fixed)
* One epoch is typically sufficient to fit the pretraining data
* The pretrained policy learns a mapping from graph structure to successful
  bias coefficients

System Configuration
~~~~~~~~~~~~~~~~~~~~

The ``system`` section specifies environment-level parameters that affect graph construction
and node feature encoding:

.. code-block:: yaml

   system:
     solvent_state: solv  # Environment type

**Solvent State**:

Specifies the simulation environment for proper graph metadata:

* ``solv`` or ``solvent``: Solvated/aqueous environment
* ``gas`` or ``vacuum``: Gas phase/vacuum environment
* ``protein``: Protein-embedded environment

This information is:

* Embedded in graph metadata for each combination
* Used by the encoder as contextual information
* Preserved in ``graph_info.json`` for pretraining data reuse

**Auto-Detection** (legacy):

Previously, the system attempted to auto-detect solvent state from directory names
(e.g., ``14benz_solv`` → ``solv``). This is now deprecated in favor of explicit 
configuration for clarity and reliability.


Full Simulation Training
~~~~~~~~~~~~~~~~~~~~~~~~~

For production training:

.. code-block:: python

   from mllf.cli.workflow import run_simulations_and_collect
   
   results = run_simulations_and_collect(
       combos=train_combos,
       encoder=encoder,
       policy=policy,
       optimizer=optimizer,
       reward_fn=reward_fn,
       compress_after=True
   )

This performs full MD simulations after each action sampling and uses
real simulation metrics for rewards.

Curriculum Learning
-------------------

.. _Curriculum Learning:

Overview
~~~~~~~~

**Curriculum learning** progressively trains the policy on increasingly complex
combinations, similar to how students learn from simple to complex problems.
Instead of training on all possible combinations at once, the policy masters
simpler tasks before advancing to harder ones.

**Benefits**:

* **Faster convergence**: Start with easier combinations that provide clearer learning signals
* **Better generalization**: Build strong foundations before tackling complex interactions
* **Reduced catastrophic forgetting**: Gradual progression prevents pretrained weights from being destroyed
* **Sample efficiency**: Focus computational resources on combinations appropriate for current skill level

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
         min_subs: 2
         max_subs: 2
         min_sites: 1
         max_sites: 1
         epochs: 50
         
       # Stage 2: Triplets at single sites
       - name: triplets_single_site
         min_subs: 3
         max_subs: 3
         min_sites: 1
         max_sites: 1
         epochs: 50
         
       # Stage 3: Cross-site combinations
       - name: pairs_two_sites
         min_subs: 4  # 2 per site
         max_subs: 4
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

* ``min_subs``, ``max_subs``: Total substituents in combination
* ``min_sites``, ``max_sites``: Number of sites represented

**Training Duration**:

* ``epochs``: Number of training epochs for this stage

**Optional Settings**:

* ``max_train_combos``: Stage-specific limit on training combinations (overrides global setting)
* ``reward_override``: Modify reward weights for this stage (e.g., emphasize transitions early)

**Example - All Stages**:

.. code-block:: yaml

   stages:
     # 1. WARM UP: Single-site pairs
     - name: pairs_single_site_easy
       min_subs: 2
       max_subs: 2
       min_sites: 1
       max_sites: 1
       epochs: 50
     
     # 2. MASTERY: More single-site pairs
     - name: pairs_single_site_full
       min_subs: 2
       max_subs: 2
       min_sites: 1
       max_sites: 1
       epochs: 50
     
     # 3. DENSITY: Single-site triplets
     - name: triplets_single_site
       min_subs: 3
       max_subs: 3
       min_sites: 1
       max_sites: 1
       epochs: 50
     
     # 4. EXTRAPOLATION: Multi-site learning
     - name: pairs_two_sites
       min_subs: 4
       max_subs: 4
       min_sites: 2
       max_sites: 2
       epochs: 50
     
     # 5. COMPLEXITY: Complex multi-site
     - name: complex_two_sites
       min_subs: 5
       max_subs: 8
       min_sites: 2
       max_sites: 2
       epochs: 50

Combination Selection
~~~~~~~~~~~~~~~~~~~~~

**Filtering Process**:

For each stage, the workflow:

1. Filters all training combinations by stage criteria (min/max subs/sites)
2. If filtered count exceeds ``max_train_combos_per_stage``, randomly selects subset
3. Uses reproducible random selection (seeded by ``split.seed + stage_index``)

**Example**:

.. code-block:: text

   Stage 1: pairs_single_site_easy (2 subs, 1 site)
   - Filtered: 75 combinations match criteria
   - Max limit: 100 combinations
   - Selected: All 75 combinations (under limit)
   
   Stage 2: triplets_single_site (3 subs, 1 site)
   - Filtered: 200 combinations match criteria  
   - Max limit: 100 combinations
   - Selected: 100 random combinations (uniform sampling)

**Important**: Random selection is uniform across all matching combinations.
If a stage allows both pairs (2 subs) and triplets (3 subs) via ``min_subs: 2,
max_subs: 3``, the 100 selected combinations will be a random mix with no
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
       min_subs: 2
       max_subs: 2
       min_sites: 1
       max_sites: 1
       epochs: 50
       reward_override:
         w_T: 0.9              # Emphasize transitions early
         min_transitions_per_site: 5  # Lower threshold for easier combinations

This allows fine-tuning the reward function to match stage difficulty.

Best Practices
~~~~~~~~~~~~~~

* **Stage ordering**: Start simple (pairs), gradually increase complexity (triplets → multi-site)
* **Combination limits**: 50-100 per stage; use ``max_train_combos_per_stage`` for large stages
* **Pretraining**: Essential for curriculum success; provides foundation for stage 1
* **Progress monitoring**: Expect reward drop at stage transitions; recovery within 10-20 epochs is normal
* **Checkpointing**: Critical for long runs (200+ epochs); save every 5-10 epochs

Troubleshooting
~~~~~~~~~~~~~~~

**Stage 1 not converging**: Check pretraining quality; increase epochs to 100 if needed

**Poor post-transition performance**: Expected; reward should recover within 20 epochs. If not, add intermediate stage

**Disk space**: Enable ``archive.per_stage: true`` to archive completed stages during training

Checkpointing and Resume
-------------------------

Overview
~~~~~~~~

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

**Location**: ``{base_dir}/checkpoint_epoch_XXX.pt``

Saved every ``checkpoint_freq`` epochs, containing:

* ``epoch``: Completed epoch number
* ``encoder_state``: Full RGCN encoder state dict
* ``policy_state``: Full edge policy state dict  
* ``optimizer_state``: Optimizer state (momentum, learning rates, etc.)
* ``stats``: Training statistics (loss, average reward)

**Purpose**: Resume entire training from a specific epoch

Per-Epoch Result Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``{base_dir}/epoch_NNN/{combo_name}/epoch_results.pt``

Saved after each simulation completes, containing:

* ``reward``: Computed reward from simulation
* ``actions``: Policy actions (bias coefficients)
* ``logp``: Log probability of actions
* ``epoch``: Epoch number
* ``combo``: Combination name

**Purpose**: Skip re-running simulations for completed combinations

Automatic Resume
~~~~~~~~~~~~~~~~

When training restarts, the workflow:

1. Scans for ``checkpoint_epoch_*.pt`` files
2. Loads the latest checkpoint (highest epoch number)
3. Restores model and optimizer state
4. Continues from the next epoch

For each combination in each epoch:

1. Checks for ``epoch_results.pt`` in the combination's directory
2. If found: loads cached reward/actions/logp, skips simulation
3. If not found: runs simulation, computes reward, saves checkpoint

Example
~~~~~~~

.. code-block:: bash

   # Start training
   sbatch training_job.sh
   
   # Training runs for epochs 1-15, then interrupted at epoch 18
   # Latest checkpoint: checkpoint_epoch_015.pt
   # Epochs 16-17 have partial epoch_results.pt files
   
   # Resume training (automatic)
   sbatch training_job.sh
   
   # Output shows:
   # === Resuming from checkpoint: .../checkpoint_epoch_015.pt ===
   # Resuming from epoch 15
   # 
   # Training continues from epoch 15, skipping combinations with
   # existing epoch_results.pt files in epochs 16-17

Benefits
~~~~~~~~

**Fault Tolerance**: Training survives:

* SLURM time limits
* System maintenance windows
* Job preemption or manual cancellation
* Hardware failures

**Efficiency**:

* No wasted computation - resume exactly where interrupted
* Per-epoch checkpoints prevent re-running completed simulations
* Granular checkpointing minimizes lost work

**Flexibility**:

* Stop/start training at any time
* Inspect checkpoints for debugging
* Adjust non-critical config between runs

File Structure Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   training_output/
   ├── checkpoint_epoch_005.pt    # Training checkpoint
   ├── checkpoint_epoch_010.pt
   ├── checkpoint_epoch_015.pt
   ├── epoch_001/
   │   ├── combo_1_2/
   │   │   ├── epoch_results.pt   # Per-epoch result
   │   │   ├── variables.py
   │   │   └── output/
   │   │       ├── transitions.txt
   │   │       └── populations.txt
   │   └── combo_1_3/
   │       └── ...
   ├── epoch_002/
   │   └── ...
   └── ...

Best Practices
~~~~~~~~~~~~~~

* **Checkpoint frequency**: Default of 5 epochs balances storage and resume granularity
* **Disk monitoring**: Per-epoch results accumulate; cleanup old checkpoints after training
* **Reproducibility**: Optimizer state ensures identical continuation after resume

Troubleshooting
~~~~~~~~~~~~~~~

**Training doesn't resume**: Check ``save_checkpoints: true`` in YAML and verify checkpoint files exist

**Out of memory**: Checkpoint device must match training device; free GPU memory if needed

**Simulations not skipping**: Verify ``epoch_results.pt`` files exist with proper permissions

Reward Function Experimentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pretraining Data Reuse**: Epoch checkpoints save raw simulation metrics
(populations, transitions) alongside computed rewards. This enables testing
different reward configurations without re-running simulations.

**Use Cases**:

* Hyperparameter tuning (weights, baselines, gamma)
* Testing alternative reward formulations
* Using existing simulations as "pretraining data"
* Rapid iteration on reward design

**Automatic Recomputation**: If you change reward parameters in
``workflow_sample.yaml`` and resume training, cached results are automatically
recomputed with the new configuration:

.. code-block:: yaml

   # Original configuration
   reward:
     w_P: 0.5
     w_T: 0.75
     gamma: 4.0
   
   # Change to emphasize transitions more
   reward:
     w_P: 0.3  # Changed
     w_T: 0.9  # Changed
     gamma: 4.0

When training resumes, rewards are recomputed from raw metrics without
re-running simulations.

**Testing Multiple Configurations**: Use ``test_reward_configs.py`` to
evaluate different reward configurations on completed epochs:

.. code-block:: bash

   # Test default configurations
   python examples/test_reward_configs.py training_output/epoch_005
   
   # Test specific configuration
   python examples/test_reward_configs.py training_output/epoch_005 \\
       --w_P 0.7 --w_T 0.3 --gamma 15.0
   
   # Test multiple configurations from YAML
   python examples/test_reward_configs.py training_output/epoch_005 \\
       --configs reward_configs_example.yaml

Output shows mean, std, min, max, and median rewards for each configuration,
ranked by mean reward.

**Example Output**:

.. code-block:: text

   Testing 5 reward configurations...
   
   Configuration: population_focused
     Parameters: w_P=0.7, w_T=0.3, gamma=10.0
     Mean reward: 45.2341 ± 8.1234
     Range: [12.3456, 67.8901]
   
   Configuration: baseline
     Parameters: w_P=0.5, w_T=0.75, gamma=4.0
     Mean reward: 42.1234 ± 7.8901
     Range: [10.2345, 65.4321]
   
   Summary Comparison:
   Config                    Mean        Std        Min        Max
   ------------------------------------------------------------------------
   population_focused      45.2341     8.1234    12.3456    67.8901
   baseline                42.1234     7.8901    10.2345    65.4321
   transition_focused      38.9012     9.2345    08.1234    63.2109
   
   Best configuration: population_focused

Archiving Combinations
-----------------------

Overview
~~~~~~~~

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

**Directory Structure**:

.. code-block:: text

   archives/
   ├── stage_1_pairs_single_site_easy/
   │   ├── comb_0001_site1_1__site1_2.tar.gz
   │   ├── comb_0002_site1_1__site1_3.tar.gz
   │   ├── ...
   │   └── archive.log                      # Archive job output
   ├── stage_2_pairs_single_site_full/
   │   ├── comb_0001_site1_1__site1_2.tar.gz
   │   └── ...
   └── stage_3_triplets_single_site/
       └── ...

**Configuration Example**:

.. code-block:: yaml

   curriculum:
     enabled: true
     stages:
       - name: pairs_single_site_easy
         min_subs: 2
         max_subs: 2
         epochs: 50
       - name: pairs_single_site_full
         min_subs: 2
         max_subs: 2
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

**Example**:

.. code-block:: text

   Before archiving:
   generated_combos/
   ├── comb_0001_site1_1__site1_2/
   ├── comb_0002_site1_1__site1_3/
   └── ...
   
   After archiving (remove_after=false):
   generated_combos/
   ├── comb_0001_site1_1__site1_2/
   ├── comb_0002_site1_1__site1_3/
   └── ...
   
   archives/
   ├── comb_0001_site1_1__site1_2.tar.gz
   ├── comb_0002_site1_1__site1_3.tar.gz
   └── ...
   
   After archiving (remove_after=true):
   archives/
   ├── comb_0001_site1_1__site1_2.tar.gz
   ├── comb_0002_site1_1__site1_3.tar.gz
   └── ...

**Configuration Example**:

.. code-block:: yaml

   archive:
     enabled: true
     per_stage: false             # Archive once at the end (default)
     pattern: 'comb_*'            # Directories to archive
     remove_after: false
     archive_dir: /path/to/archives

Use Cases
~~~~~~~~~

**Disk Space Management**: Large training runs can generate significant data.
Archiving allows:

* Preserving all simulation outputs for later analysis
* Reducing active disk usage by 50-90% (typical compression ratio)
* Organizing completed runs for long-term storage

**When to Use Per-Stage Archiving**:

* Long curriculum training (multiple stages over many days)
* Limited disk space on compute clusters
* Need to free space continuously as training progresses
* Want stage-specific organization for analysis

**When to Use Post-Training Archiving**:

* Short training runs (single stage or few epochs)
* Sufficient disk space for full training run
* Want to keep all data accessible during training
* Simpler workflow without background jobs

**Best Practices**:

* Set ``remove_after: false`` initially to verify archives are valid
* Test extracting an archive to ensure contents are intact
* Monitor background archive jobs via ``archive.log`` files (per-stage mode)
* Use ``remove_after: true`` for production runs with proven archiving
* Keep archives and checkpoints separate (archives for data, checkpoints for resume)

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

Monitoring Per-Stage Archives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For per-stage archiving, monitor the background jobs:

.. code-block:: bash

   # Check archive logs for each stage
   tail -f archives/stage_1_pairs_single_site_easy/archive.log
   
   # Training output shows archive job status
   # === Archiving Stage 1 Combinations ===
   #   Archiving 41 combinations to .../archives/stage_1_pairs_single_site_easy
   #   Running archive in background (see .../archive.log)
   #   Archive job started (PID: 12345)
   
   # After training completes, workflow waits for archive jobs
   # === Waiting for Background Archive Jobs ===
   #   Waiting for stage 'pairs_single_site_easy' archive (PID: 12345)...
   #     ✓ Stage 'pairs_single_site_easy' archived successfully
   #     Log: .../archives/stage_1_pairs_single_site_easy/archive.log

Extracting Archives
~~~~~~~~~~~~~~~~~~~

To extract an archived combination:

.. code-block:: bash

   # Extract a single combination
   tar -xzf archives/comb_0001_site1_1__site1_2.tar.gz -C .
   
   # Extract all archives
   cd archives
   for f in *.tar.gz; do tar -xzf "$f" -C ../; done

Simulation Execution
--------------------

Launching Simulations
~~~~~~~~~~~~~~~~~~~~~

Simulations are launched via subprocess:

.. code-block:: python

   import subprocess
   
   # Write variables.py first (from CB policy)
   write_variables_from_actions(combo_dir, data, extras, actions)
   
   # Run CHARMM simulation
   subprocess.run(['charmm', '-i', 'input.inp'], cwd=combo_dir)

The simulator reads bias coefficients from ``variables.py`` and outputs
transition counts and population distributions.

Output Parsing
~~~~~~~~~~~~~~

After simulation:

.. code-block:: python

   from mllf.file_handling.parse_sim_output import parse_transitions, parse_population
   
   transitions = parse_transitions(f'{combo_dir}/output/transitions.txt')
   population = parse_population(f'{combo_dir}/output/populations.txt')
   
   reward = compute_reward(transitions, population)

Typical reward functions:

.. code-block:: python

   def default_env_reward(transitions, population):
       # Count total successful transitions
       total_trans = sum(transitions.values())
       
       # Measure coverage of state space
       visited_states = len([p for p in population.values() if p['counts'] > 0])
       total_states = len(population)
       coverage = visited_states / total_states if total_states > 0 else 0
       
       # Weighted combination
       return 0.7 * total_trans + 0.3 * coverage * 1000

Compression
~~~~~~~~~~~

To save disk space, compress simulation outputs:

.. code-block:: python

   from mllf.cli.workflow import compress_runs
   
   compress_runs(combos, patterns=['*.dcd', '*.coor', '*.vel'])

This creates ``output.tar.gz`` in each combo directory and removes
the specified file patterns.

Complete Workflow Example
--------------------------

Full Pipeline Script
~~~~~~~~~~~~~~~~~~~~

The main training workflow is implemented in ``examples/run_workflow.py``:

.. code-block:: bash

   cd examples
   python run_workflow.py workflow_sample.yaml

This executes:

1. Combination generation (if ``create_combos`` specified)
2. Train/val/test split based on ``split`` configuration
3. Model initialization (RGCN encoder + edge policy)
4. Checkpoint detection and resume (if checkpoints exist)
5. Training loop with SLURM job submission
6. Checkpoint saving at ``checkpoint_freq`` intervals
7. Archiving combinations (if ``archive.enabled`` is true)

.. _Complete Configuration Example:

Configuration File
~~~~~~~~~~~~~~~~~~

A complete workflow configuration (``workflow_14benz.yaml``) includes:

.. code-block:: yaml

   # System environment
   system:
     solvent_state: solv
   
   # Generate combinations
   create_combos:
     input_dir: /path/to/14benz
     out_dir: /path/to/generated_combos
     include_patterns: [msld_flat.py]
   
   # Data splitting
   split:
     train_frac: 0.9
     val_frac: 0.1
     seed: 42
   
   # Pretraining (optional but recommended)
   pretrain:
     model_path: models/pretrained_policy.pt
   
   # Curriculum learning
   curriculum:
     enabled: true
     max_train_combos_per_stage: 100
     stages:
       - name: pairs_single_site
         min_subs: 2
         max_subs: 2
         epochs: 50
       - name: triplets_single_site
         min_subs: 3
         max_subs: 3
         epochs: 50
     progression:
       type: epoch
   
   # Model architecture
   training:
     encoder:
       hidden_dims: [64, 64]
       out_dim: 32
     policy:
       mlp_hidden: 64
     value_network:
       hidden_dims: [64, 32]
       lr: 0.001
     optimizer:
       lr: 0.0001
   
   # Simulation settings
   run_sims: true
   max_concurrent_jobs: 60
   timeout: 1200
   
   # Reward function
   reward:
     w_P: 0.5
     w_T: 0.75
     w_U: 0.3
     gamma: 4.0
     lambda_entropy: 0.5
   
   # Checkpointing
   output:
     base_dir: /path/to/training_output
     save_checkpoints: true
     checkpoint_freq: 5
   
   # Per-stage archiving
   archive:
     enabled: true
     per_stage: true
     archive_dir: /path/to/archives

For detailed explanations of each section, see the relevant subsections below.

Training Loop Structure
~~~~~~~~~~~~~~~~~~~~~~~

The main training loop implements epoch-based training with concurrent SLURM job management:

**High-Level Flow**:

1. For each combination in training set:
   
   - Check if epoch already completed (cached results)
   - Build graph from RTF fragments
   - Sample bias coefficients from policy (stochastic)
   - Write ``variables.py`` with sampled coefficients
   - Submit SLURM job for simulation

2. Wait for all jobs to complete (up to ``max_concurrent_jobs`` running)
3. Parse simulation outputs (transitions, populations)
4. Compute rewards from simulation metrics
5. Update value network: minimize :math:`(V(s) - R)^2`
6. Update policy: maximize :math:`\log \pi(a|s) \cdot (R - V(s))`
7. Save checkpoints at specified intervals

**Key Features**:

* **Cached results**: Skip simulations if ``epoch_results.pt`` exists
* **Concurrent jobs**: Manages SLURM queue with ``max_concurrent_jobs`` limit
* **Resume capability**: Automatically resumes from latest checkpoint
* **Reward recomputation**: Can update rewards with new config without re-running sims

Custom Training Loop
~~~~~~~~~~~~~~~~~~~~

For custom workflows, import the key components:

.. code-block:: python

   from mllf.file_handling.generate_combinations import create_combination_dirs
   from mllf.cli.workflow import build_data_and_targets_from_combo, write_variables_from_actions
   from mllf.cb.rgcn import RGCNEncoder
   from mllf.cb.policy import EdgePolicy
   from mllf.cb.value_net import ValueNetwork
   from mllf.cb.train_improved import compute_reward_from_raw_metrics
   
   # Generate combinations
   combos = create_combination_dirs(
       input_dir=Path('14benz'),
       output_dir=Path('generated_combos'),
       include_patterns=['msld_flat.py']
   )
   
   # Initialize actor-critic model
   sample_data, _, sample_extras = build_data_and_targets_from_combo(combos[0])
   
   encoder = RGCNEncoder(in_dim=sample_data.x.size(1), hidden_dims=[64, 64], out_dim=32,
                         num_relations=sample_data.edge_type.max().item() + 1)
   
   policy = EdgePolicy.from_pyg_data(encoder=encoder, emb_dim=32, data=sample_data,
                                      mlp_hidden=64, mlp_out_dim=len(sample_extras['relation_names']) // 2)
   
   value_network = ValueNetwork(emb_dim=32, hidden_dims=[64, 32])
   
   optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)
   value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.001)
   
   # Training loop - see run_workflow.py for full implementation
* :doc:`examples` - Example workflows and usage patterns
* :doc:`api` - API reference for workflow modules
