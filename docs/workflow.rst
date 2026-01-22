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

A workflow config specifies which operations to run and their parameters:

.. code-block:: yaml

   # Combination generation
   create_combos:
     input_dir: examples/cb/14benz_solv_5.5  # Directory with site/sub fragment files
     out_dir: examples/cb/generated_combos   # Output directory for combinations
     include_patterns:
       - msld_flat.py  # Additional files to copy to each combination
   
   # OR use existing combinations
   manifest: examples/manifest_example.txt
   
   # Split into train/val/test
   split:
     manifest: examples/manifest_example.txt
     train_manifest: examples/train.txt
     val_manifest: examples/val.txt
     test_manifest: examples/test.txt
     train_fraction: 0.7
     val_fraction: 0.15
   
   # Optional: Pretrain from existing simulations
   pretrain:
     data_dir: /path/to/pretraining_data     # Directory with completed runs
     num_epochs: 1                            # Usually 1 epoch is sufficient
     model_path: models/pretrained_policy.pt  # Save pretrained model here
   
   # Training configuration
   training:
     num_epochs: 50
     load_pretrained: models/pretrained_policy.pt  # Optional: load pretrained model
     encoder:
       hidden_dims: [64, 64]
       out_dim: 32
     policy:
       mlp_hidden: 64
     optimizer:
       lr: 0.001
   
   # Output and checkpointing
   output:
     base_dir: /path/to/training_output
     save_checkpoints: true    # Enable automatic checkpoint/resume
     checkpoint_freq: 5         # Save checkpoint every 5 epochs
   
   # Run MD simulations
   run_sims: true
   compress_after: true  # Archive outputs after each simulation

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

**Example**: For a 14,211-combination system with 1% train/1% val split:

* Without lazy creation: ~14,211 directories created upfront
* With lazy creation: ~284 directories created on-demand (only those used)
* Space savings: ~98% fewer directories

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
   │       ├── full_ligand.rtf
   │       ├── full_ligand.prm
   │       ├── top_all36_msld.rtf
   │       ├── par_all36_msld.prm
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

**Key features**:

* ``nsubs`` is a list showing substituent count per site (e.g., ``[3, 3]`` for 
  3 subs at each of 2 sites)
* ``nblocks`` is computed as the sum of ``nsubs`` (total substituents)
* Used by ``msld_flat.py`` to determine site structure: ``nsites = len(info['nsubs'])``

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
   
   bias = load_bias_from_variables('examples/cb/14benz_solv_5.5/variables.py')
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

**Training Reward (Reinforcement Learning)**

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

* :math:`w_T` is the transition weight (default: 0.5)
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

**Tier 1: "Death Floor"** (0 transitions):

.. math::

   \text{penalty} = -40.0 \quad \text{(per site with 0 transitions)}

Worst possible state, signaling total inactivity is unacceptable.

**Tier 2: "Climbing Ramp"** (1-9 transitions):

.. math::

   \text{penalty} = -\left(5.0 + 2.8 \times \text{deficit}\right)

where :math:`\text{deficit} = 10 - T_s` for site :math:`s`. This creates a linear gradient:

* 1 transition: :math:`-5.0 - 2.8 \times 9 = -30.2`
* 5 transitions: :math:`-5.0 - 2.8 \times 5 = -19.0`
* 9 transitions: :math:`-5.0 - 2.8 \times 1 = -7.8`

Each additional transition improves the reward by ~2.8 points, providing continuous feedback.

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

     \text{penalty} = -\gamma \times 15.0 \quad \text{(per concentrated site)}

* **Simulation failure**: :math:`R = -100 \times \gamma` if simulation does not terminate normally

**Default Hyperparameters**:

.. code-block:: yaml

   reward:
     w_P: 0.5                                # Population weight
     w_T: 0.5                                # Transition weight
     w_U: 0.3                                # Uniformity weight
     gamma: 4.0                              # Base penalty coefficient
     P_baseline: 500.0                       # Population normalization
     T_baseline: 50.0                        # Transition normalization
     min_transitions_per_site: 10            # Tier 3 threshold
     min_coverage_ratio: 0.5                 # Minimum fraction of substituents to visit
     entropy_bonus: 8.0                      # Entropy bonus coefficient
     concentration_penalty_threshold: 0.8    # Single-substituent dominance threshold

**Policy Gradient Update**:

The policy is updated using REINFORCE with the advantage function:

.. math::

   \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t) \cdot A_t \right]

where:

* :math:`\mathbf{a}_t` is the action (bias coefficients) at time :math:`t`
* :math:`\mathbf{s}_t` is the state (graph representation) at time :math:`t`
* :math:`A_t = R_{\text{total}} - b` is the advantage (with baseline :math:`b`)

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

Organize pretraining data by copying completed runs into a unified directory:

.. code-block:: bash

   pretraining/
   ├── system1_run001/
   │   ├── variables.py         # Bias coefficients used
   │   ├── info.py              # System configuration
   │   └── res/
   │       └── *_flat.lmd       # Lambda trajectory
   ├── system1_run002/
   ├── system2_run001/
   └── ...

Each directory should contain:

* ``variables.py``: Bias coefficients that were used for the simulation
* ``info.py``: System metadata (nsubs, nblocks, temp)
* ``res/*_flat.lmd``: Lambda dynamics trajectory file for computing rewards

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

**Multi-System Pretraining**:

Pretraining can combine data from multiple systems to learn generalizable
patterns:

.. code-block:: bash

   pretraining/
   ├── 14benz_solv/      # Benzene derivatives in solvent
   │   └── run_*/
   ├── 14benz_vac/       # Same system in vacuum
   │   └── run_*/
   ├── indole_prot/      # Indole derivatives in protein
   │   └── run_*/
   └── indole_solv/      # Indole derivatives in solvent
       └── run_*/

The policy learns to adapt bias coefficients based on:

* System size (number of sites, substituents)
* Environment type (vacuum, solvent, protein)
* Chemical properties (atom types, charge)

Quick Epoch
~~~~~~~~~~~

For rapid prototyping, run a single epoch without full simulations:

.. code-block:: python

   from mllf.cli.workflow import run_quick_epoch_for_combo
   
   encoder, policy, optimizer = initialize_model(...)
   
   for combo_dir in train_combos:
       loss = run_quick_epoch_for_combo(
           encoder, policy, optimizer,
           combo_dir, reward_fn
       )

This samples actions, writes ``variables.py``, computes a dummy reward,
and updates the policy once per combo.

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

**Checkpoint Frequency**: Balance storage vs resume granularity

* Lower ``checkpoint_freq`` → more frequent saves, less lost work
* Higher ``checkpoint_freq`` → less storage, more potential lost work
* Default of 5 is reasonable for most workflows

**Disk Space**: Monitor for long runs

* Each training checkpoint: ~MB (relatively small)
* Per-epoch results accumulate over time
* Consider cleanup of old checkpoints after training completes

**Reproducibility**: Optimizer state is saved

* Resumed training continues identically
* Same momentum, learning rate schedules, etc.
* No discontinuity in training dynamics

Troubleshooting
~~~~~~~~~~~~~~~

**Training doesn't resume from checkpoint**

* Verify ``save_checkpoints: true`` in YAML
* Check checkpoint directory exists with ``checkpoint_epoch_*.pt`` files
* Ensure file permissions allow reading checkpoints

**Out of memory when loading checkpoint**

* Checkpoint loads to same device (CPU/GPU) as training
* May need to adjust ``device`` in config or free GPU memory

**Per-epoch results not skipping simulations**

* Verify ``epoch_results.pt`` files exist in epoch directories
* Check file permissions
* Look for errors in log files about loading checkpoints

Testing Checkpoint Functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify checkpoint functionality is working correctly:

.. code-block:: bash

   python tests/tools/test_checkpoint_resume.py

This test script verifies:

* YAML configuration has checkpointing enabled
* Training checkpoints contain all required keys
* Per-epoch result checkpoints are saved correctly
* Checkpoint structure matches expected format

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
     w_T: 0.5
     gamma: 10.0
   
   # Change to emphasize transitions more
   reward:
     w_P: 0.3  # Changed
     w_T: 0.7  # Changed
     gamma: 10.0

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
     Parameters: w_P=0.5, w_T=0.5, gamma=10.0
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

After training completes, combination directories can be automatically archived
to save disk space. Each combination directory is compressed into a ``.tar.gz``
file, optionally removing the original directory.

Configuration
~~~~~~~~~~~~~

Enable archiving in your workflow YAML:

.. code-block:: yaml

   archive:
     enabled: true              # Enable post-training archiving
     pattern: 'comb_*'          # Glob pattern for directories to archive
     remove_after: false        # Remove originals after successful archiving
     archive_dir: /path/to/archives  # Where to store .tar.gz files

Behavior
~~~~~~~~

When ``archive.enabled`` is ``true``:

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

Use Cases
~~~~~~~~~

**Disk Space Management**: Large training runs can generate significant data.
Archiving allows:

* Preserving all simulation outputs for later analysis
* Reducing active disk usage by 50-90% (typical compression ratio)
* Organizing completed runs for long-term storage

**Best Practices**:

* Set ``remove_after: false`` initially to verify archives are valid
* Test extracting an archive to ensure contents are intact
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

Workflow Structure
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   examples/
   ├── run_workflow.py           # Main training script
   ├── workflow_sample.yaml      # Configuration file
   ├── training_test.sh          # SLURM submission script
   └── cb/
       ├── 14benz_solv_5.5/      # Base system with fragments
       │   ├── site1_sub*.rtf
       │   ├── site2_sub*.rtf
       │   ├── msld_flat.py      # Simulation script
       │   └── prep/
       └── generated_combos/      # Generated combinations
           ├── manifest.txt
           ├── train_manifest.txt
           ├── val_manifest.txt
           ├── test_manifest.txt
           └── comb_XXXX_*/       # Individual combinations

Configuration File
~~~~~~~~~~~~~~~~~~

The ``workflow_sample.yaml`` file controls all aspects of the workflow:

.. code-block:: yaml

   # Generate combinations from fragments
   create_combos:
     input_dir: /path/to/14benz_solv_5.5
     out_dir: /path/to/generated_combos
     include_patterns:
       - msld_flat.py
   
   # Split data
   split:
     train_frac: 0.70
     val_frac: 0.15
     seed: 42
   
   # Optional: Pretrain from existing simulations
   pretrain:
     data_dir: /path/to/pretraining_data
     num_epochs: 1
     model_path: models/pretrained_policy.pt
   
   # Model configuration
   training:
     num_epochs: 50
     load_pretrained: models/pretrained_policy.pt  # Use pretrained model
     encoder:
       hidden_dims: [64, 64]
       out_dim: 32
     policy:
       mlp_hidden: 64
     optimizer:
       lr: 0.001
   
   # SLURM settings
   run_sims: true
   wait_for_jobs: true
   max_concurrent_jobs: 30
   timeout: 600
   
   # Reward function
   reward:
     w_P: 0.5
     w_T: 0.5
     gamma: 10.0
     P_baseline: 1000.0
     T_baseline: 100.0
     lambda_entropy: 0.01
   
   # Output and checkpoints
   output:
     base_dir: /path/to/training_output
     save_checkpoints: true
     checkpoint_freq: 5
   
   # Archive combinations after training
   archive:
     enabled: true              # Compress combinations after training
     pattern: 'comb_*'          # Directories to archive
     remove_after: false        # Remove originals after archiving
     archive_dir: /path/to/archives  # Where to store .tar.gz files

Training Loop Details
~~~~~~~~~~~~~~~~~~~~~

The main training loop in ``run_workflow.py`` implements:

.. code-block:: python

   # Pseudocode showing the training loop structure
   for epoch in range(start_epoch, num_epochs):
       
       for combo_dir in train_combos:
           # Create epoch-specific output directory
           epoch_dir = combo_dir / f"run_{epoch:03d}"
           
           # Check for cached results (resume capability)
           if (epoch_dir / 'epoch_results.pt').exists():
               # Load cached reward, actions, logp
               # Apply REINFORCE update
               # Continue to next combo
           
           # Build graph from RTF fragments
           data, targets, extras = build_data_and_targets_from_combo(combo_dir)
           
           # Sample actions from policy
           actions, logp, mean, log_std = policy.get_actions(
               data.x, data.edge_index, data.edge_type, data.edge_attr,
               deterministic=False
           )
           
           # Write epoch-specific variables.py
           write_variables_from_actions(
               combo_dir, data, extras, actions,
               out_name=f'run_{epoch:03d}/variables.py'
           )
           
           # Submit SLURM job (non-blocking)
           # Manages up to max_concurrent_jobs
           job_id = submit_simulation_job(epoch_dir)
           job_queue.append((combo_dir, epoch_dir, job_id, actions, logp))
       
       # Wait for all jobs to complete
       for combo_dir, epoch_dir, job_id, actions, logp in job_queue:
           # Parse simulation outputs
           transitions, populations = parse_msld_output(epoch_dir)
           
           # Compute reward
           reward = compute_msld_reward(
               transitions, populations,
               w_P=0.5, w_T=0.5, gamma=10.0,
               P_baseline=1000.0, T_baseline=100.0
           )
           
           # Save epoch results (for resume)
           torch.save({
               'reward': reward,
               'actions': actions.detach().cpu(),
               'logp': logp.detach().cpu(),
               'epoch': epoch,
               'combo': combo_dir.name
           }, epoch_dir / 'epoch_results.pt')
           
           # REINFORCE update
           baseline = mean(epoch_rewards)
           advantage = reward - baseline
           loss = -(logp.sum() * advantage)
           
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
       
       # Save training checkpoint
       if (epoch + 1) % checkpoint_freq == 0:
           save_checkpoint(epoch + 1, encoder, policy, optimizer, stats)

Key Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Concurrent Job Management**: The workflow maintains a queue of active SLURM
jobs and waits when ``max_concurrent_jobs`` is reached:

.. code-block:: python

   while len(job_queue) >= max_concurrent_jobs:
       # Poll squeue to check for completed jobs
       completed_jobs = check_job_status(job_queue)
       # Remove completed jobs from queue
       time.sleep(5)  # Wait before checking again

**SLURM Job Script**: Each epoch creates a job script that:

1. Changes to the combination directory
2. Runs ``msld_flat.py`` with epoch-specific ``variables.py``
3. Writes outputs to epoch-specific directory
4. Captures stdout/stderr

**Reward Computation**: The reward function in ``src/mllf/cb/train.py``:

.. code-block:: python

   def compute_msld_reward(transitions, populations, w_P, w_T, gamma, ...):
       # Transition-based reward (higher is better)
       total_transitions = sum(transitions.values())
       R_T = total_transitions / T_baseline
       
       # Population-based reward (balanced is better)
       nonzero_pops = [p for p in populations.values() if p > 0]
       balance_factor = exp(-coefficient_of_variation(nonzero_pops))
       R_P = len(nonzero_pops) / len(populations) * balance_factor
       
       # Worst-case penalty
       if max(nonzero_pops) / sum(nonzero_pops) > 0.9 and total_transitions < 10:
           penalty = -2 * gamma
       
       # Scalarized reward
       return w_P * R_P + w_T * R_T + penalty + gamma * len(nonzero_pops)

Custom Training Loop
~~~~~~~~~~~~~~~~~~~~

For custom workflows, the key components are:

.. code-block:: python

   from mllf.file_handling.generate_combinations import create_combination_dirs
   from mllf.cli.workflow import (
       build_data_and_targets_from_combo,
       write_variables_from_actions
   )
   from mllf.cb.rgcn import RGCNEncoder
   from mllf.cb.policy import EdgePolicy
   from mllf.cb.train import compute_msld_reward
   
   # 1. Generate combinations
   combos = create_combination_dirs(
       input_dir=Path('14benz_solv_5.5'),
       output_dir=Path('generated_combos'),
       include_patterns=['msld_flat.py'],
       max_subs_per_site=10  # Limit combination size (default: 10)
   )
   
   # Note: max_subs_per_site limits each combination to at most N substituents per site.
   # All substituents can still participate, but individual combinations are capped.
   # Increase this value if you need larger combinations (may significantly increase
   # total number of combinations generated).
   
   # 2. Initialize model
   sample_data, _, sample_extras = build_data_and_targets_from_combo(combos[0])
   
   encoder = RGCNEncoder(
       in_dim=sample_data.x.size(1),
       hidden_dims=[64, 64],
       out_dim=32,
       num_relations=sample_data.edge_type.max().item() + 1
   )
   
   policy = EdgePolicy.from_pyg_data(
       encoder=encoder,
       emb_dim=32,
       data=sample_data,
       mlp_hidden=64,
       mlp_out_dim=len(sample_extras['relation_names']) // 2
   )
   
   optimizer = torch.optim.Adam(
       list(encoder.parameters()) + list(policy.parameters()),
       lr=0.001
   )
   
   # 3. Training loop (see pseudocode above for details)
   for epoch in range(num_epochs):
       # Your custom training logic here
       pass

See Also
--------

* :doc:`cb_setup` - CB architecture and graph representation details
* :doc:`examples` - Example workflows and usage patterns
* :doc:`api` - API reference for workflow modules
