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
   
   # Training configuration
   training:
     num_epochs: 50
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

Directory Structure
~~~~~~~~~~~~~~~~~~~

Each combination creates a directory with a standardized name:

.. code-block:: text

   generated_combos/
   ├── comb_0001_site2_1__site2_2/          # Within-site: site2 with subs 1,2
   │   └── prep/
   │       ├── site2_sub1_label.rtf
   │       ├── site2_sub1_label.pdb
   │       ├── site2_sub2_label.rtf
   │       ├── site2_sub2_label.pdb
   │       └── support_files...
   ├── comb_0075_site1_5__site1_1__site1_2/  # Within-site: site1 with subs 5,1,2
   │   └── prep/
   ├── comb_0262_site1_1__site1_2__site2_1__site2_2/  # Cross-site combination
   │   ├── info.py                          # Configuration metadata
   │   ├── mapping.json                     # File renumbering mapping
   │   ├── run.sh                           # Job submission script
   │   └── prep/
   │       ├── site1_sub1_label.rtf         # Files from both sites
   │       ├── site1_sub2_label.rtf
   │       ├── site2_sub1_label.rtf
   │       └── site2_sub2_label.rtf
   └── ...

Manifest Files
~~~~~~~~~~~~~~

A manifest lists combination directories (one per line):

.. code-block:: text

   examples/cb/14benz_solv_5.5
   examples/cb/14benz_solv_6.6
   examples/cb/14benz_solv_5.6

Manifests enable reproducible splits and batch operations.

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

Graph Structure
^^^^^^^^^^^^^^^

The ``Graph`` object stores:

* ``nodes``: List of node labels (e.g., ``['site1_2', 'site1_3', 'site1_4']``)
* ``edge_coeffs``: ``EdgeCoeffs`` object mapping edge types to node pairs

Edge types include:

* ``linear``: Per-node bias (actually stored per edge, aggregated to nodes)
* ``quadratic``: Quadratic interaction
* ``skew``: Skew bias
* ``end``: End-state bias

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

For neural network training:

.. code-block:: python

   from mllf.cb import graph_utils
   
   data, extras = graph_utils.build_pyg_graph_from_mllf_graph(graph)

Key properties of ``data``:

* ``x``: Node features ``[num_nodes, 4 + vocab_size]`` containing:
  
  - ``total_charge``: Molecular charge (float)
  - ``is_vacuum``: Binary indicator for vacuum/gas environment
  - ``is_solvent``: Binary indicator for solvent/water environment
  - ``is_protein``: Binary indicator for protein environment
  - Multi-hot encoding of distinct atom types (e.g., CG2R61, HGR61, NG2R60)
  
  The atom type vocabulary is loaded from CHARMM toppar files (default: 333 types)
  ensuring consistent feature dimensions across all training runs.

* ``edge_index``: COO format edge indices ``[2, num_edges*2]``
* ``edge_type``: Relation type per edge (int indices)
* ``edge_attr``: Optional edge attributes

**Directed Edge Expansion**: Each undirected edge becomes two directed edges
with forward/backward relation types.

Training Pipeline
-----------------

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
   
   # Model configuration
   training:
     num_epochs: 50
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
       include_patterns=['msld_flat.py']
   )
   
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
