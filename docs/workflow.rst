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
     sites_file: examples/sites.txt
     substituents_file: examples/substituents.txt
     output_dir: examples/cb
     r: 3  # Number of substituents per combination
     base_dir: examples/14benz_solv_base
   
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
   
   # Run MD simulations
   run_sims: true
   compress_after: true  # Archive outputs after each simulation

Combination Generation
----------------------

Principles
~~~~~~~~~~

Combinations are generated from two input files:

* **sites.txt**: Available λ-sites on the molecule (e.g., ``site1``, ``site2``)
* **substituents.txt**: Available substituent groups (e.g., ``sub1``, ``sub2``)

The generator creates all valid r-combinations with constraints:

1. **First element fixed**: Permutations are ordered by the first site
2. **Tail unordered**: Remaining (r-1) elements form an unordered set

This prevents duplicate combinations that differ only by tail permutation order.

Example
~~~~~~~

For sites ``[1, 2, 3, 4]`` and ``r=3``:

* ✓ Generated: ``[1, 2, 3]``, ``[1, 2, 4]``, ``[1, 3, 4]``, ``[2, 3, 4]``
* ✗ Not generated: ``[1, 3, 2]`` (same as ``[1, 2, 3]``), ``[1, 4, 2]`` (same as ``[1, 2, 4]``)

For 4 sites, 4 substituents per site, r=3, this generates 261 combinations
instead of 2270 if all permutations were included.

Directory Structure
~~~~~~~~~~~~~~~~~~~

Each combination creates a directory:

.. code-block:: text

   examples/cb/
   ├── 14benz_solv_5.5/              # Example: site1_2 + site1_3 + site1_4
   │   ├── site1_2_solv_pres.rtf
   │   ├── site1_3_solv_pres.rtf
   │   ├── site1_4_solv_pres.rtf
   │   ├── variables.py               # CB training writes here
   │   └── output/                    # Simulation outputs
   │       ├── transitions.txt
   │       └── populations.txt
   └── 14benz_solv_6.6/              # Another combination
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

* ``x``: Node features (typically one-hot or constant)
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

.. code-block:: python

   from mllf.cli.workflow import run_from_config
   
   # Single command runs everything
   run_from_config('examples/workflow_sample.yaml')

This executes:

1. Combination generation (if ``create_combos`` specified)
2. Manifest splitting (if ``split`` specified)
3. Training (if ``train`` specified)
4. Simulation (if ``run_sims: true``)
5. Compression (if ``compress_after: true``)

Custom Training Loop
~~~~~~~~~~~~~~~~~~~~

For more control:

.. code-block:: python

   from mllf.cli.workflow import (
       create_and_manifest, split_manifest,
       build_data_and_targets_from_combo,
       write_variables_from_actions
   )
   from mllf.cb.rgcn import RGCNEncoder
   from mllf.cb.policy import EdgePolicy
   import torch
   
   # 1. Generate combinations
   create_and_manifest(
       sites_file='examples/sites.txt',
       substituents_file='examples/substituents.txt',
       output_dir='examples/cb',
       r=3,
       base_dir='examples/14benz_solv_base',
       manifest_path='examples/manifest.txt'
   )
   
   # 2. Split into train/val/test
   split_manifest(
       manifest='examples/manifest.txt',
       train_out='examples/train.txt',
       val_out='examples/val.txt',
       test_out='examples/test.txt',
       train_fraction=0.7,
       val_fraction=0.15
   )
   
   # 3. Initialize model
   train_combos = [line.strip() for line in open('examples/train.txt')]
   sample_data, _, sample_extras = build_data_and_targets_from_combo(train_combos[0])
   
   encoder = RGCNEncoder(
       in_dim=sample_data.x.size(1),
       hidden_dims=[64, 64],
       out_dim=32,
       num_relations=sample_data.edge_type.max().item() + 1
   )
   
   policy = EdgePolicy.from_pyg_data(
       encoder=encoder,
       hidden_dim=32,
       sample_data=sample_data,
       mlp_hidden=64,
       mlp_out_dim=len(sample_extras['relation_names']) // 2
   )
   
   optimizer = torch.optim.Adam(
       list(encoder.parameters()) + list(policy.parameters()),
       lr=1e-3
   )
   
   # 4. Training loop
   for epoch in range(10):
       epoch_loss = 0.0
       for combo_dir in train_combos:
           data, targets, extras = build_data_and_targets_from_combo(combo_dir)
           
           # Forward pass
           actions, logp, mean, log_std = policy.get_actions(
               data.x, data.edge_index, data.edge_type, data.edge_attr,
               deterministic=False
           )
           
           # Write variables for simulation
           write_variables_from_actions(combo_dir, data, extras, actions)
           
           # Run simulation (example command)
           import subprocess
           subprocess.run(['charmm', '-i', 'input.inp'], cwd=combo_dir)
           
           # Compute reward from simulation output
           reward = compute_reward_from_output(combo_dir)
           
           # REINFORCE update
           loss = -(logp.sum() * reward)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           epoch_loss += loss.item()
       
       print(f'Epoch {epoch}: Loss = {epoch_loss / len(train_combos):.4f}')

See Also
--------

* :doc:`cb_setup` - CB architecture and graph representation details
* :doc:`examples` - Example workflows and usage patterns
* :doc:`api` - API reference for workflow modules
