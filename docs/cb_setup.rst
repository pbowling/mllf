Contextual Bandit Setup
=======================

Overview
--------

The contextual bandit (CB) training framework provides a reinforcement learning approach to
optimizing bias coefficients for multisite λ-dynamics simulations. Instead of hand-tuning
bias parameters, we use graph neural networks to predict optimal coefficients based on the
molecular graph structure.

Core Components
---------------

Graph Representation
~~~~~~~~~~~~~~~~~~~~

Molecular systems are represented as undirected graphs where:

* **Nodes** represent λ-sites (substituent positions on the molecule)
* **Edges** represent interactions between sites with associated bias coefficients

Each edge can have multiple bias types:

* ``linear`` (b): Per-node linear bias ensuring equal population of all perturbations 
  at each site when correctly parameterized
* ``quadratic`` (c): Pairwise interaction bias that removes barriers in alchemical space 
  due to electrostatic interactions between sites
* ``skew`` (x): Asymmetry correction bias that fits residuals beyond quadratic and end 
  biases, particularly important after soft-core introduction
* ``end`` (s): End-state bias compensating for the entropic and surface tension cost 
  of displacing solvent and nearby molecules when substituents appear

**Key Property**: All bias matrices are antisymmetric:

.. math::

   B_{ij} = -B_{ji}

This ensures that the bias interaction from site i→j is the negative of j→i,
maintaining physical consistency in the bidirectional interactions.

Graph Construction
^^^^^^^^^^^^^^^^^^

Graphs are built using one of two methods:

1. **From RTF Fragments** (preferred):
   
   .. code-block:: python
   
      from mllf.file_handling.read_rtf import parse_rtf_dir
      from mllf.cb.graph import Graph
      
      rtf_results = parse_rtf_dir('combo_dir')
      graph = Graph.from_rtf_results(rtf_results)
      
      # Optional: Override environment type for all nodes
      # graph = Graph.from_rtf_results(rtf_results, solvent_override='protein')

   This method reads ``site*_sub*_*_pres.rtf`` files and extracts connectivity
   information from the RTF topology fragments.
   
   **Environment Detection**: The environment type (vacuum, solvent, or protein) is 
   automatically detected from filenames. You can override this behavior using the 
   ``solvent_override`` parameter with values: ``'gas'``/``'vacuum'``, ``'solv'``/``'solvent'``, 
   or ``'protein'``.
   
   .. note::
      While ``'gas'``/``'vacuum'`` is still accepted for compatibility, it has been 
      deprecated as a distinct environment encoding in node features. Vacuum environments 
      are now represented as neither solvent nor protein (both flags set to 0.0).

2. **From variables.py** (fallback):
   
   .. code-block:: python
   
      from mllf.cli.workflow import load_bias_from_variables, graph_from_bias
      
      bias = load_bias_from_variables('combo_dir/variables.py')
      graph = graph_from_bias(bias)

   This parses existing bias matrices from a ``variables.py`` file containing
   a triple-quoted YAML ``bias_string``.

PyTorch Geometric Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For neural network training, graphs are converted to PyTorch Geometric format:

.. code-block:: python

   from mllf.cb import graph_utils
   
   data, extras = graph_utils.build_pyg_graph_from_mllf_graph(graph)

**Node Features**: Each node is represented by a feature vector with:

* Total molecular charge (float) - sum of partial charges in the substituent
* Binary indicators for environment type:
  
  - ``is_solvent``: 1.0 for solvent/water environment, 0.0 otherwise
  - ``is_protein``: 1.0 for protein environment, 0.0 otherwise
  - Note: Vacuum/gas environments have both flags set to 0.0 (deprecated as explicit encoding)

* Multi-hot encoding of chemical elements present in the substituent 
  (e.g., C, H, N, O from the periodic table)
* Multi-hot encoding of distinct CHARMM atom types present in the substituent
  (e.g., CG2R61, HGR61, NG2R60)

This two-level encoding provides both coarse-grained (element) and fine-grained 
(atom type) chemical information while being more efficient than a single large 
encoding.

**Vocabularies**: The vocabularies are loaded from CHARMM toppar files
in the ``toppar/`` directory, which contain MASS entries defining atom types 
and their corresponding elements. By default, only CGenFF is loaded:

* **Element vocabulary**: 14 elements (Al, B, Br, C, Cl, F, H, I, N, O, P, S, Se, X)
* **Atom type vocabulary**: 161 CGenFF atom types
* **Total feature dimensions**: ``3 + 14 + 161 = 178`` for CGenFF only

This ensures:

* Consistent feature dimensions across all graphs
* No vocabulary mismatch between training and inference
* Support for unseen atom types during deployment (with warnings)

**Important**: Each undirected edge is expanded into **two directed edges**:

* Forward relation: ``{base}_fwd`` (e.g., ``quadratic_fwd``)
* Backward relation: ``{base}_bwd`` (e.g., ``quadratic_bwd``)

This doubling is intentional and allows the model to learn directional interactions
while maintaining the antisymmetry constraint during variable writing.

The ``extras`` dict contains:

* ``relation_names``: List mapping relation indices to names
* ``base_relation_map``: Dict mapping base types to (forward, backward) relation names
* ``atom_type_vocab``: Dict mapping atom type strings to feature indices
* ``element_vocab``: Dict mapping element symbols to feature indices
* ``atom_to_element``: Dict mapping atom types to their elements

Policy Network Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RGCN Encoder
^^^^^^^^^^^^

Node embeddings are computed using a Relational Graph Convolutional Network:

.. code-block:: python

   from mllf.cb.rgcn import RGCNEncoder
   
   # input_features = 3 + len(element_vocab) + len(atom_type_vocab)
   # For CGenFF only: 3 + 14 + 161 = 178
   encoder = RGCNEncoder(
       in_dim=input_features,
       hidden_dims=[64, 64],
       out_dim=32,
       num_relations=num_edge_types
   )
   
   node_embeddings = encoder(x, edge_index, edge_type)

The RGCN handles different edge types (relation types) explicitly, learning
separate transformation matrices for each interaction type.

Edge Policy
^^^^^^^^^^^

Per-edge coefficients are predicted by an edge-level MLP:

.. code-block:: python

   from mllf.cb.policy import EdgePolicy
   
   policy = EdgePolicy.from_pyg_data(
       encoder=encoder,
       hidden_dim=32,
       sample_data=data,
       mlp_hidden=64,
       mlp_out_dim=num_bases  # e.g., 4 for linear/quadratic/skew/end
   )
   
   # Sample actions (stochastic)
   actions, logp, mean, log_std = policy.get_actions(
       x, edge_index, edge_type, edge_attr,
       deterministic=False
   )

The policy outputs:

* ``actions``: Sampled coefficient values (one per directed edge)
* ``logp``: Log-probabilities for REINFORCE updates
* ``mean``: Mean of the Gaussian distribution per edge
* ``log_std``: Log standard deviation per edge

Each directed edge receives its own Gaussian distribution, and actions are sampled
independently per edge:

.. math::

   v_{ij} \\sim \\mathcal{N}(\\mu_{ij}, \\sigma_{ij}^2)

Training with REINFORCE
~~~~~~~~~~~~~~~~~~~~~~~

The policy is trained using the REINFORCE algorithm:

1. **Sample actions** from the policy for all edges
2. **Write variables.py** with the sampled coefficients
3. **Run simulation** to collect metrics (transitions, populations)
4. **Compute reward** from simulation results
5. **Update policy** using policy gradient:

.. math::

   \\nabla_\\theta J = \\mathbb{E}\\left[\\sum_{edges} \\nabla_\\theta \\log \\pi_\\theta(a_e) \\cdot R\\right]

Training Loop Example
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   for epoch in range(num_epochs):
       for combo_dir in combos:
           # Build graph and targets
           data, targets, extras = build_data_and_targets_from_combo(combo_dir)
           
           # Sample actions from policy
           actions, logp, _, _ = policy.get_actions(
               data.x, data.edge_index, data.edge_type, data.edge_attr,
               deterministic=False
           )
           
           # Write variables.py for simulator
           write_variables_from_actions(combo_dir, data, extras, actions)
           
           # Run simulation and get reward
           run_simulation_command(combo_dir)
           sim_results = parse_simulation_results(combo_dir)
           reward = compute_reward(sim_results)
           
           # Update policy (REINFORCE)
           loss = -(logp.sum() * reward)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

Reward Function
^^^^^^^^^^^^^^^

The reward function balances multiple simulation objectives using a tiered system with
protections against double jeopardy and unreliable data.

**Improved Reward Structure** (``compute_msld_reward_improved``):

.. code-block:: python

   def compute_msld_reward_improved(
       run_dir,
       w_P=0.5,           # Population balance weight
       w_T=0.5,           # Transition count weight  
       w_U=0.3,           # Coverage uniformity weight
       gamma=4.0,         # Penalty scaling factor
       P_baseline=500.0,  # Population normalization (lower = higher rewards)
       T_baseline=50.0,   # Transition normalization (lower = higher rewards)
       min_transitions_per_site=10,  # Threshold for "Success Zone"
       min_coverage_ratio=0.5,       # Minimum % of substituents visited
       entropy_bonus=8.0,            # Bonus for uniform distributions
       concentration_penalty_threshold=0.8  # Max concentration ratio
   ):
       # Parse simulation outputs
       populations = parse_populations(run_dir)
       transitions = parse_transitions(run_dir)
       
       # Track minimum transitions across all sites
       min_transitions_across_sites = min(transitions.values())
       
       # === PENALTY SHIELD (I_dead) ===
       # Prevents "double jeopardy" for frozen simulations
       I_dead = 1 if min_transitions_across_sites == 0 else 0
       
       # === TIERED TRANSITION PENALTIES ===
       penalties = 0.0
       for site, trans in transitions.items():
           if trans == 0:
               penalties -= 40.0  # Tier 1: "Death Floor"
           elif trans < min_transitions_per_site:
               deficit = min_transitions_per_site - trans
               penalties -= (5.0 + 2.8 * deficit)  # Tier 2: "Climbing Ramp"
           # Tier 3: "Success Zone" (>=10) gets no penalty
       
       # === CONFIDENCE FACTOR (C_F) ===
       # Scale population rewards by data reliability
       confidence_factor = min(1.0, min_transitions_across_sites / (2.0 * min_transitions_per_site))
       
       # === REWARD COMPONENTS ===
       
       # R_P: Population balance (scaled by confidence)
       pop_array = np.array(list(populations.values()))
       nonzero_pops = pop_array[pop_array > 0]
       if len(nonzero_pops) > 1:
           cv = np.std(nonzero_pops) / np.mean(nonzero_pops)
           balance_factor = np.exp(-cv)
           total_pop_normalized = sum(p / P_baseline for p in nonzero_pops)
           R_P = w_P * total_pop_normalized * balance_factor * confidence_factor
       else:
           R_P = 0.0
       
       # R_T: Transitions (only if all sites meet threshold)
       sites_below_threshold = sum(1 for t in transitions.values() 
                                  if t < min_transitions_per_site)
       if sites_below_threshold == 0:
           total_trans = sum(transitions.values())
           R_T = w_T * (total_trans / T_baseline)
       else:
           R_T = 0.0
       
       # R_U: Coverage uniformity
       coverage_ratio = len(nonzero_pops) / len(populations)
       R_U = w_U * coverage_ratio
       
       # R_entropy: Shannon entropy bonus
       pop_probs = pop_array / pop_array.sum()
       entropy = -np.sum(pop_probs * np.log(pop_probs + 1e-10))
       max_entropy = np.log(len(pop_probs))
       R_entropy = entropy_bonus * (entropy / max_entropy)
       
       # === SECONDARY PENALTIES (only when shield inactive) ===
       if (1 - I_dead):
           # Coverage penalty
           if coverage_ratio < min_coverage_ratio:
               deficit = min_coverage_ratio - coverage_ratio
               penalties -= gamma * 20.0 * deficit
           
           # Concentration penalty (per-site check)
           for site_pops in get_site_populations(populations):
               concentration = max(site_pops) / sum(site_pops)
               if concentration > concentration_penalty_threshold:
                   penalties -= gamma * 5.0 * (concentration - concentration_penalty_threshold)
       
       # Total reward
       reward = R_P + R_T + R_U + R_entropy + penalties
       return reward

**Key Mechanisms**:

* **Penalty Shield (I_dead)**: When ``min_transitions_across_sites = 0``, disables secondary
  penalties (coverage, concentration) to prevent "double jeopardy" on frozen simulations.

* **Confidence Factor (C_F)**: Scales population rewards by ``min(1.0, min_transitions / (2*N_req))``.
  Low-transition runs (1-5 transitions) have unreliable population distributions and receive
  reduced ``R_P`` accordingly.

* **Tiered Transition Penalties**: Provides continuous gradient feedback instead of binary
  thresholds:
  
  - Tier 1 "Death Floor" (0 trans): -40.0 penalty per site
  - Tier 2 "Climbing Ramp" (1-9 trans): -30.2 to -7.8 (linear gradient)
  - Tier 3 "Success Zone" (≥10 trans): 0.0 penalty, unlocks ``R_T`` reward

**Default Configuration** (higher_rewards_v1):

Tested on indolizine pretraining data with 64.10 point separation between good and bad runs:

* ``w_P=0.5, w_T=0.5, w_U=0.3``
* ``gamma=4.0, P_baseline=500, T_baseline=50``
* ``entropy_bonus=8.0, min_transitions_per_site=10``

Higher rewards indicate better bias coefficients that improve sampling efficiency.

Variables.py Format
~~~~~~~~~~~~~~~~~~~

The simulator reads bias coefficients from a ``variables.py`` file:

.. code-block:: python

   # Auto-generated variables.py — bias_string contains YAML for bias matrices
   bias_string = '''
   b:  # Per-node linear bias vector (length N)
   - 0.1
   - 0.2
   - -0.05
   c:  # NxN quadratic bias matrix (antisymmetric)
   - [0.0, 0.3, -0.1]
   - [-0.3, 0.0, 0.2]
   - [0.1, -0.2, 0.0]
   x:  # NxN skew bias matrix (antisymmetric)
   - [0.0, 0.05, 0.0]
   - [-0.05, 0.0, 0.0]
   - [0.0, 0.0, 0.0]
   s:  # NxN end bias matrix (antisymmetric)
   - [0.0, 0.0, 0.0]
   - [0.0, 0.0, 0.0]
   - [0.0, 0.0, 0.0]
   '''

Forward-Only Canonical Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When writing variables.py from edge-level predictions:

1. Each undirected pair (i,j) stores **one canonical forward value**
2. If only the backward relation exists, use its negative: ``forward = -backward``
3. Build antisymmetric matrices: ``matrix[i][j] = v``, ``matrix[j][i] = -v``

For linear bias, per-edge predictions are aggregated into per-node values:

.. code-block:: python

   # Average linear values from incident edges
   for (i, j), value in edge_linear_values.items():
       b[i] += value
       b[j] += value
       count[i] += 1
       count[j] += 1
   
   b = [b[i] / count[i] if count[i] > 0 else 0.0 for i in range(N)]

See Also
--------

* :doc:`workflow` - Complete workflow from combo generation to training
* :doc:`examples` - Running the full training workflow
* :doc:`api` - API reference for CB modules
* ``examples/run_workflow.py`` - Full training implementation
* ``examples/workflow_sample.yaml`` - Configuration file template
