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

* ``linear`` (b): Per-node linear bias vector ensuring equal population of all perturbations 
  at each site when correctly parameterized
* ``quadratic`` (c): Pairwise interaction bias that removes barriers in alchemical space 
  due to electrostatic interactions between sites (antisymmetric: :math:`c_{ij} = -c_{ji}`)
* ``skew`` (x): Asymmetry correction bias that fits residuals beyond quadratic and end 
  biases, particularly important after soft-core introduction
* ``end`` (s): End-state bias compensating for the entropic and surface tension cost 
  of displacing solvent and nearby molecules when substituents appear

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
   
   **Environment Type**: The environment (vacuum, solvent, or protein) can be 
   specified via the ``solvent_override`` parameter:
   
   * ``'solv'`` or ``'solvent'``: Solvated/aqueous environment
   * ``'gas'`` or ``'vacuum'``: Gas phase/vacuum environment
   * ``'protein'``: Protein-embedded environment
   
   If not specified, the system attempts auto-detection from directory names
   (e.g., ``14benz_solv`` → ``solv``). For production use, explicit configuration
   via workflow YAML is recommended:
   
   .. code-block:: yaml
   
      system:
        solvent_state: solv  # Explicit environment specification

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
  
  - ``is_solvent``: 1 for solvent/water environment, 0 otherwise
  - ``is_protein``: 1 for protein environment, 0 otherwise
  - Note: Vacuum/gas environments have both flags set to 0 (deprecated as explicit encoding)

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

**Important**: Each undirected edge is expanded into **two directed edges**:

* Forward relation: ``{base}_fwd`` (e.g., ``quadratic_fwd``)
* Backward relation: ``{base}_bwd`` (e.g., ``quadratic_bwd``)

This doubling is intentional and allows the model to learn directional interactions.
During variable writing, quadratic maintains antisymmetry (:math:`c_{ij} = -c_{ji}`),
while skew and end store both directions independently.

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

Per-edge coefficients are predicted by an edge-level policy network with **separate heads**
architecture. This design allows specialized predictions for each bias type while sharing
common feature representations.

**Architecture Overview**:

The ``EdgeValueMLP`` uses a two-stage design:

1. **Shared Trunk**: Two-layer MLP that processes concatenated node embeddings
   
   * Input: Concatenated node features [h_i, h_j] from encoder
   * Layer 1: Linear(in_dim → 64) + ReLU
   * Layer 2: Linear(64 → 64) + ReLU
   * Output: 64-dimensional shared representation

2. **Separate Heads**: Independent linear layers per bias type
   
   * 4 heads (one per bias type: linear, quadratic, skew, end)
   * Each head: Linear(64 → 2) outputting [mean, log_std]
   * Total output: 8 values per edge (4 means + 4 log_stds)

**Key Features**:

* **Specialized Predictions**: Each bias type gets its own predictor head
  
  - Reduces interference between different bias types
  - Allows learning type-specific patterns
  - Improves sample efficiency

* **Output Scaling**: Mean predictions use bias-specific scale factors via ``tanh(mean) * scale_factors``
  
  - **Linear**: ±61.4, **Quadratic**: ±70.5, **Skew**: ±6.6, **End**: ±3.6
  - Derived from analysis of 5,332 coefficients across 52 pretraining systems (95th percentile + 20% margin)
  - Actions clipped to 1.05× scale factors during sampling: [±64.5, ±74.0, ±6.9, ±3.8]
  - Ensures bias magnitudes are in physically meaningful ranges based on empirical data

* **Enhanced Exploration**: Log standard deviation clamped to [-20, 2.0]
  
  - Standard deviation range: [~0, 7.4]
  - Provides exploration while preventing extreme outliers
  - Higher values (e.g., 3.5 → std≈33) can produce samples far beyond intended ranges

**Usage Example**:

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

* ``actions``: Sampled coefficient values (shape: [num_edges, 4])
* ``logp``: Log-probabilities for REINFORCE updates
* ``mean``: Mean of the Gaussian distribution per edge per bias type (scaled to [-20, 20])
* ``log_std``: Log standard deviation per edge per bias type (clamped to [-20, 3.5])

Each directed edge receives 4 independent Gaussian distributions (one per bias type),
and actions are sampled independently:

.. math::

   v_{ij}^{(k)} \\sim \\mathcal{N}(\\mu_{ij}^{(k)}, (\\sigma_{ij}^{(k)})^2)

where :math:`k \\in \\{\\text{linear, quadratic, skew, end}\\}`.

Training with Actor-Critic (REINFORCE + Value Network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The policy is trained using an Actor-Critic architecture that combines REINFORCE 
with a learned value network for variance reduction.

**Components**:

* **Actor (Policy Network)**: Predicts bias coefficients from graph structure
* **Critic (Value Network)**: Predicts expected reward to provide state-dependent baselines

Value Network Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mllf.cb.value_net import ValueNetwork
   
   value_network = ValueNetwork(
       emb_dim=32,           # Node embedding dimension from encoder
       hidden_dims=[64, 32]  # MLP layers: input → 64 → 32 → 1
   )
   
   # Predict expected reward for a combination
   node_embeddings = encoder(data.x, data.edge_index, data.edge_type)
   predicted_value = value_network(node_embeddings)  # Scalar prediction

The value network uses:

1. **Global mean pooling** over node embeddings to get graph-level representation
2. **MLP prediction** of scalar expected reward
3. **MSE loss** to minimize :math:`(V(s) - R)^2`

**Benefits**:

* **Lower variance**: State-dependent baselines reduce gradient noise
* **Faster learning**: Value network trains 10x faster than policy (higher LR)
* **Better credit assignment**: Easy vs hard combinations get appropriate baselines
* **Stability**: Prevents catastrophic forgetting of pretrained weights

Training Loop Example
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch.nn.functional as F
   
   for epoch in range(num_epochs):
       for combo_dir in combos:
           # Build graph and targets
           data, targets, extras = build_data_and_targets_from_combo(combo_dir)
           
           # Encode graph
           node_embeddings = encoder(data.x, data.edge_index, data.edge_type)
           
           # Sample actions from policy
           actions, logp, _, _ = policy.get_actions(
               data.x, data.edge_index, data.edge_type, data.edge_attr,
               deterministic=False
           )
           
           # Write variables.py and run simulation
           write_variables_from_actions(combo_dir, data, extras, actions)
           run_simulation_command(combo_dir)
           reward = compute_reward(parse_simulation_results(combo_dir))
           
           # Predict expected value
           predicted_value = value_network(node_embeddings)
           advantage = reward - predicted_value.item()
           
           # Update value network
           value_loss = F.mse_loss(predicted_value, torch.tensor([reward]))
           value_optimizer.zero_grad()
           value_loss.backward()
           value_optimizer.step()
           
           # Update policy with advantage
           policy_loss = -(logp.sum() * advantage)
           optimizer.zero_grad()
           policy_loss.backward()
           optimizer.step()

**Key Insight**: The advantage :math:`A = R - V(s)` tells the policy whether the 
actual reward was better or worse than expected for this specific combination, 
providing more informative gradients than a global average baseline.

Reward Function
^^^^^^^^^^^^^^^

The reward function balances multiple simulation objectives using a tiered system with
protections against double jeopardy and unreliable data.

**Improved Reward Structure** (``compute_msld_reward_improved``):

.. code-block:: python

   def compute_msld_reward_improved(
       run_dir,
       w_P=0.5,           # Population balance weight
       w_T=0.75,          # Transition count weight  
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
       sites_below_threshold = sum(1 for t in transitions.values() if t < min_transitions_per_site)
       
       # === TIERED TRANSITION PENALTIES (multi-site aware) ===
       # Base penalty determined by worst site (minimum transitions)
       penalties = 0.0
       if min_transitions_across_sites == 0:
           base_penalty = 40.0  # Tier 1: "Death Floor" (0 transitions)
       elif min_transitions_across_sites == 1:
           base_penalty = 32.0  # Tier 1: "Death Floor" (1 transition)
       elif min_transitions_across_sites == 2:
           base_penalty = 24.0  # Tier 1: "Death Floor" (2 transitions)
       elif min_transitions_across_sites < min_transitions_per_site:
           deficit = min_transitions_per_site - min_transitions_across_sites
           base_penalty = 2.0 + 2.0 * deficit  # Tier 2: "Climbing Ramp"
       else:
           base_penalty = 0.0  # Tier 3: "Success Zone"
       
       # Multi-site degradation: add incremental penalty for each additional bad site
       if sites_below_threshold > 1:
           multisite_penalty = (sites_below_threshold - 1) * 4.0
           penalties -= (base_penalty + multisite_penalty)
       elif sites_below_threshold == 1:
           penalties -= base_penalty
       
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
       
       # === SECONDARY PENALTIES ===
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

* **Confidence Factor (C_F)**: Scales population rewards by ``min(1.0, min_transitions / (2*N_req))``.
  Low-transition runs (1-5 transitions) have unreliable population distributions and receive
  reduced ``R_P`` accordingly.

* **Tiered Transition Penalties**: Provides continuous gradient feedback instead of binary
  thresholds:
  
  - Tier 1 "Death Floor" (0-2 trans): -40.0, -32.0, -24.0 fixed penalties
  - Tier 2 "Climbing Ramp" (3-9 trans): -16.0 to -4.0 via ``-2.0 - (2.0 × deficit)``
  - Tier 3 "Success Zone" (≥10 trans): 0.0 penalty, unlocks ``R_T`` reward

**Default Configuration** (higher_rewards_v1):

Tested on indolizine pretraining data with 64.10 point separation between good and bad runs:

* ``w_P=0.5, w_T=0.75, w_U=0.3``
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
   c:  # NxN quadratic bias matrix (antisymmetric: c[j][i] = -c[i][j])
   - [0.0, 0.3, -0.1]
   - [-0.3, 0.0, 0.2]
   - [0.1, -0.2, 0.0]
   x:  # NxN skew bias matrix (NOT antisymmetric: both directions independent)
   - [0.0, 0.05, -0.02]
   - [-0.05, 0.0, 0.03]
   - [0.02, -0.03, 0.0]
   s:  # NxN end bias matrix (NOT antisymmetric: both directions independent)
   - [0.0, 0.1, -0.05]
   - [-0.1, 0.0, 0.08]
   - [0.05, -0.08, 0.0]
   '''

Forward-Only Canonical Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When writing variables.py from edge-level predictions:

**For linear bias (per-node vector)**:

Per-edge predictions are aggregated into per-node values:

.. code-block:: python

   # Average linear values from incident edges
   for (i, j), value in edge_linear_values.items():
       b[i] += value
       b[j] += value
       count[i] += 1
       count[j] += 1
   
   b = [b[i] / count[i] if count[i] > 0 else 0.0 for i in range(N)]

**For quadratic bias (antisymmetric)**:

1. Each undirected pair (i,j) stores **one canonical forward value**
2. If only the backward relation exists, use its negative: ``forward = -backward``
3. Build antisymmetric matrix: ``c[i][j] = v``, ``c[j][i] = -v``

**For skew and end biases (NOT antisymmetric)**:

1. Each directed pair (i→j) stores its **own independent value**
2. Forward and backward relations are stored separately
3. Build full matrix: ``x[i][j] = v_ij``, ``x[j][i] = v_ji`` (where v_ij ≠ -v_ji)

See Also
--------

* :doc:`workflow` - Complete workflow from combo generation to training
* :doc:`examples` - Running the full training workflow
* :doc:`api` - API reference for CB modules
* ``examples/run_workflow.py`` - Full training implementation
* ``examples/workflow_sample.yaml`` - Configuration file template
