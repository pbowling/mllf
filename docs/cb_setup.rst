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

Node embeddings are computed using a 3-layer Relational Graph Convolutional Network 
that handles different edge types explicitly by learning separate transformation matrices 
for each relation type. The standard architecture uses:

.. math::

   \text{RGCN}: \mathbb{R}^{178} \to \mathbb{R}^{64} \to \mathbb{R}^{64} \to \mathbb{R}^{32}

where the input dimension is 178 for CGenFF (3 scalar features + 14 elements + 161 atom types) 
and the output produces 32-dimensional node embeddings used by the policy and value networks.

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

Training and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

The policy network is trained using an **Actor-Critic** architecture (REINFORCE with learned 
value network) that provides state-dependent baselines for variance reduction.

**Actor-Critic Components**:

* **Actor (Policy Network)**: RGCN encoder + EdgeValueMLP that predicts bias coefficients from graph structure
* **Critic (Value Network)**: 3-layer MLP that predicts expected reward for a combination

**Value Network Architecture**:

The value network maps node embeddings to expected rewards via global pooling and MLP:

.. math::

   V(\mathbf{s}) = \text{MLP}_{32 \to 64 \to 32 \to 1}\left(\text{GlobalMeanPool}(\mathbf{H})\right)

where :math:`\mathbf{H} \in \mathbb{R}^{N \times 32}` are node embeddings from the RGCN encoder. 
The network is trained to minimize :math:`(V(s) - R)^2` where :math:`R` is the actual reward.

**Training Updates**:

For each combination:

1. Encode graph to node embeddings :math:`\mathbf{H}` via RGCN
2. Sample bias coefficients :math:`\mathbf{a} \sim \pi_\theta(\cdot | \mathbf{s})` from policy
3. Run simulation and compute reward :math:`R` from metrics
4. Compute advantage: :math:`A = R - V_\phi(\mathbf{s})`
5. Update value network: minimize :math:`(V_\phi(\mathbf{s}) - R)^2`
6. Update policy: maximize :math:`\log \pi_\theta(\mathbf{a} | \mathbf{s}) \cdot A`

**Benefits**:

* **Lower variance**: State-dependent baselines reduce gradient noise compared to fixed baselines
* **Stability**: Prevents catastrophic forgetting of pretrained weights during RL training
* **Better credit assignment**: Difficult combinations get lower baseline expectations

For details on reward function components, curriculum learning, and workflow configuration, 
see :doc:`workflow`.

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
   c:  # NxN quadratic bias matrix (antisymmetric: c[j][i] = -c[i][j], use only upper triangle)
   - [0.0, 0.3, -0.1]
   - [0.0, 0.0, 0.2]
   - [0.0, 0.0, 0.0]
   x:  # NxN skew bias matrix (NOT antisymmetric: both directions independent)
   - [0.0, 0.05, -0.02]
   - [-0.05, 0.0, 0.03]
   - [0.02, -0.03, 0.0]
   s:  # NxN end bias matrix (NOT antisymmetric: both directions independent)
   - [0.0, 0.1, -0.05]
   - [-0.1, 0.0, 0.08]
   - [0.05, -0.08, 0.0]
   '''

Edge-to-Matrix Mapping
^^^^^^^^^^^^^^^^^^^^^^

The policy network predicts per-edge coefficients which are mapped to bias matrices:

* **Linear bias (b)**: Per-edge predictions are averaged at each node to produce the 
  per-node linear bias vector (length N)

* **Quadratic bias (c)**: Antisymmetric matrix where each undirected pair (i,j) uses 
  one canonical forward value to set ``c[i][j] = v`` and ``c[j][i] = -v``

* **Skew (x) and End (s) biases**: Full matrices where forward and backward directions 
  are stored independently (NOT antisymmetric): ``x[i][j]`` and ``x[j][i]`` are separate values

See Also
--------

* :doc:`workflow` - Complete workflow from combo generation to training
* :doc:`examples` - Running the full training workflow
* :doc:`api` - API reference for CB modules
* ``examples/run_workflow.py`` - Full training implementation
* ``examples/workflow_sample.yaml`` - Configuration file template
