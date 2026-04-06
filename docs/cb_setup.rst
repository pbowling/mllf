Contextual Bandit Setup
=======================

Overview
--------

The contextual bandit (CB) training framework provides a reinforcement learning approach to
optimizing bias coefficients for multisite λ-dynamics simulations. Instead of hand-tuning
bias parameters, we use graph neural networks to predict optimal coefficients based on the
molecular graph structure.

**Architecture Pipeline**:

The policy network uses a two-stage architecture to predict bias coefficients:

1. **RGCN Encoder**: Processes the molecular graph with DeepSet embeddings (64D) as initial 
   node features, producing refined context-aware node embeddings (32D). The RGCN learns 
   separate transformations for different bias types, allowing it to capture bias-specific 
   molecular interactions.

2. **EdgeValueMLP Policy**: Takes pairs of RGCN-processed node embeddings and predicts 
   bias coefficients for each edge. Uses a shared trunk with separate heads per bias type,
   outputting Gaussian distributions (mean, std) for sampling actions.

This separation allows the encoder to learn general molecular representations while the 
policy network specializes in predicting bias-specific coefficients for individual transitions.

Core Components
---------------

Graph Representation
~~~~~~~~~~~~~~~~~~~~

Molecular systems are represented as directed graphs where:

* **Nodes** represent individual substituents at each λ-site
* **Edges** represent transitions between substituents with associated bias coefficients

For a 2-site system with 3 substituents at site 1 and 2 substituents at site 2, the graph
contains 5 nodes total (one per substituent). Edges connect substituents within the same site,
allowing the model to predict bias coefficients for all possible transitions.

**Bias Types**:

Each edge can have multiple bias coefficient types:

* **Linear (b)**: Per-node bias ensuring equal population of all substituents at each site
  when correctly parameterized.

* **Quadratic (c)**: Pairwise interaction bias removing alchemical barriers due to 
  electrostatic interactions between sites. Antisymmetric: :math:`c_{ij} = -c_{ji}`, 
  meaning the forward and backward transitions have equal magnitude but opposite sign.

* **Skew (x)**: Asymmetry correction fitting residuals beyond quadratic and end biases,
  particularly important after soft-core introduction. Forward and backward transitions
  are independent (not antisymmetric).

* **End (s)**: End-state bias compensating for entropic and surface tension costs of
  displacing solvent and nearby molecules when substituents appear. Forward and backward
  transitions are independent (not antisymmetric).

Graph Construction
^^^^^^^^^^^^^^^^^^

Graphs are constructed with **DeepSet embeddings** as the primary node features, representing
each substituent's 3D atomic structure and chemical composition as a learned 64-dimensional
vector. These embeddings replace manual feature engineering with neural representations
pretrained on diverse molecular data.

**DeepSet-Based Construction**:

The standard construction pipeline:

1. Parse RTF files to identify substituents and extract metadata (site numbers, charges, atom types)
2. Build graph topology: one node per substituent, edges connecting substituents within each site
3. Compute DeepSet embeddings for each node from PDB coordinates and RTF charges
4. Store embeddings as node features for neural network input

The DeepSet embeddings capture rich molecular information automatically:

* **Spatial structure**: Bond lengths, angles, 3D conformations from atomic coordinates
* **Chemical composition**: Element types, functional groups, charge distributions
* **Environmental context**: Nearby protein atoms and core structure atoms (via context-aware AEV computation)

See :doc:`deepset_pretraining` for technical details on the 4-step pretraining pipeline
(atom-level AEV features → autoencoder → sum-pooling aggregation).

**Environmental Context Encoding**:

The environment type influences how DeepSet embeddings are computed. When a
``minimized.pdb`` file is present in the prep directory, post-minimization coordinates
are used to provide the most accurate representation of each atom's environment—the
minimized geometry reflects the actual sampled ensemble rather than the initial placement.

* **Protein systems**: All protein atoms within 5.1 Å of the substituent are extracted
  from ``minimized.pdb`` and included in AEV computation. This encodes protein-specific
  interactions (hydrogen bonds, hydrophobic contacts, electrostatics) directly into the
  molecular representation. Falls back to a standalone ``protein.pdb`` if no
  ``minimized.pdb`` is found.

* **Solvent systems**: Water molecules within 5.1 Å of the substituent are extracted
  from ``minimized.pdb`` and included as solvent context, capturing the immediate
  solvation shell. Without ``minimized.pdb``, only the core and nearby substituents
  from other sites contribute.

* **Vacuum systems**: No additional environment atoms (core + other-site substituents
  within cutoff only). ``minimized.pdb`` is checked but not used for extra context.


This context-aware approach eliminates the need for explicit environment flags as node
features—the environmental information is implicitly encoded in the embeddings themselves.

**Legacy RTF-Only Construction**:

Graphs can also be built directly from RTF topology fragments without DeepSet embeddings,
using manually engineered features (atom counts, charge, element compositions). This approach
is maintained for backward compatibility and systems where PDB coordinates are unavailable,
but the DeepSet-based method is strongly preferred for production use due to its superior
representation quality.


Neural Network Graph Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For neural network processing, the molecular graph (with its DeepSet node embeddings) is
converted to PyTorch Geometric format. This conversion handles the technical details of
edge expansion and relation type encoding for the RGCN policy network.

**Node Features**:

The 64-dimensional DeepSet embeddings computed during graph construction become the node
feature matrix. Each row represents one substituent with its learned molecular representation
already encoding all necessary information (structure, chemistry, environment). No additional
feature engineering is required—the embeddings serve as direct input to the RGCN.

**Edge Expansion**:

Each undirected molecular edge is expanded into **directed relation edges** based on bias type:

* **Linear bias**: Only edges FROM reference substituent (sub1) TO others
  
  - Creates one directed edge per transition (e.g., sub1→sub2, sub1→sub3)
  - No backward edges (sub2→sub1, sub3→sub1) since linear bias is node-level

* **Quadratic bias**: Only upper-triangle edges (i→j where i < j)
  
  - Creates one directed edge per undirected pair
  - Antisymmetry enforced during coefficient mapping (forward value negated for backward)

* **Skew and End biases**: Both forward AND backward edges (i→j and j→i)
  
  - Creates two directed edges per undirected pair
  - Independent values for each direction (no symmetry constraint)

Each directed edge has a relation type (``linear_fwd``, ``quadratic_fwd``, ``skew_bwd``, etc.)
that identifies which bias type and direction it represents. The RGCN learns separate
transformation matrices for each relation type, allowing bias-specific edge processing.

Policy Network Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RGCN Encoder
^^^^^^^^^^^^

Node embeddings are computed using a 3-layer Relational Graph Convolutional Network 
that handles different edge types explicitly by learning separate transformation matrices 
for each relation type. The standard architecture uses:

.. math::

   \text{RGCN}: \mathbb{R}^{64} \to \mathbb{R}^{64} \to \mathbb{R}^{64} \to \mathbb{R}^{32}

The input to the RGCN is the **64-dimensional DeepSet embedding** for each substituent node.
A **LayerNorm** layer is applied to these embeddings before the first convolution. This
normalises the per-feature mean and variance across nodes, stabilising training when
sum-pool magnitudes vary with substituent atom count (5–50 atoms). The learnable γ/β
parameters preserve size-related information after normalisation.

The RGCN then processes the normalised embeddings through 3 layers of relational graph
convolutions, where each layer learns separate transformation matrices for different bias
types (linear, quadratic, skew, end). The final output produces 32-dimensional node
embeddings used by the policy and value networks.


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
  
  - **Linear**: ±305, **Quadratic**: ±520, **Skew**: ±85, **End**: ±30
  - Derived from a full scan of 20,000+ pretraining runs with margin above the empirical maximum
  - Covers the full observed range: linear max 235, quadratic max 470, skew max 77, end max 27
  - Earlier bounds (±61–±70) were derived from a small biased sample and clipped ~17–34% of targets

* **Enhanced Exploration**: Log standard deviation clamped to [-20, 2.0]
  
  - Standard deviation range: [~0, 7.4]
  - Provides exploration while preventing extreme outliers
  - Higher values (e.g., 3.5 → std≈33) can produce samples far beyond intended ranges

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

For details on reward function components, curriculum learning, and workflow configuration, 
see :doc:`workflow`.

Variables.py Format
~~~~~~~~~~~~~~~~~~~

MSLD simulation setup files read bias coefficients from ``variables.py`` files containing YAML-formatted
bias matrices. The policy network's per-edge predictions are assembled into these matrices
following specific composition rules for each bias type.

**Matrix Format**:

Bias coefficients are organized as:

* **b (linear)**: 1D vector of length N (one value per substituent)
* **c (quadratic)**: N×N antisymmetric matrix (upper triangle stored)
* **x (skew)**: N×N full matrix (both triangles stored independently)
* **s (end)**: N×N full matrix (both triangles stored independently)

For a system with N=5 substituents (e.g., 3 at site 1, 2 at site 2), the matrices have
shapes [5], [5×5], [5×5], and [5×5] respectively.

**Example Structure**:

.. code-block:: python

   # Auto-generated variables.py
   bias_string = '''
   b:  # Per-node linear bias vector (length N)
   - 0.1
   - 0.2
   - -0.05
   c:  # NxN quadratic bias matrix (antisymmetric: c[j][i] = -c[i][j])
   - [0.0, 0.3, -0.1]
   - [0.0, 0.0, 0.2]
   - [0.0, 0.0, 0.0]
   x:  # NxN skew bias matrix (both directions independent)
   - [0.0, 0.05, -0.02]
   - [-0.05, 0.0, 0.03]
   - [0.02, -0.03, 0.0]
   s:  # NxN end bias matrix (both directions independent)
   - [0.0, 0.1, -0.05]
   - [-0.1, 0.0, 0.08]
   - [0.05, -0.08, 0.0]
   '''

Edge-to-Matrix Mapping
^^^^^^^^^^^^^^^^^^^^^^

The policy network operates on directed graph edges and predicts coefficients for each
edge-bias type combination. These per-edge predictions are assembled into simulation-ready
matrices using bias-specific composition rules:

**Linear Bias Composition**:

Linear bias values are predicted for edges FROM the reference substituent (sub1 at each site)
TO other substituents at the same site. Since linear bias is fundamentally per-node rather
than per-edge, the individual edge predictions are averaged at each target node:

* Edge sub1→sub2 predicts value v₁₂
* Edge sub1→sub3 predicts value v₁₃  
* Node 2 receives: b[2] = mean(v₁₂)
* Node 3 receives: b[3] = mean(v₁₃)

This averaging provides robustness when multiple edges target the same node in complex graphs.

**Quadratic Bias Composition**:

Quadratic bias is antisymmetric: forward and backward transitions have equal magnitude but
opposite sign. Only upper-triangle edges (i→j where i<j) are created in the graph. The
predicted forward value defines both matrix entries:

* Edge i→j predicts forward value v
* Matrix stores: c[i][j] = v and c[j][i] = -v


**Skew and End Bias Composition**:

Skew and end biases are NOT antisymmetric—forward and backward transitions are physically
independent. Both directed edges exist in the graph, and predictions are stored directly:

* Edge i→j predicts forward value v_fwd
* Edge j→i predicts backward value v_bwd
* Matrix stores: x[i][j] = v_fwd and x[j][i] = v_bwd

This allows the model to learn asymmetric transition barriers without symmetry constraints.

**Matrix Assembly**:

During simulation preparation:

1. Policy network samples coefficients for all directed edges
2. Edge coefficients are grouped by bias type
3. Each bias type is assembled into its matrix format using the rules above
4. Matrices are serialized to YAML in ``variables.py``
5. CHARMM reads the file and applies biases during λ-dynamics simulation

See Also
--------

* :doc:`file_handling` - File format documentation (RTF, PDB, bias coefficients)
* :doc:`deepset_pretraining` - DeepSet pretraining for node embeddings
* :doc:`cb_pretraining` - Behavior cloning from expert bias coefficients
* :doc:`workflow` - Complete workflow from combo generation to training
* :doc:`examples` - Running the full training workflow
* :doc:`api` - API reference for CB modules
* ``examples/run_workflow_deepset.py`` - Full training implementation (DeepSet + pretrained weights)
* ``examples/workflow_14benz.yaml`` - Configuration file for the 14benz system
* ``examples/workflow_deepset.yaml`` - Alternate configuration file template
