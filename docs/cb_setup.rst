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

1. **AtomBondGNN (Phase 1, frozen)**: A graph neural network pretrained on diverse molecular
   data that encodes each substituent's 3D atomic structure into a 64-dimensional vector.
   It uses a dual-stream GINEConv architecture — a substituent stream and a scaffold-context
   (core) stream — with scaffold-aware attentional pooling. Weights are loaded from a
   pre-trained checkpoint and held fixed during all CB training.

2. **SitePoolMLPPolicy (Phase 2, trained by RL)**: A pairwise edge MLP that receives the
   AtomBondGNN P1 embeddings for both endpoint substituents plus a site-level mean-pool
   context vector — 192D total — and outputs independent Gaussian distributions over bias
   coefficients via four completely decoupled ``BiasHeadMLP`` networks (one per bias type).
   Each edge is routed exclusively to the MLP for its bias type (``edge_type // 2``),
   preventing gradient cross-contamination between bias types.

This design eliminates the intermediate RGCN encoder, using instead a lightweight
site-pool context signal computed directly from the frozen Phase 1 embeddings.

.. _Architecture Diagram:

**Full Architecture**:

.. code-block:: text

   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  PHASE 1 — AtomBondGNN  [FROZEN in all training phases]                ║
   ║                                                                          ║
   ║  Per substituent (dual-stream):                                          ║
   ║  AEV[2288] + charge[1] + atom_id[11] = 2300D per atom                  ║
   ║      │  sub_input_proj: 2300 → 256 → 256 + ReLU (sub atoms)            ║
   ║      │  core_input_proj: 2300 → 256 → 256 + ReLU (core+ref atoms)      ║
   ║      │  sub_gin_layers: 4× GINEConv(256→256, edge_dim=1) + ReLU        ║
   ║      │  core_gin_layers: 4× GINEConv(256→256, edge_dim=1) + ReLU       ║
   ║      │  core_summary = mean-pool(core_gin_layers output)                ║
   ║      │  gate = σ(Linear(concat(sub_h, core_summary), 1))               ║
   ║      │  GlobalAttentionPool: weighted sum of pool_nn(sub_h)             ║
   ║      ▼                                                                   ║
   ║   64D  P1 embedding  (data.x)                                           ║
   ╚══════════════════════════════════════════════════════════════════════════╝
                       │
                       │  data.x [N, 64]  (one row per substituent node)
                       │  data.site_index [N]  (which λ-site each node belongs to)
                       ▼
   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  PHASE 2 — SitePoolMLPPolicy  [trained by BC pretraining and RL]       ║
   ║                                                                          ║
   ║  site_pool = mean(P1 embeddings at the same λ-site)  [N, 64]           ║
   ║                                                                          ║
   ║  Per directed edge (Sub_A → Sub_B):                                     ║
   ║  concat(P1_A[64], P1_B[64], site_pool_A[64]) = 192D                    ║
   ║      │   block dropout on site_pool_A slice (p=0.3) during training     ║
   ║      │   edge_type // 2 → routes edge to its own BiasHeadMLP            ║
   ║      ▼                                                                   ║
   ║  ┌── BiasHeadMLP (linear) ──┐  ┌── BiasHeadMLP (quadratic) ──┐        ║
   ║  │  192 → 128 → 64 → 32 → 2│  │  192 → 128 → 64 → 32 → 2   │        ║
   ║  └──────────────────────────┘  └─────────────────────────────┘        ║
   ║  ┌── BiasHeadMLP (skew) ────┐  ┌── BiasHeadMLP (end) ─────────┐       ║
   ║  │  192 → 128 → 64 → 32 → 2│  │  192 → 128 → 64 → 32 → 2    │       ║
   ║  └──────────────────────────┘  └──────────────────────────────┘       ║
   ║      ▼                                                                   ║
   ║  (μ_d, log σ_d) for the relevant bias type d of each edge               ║
   ║  Output scaled: tanh(μ_d) × scale_d                                     ║
   ║    scale: [305, 520, 85, 30] for [linear, quadratic, skew, end]         ║
   ╚══════════════════════════════════════════════════════════════════════════╝
                               ▼
   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  PER-EDGE PER-DIM REWARD  (from simulation output)  [E, 4]             ║
   ║                                                                          ║
   ║  Unvisited pair (DDG None/NaN/Inf):  all 4 dims = -1.0                 ║
   ║  Visited pair (finite DDG):                                              ║
   ║    dim 0 (linear):    minority_frac = min(p_i,p_j)/(p_i+p_j) ∈[0,0.5] ║
   ║    dim 1 (quadratic): (pair_visited + trans_quality) / 2  ∈[0.5,1.0]  ║
   ║    dim 2 (skew):      combined proxy ∈[0.5,0.75]                       ║
   ║    dim 3 (end):       combined proxy ∈[0.5,0.75]                       ║
   ║                                                                          ║
   ║  policy_loss = -∑_e R_pair[e, d_e] × logp_e[d_e]   per-head REINFORCE ║
   ╚══════════════════════════════════════════════════════════════════════════╝

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

Graphs are constructed with **AtomBondGNN embeddings** as the primary node features, representing
each substituent's 3D atomic structure and chemical composition as a learned 64-dimensional
vector. These embeddings replace manual feature engineering with neural representations
pretrained on diverse molecular data.

**AtomBondGNN-Based Construction**:

The standard construction pipeline:

1. Parse RTF files to identify substituents and extract metadata (site numbers, charges, atom types)
2. Build graph topology: one node per substituent, edges connecting substituents within each site
3. Compute AtomBondGNN embeddings for each node from PDB coordinates, RTF charges, and bond topology
4. Store embeddings as node features for neural network input

The AtomBondGNN embeddings capture rich molecular information automatically:

* **Spatial structure**: Bond lengths, angles, 3D conformations from atomic coordinates
* **Chemical composition**: Element types, functional groups, charge distributions
* **Bond-topology context**: GINEConv message passing propagates bonded-neighbor information before pooling
* **Scaffold-aware context**: A parallel core stream processes the shared ligand core, and its
  mean-pooled summary modulates the attention gate — encoding scaffold identity without
  contaminating the substituent feature path
* **Environmental context**: Nearby protein atoms and core structure atoms (via context-aware AEV computation)

See :doc:`deepset_pretraining` for technical details on the AtomBondGNN pretraining
pipeline (atom-level AEV features + bond topology → dual GINEConv streams → scaffold-aware AttentionPool).

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

The 64-dimensional AtomBondGNN embeddings computed during graph construction become the
node feature matrix. Each row represents one substituent with its learned molecular
representation encoding structure, chemistry, and environment. A ``site_index`` tensor
(one integer per node identifying its λ-site) is stored alongside the embeddings and
provides the ``SitePoolMLPPolicy`` with its system-context signal via mean-pooling.

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

SitePoolMLPPolicy
^^^^^^^^^^^^^^^^^

Node embeddings from the frozen AtomBondGNN are used directly by the policy without
an intermediate graph convolution step. The ``SitePoolMLPPolicy`` computes a
site-level mean-pool context on the fly:

.. math::

   \text{site\_pool}_i = \frac{1}{|\mathcal{S}_{\text{site}(i)}|}
   \sum_{j \in \mathcal{S}_{\text{site}(i)}} \mathbf{P1}_j

where :math:`\mathcal{S}_{\text{site}(i)}` is the set of all substituent nodes at the
same λ-site as node :math:`i`. For each directed edge :math:`(A \to B)` the input is:

.. math::

   \mathbf{e}_{AB} = [\mathbf{P1}_A,\; \mathbf{P1}_B,\; \text{site\_pool}_A] \in \mathbb{R}^{192}

During training, the entire ``site_pool`` block is zeroed with probability 0.3
(**block dropout**), forcing the policy to remain useful without context and
preventing over-reliance on the site-pool signal.

Edge Policy
^^^^^^^^^^^

Per-edge coefficients are predicted by four **completely independent** ``BiasHeadMLP``
networks, one per MSLD bias type. Each head owns its entire feature-extraction stack
so that gradient from one bias type (e.g. the noisy end-state signal) cannot overwrite
features learned by another type (e.g. the linear population-balance signal).

**Architecture Overview**:

Each ``BiasHeadMLP`` maps the 192D edge input directly to a (mean, log_std) pair:

.. code-block:: text

   Input [E, 192]
       Linear(192 → 128) + ReLU
       Linear(128 →  64) + ReLU
       Linear( 64 →  32) + ReLU
       Linear( 32 →   2)          → (mean_raw, log_std_raw) per edge

The ``EdgeValueMLP`` container holds all four heads and routes each edge to the
correct head via ``edge_type // 2``:

* ``edge_type`` 0 or 1 (linear fwd/bwd) → ``mlps[0]``
* ``edge_type`` 2 or 3 (quadratic fwd/bwd) → ``mlps[1]``
* ``edge_type`` 4 or 5 (skew fwd/bwd) → ``mlps[2]``
* ``edge_type`` 6 or 7 (end fwd/bwd) → ``mlps[3]``

In the routed forward pass only the matching head processes each edge; output slots
for other heads remain zero so the zero-gradient property is enforced by construction.

**Key Features**:

* **Full gradient isolation**: Four separate parameter stacks; no shared weights that
  could mix reward signals between bias types.

* **Output scaling**: Mean predictions scaled via ``tanh(mean_raw) * scale_factors``
  
  - **Linear**: ±305, **Quadratic**: ±520, **Skew**: ±85, **End**: ±30
  - Covers the empirical maximum from 20,000+ pretraining runs with ~10% headroom.

* **Enhanced exploration**: Log standard deviation clamped to [-20, 2.0]
  (standard deviation range: [~0, 7.4])

The policy outputs:

* ``actions``: Sampled coefficient values (shape: [E, 4])
* ``logp``: Log-probabilities for REINFORCE updates
* ``mean``: Mean of the Gaussian distribution per edge per bias type
* ``log_std``: Log standard deviation per edge per bias type

Each directed edge receives one Gaussian distribution for its relevant bias type.
Actions are sampled and log-probabilities are computed with dimension masking so
that only the head responsible for an edge's bias type contributes a gradient:

.. math::

   v_{ij}^{(d_{ij})} \sim \mathcal{N}(\mu_{ij}^{(d_{ij})}, (\sigma_{ij}^{(d_{ij})})^2)

where :math:`d_{ij} = \text{edge\_type}_{ij} // 2` is the bias-type index for edge :math:`(i,j)`.

Training and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

The policy network is trained using **REINFORCE** with per-edge per-dimension reward
signals. Each ``BiasHeadMLP`` receives gradient only from simulations where the
corresponding bias type was responsible for the edge, providing clean per-type credit
assignment without a value network.

**REINFORCE Components**:

* **Policy (SitePoolMLPPolicy)**: Predicts independent Gaussian distributions for each
  bias type from 192D per-edge inputs. Only the SitePoolMLPPolicy has its weights
  updated by RL; the AtomBondGNN (Phase 1) is frozen throughout.
* **No Q-Critic or value baseline**: The per-bias-type reward tensor (``compute_pair_reward``
  returning ``[E, 4]``) provides direct per-head reward signals, making a separate
  critic unnecessary. Dimension masking in ``evaluate_logp`` ensures each MLP head
  receives gradient only from its own reward dimension.

**Per-Edge Per-Dimension Reward**:

Rather than a single scalar reward, each directed edge :math:`(i, j)` receives a
4-dimensional reward vector matching the four bias types. Each dimension captures a
different physical signal from the simulation:

.. math::

   R_{ij}^{(d)} = \begin{cases}
     -1.0 & \text{if } \Delta\Delta G_{ij} \text{ is None, NaN, or } \pm\infty \\
     \text{dim-specific signal} & \text{if } \Delta\Delta G_{ij} \text{ is finite}
   \end{cases}

For visited pairs (finite DDG):

* **dim 0 (linear)**: ``minority_frac`` = :math:`\min(p_i, p_j)/(p_i + p_j)` ∈ [0, 0.5] —
  rewards population balance
* **dim 1 (quadratic)**: ``(pair_visited + trans_quality) / 2`` ∈ [0.5, 1.0] — rewards
  barrier removal (per-pair crossing + combo-level transition quality)
* **dim 2 (skew)**: combined proxy from linear and quadratic signals
* **dim 3 (end)**: same combined proxy as skew

**Policy Loss**:

The loss uses dimension masking so each head's gradient comes exclusively from its
own reward dimension:

.. math::

   \mathcal{L}_{\text{policy}} = -\sum_{e} R_{e}^{(d_e)} \cdot \log \pi_\theta(a_e^{(d_e)} \mid s)

where :math:`d_e = \text{edge\_type}_e // 2` is the bias-type index for edge :math:`e`.

**Training Updates**:

For each combination:

1. Encode graph via AtomBondGNN (frozen) → 64D P1 node embeddings
2. Compute site-pool context: mean-pool P1 embeddings per λ-site → [N, 64]
3. Build 192D per-edge inputs: ``[P1_src, P1_dst, site_pool_src]``
4. Route each edge to its ``BiasHeadMLP`` via ``edge_type // 2``
5. Sample bias coefficients :math:`a \sim \pi_\theta(\cdot | s)` from ``SitePoolMLPPolicy``
6. Run simulation; parse DDG pairs and block populations
7. Compute per-edge, per-dim rewards via ``compute_pair_reward`` → ``[E, 4]``
8. Evaluate ``log π_θ(a | s)`` with dimension masking (each edge contributes gradient
   only to its own ``BiasHeadMLP``)
9. Update ``SitePoolMLPPolicy``: maximise
   :math:`\sum_e R_e^{(d_e)} \cdot \log \pi_\theta(a_e^{(d_e)} \mid s)`

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
* ``examples/run_workflow_deepset.py`` - Full training implementation
* ``examples/workflow_14benz.yaml`` - Configuration file for the 14benz system
* ``examples/workflow_deepset.yaml`` - Alternate configuration file template
