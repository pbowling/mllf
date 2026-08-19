Contextual Bandit Setup
=======================

Overview
--------

The contextual bandit (CB) training framework provides a reinforcement learning approach to
optimizing bias coefficients for multisite λ-dynamics simulations. Instead of hand-tuning
bias parameters, a policy network predicts optimal coefficients directly from pretrained
molecular representations of each substituent.

**Architecture**:

``UnimolPolicy`` (``mllf.cb.policy``) is a single-stage, feed-forward policy with **no
graph encoder**:

1. **Node features**: Each substituent's node feature is a pretrained, frozen **Uni-Mol**
   embedding (see :doc:`unimol_representation`) — 512D in standard mode, or 1024D
   (``[ligand_only, full]``) in **dual embedding mode**, which is what online training and
   pretraining both use by default. Uni-Mol is never fine-tuned; only the policy heads
   downstream of it are trained.

2. **Per-edge policy heads**: For every directed pair of substituents at the same λ-site,
   the node embeddings are combined into a 1024D edge input and passed to four completely
   independent ``BiasHeadMLP`` networks (one per MSLD bias type: linear, quadratic, skew,
   end), each producing a ``(mean, log_std)`` pair. There is no shared trunk across bias
   types and no intermediate graph convolution — this eliminates the RGCN encoder and
   AtomBondGNN/DeepSet pretraining stage used by earlier versions of this framework (see
   :doc:`unimol_representation` for that history).

.. _Architecture Diagram:

**Full Architecture** (dual embedding mode):

.. code-block:: text

   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  Uni-Mol  [FROZEN, pretrained on 1.1B PubChem molecules]                ║
   ║                                                                          ║
   ║  Per substituent, two embeddings:                                       ║
   ║    ligand-only   = UniMol(core + sub)                        [512]     ║
   ║    full          = UniMol(core + sub + ref_subs + env)       [512]     ║
   ║  node feature = [ligand_only, full]                          [1024]    ║
   ╚══════════════════════════════════════════════════════════════════════════╝
                       │  data.x [N, 1024]  (one row per substituent node)
                       ▼
   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  UnimolPolicy._build_edge_input   (per directed edge A → B)             ║
   ║                                                                          ║
   ║  diff_ligand = ligand_only_A − ligand_only_B    (antisymmetric)  [512]  ║
   ║  mean_full   = (full_A + full_B) / 2            (symmetric)      [512]  ║
   ║  edge_input  = [diff_ligand, mean_full]                         [1024]  ║
   ╚══════════════════════════════════════════════════════════════════════════╝
                       ▼
   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  EdgeValueMLP — four independent BiasHeadMLP  [trained by BC + RL]      ║
   ║                                                                          ║
   ║  ┌── BiasHeadMLP (linear) ─────┐  ┌── BiasHeadMLP (quadratic) ────┐    ║
   ║  │ trunk: 1024→512→256→64+ReLU│  │ trunk: 1024→512→256→64+ReLU  │    ║
   ║  │ readout: Linear(64, 2)      │  │ readout: Linear(64, 2)        │    ║
   ║  └──────────────────────────────┘  └────────────────────────────────┘    ║
   ║  ┌── BiasHeadMLP (skew) ───────┐  ┌── BiasHeadMLP (end) ──────────┐    ║
   ║  │ trunk: 1024→512→256→64+ReLU│  │ trunk: 1024→512→256→64+ReLU  │    ║
   ║  │ readout: Linear(64, 2)      │  │ readout: Linear(64, 2)        │    ║
   ║  └──────────────────────────────┘  └────────────────────────────────┘    ║
   ║      ▼                                                                   ║
   ║  (μ_d, log σ_d) per bias type d, for every edge                          ║
   ║  Output scaled: softsign(μ_d) × scale_d                                  ║
   ║    scale: [305, 520, 85, 30] for [linear, quadratic, skew, end]          ║
   ║  log σ_d clamped to [-20, 2.0]                                           ║
   ╚══════════════════════════════════════════════════════════════════════════╝
                               ▼
   ╔══════════════════════════════════════════════════════════════════════════╗
   ║  PER-EDGE PER-DIM REWARD  (compute_pair_reward, from sim output) [E, 4] ║
   ║                                                                          ║
   ║  Unvisited pair (no lambda-space crossing, or both subs unsampled):     ║
   ║      all 4 dims = -1.0                                                  ║
   ║  Visited pair:                                                           ║
   ║    dim 0 (linear):    minority_frac = min(p_i,p_j)/(p_i+p_j) ∈ [0,0.5] ║
   ║    dim 1 (quadratic): (pair_visited + trans_quality) / 2   ∈ [0.5,1.0] ║
   ║    dim 2 (skew):      combined population/quality proxy                ║
   ║    dim 3 (end):       fraction-physical-ligand proxy (or combined       ║
   ║                       fallback if not available)                        ║
   ║                                                                          ║
   ║  policy_loss = -∑_e R[e, :] · logp_e[:]     each head trained only on   ║
   ║                                              its own reward dimension    ║
   ╚══════════════════════════════════════════════════════════════════════════╝

**NeuralLinear + Thompson Sampling (optional)**: setting ``use_bayesian_heads=True`` swaps
each ``BiasHeadMLP``'s deterministic ``readout`` for a
:class:`~mllf.cb.bayesian_head.BayesianLinearHead` — a closed-form Bayesian linear
regression on the same 64D trunk features. See `NeuralLinear + Thompson Sampling`_ below.

Core Components
---------------

Graph Representation
~~~~~~~~~~~~~~~~~~~~

Molecular systems are represented as directed graphs where:

* **Nodes** represent individual substituents at each λ-site
* **Edges** represent transitions between substituents with associated bias coefficients

For a 2-site system with 3 substituents at site 1 and 2 substituents at site 2, the graph
contains 5 nodes total (one per substituent). ``build_directed_pairs()``
(``mllf.cb.graph_utils``) generates **both directions** (i→j and j→i) for every pair of
substituents *within the same site* — no cross-site edges. This "fully-connected within
site" scheme is what both online training (``examples/run_workflow.py``'s
``build_graph_and_data``) and pretraining (``build_fully_connected_graph_for_pretraining``)
use: one edge per ordered pair, each edge carrying predictions for **all four** bias types
at once (``UnimolPolicy`` outputs ``[E, 4]`` means/log-stds per call, since the current
graph construction passes ``edge_type=None`` and lets every ``BiasHeadMLP`` process every
edge — there is no per-relation-type edge subset to route).

.. note::
   An older, sparser graph construction still exists in the codebase
   (``mllf.cb.graph_utils.build_pyg_graph_from_mllf_graph``) that expands each undirected
   pair into up to eight *relation-typed* directed edges (``linear_fwd``/``bwd``,
   ``quadratic_fwd``/``bwd``, etc.) and routes each edge to a single ``BiasHeadMLP`` via
   ``edge_type // 2``. This is used by the legacy, encoder-agnostic ``EdgePolicy`` class
   (paired with ``mllf.cli.workflow.build_data_and_targets_from_combo``) for callers that
   supply their own node encoder. Current examples do not use this path.

**Bias Types**:

Each edge can have multiple bias coefficient types:

* **Linear (b)**: Per-node bias ensuring equal population of all substituents at each site
  when correctly parameterized.

* **Quadratic (c)**: Pairwise interaction bias removing alchemical barriers due to
  electrostatic interactions between sites. Symmetric: :math:`c_{ij} = c_{ji}`, meaning
  the forward and backward transitions share the same value (only the upper triangle
  ``i < j`` is stored; the lower triangle is understood to hold the same value, not its
  negation).

* **Skew (x)**: Asymmetry correction fitting residuals beyond quadratic and end biases,
  particularly important after soft-core introduction. Forward and backward transitions
  are independent (not antisymmetric).

* **End (s)**: End-state bias compensating for entropic and surface tension costs of
  displacing solvent and nearby molecules when substituents appear. Forward and backward
  transitions are independent (not antisymmetric).

Graph Construction
^^^^^^^^^^^^^^^^^^

For each combination directory, graph construction:

1. Parses RTF/PDB files to identify substituents at each site and locate the core scaffold
2. Builds a per-site consensus environment atom set from the core (see `Environment
   Consensus <unimol_representation.html#environment-consensus>`__ in
   :doc:`unimol_representation`), skipped for gas-phase systems
3. Computes a dual Uni-Mol embedding ``[ligand_only (512), full (512)]`` for every
   substituent at every site (not just the "active" ones in the combo name)
4. Builds directed edges within each site via ``build_directed_pairs()``
5. Stores node embeddings (``data.x``), ``edge_index``, and ``site_index`` (which λ-site
   each node belongs to) for the policy forward pass

**Environmental Context Encoding**: the environment type (``solvent_state``: ``solv``,
``protein``, or ``gas``/``vacuum``) controls what — if anything — is included as
environment atoms during embedding computation (protein atoms, solvent atoms, or none).
See :doc:`unimol_representation` for the full fallback chain and periodic-boundary
handling. This eliminates the need for explicit environment flags as node features — the
environmental information is implicitly encoded in the embeddings themselves.

**Legacy RTF-Only Construction**: graphs can also be built directly from RTF topology
fragments without Uni-Mol embeddings, using manually engineered features (atom type/element
counts, charge) via ``build_pyg_graph_from_mllf_graph`` and ``mllf.cb.atom_vocab``. This
path is what feeds the legacy ``EdgePolicy``/relation-typed-edge pipeline described above,
and remains available for systems where PDB coordinates are unavailable, but Uni-Mol-based
``UnimolPolicy`` is what production training uses.

Policy Network Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

UnimolPolicy
^^^^^^^^^^^^

``UnimolPolicy`` (``mllf.cb.policy``) takes pre-computed Uni-Mol node embeddings directly —
there is no graph encoder or site-pooling step. For each directed edge :math:`(A \to B)`:

.. math::

   \mathbf{e}_{AB} = [\,\text{ligand}_A - \text{ligand}_B,\;\; \tfrac{1}{2}(\text{full}_A + \text{full}_B)\,]
   \in \mathbb{R}^{1024}

(standard, non-dual mode instead concatenates ``[emb_A - emb_B, (emb_A + emb_B)/2]`` from a
single 512D embedding per node). The antisymmetric half captures how the two substituents
differ; the symmetric half captures the environment they share.

Edge Policy Heads
^^^^^^^^^^^^^^^^^

Per-edge coefficients are predicted by four **completely independent** ``BiasHeadMLP``
networks, one per MSLD bias type. Each head owns its entire feature-extraction stack (a
"trunk") so that gradient from one bias type cannot overwrite features learned by another.

.. code-block:: text

   Input [E, 1024]
       Linear(1024 → 512) + ReLU
       Linear( 512 → 256) + ReLU
       Linear( 256 →  64) + ReLU        ← trunk output z = phi(x), 64D
       Linear(  64 →   2)               → (mean_raw, log_std_raw) per edge   [deterministic mode]
       — or —
       BayesianLinearHead(64 → 1)       → (posterior mean, 0.5·log(var))     [NeuralLinear mode]

``EdgeValueMLP`` holds all four heads. When ``edge_type`` is supplied (the legacy
relation-typed path), each edge is routed to exactly one head via ``edge_type // 2``; when
``edge_type`` is ``None`` (the current fully-connected pairwise scheme), every head
processes every edge and the results are stacked into one ``[E, 4]`` mean/log-std tensor —
credit assignment per bias type then comes entirely from ``compute_pair_reward``'s
per-dimension reward (see `Training and Optimization`_), not from edge masking.

**Key Features**:

* **Full gradient isolation between bias types**: four separate parameter stacks, no shared
  weights.
* **Output scaling**: mean predictions scaled via ``softsign(mean_raw) * scale_factors``
  (softsign, not tanh — its gradient ``1/(1+|z|)^2`` stays alive at large ``|z|``, avoiding
  the "dead head" saturation collapse tanh could produce after many training epochs).

  - **Linear**: ±305, **Quadratic**: ±520, **Skew**: ±85, **End**: ±30
  - Empirical maximum from 20,000+ pretraining runs with ~10% headroom.

* **Exploration**: ``log_std`` clamped to ``[-20, 2.0]`` (std range ``[~0, 7.4]``); sampled
  actions are additionally clamped to ``scale_factors × 1.05``.

The policy outputs:

* ``actions``: sampled coefficient values, shape ``[E, 4]``
* ``mean`` / ``log_std``: Gaussian parameters per edge per bias type
* ``logp``: only for REINFORCE mode (``None`` when ``use_bayesian_heads=True`` — Thompson
  sampling has no log-prob concept)

NeuralLinear + Thompson Sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an alternative to REINFORCE, each ``BiasHeadMLP`` can use a
:class:`~mllf.cb.bayesian_head.BayesianLinearHead` in place of its deterministic
``readout``: a closed-form Bayesian linear regression ``r ~ N(z @ w, σ²)`` on the trunk's
64D features ``z``, maintaining a full Gaussian posterior ``w ~ N(μ, Λ⁻¹)`` that is updated
analytically from observed ``(z, r)`` pairs — no backward pass or optimizer step for the
head's last layer. Acting draws one weight sample per decision (Thompson sampling) via
``UnimolPolicy.get_actions_thompson``; the trunk itself is still a regular ``nn.Module``
(shared with the deterministic mode).

This mode is opt-in: set ``training.policy.use_bayesian_heads: true`` and
``bandit.algorithm: neurallinear_ts`` in the workflow config (see :doc:`workflow`). It is
intended for online fine-tuning where posterior uncertainty can drive combo selection
(``select_combos_by_uncertainty`` in ``mllf.cb.workflow_utils``) rather than pure random or
curriculum sampling. A deterministic pretrained policy can be converted to a Bayesian
sibling via ``pretrain_policy.py --bayesian-heads`` (see :doc:`cb_pretraining`), which
warm-starts each posterior mean from the trained deterministic readout.

Training and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

The policy network is trained using **REINFORCE** (or, in NeuralLinear + TS mode, by
folding observed rewards directly into each head's Bayesian posterior — no REINFORCE
gradient at all for that mode). Each ``BiasHeadMLP`` receives gradient only from the reward
dimension matching its bias type, providing clean per-type credit assignment without a
value network or Q-critic.

**Per-Edge Per-Dimension Reward** (``compute_pair_reward`` in ``mllf.cb.workflow_utils``):

Rather than a single scalar reward, each directed edge :math:`(i, j)` receives a
4-dimensional reward vector, one entry per bias type:

.. math::

   R_{ij}^{(d)} = \begin{cases}
     -1.0 & \text{pair not visited (DDG None/NaN/}\pm\infty\text{), or both subs unsampled} \\
     \text{dim-specific signal (below)} & \text{otherwise}
   \end{cases}

For visited pairs:

* **dim 0 (linear)**: ``minority_frac`` = :math:`\min(p_i, p_j)/(p_i + p_j)` ∈ [0, 0.5] —
  rewards population balance
* **dim 1 (quadratic)**: ``(pair_visited + trans_quality) / 2`` ∈ [0.5, 1.0], where
  ``trans_quality = min(total_transitions, T_baseline) / T_baseline`` — rewards barrier
  removal via per-pair DDG existence plus a combo-level transition-count quality signal
* **dim 2 (skew)**: the mean of the (normalized) population-balance and quadratic-quality
  signals above — a combined proxy, since no simulation observable yet isolates barrier
  asymmetry specifically
* **dim 3 (end)**: ``min(1.0, fraction_physical × total_pairs)`` — the combo-level
  "fraction physical ligand" diagnostic (fraction of the trajectory in a fully-resolved,
  single-substituent-per-site state) scaled by the combo's substituent-pair count so combos
  of different sizes are graded comparably; falls back to the dim-2 ``combined`` proxy when
  the diagnostic wasn't parsed (e.g. older cached results)

Edges whose pair had no observed λ-space transition at all (as opposed to a *finite* DDG)
can additionally be down-weighted via ``build_edge_weights`` /
``reward.no_transition_weight`` (default 0.15) rather than fully zeroed, so unvisited pairs
still contribute a little training signal without dominating the gradient.

**Policy Loss**:

.. math::

   \mathcal{L}_{\text{policy}} = -\sum_{e} \sum_{d} R_{e}^{(d)} \cdot \log \pi_\theta(a_e^{(d)} \mid s)

Because each dimension :math:`d` is produced by an independent ``BiasHeadMLP``, this sum
naturally isolates gradient per head — no additional masking is required when
``edge_type=None`` (see `Edge Policy Heads`_).

**Training Updates** (one combination, REINFORCE mode):

1. Compute dual Uni-Mol embeddings for every substituent → ``[N, 1024]`` node features
2. Build directed edges within each site via ``build_directed_pairs``
3. Sample bias coefficients :math:`a \sim \pi_\theta(\cdot \mid s)` from ``UnimolPolicy``
4. Write ``variables.py`` and run the MSLD simulation
5. Parse per-pair DDG and per-substituent populations from the simulation output
6. Compute per-edge, per-dimension rewards via ``compute_pair_reward`` → ``[E, 4]``
7. Evaluate :math:`\log \pi_\theta(a \mid s)` for the actions actually submitted
8. Update ``UnimolPolicy``: maximise :math:`\sum_e \sum_d R_e^{(d)} \cdot \log \pi_\theta(a_e^{(d)} \mid s)`

For reward hyperparameters, curriculum learning, and full workflow configuration, see
:doc:`workflow`.

Variables.py Format
~~~~~~~~~~~~~~~~~~~

MSLD simulation setup files read bias coefficients from ``variables.py`` files containing
YAML-formatted bias matrices. ``write_variables_from_actions`` (``mllf.cli.workflow``)
assembles the policy's per-edge predictions into these matrices.

**Matrix Format**:

* **b (linear)**: 1D vector of length N (one value per substituent)
* **c (quadratic)**: N×N symmetric matrix (upper triangle stored; ``c[j][i]`` is understood
  to equal ``c[i][j]``, not its negation)
* **x (skew)**: N×N full matrix (both triangles stored independently)
* **s (end)**: N×N full matrix (both triangles stored independently)

**Quadratic Composition**: quadratic is genuinely symmetric (:math:`c_{ij} = c_{ji}`), so
forward and backward predictions for the same undirected pair are **averaged**, and the
result is stored once as ``c[i][j] = v`` (upper triangle only; ``c[j][i]`` is left at 0.0
in the file, with the simulator treating the stored upper-triangle value as shared by both
directions — not negated for the lower triangle, unlike the genuinely antisymmetric linear
case below).

**Skew and End Composition**: these are *not* antisymmetric — forward and backward
transitions are physically independent, so both directed predictions are stored directly:
``x[i][j] = v_fwd``, ``x[j][i] = v_bwd``.

**Linear Composition**: linear predictions for a directed edge :math:`(i \to j)`
approximate :math:`b_j - b_i` — an **antisymmetric** quantity, unlike quadratic. The
per-node vector :math:`b` is reconstructed relative to each site's reference substituent
(``sub1``, with :math:`b_{\text{ref}} = 0`): for each other substituent :math:`j` at that
site, the forward prediction :math:`v_{\text{fwd}} \approx b_j - b_{\text{ref}}` and
backward prediction :math:`v_{\text{bwd}} \approx b_{\text{ref}} - b_j` are combined as
:math:`b_j = \tfrac{1}{2}(v_{\text{fwd}} - v_{\text{bwd}})` when both are available (falling
back to whichever single direction exists). Averaging the *raw* forward/backward values
together (as if linear were symmetric like quadratic) would cancel this antisymmetric
signal to ~0, so it must be inverted relative to the reference, not averaged directly.

See Also
--------

* :doc:`file_handling` - File format documentation (RTF, PDB, bias coefficients)
* :doc:`unimol_representation` - Uni-Mol embeddings and environment consensus
* :doc:`cb_pretraining` - Behavior cloning from expert bias coefficients
* :doc:`workflow` - Complete workflow from combo generation to training
* :doc:`examples` - Running the full training workflow
* :doc:`api` - API reference for CB modules
* ``examples/run_workflow.py`` - Full online-training implementation
* ``examples/workflow_14benz.yaml`` - Minimal configuration file for the 14benz system
