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
   types, no graph encoder, and no intermediate graph convolution.

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
   ║    dim 1 (quadratic): barrier_quality(midpoint barrier)    ∈ [0.5,1.0] ║
   ║    dim 2 (skew):      gaussian_quality(skew asymmetry, 0)  ∈ [0.5,1.0] ║
   ║    dim 3 (end):       gaussian_quality(end asymmetry, 0)   ∈ [0.5,1.0] ║
   ║                       (legacy proxies when no λ trajectory)             ║
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
   An alternative, sparser graph construction is also available
   (``mllf.cb.graph_utils.build_pyg_graph_from_mllf_graph``, with manually engineered atom
   type/charge features from ``mllf.cb.atom_vocab`` and no Uni-Mol embeddings): it expands
   each undirected pair into up to eight *relation-typed* directed edges
   (``linear_fwd``/``bwd``, ``quadratic_fwd``/``bwd``, etc.) and routes each edge to a
   single ``BiasHeadMLP`` via ``edge_type // 2``. It pairs with the encoder-agnostic
   ``EdgePolicy`` class (and ``mllf.cli.workflow.build_data_and_targets_from_combo``) for
   callers that supply their own node encoder or lack PDB coordinates. The examples and
   production training use the Uni-Mol path above, not this one.

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
handling. Because the environment is baked into the embedding, no explicit environment
flags are needed as node features.

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

``EdgeValueMLP`` holds all four heads. When ``edge_type`` is supplied (the relation-typed
path noted under `Graph Representation`_), each edge is routed to exactly one head via
``edge_type // 2``; when ``edge_type`` is ``None`` (the fully-connected pairwise scheme the
examples use), every head processes every edge and the results are stacked into one
``[E, 4]`` mean/log-std tensor —
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

**Hindsight targets** (``bandit.update_target: hindsight``, :mod:`mllf.cb.hindsight`): by
default (``awr``) the posterior regresses toward the action each run sampled, weighted by
``exp(r/β)``. A scalar reward says whether a run's biases were good, but not which way
they were off or by how much. Because the samples sit around the current mean, the
update also barely moves the posterior. With ``hindsight``, each run is relabelled with
the biases it *measured* it should have used, and the heads in ``hindsight_dims``
(default: all four) regress toward those. Each signal comes from the run's own λ
trajectory (``read_output.parse_pair_profile``), and each response is from the
pretraining bias-sensitivity analysis:

* linear, directed edge :math:`(i \to j)`:
  :math:`\Delta b^* = \Delta b - kT \ln(p_j / p_i)`. When one substituent is unsampled,
  this is a capped step toward it.
* quadratic: :math:`c^* = c + (B - B_0)/|s_c|`, from the pair's midpoint barrier
  :math:`B`, where :math:`s_c` is the barrier's response to c.
* skew: the step that removes the pair's skew asymmetry, split between
  :math:`x_{lo,hi}` and :math:`x_{hi,lo}`.
* end: the step that removes the pair's end asymmetry, split between
  :math:`s_{lo,hi}` and :math:`s_{hi,lo}`. When the trajectory crowds up to an endpoint
  but does not hold it (many frames just short of it, almost none at it), the end term
  there is holding λ off. That s (:math:`s_{hi,lo}` at the lo end, :math:`s_{lo,hi}` at
  the hi end) then moves part of the way toward a typical converged value
  (``end_reference_s``), so even a large error shrinks within a few runs. This signal
  exists exactly when the others do not: the pair never reached an endpoint.

Iterative flattening methods also correct biases from each simulation. Those
corrections apply only to the system being simulated, so every new system starts from
scratch. This implementation differs in four ways:

* **Transfer.** The corrected targets train a shared model conditioned on the
  substituents' Uni-Mol embeddings, not the biases of one system. What is learned on
  one pair improves the first guess for pairs and systems that have never been
  simulated.
* **Calibrated weighting.** Each target enters the Bayesian posterior with its own
  measurement variance (crossings for b, the scatter of the barrier, skew or end signal,
  and a term that grows with the size of the linearized step). Well-measured runs count
  for more than noisy ones, with no fixed per-coefficient regularization.
  ``hindsight_weight_scale`` sets how strongly the shared model follows the training
  pairs relative to the per-pair offsets below.
* **Pair-level Bayesian offsets and exploration.** Each pair's own measurements are
  combined with the model's prediction as a prior, and proposals *sample* from that
  per-pair posterior (below), so exploration is targeted where it is needed.
* **Active selection.** The shared posterior's epistemic uncertainty and each pair's
  own state choose which pairs to simulate next.

Each step is still capped (``hindsight_linear_step_cap``,
``hindsight_quadratic_step_cap``, ``hindsight_skew_step_cap``,
``hindsight_end_step_cap``), because the linearized responses hold only near sensible
biases.
A pair that keeps stepping b toward an unsampled substituent in the same direction is far
from its working b. Its per-pair steps then double with each consecutive step, up to
``linear_step_max``, and reset once both substituents are sampled. The shared model keeps
the base step.

``hindsight_weight_scale`` can be given per head (``{linear: …, quadratic: …, skew: …,
end: …}``). A head whose shared update would shift predictions for every pair, including
ones it never trained on, can be damped strongly while the per-pair offsets still correct
that coefficient pair by pair.

With ``bandit.pair_offsets: true``, each training pair's last ``offset_history`` targets
are stored in ``bandit_state.json`` under its original site/sub numbers, per coefficient
(per direction for x and s, which are independent parameters). Its next proposal combines
them with the model prediction, which acts as a Bayesian prior of sd
``offset_prior_sd``: a pair the model already predicts well moves little, and one it
predicts poorly is corrected from its own data without distorting the shared model. The
shared model carries what generalizes; the offsets hold what is specific to a pair.
Validation passes never apply offsets, so they measure the shared model alone.

With ``bandit.offset_sampling: true``, proposals draw each pair's offset rather than
using its mean:

* A coefficient the pair has not measured is explored with sd
  ``offset_base_sd × factor`` on top of the shared model's Thompson sample.
* A coefficient it has measured is proposed at the model mean plus a draw from the
  pair's own offset posterior. Exploration then narrows as the pair's data accumulate,
  instead of following the shared model's noise.
* When a run fails without giving a direction for *any* of the pair's coefficients
  (a failure no signal explains), the pair's factors double, up to
  ``explore_factor_max``, so the next proposals search further. A success halves them.
  A single failure that does give a direction leaves the factors alone: the targets
  address it.
* A pair that fails ``bandit.explore_after_failures`` runs in a row (default 3) is not
  being fixed by its targets, so all of its factors double, measured coefficients
  included. A measured coefficient's spread stays capped by its offset posterior sd.
  Because b gets a target whenever either substituent is sampled, almost every failure
  gives a direction, so this is the usual way a stuck pair widens its search.

This is how the method copes with a new system whose failure mode was not anticipated:
diagnosable failures are corrected directly, and undiagnosable ones widen the search
until something measurable happens.

With ``bandit.selection_attention_weight`` > 0, uncertainty-driven selection ranks
candidates by the percentile of their shared posterior variance plus this weight × the
pair's own state (+1 if its last run failed, up to +1 for widened exploration), so pairs
that are stuck or still converging keep being revisited until they succeed.

**Full-network hindsight regression** (``bandit.shared_update: regression``,
:mod:`mllf.cb.hindsight_regression`): the default ``posterior`` update moves only each
head's last linear layer, in closed form. With ``regression``, the epoch's hindsight
targets instead train the whole policy, every head's trunk and output weights, by
gradient descent:

* the loss pulls each prediction toward its hindsight target, weighted by that target's
  own variance (the regression counterpart of the posterior update);
* pretraining runs are replayed in the same loss (``regression.replay_weight``), with the
  data-hygiene weights, so fitting new pairs doesn't overwrite what the model knows. Each
  source's error is measured in its own unit (its starting residual), so the noisier
  pretraining targets don't swamp the online signal;
* online targets from the last ``regression.online_history`` epochs are kept, so each
  step sees more than one epoch's pairs;
* afterwards each Bayesian head is re-seeded with the trained output weights and its
  calibrated prior precision, so Thompson sampling and uncertainty-driven selection keep
  working on the new features. The precision is not updated from online runs, so the
  posterior variance that selection ranks by reflects distance from the pretraining
  data; the attention term (``selection_attention_weight``) is what brings back pairs
  that are failing online. ``min_observations_before_confident`` does not apply in this
  mode, since no posterior observations are added;
* with ``regression.validation_gate``, a validation pass that scores worse than the last
  accepted one by more than ``gate_margin`` restores that accepted model (per-pair
  offsets are kept).

Unlike REINFORCE, which only learns whether a sampled action scored above or below
average, each run here supplies a direction and size for every coefficient it measured.
REINFORCE and the closed-form posterior update are unchanged and selectable as before.
The replay set can be prebuilt with ``training/build_regression_replay.sh``.

**Uncertainty-driven selection with plain REINFORCE**: ``bandit.selection: uncertainty``
also works without ``neurallinear_ts`` — ``select_combos_by_uncertainty`` accepts a
``score_fn`` and defaults to ``action_std_score`` for any other ``bandit.algorithm``, which
ranks candidate combos by the policy's own mean action std (``exp(log_std)``) instead of a
Bayesian posterior variance. This needs no extra model state, but it is *not* epistemic
uncertainty — ``log_std`` is a learned exploration parameter, so it measures how much the
policy is currently exploring a combo rather than how much it would learn from it. Default
remains ``random`` (stratified/uniform combo sampling) for REINFORCE; both strategies have
merit and the choice is left to the config.

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
* **dim 1 (quadratic)**: ``barrier_quality(barrier)`` =
  :math:`0.5 + 0.5\exp[-\tfrac{1}{2}((B - B_0)/\sigma)^2]` ∈ [0.5, 1.0], where
  :math:`B` is the pair's biased midpoint free-energy barrier from the λ trajectory,

  .. math::

     B = G(\lambda \approx 0.5) - \tfrac{1}{2}\left[G(\lambda \approx 0) + G(\lambda \approx 1)\right],
     \qquad G = -kT \ln P(\lambda)

  (``parse_pair_barriers`` in ``mllf.file_handling.read_output``; for 3+ substituents,
  frames on the pair's edge of the simplex). The quadratic term contributes
  :math:`-c\,\lambda_i\lambda_j`, which is zero at both endpoints and :math:`-c/4` at the
  midpoint, so :math:`B` moves by :math:`-c/4` per unit :math:`c`, while the linear bias
  cancels between the midpoint and the mean of the endpoints. The signal is therefore
  per-pair and specific to the quadratic coefficient, and two-sided: an over-flattened
  barrier (run stuck at intermediate λ) and a too-high barrier (few crossings) both score
  lower. :math:`B_0` / :math:`\sigma` are ``reward.barrier_target`` /
  ``reward.barrier_width`` (defaults ``BARRIER_TARGET`` / ``BARRIER_WIDTH`` in
  ``workflow_utils``, set from ``pretraining/1_analysis_scripts/analyze_bias_sensitivity.py``).
  When no barrier is available (outputs without a λ trajectory, or cached results /
  ``simulation_results.json`` files without ``barrier_pairs``), falls back to the legacy
  ``(pair_visited + trans_quality) / 2`` with
  ``trans_quality = min(total_transitions, T_baseline) / T_baseline``
* **dim 2 (skew)** and **dim 3 (end)**: ``gaussian_quality(v, 0, σ)`` ∈ [0.5, 1.0], where
  :math:`v` is an *asymmetry of the biased* λ profile along the pair's edge — what remains
  after the biases have (or have not) cancelled the physical surface's asymmetry. The
  physical surface is not symmetric; that is why the skew and end coefficients exist. The
  target 0 means they have cancelled it, so the simulation samples a profile that is
  symmetric in that respect (runs whose sampled profile kept less asymmetry succeed more
  often, across 26 systems). Both come from ``parse_pair_profile``, with
  :math:`r = \lambda_{hi}/(\lambda_{lo}+\lambda_{hi})` and :math:`G = -kT\ln` (density):

  - **end** = :math:`K_{hi} - K_{lo}`, each endpoint's sharpness
    :math:`K = G(\text{near-end shoulder}) - G(\text{endpoint core})` ([0.95, 0.99) vs
    ≥ 0.99, and [0.01, 0.05) vs < 0.01). The end term :math:`s_{ij}` peaks within a few
    percent of an endpoint and vanishes at it, so :math:`K_{hi}` moves ≈ −0.44 per unit
    ``s[lo][hi]`` and :math:`K_{lo}` ≈ −0.46 per unit ``s[hi][lo]`` (``x`` leaks ≈ 0.09;
    b, c ≈ 0).
  - **skew** = :math:`[G(r{\approx}0.7) - G(r{\approx}0.3)] - 0.4\,\text{tilt} - 0.687\,\text{end}`.
    The skew term ``x[lo][hi]`` is a broad hump peaking near :math:`r \approx 0.7`
    (``x[hi][lo]`` near 0.3), so the flank difference tracks ``x[lo][hi] − x[hi][lo]``
    (≈ ∓0.19 per unit); subtracting the linear tilt removes b, and 0.687·end removes the end
    terms' reach into the flanks (residual ≈ 0.01 per unit ``s``).

  Widths ``reward.skew_width`` / ``reward.end_width`` (defaults ``SKEW_WIDTH`` = 0.78 and
  ``END_WIDTH`` = 0.58 kcal/mol) put the reward at 0.75 where P(success) halves in a pooled
  fit. Both directed edges of a pair get the same value (the reward uses the magnitude).
  Fallbacks when no λ-profile signal is available: skew → the mean of the normalized
  population-balance signal and the legacy transition-based quality signal (``combined``);
  end → ``min(1.0, fraction_physical × total_pairs)``, else ``combined``. (The fraction
  physical ligand is mostly set by the barrier, i.e. by c, which is why the end asymmetry
  replaced it.)

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
