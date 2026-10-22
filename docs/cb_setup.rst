Contextual Bandit Setup
=======================

Overview
--------

The model predicts all four bias coefficients for every pair of substituents at a λ-site,
from pretrained molecular representations of those substituents. ``UnimolPolicy``
(``mllf.cb.policy``) is deliberately simple: frozen Uni-Mol embeddings as node features,
one edge input per ordered pair, and four independent heads, one per bias type. No graph
encoder or message passing is involved. What the model knows about a substituent's
chemistry and environment comes from Uni-Mol (:doc:`unimol_representation`), and what it
knows about biases comes from pretraining (:doc:`cb_pretraining`) and online corrections
(below).

.. _Architecture Diagram:

.. code-block:: text

   Uni-Mol [frozen]           per substituent:
                                ligand_only = UniMol(core + sub)                   [512]
                                full        = UniMol(core + sub + ref_subs + env)  [512]
                                node feature = [ligand_only, full]                 [1024]
          │
          ▼
   Edge input (A → B)         [ligand_only_A − ligand_only_B,  (full_A + full_B)/2] [1024]
          │
          ▼
   Four BiasHeadMLP heads     linear, quadratic, skew, end; each:
   (no shared weights)          trunk 1024 → 512 → 256 → 64 (ReLU)
                                readout Linear(64, 2) → (μ, log σ)
                                  or BayesianLinearHead(64 → 1)  [NeuralLinear mode]
          │
          ▼
   Output per edge            softsign(μ) × scale,  scale = [305, 520, 85, 30]   [E, 4]

Graph and Edge Inputs
---------------------

Each substituent is a node; ``build_directed_pairs()`` (``mllf.cb.graph_utils``) adds an
edge in both directions for every pair of substituents *at the same site*. There are no
cross-site edges, so cross-site couplings are not predicted. Every edge carries all four
coefficients. ``build_graph_and_data`` (``mllf.cb.graph_build``) builds this graph for
online training, and ``build_fully_connected_graph_for_pretraining`` builds it for
pretraining. Both:

1. parse the RTF/PDB files to find each site's substituents and the core
2. build the site's consensus environment from the core (skipped in gas phase)
3. compute a dual Uni-Mol embedding for every substituent at every site, not only those
   in the combination name
4. store ``data.x``, ``edge_index`` and ``site_index`` for the forward pass

The edge input splits into two halves with different symmetry. The difference of the
ligand-only embeddings is antisymmetric and carries how the two substituents differ. The
mean of the full embeddings is symmetric and carries the environment they share. In
non-dual mode, the edge input is ``[emb_A − emb_B, (emb_A + emb_B)/2]`` from one 512D
embedding per node.

.. note::
   ``mllf.cb.graph_utils.build_pyg_graph_from_mllf_graph`` provides a sparser alternative
   with hand-built atom type/charge features (``mllf.cb.atom_vocab``) and no Uni-Mol. It
   expands each pair into relation-typed edges, each routed to one head
   (``edge_type // 2``), and pairs with the encoder-agnostic ``EdgePolicy``. It is kept for
   callers without PDB coordinates; training uses the Uni-Mol path.

Bias Heads
----------

Each bias type has its own head with its own trunk, so gradient from one coefficient
cannot overwrite features another has learned. When ``edge_type`` is ``None`` (the
Uni-Mol path), every head processes every edge and the outputs are stacked into ``[E, 4]``.

* **Output scale.** Means are ``softsign(μ_raw) × scale`` with scales ±305 (linear),
  ±520 (quadratic), ±85 (skew) and ±30 (end): the largest values in more than 20,000
  pretraining runs plus about 10%. Softsign rather than tanh keeps a usable gradient at
  large inputs (:math:`1/(1+|z|)^2`), so heads do not saturate after long training.
* **Exploration.** ``log_std`` is clamped to ``[-20, 2.0]``, and sampled actions to
  1.05 × the scale.
* **Outputs.** ``actions`` and ``mean`` / ``log_std`` (all ``[E, 4]``), and ``logp`` for
  REINFORCE (``None`` with Bayesian heads, since Thompson sampling has no log-probability).

From Edge Predictions to Biases
-------------------------------

``write_variables_from_actions`` (``mllf.cli.workflow``) turns the per-edge predictions
into the per-site matrices of ``variables.py`` (storage conventions in
:doc:`file_handling`). Each coefficient is combined according to its symmetry:

* **Quadratic** is symmetric (:math:`c_{ij} = c_{ji}`). The two directed predictions are
  averaged and stored once in the upper triangle.
* **Skew and end** are independent per direction and stored directly:
  ``x[i][j] = v_fwd``, ``x[j][i] = v_bwd``.
* **Linear** predictions approximate :math:`b_j - b_i`, which is antisymmetric. Averaging
  the two directions would cancel the signal, so b is reconstructed relative to the site's
  reference (:math:`b_{\text{ref}} = 0`), with
  :math:`b_j = \tfrac{1}{2}(v_{\text{fwd}} - v_{\text{bwd}})`. Production export instead
  fits b per site from every within-site edge, weighted by each pair's offset variance
  (:doc:`workflow`).

Training Signals
----------------

Per-Edge Reward
~~~~~~~~~~~~~~~

REINFORCE and the default ``awr`` posterior update learn from a reward per edge and per
coefficient (``compute_pair_reward``, ``mllf.cb.workflow_utils``), not from one number per
run. Each head is trained only on its own dimension, so credit is assigned per
coefficient without masking or a critic:

.. math::

   \mathcal{L}_{\text{policy}} = -\sum_{e} \sum_{d} R_{e}^{(d)} \cdot \log \pi_\theta(a_e^{(d)} \mid s)

An edge whose pair was never visited (no finite ΔΔG, or neither substituent sampled) gets
−1 in every dimension. For a visited pair, each dimension measures the part of the λ
profile its coefficient controls:

* **Linear**: the minority fraction :math:`\min(p_i, p_j)/(p_i + p_j) \in [0, 0.5]`.
* **Quadratic**: :math:`0.5 + 0.5\exp[-\tfrac{1}{2}((B - B_0)/\sigma)^2]`, from the pair's
  biased midpoint barrier

  .. math::

     B = G(\lambda \approx 0.5) - \tfrac{1}{2}\left[G(\lambda \approx 0) + G(\lambda \approx 1)\right],
     \qquad G = -kT \ln P(\lambda)

  (``parse_pair_barriers``; for three or more substituents, the frames on the pair's edge of
  the simplex). The quadratic term is :math:`-c\,\lambda_i\lambda_j`: zero at both endpoints
  and :math:`-c/4` at the midpoint, while the linear term cancels between the midpoint and
  the mean of the endpoints. B therefore responds to c alone, and the reward is
  two-sided: an over-flattened barrier and a too-high one both score lower.
  :math:`B_0` and :math:`\sigma` are ``reward.barrier_target`` / ``reward.barrier_width``,
  set from the pretraining bias-sensitivity analysis.
* **Skew and end**: :math:`0.5 + 0.5\exp[-\tfrac{1}{2}(v/w)^2]`, where :math:`v` is an
  asymmetry of the *biased* λ profile along the pair's edge (``parse_pair_profile``). The
  physical surface is asymmetric, which is why these coefficients exist; :math:`v = 0`
  means they have cancelled that asymmetry. In the pretraining data, runs that kept less
  asymmetry succeeded more often across 26 systems.

  - **end** :math:`= K_{hi} - K_{lo}`, the difference in sharpness of the two endpoints,
    :math:`K = G(\text{shoulder}) - G(\text{core})` (λ in [0.95, 0.99) vs ≥ 0.99). The end
    term peaks just short of an endpoint and vanishes at it, so each K responds to one
    direction of s (≈ −0.45 per unit) and barely to b, c or x.
  - **skew** :math:`= [G(r{\approx}0.7) - G(r{\approx}0.3)] - 0.4\,\text{tilt} - 0.687\,\text{end}`,
    with :math:`r = \lambda_{hi}/(\lambda_{lo}+\lambda_{hi})`. The skew term is a broad hump
    near :math:`r \approx 0.7` (0.3 for the other direction), so the flank difference tracks
    ``x[lo][hi] − x[hi][lo]``; subtracting the tilt removes b, and the end correction
    removes the end terms' reach into the flanks.

  The widths ``reward.skew_width`` (0.78) and ``reward.end_width`` (0.58 kcal/mol) put the
  reward at 0.75 where the success probability halves in a pooled fit.

Without a λ trajectory, all three nonlinear dimensions fall back to visit, transition and
population-balance signals. Edges of unvisited pairs can be
down-weighted (``reward.no_transition_weight``, default 0.15) rather than removed.

NeuralLinear + Thompson Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``training.policy.use_bayesian_heads: true`` and ``bandit.algorithm:
neurallinear_ts``, each head's readout is a :class:`~mllf.cb.bayesian_head.BayesianLinearHead`:
a Bayesian linear regression on the trunk's 64D features with a full Gaussian posterior,
updated in closed form. Acting draws one weight sample per decision (Thompson sampling,
``UnimolPolicy.get_actions_thompson``), and the posterior variance tells selection where
the model is uncertain. ``pretrain_policy.py --bayesian-heads`` converts a pretrained
policy, starting each posterior mean from the trained readout (:doc:`cb_pretraining`).

Hindsight Targets
~~~~~~~~~~~~~~~~~

A reward says whether a run's biases were good, but not which way they were off or by how
much. With the default ``awr`` update, the posterior moves toward the actions sampled,
which sit near the current mean, so it hardly moves. With ``bandit.update_target:
hindsight`` (:mod:`mllf.cb.hindsight`), each run is relabelled with the biases it
*measured* it should have used, and the heads in ``hindsight_dims`` (default all four)
regress toward those. Each target comes from the run's own λ profile, using responses from
the pretraining bias-sensitivity analysis:

* **linear**, edge :math:`(i \to j)`: :math:`\Delta b^* = \Delta b - kT \ln(p_j / p_i)`;
  when one substituent is unsampled, a capped step toward it.
* **quadratic**: :math:`c^* = c + (B - B_0)/|s_c|`, where :math:`s_c` is the barrier's
  response to c.
* **skew / end**: the step that removes the pair's skew or end asymmetry, split between the
  two directions. When λ crowds up to an endpoint without holding it, the end term there
  is holding λ off, and that s moves part of the way toward a typical converged value
  (``end_reference_s``). This signal exists exactly when the others do not.

Steps are capped (``hindsight_{linear,quadratic,skew,end}_step_cap``) because the
linearized responses hold only near sensible biases.

Iterative flattening also corrects biases from each simulation, but only for the system
being simulated. This implementation differs in four ways:

* **Transfer.** The targets train a shared model conditioned on Uni-Mol embeddings, so what
  is learned on one pair improves the first guess for pairs and systems never simulated.
* **Calibrated weighting.** Each target enters with its own measurement variance
  (crossings for b, scatter of the barrier or asymmetry, and a term growing with step
  size), so well-measured runs count for more without fixed regularization.
  ``hindsight_weight_scale`` (per head if given as a dict) sets how strongly the shared
  model follows the training pairs; a head whose update would shift every pair can be
  damped while the per-pair offsets still correct it pair by pair.
* **Per-pair offsets.** Each pair's own measurements are combined with the model as a
  prior, and proposals sample from that posterior.
* **Active selection.** The shared posterior's uncertainty and each pair's state choose
  which pairs to run next.

Assigning a Failed Run to b or c
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pair run that never crosses has one of two problems: b favors one substituent, or the
barrier is too high to cross. Raising c helps only in the second case; in the first it
makes the landscape harder to sample. The λ trajectory usually tells them apart:

* **Pinned**: λ never left its starting substituent (≥ 99% of frames, nothing else
  sampled). The barrier holds λ, so this is the evidence for raising c.
* **Stranded**: nothing at the site was sampled; λ stayed between endpoints. The quadratic
  bias flattens too much, so c steps down (unless the run measured a barrier).
* **One-sided**: one substituent sampled, the other not. b steps toward the unsampled one.

Repeated one-sided steps in the same direction double, up to ``linear_step_max``, for the
pair's offset (the shared model keeps the base step). Each pair also keeps a **b
bracket**: the b values at which it was stuck on either side. Once both bounds exist, the
next target is their midpoint. If the bracket has closed to within ``barrier_trap_width``
and λ still does not cross, b is no longer the problem: the **barrier trap** steps c up. c
steps double the same way, up to ``quadratic_step_max``, and bisect between a c that was
too low and one too high.

A bracket can **invert**: stuck needing a higher b at a b above one where it was stuck
needing a lower b. λ then ends wherever the barrier catches it, not where b points, and
bisecting would fix b at a tight spread that only a crossing could release. An inverted b
bracket is cleared instead: b goes to its midpoint with the ordinary step's spread, and
the run counts as a barrier trap, so c steps up. An inverted c bracket (told to go both
up and down over the same range) is cleared too, and c takes an ordinary step from there.

Two switches set how strictly b is tried first:

* ``bandit.b_direction_requires_crossing`` (default ``true``): a run that never crossed
  gives no b direction and steps c up, like a pinned run. This is cautious when the
  trajectory cannot separate the cases, but it raises c on every one-sided run.
* ``bandit.c_up_requires_b_bracket`` (default ``false``): with ``true``, even a pinned run
  steps b first, and c rises only through the barrier trap.

The staged configs set ``false`` and ``true``. b is measured directly by the population
ratio and is cheaper to get right, while a c raised too early must be walked back down by
stranded runs.

Per-Pair Offsets and Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``bandit.pair_offsets: true``, each training pair's last ``offset_history`` targets
are stored in ``bandit_state.json`` under its original site/sub numbers, per coefficient
(and per direction for x and s). The next proposal combines them with the model's
prediction as a prior of sd ``offset_prior_sd``. A pair the model predicts well moves
little; one it predicts poorly is corrected from its own data without distorting the
shared model. The shared model carries what generalizes, and the offsets hold what is
specific to a pair. Validation never applies offsets, so it measures the shared model
alone.

With ``bandit.offset_sampling: true``, proposals draw each pair's offset instead of using
its mean:

* An unmeasured coefficient is explored with sd ``offset_base_sd × factor`` on top of the
  Thompson sample; a measured one is drawn from the pair's offset posterior, so
  exploration narrows as data accumulate.
* A failure with no direction for *any* coefficient doubles the pair's factors, up to
  ``explore_factor_max``; a success halves them. A failure that does give a direction
  leaves them alone, since the targets address it.
* ``bandit.explore_after_failures`` failures in a row (default 3) double all factors,
  measured coefficients included (capped by the offset sd). Because b gets a target
  whenever either substituent is sampled, this is the usual way a stuck pair widens its
  search.

Diagnosable failures are thus corrected directly, and undiagnosable ones widen the search
until something measurable happens. That is how a new system whose failure mode was not
anticipated still converges.

Selection
~~~~~~~~~

With ``bandit.selection: uncertainty``, combos are ranked by the percentile of their
shared posterior variance, plus ``selection_attention_weight`` × the pair's own state (+1
if its last run failed, up to +1 for widened exploration), so stuck pairs keep coming
back. Without ``neurallinear_ts``, the score falls back to the policy's mean action std.
That needs no extra state, but it measures how much the policy is exploring a combo, not
how much it would learn from it. REINFORCE defaults to ``random`` (stratified) selection.

**Revisits** (``bandit.revisit``). The per-pair corrections above need *consecutive*
visits, and in a large pool, selection may not draw a pair again for many epochs. With
``revisit.enabled: true``, pairs in progress are rerun each epoch, taking up to
``revisit.fraction`` of each system's ``min_combos_per_system`` slots. A pair leaves the
revisit pool once *satisfied* (``revisit.successes`` good runs, default 2) or *exhausted*
(``revisit.max_visits`` visits, default 8), so a pair that cannot be fixed stops holding a
slot. Revisits need ``neurallinear_ts`` and a per-epoch combo budget.

Full-Network Regression
~~~~~~~~~~~~~~~~~~~~~~~

The default ``posterior`` update moves only each head's last layer. With
``bandit.shared_update: regression`` (:mod:`mllf.cb.hindsight_regression`), the epoch's
hindsight targets train the whole policy by gradient descent:

* each prediction is pulled toward its target, weighted by the target's variance;
* pretraining runs are replayed in the same loss (``regression.replay_weight``, with the
  data-hygiene weights), so fitting new pairs does not overwrite what the model knows.
  Each source's error is measured against its own starting residual, so the noisier
  pretraining targets do not swamp the online signal. With
  ``regression.replay_pretraining_only`` (default ``true``), online runs collected into the
  datasets later (tagged with ``collection_tag``, :doc:`cb_pretraining`) are not replayed,
  so a new test does not replay earlier tests. Set it to ``false`` for a model that was
  pretrained on those runs;
* which runs are replayed is set by ``regression.replay_selection``. ``random`` (default)
  takes one random run per combination directory, in turn across datasets, up to
  ``replay_runs``. ``coverage`` (:mod:`mllf.cb.replay_selection`) gives each system a
  share of ``replay_runs`` proportional to its substituents (at least
  ``replay_min_per_system``) and fills it with the runs that cover the system's pairs
  best. A pair's score in a run is its population balance times its crossings relative
  to the run length's success test. Runs are picked greedily by how much pair coverage
  they add (a pair counts as covered at score 0.5), so a group run that samples many pairs
  well wins over pair runs that each sample one slightly better. A system stops below its
  share once nothing adds coverage, and the share it leaves goes to systems that used all
  of theirs, in proportion to their uncovered pairs. Each run keeps weight only on the
  pairs it is the best chosen run for. A system left with fewer than
  ``replay_min_per_system`` graphs (default 3; one run sampled all its pairs) gets more,
  chosen to give its pairs a second good measurement, each weighted on every pair it
  measured well. Substituents are matched across ALF groups, combinations and
  online runs by fragment content against ``replay_systems_root/<system>/prep``. A
  per-pair report (``<replay_cache>_pairs.csv``) lists each pair's run, source and score;
* targets from the last ``regression.online_history`` epochs are kept;
* each Bayesian head is then re-seeded with the trained weights and its calibrated prior
  precision. The precision is not updated from online runs, so the posterior variance
  reflects distance from the pretraining data, and the attention term is what brings back
  pairs failing online (``min_observations_before_confident`` does not apply);
* with ``regression.validation_gate``, a validation pass worse than the last accepted one by
  more than ``gate_margin`` restores that model (offsets are kept).

Unlike REINFORCE, which learns only whether an action scored above or below average,
each run here supplies a direction and size for every coefficient it measured. The replay
set can be prebuilt with ``training/build_regression_replay.sh``.

See Also
--------

* :doc:`workflow` - stages, configuration and outputs of online training
* :doc:`unimol_representation` - node embeddings and environment consensus
* :doc:`cb_pretraining` - behavior cloning and data hygiene
* :doc:`file_handling` - ``variables.py`` storage conventions
