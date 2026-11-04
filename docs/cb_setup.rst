Contextual Bandit Setup
=======================

Overview
--------

The model predicts all four bias coefficients for every pair of substituents at a λ-site
from fixed molecular representations of those substituents. ``UnimolPolicy``
(``mllf.cb.policy``) takes frozen Uni-Mol embeddings as node features, builds one input
per ordered pair, and passes it to four independent heads, one per bias type, with no
graph encoder or message passing. Its knowledge of each substituent's chemistry comes from
Uni-Mol (:doc:`unimol_representation`), and its knowledge of biases comes from
pretraining (:doc:`cb_pretraining`) and from the online corrections described below.

Architecture
------------

.. _Architecture Diagram:

.. code-block:: text

   Uni-Mol [frozen]        per substituent (512 each):
                             ligand = UniMol(core + sub [+ other sites' reference subs])
                             second half, by unimol.env_input:
                               dual:  UniMol(core + sub + reference subs + environment)
                               phase: [condensed, protein] flags (+ site size), zeros
          │
          ▼
   Edge input (A → B)      dual:          [ligand_A − ligand_B, (full_A + full_B)/2]   1024
                           phase:         [ligand_A − ligand_B, flags (+ site size)]   514 (515)
                           phase_ligmean: phase + (ligand_A + ligand_B)/2               1026 (1027)
          │
          ▼
   Four BiasHeadMLP heads  linear, quadratic, skew, end; no shared weights
                             trunk  in → in/2 → in/4 → 64 (ReLU)
                             readout Linear(64, 2) → (μ, log σ), or BayesianLinearHead(64)
          │
          ▼
   Output per edge         softsign(μ) × scale,  scale = [305, 520, 85, 30]          [E, 4]

The edge input combines halves of different symmetry. The difference of the ligand
embeddings is antisymmetric and carries how the two substituents differ. The second half
is symmetric and carries what they share. In ``dual`` mode, it is the mean of the
context embeddings (substituent, reference substituents and environment). In ``phase``
mode, it is two flags, condensed and protein, so vacuum, solvent and protein legs are told
apart directly rather than through an environment embedding that would vary between
systems for reasons unrelated to the biases. ``phase_ligmean`` adds the pair's mean ligand
embedding as symmetric chemical context.

With ``unimol.site_size_input``, the phase flags are followed by :math:`1/(n-1)` for the
:math:`n` substituents at the edge's site in the current run (1 for a pair, 0.14 for eight
substituents). In a group run, the end terms of every partner act together at a shared
endpoint, so a pair's x and s that sample well in a pair run can hold that endpoint off in
a group. Without the site size, the model sees both as the same edge.

The ``unimol`` settings must match those of the pretrained model. The driver checks
``ligand_other_sites`` and ``site_size_input`` against the model's
``pretrain_metadata.json``.

Graph
~~~~~

Each substituent is a node, and ``build_directed_pairs`` (``mllf.cb.graph_utils``) adds an
edge in each direction for every pair of substituents at the same site. There are no
cross-site edges, so cross-site couplings are not predicted. ``build_graph_and_data``
(``mllf.cb.graph_build``) parses the combination's RTF and PDB files, builds each site's
consensus environment (in ``dual`` mode), embeds every substituent at every site, and
stores ``data.x``, ``edge_index`` and ``site_index`` for the forward pass.

Bias Heads
~~~~~~~~~~

Each bias type has its own trunk, so the gradient of one coefficient cannot overwrite
features another has learned. The output scales (±305 linear, ±520 quadratic, ±85 skew,
±30 end) are the largest values in more than 20,000 pretraining runs plus ∼10%. Softsign
rather than tanh keeps a usable gradient at large inputs (:math:`1/(1+|z|)^2`), so heads do
not saturate over long training. The exploration σ of each head is bounded between 0.2%
and 25% of its scale, which gives every head the same proportional room to explore.

From Edge Predictions to Biases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``write_variables_from_actions`` (``mllf.cli.workflow``) turns the per-edge predictions
into the per-site matrices of ``variables.py`` (storage conventions in
:doc:`file_handling`), combining each coefficient according to its symmetry.

* **Quadratic** is symmetric (:math:`c_{ij} = c_{ji}`). The two directed predictions are
  averaged and stored once, in the upper triangle.
* **Skew and end** are independent per direction and stored directly
  (``x[i][j] = v_fwd``, ``x[j][i] = v_bwd``).
* **Linear** predictions approximate :math:`b_j - b_i`, which is antisymmetric, so
  averaging the two directions would cancel the signal. b is instead fitted per site from
  every within-site edge, weighted by each pair's linear-offset variance, with the site's
  reference at 0.

Training Signals
----------------

Per-Edge Reward
~~~~~~~~~~~~~~~

REINFORCE, the ``awr`` update target and pretraining learn from a reward per edge and per
coefficient (``compute_pair_reward``, ``mllf.cb.workflow_utils``) rather than one number
per run. Each head is trained only on its own dimension, so credit is assigned per
coefficient without masking or a critic:

.. math::

   \mathcal{L}_{\text{policy}} = -\sum_{e} \sum_{d} R_{e}^{(d)} \cdot \log \pi_\theta(a_e^{(d)} \mid s)

An edge whose pair was never visited (no finite ΔΔG, or neither substituent sampled) gets
−1 in every dimension, and its weight is ``reward.no_transition_weight`` (default 0.2).
For a visited pair, each dimension measures the part of the λ profile its coefficient
controls.

* **Linear**: the minority fraction :math:`\min(p_i, p_j)/(p_i + p_j) \in [0, 0.5]`.
* **Quadratic**: :math:`0.5 + 0.5\exp[-\tfrac{1}{2}((B - B_0)/\sigma)^2]`, from the pair's
  biased midpoint barrier

  .. math::

     B = G(\lambda \approx 0.5) - \tfrac{1}{2}\left[G(\lambda \approx 0) + G(\lambda \approx 1)\right],
     \qquad G = -kT \ln P(\lambda)

  (``parse_pair_barriers``; for three or more substituents, the frames on the pair's edge
  of the simplex). The quadratic term :math:`-c\,\lambda_i\lambda_j` is zero at both
  endpoints and :math:`-c/4` at the midpoint, while the linear term cancels between the
  midpoint and the mean of the endpoints, so B responds to c alone. The reward is
  two-sided, since an over-flattened barrier and one too high both score lower.
  :math:`B_0` and :math:`\sigma` (``reward.barrier_target``, 1.2 kcal/mol, and
  ``reward.barrier_width``, 1.29 kcal/mol) come from the pretraining bias-sensitivity
  analysis.
* **Skew and end**: :math:`0.5 + 0.5\exp[-\tfrac{1}{2}(v/w)^2]`, where :math:`v` is an
  asymmetry of the *biased* λ profile along the pair's edge (``parse_pair_profile``). The
  physical surface is asymmetric, which is why these coefficients exist, and
  :math:`v = 0` means they have cancelled that asymmetry. Across 26 pretraining systems,
  runs that kept less asymmetry succeeded more often.

  - **end** :math:`= K_{hi} - K_{lo}`, the difference in sharpness of the two endpoints,
    with :math:`K = G(\text{shoulder}) - G(\text{core})` (λ in [0.95, 0.99) against
    ≥ 0.99). The end term peaks just short of an endpoint and vanishes at it, so each K
    responds to one direction of s (≈ −0.45 per unit) and barely to b, c or x.
  - **skew** :math:`= [G(r{\approx}0.7) - G(r{\approx}0.3)] - 0.4\,\text{tilt} - 0.687\,\text{end}`,
    with :math:`r = \lambda_{hi}/(\lambda_{lo}+\lambda_{hi})`. The skew term is a broad
    hump near :math:`r \approx 0.7` (0.3 for the other direction), so the flank difference
    tracks ``x[lo][hi] − x[hi][lo]``. Subtracting the tilt removes b, and the end
    correction removes the end terms' reach into the flanks.

  The widths ``reward.skew_width`` (0.78) and ``reward.end_width`` (0.58 kcal/mol) put the
  reward at 0.75 where the success probability halves in a pooled fit.

Without a λ trajectory, the three nonlinear dimensions fall back to visit, transition and
population-balance signals.

Bayesian Heads and Thompson Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``bandit.algorithm: hindsight``, each head's readout is a
:class:`~mllf.cb.bayesian_head.BayesianLinearHead`, a Bayesian linear regression on the
trunk's 64-D features with a full Gaussian posterior. Acting draws one weight sample per
decision (Thompson sampling, ``UnimolPolicy.get_actions_thompson``), scaled by
``bandit.exploration_scale``, and the posterior variance tells selection where the model
is uncertain. ``pretrain_policy --bayesian-heads`` converts a pretrained policy, starting
each posterior mean from the trained readout and calibrating its precision on the
pretraining data (:doc:`cb_pretraining`).

.. _Bandit Defaults:

Bandit Defaults
~~~~~~~~~~~~~~~

With ``algorithm: hindsight``, settings a config leaves out take the values below
(``mllf.cb.bandit_config``). A key set in the config wins, ``null`` turns off ``c_cap``
and ``group_holdoff``, and ``group_correction: false`` turns off the correction. The
sections that follow explain each setting.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Default
   * - ``update_target`` / ``shared_update``
     - ``hindsight`` / ``regression``
   * - ``exploration_scale``
     - 0.4
   * - ``offset_sampling`` / ``selection_attention_weight``
     - true / 0.5
   * - ``b_direction_requires_crossing`` / ``c_up_requires_b_bracket``
     - false / true
   * - ``linear_step_max``
     - 12.0
   * - ``hindsight_weight_scale``
     - linear 0.1, quadratic 0.01, skew 0.1, end 0.1
   * - ``c_cap``
     - floor 15.0, fraction 0.5
   * - ``group_holdoff``
     - dominant_min 0.7, at_max 0.05, partner_min 0.02, min_partners 2
   * - ``group_correction``
     - step 0.25, floor 0.5
   * - ``revisit``
     - enabled, max_visits 12
   * - ``regression.replay_selection`` / ``regression.replay_pretraining_only``
     - ``coverage`` / false
   * - stage ``bracketed_runs_sticky`` / ``partial_runs_to_resolve`` / ``group_pass_window``
     - true / 3 / 3 (:doc:`workflow`)
   * - ``training.policy.use_bayesian_heads``
     - true

Hindsight Targets
~~~~~~~~~~~~~~~~~

A reward says whether a run's biases were good, but not which way they were off or by how
much, and the ``awr`` target (which regresses toward the sampled actions, weighted by
reward) moves little because those actions sit near the current mean. With
``update_target: hindsight`` (:mod:`mllf.cb.hindsight`), each run is relabelled with the
biases it *measured* it should have used, and the heads in ``hindsight_dims`` (all four by
default) regress toward them. Each target comes from the run's own λ profile, using the
responses measured in the pretraining bias-sensitivity analysis.

* **linear**, edge :math:`(i \to j)`: :math:`\Delta b^* = \Delta b - kT \ln(p_j / p_i)`,
  or a capped step toward an unsampled substituent.
* **quadratic**: :math:`c^* = c + (B - B_0)/|s_c|`, where :math:`s_c` is the barrier's
  response to c.
* **skew / end**: the step that removes the pair's skew or end asymmetry, split between
  the two directions. When λ crowds up to an endpoint without holding it, the end term
  there is holding λ off, and that s moves part of the way toward a typical converged
  value (``end_reference_s``, 1.5). This signal exists exactly when the others do not.

Steps are capped (``hindsight_{linear,quadratic,skew,end}_step_cap``, 3.0, 12.0, 3.0 and
1.5) because the linearized responses hold only near sensible biases.

Iterative flattening also corrects biases from each simulation, but only for the system
being simulated. Hindsight targets add four things to that correction.

* **Transfer.** The targets train a shared model conditioned on Uni-Mol embeddings, so a
  correction learned on one pair improves the first guess for pairs and systems never
  simulated.
* **Calibrated weighting.** Each target enters with its own measurement variance
  (crossings for b, scatter of the barrier or asymmetry, and a term that grows with step
  size), so well-measured runs count for more without fixed regularization.
  ``hindsight_weight_scale`` sets how strongly the shared model follows the training
  pairs, per head. A head whose update would shift every pair can be damped while the
  per-pair offsets still correct it pair by pair.
* **Per-pair offsets.** Each pair's own measurements are combined with the model as a
  prior, and proposals sample from that posterior.
* **Active selection.** The shared posterior's uncertainty and each pair's state choose
  which pairs to run next.

Assigning a Failed Run to b or c
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pair run that never crosses has one of two problems. Either b favors one substituent, or
the barrier is too high to cross. Raising c helps only in the second case and makes the
landscape harder to sample in the first, so the λ trajectory is used to tell them apart.

* **Pinned**: λ never left its starting substituent (≥ 99% of frames, nothing else
  sampled), which is evidence that the barrier holds λ.
* **Stranded**: nothing at the site was sampled, and λ stayed between endpoints. The
  quadratic bias flattens too much, so c steps down (unless the run measured a barrier).
* **One-sided**: one substituent sampled and the other not, so b steps toward the
  unsampled one.

Repeated one-sided steps in the same direction double, up to ``linear_step_max``, for the
pair's offset (the shared model keeps the base step). Each pair also keeps a **b
bracket**, the b values at which it was stuck on either side. Once both bounds exist, the
next target is their midpoint. If the bracket has closed to within ``barrier_trap_width``
(2.0) and λ still does not cross, b is no longer the problem, and the **barrier trap**
steps c up. c steps double the same way, up to ``quadratic_step_max`` (48), and bisect
between a c that was too low and one that was too high.

A bracket can **invert**, stuck needing a higher b at a b above one where it was stuck
needing a lower b. λ then ends wherever the barrier catches it rather than where b points,
and bisecting would fix b at a tight spread that only a crossing could release. An
inverted b bracket is therefore cleared, b goes to its midpoint with the ordinary step's
spread, and the run counts as a barrier trap, so c steps up. An inverted c bracket is
cleared in the same way, and c takes an ordinary step from there.

By default, b is tried first. With ``b_direction_requires_crossing: false``, a one-sided
run that never crossed still gives b a direction, and with
``c_up_requires_b_bracket: true``, even a pinned run steps b first, so c rises only
through the barrier trap. We prefer this order because b is measured directly by the
population ratio and is cheaper to get right, whereas a c raised too early must be walked
back down by stranded runs.

The c cap (``c_cap``) bounds the other direction. Each pair's c stays within
:math:`\max(\text{floor}, \text{fraction} \times |c_{\text{model}}|)` of the model's
prediction in every proposal and in the exported biases, while the pair's offset itself
is kept. Repeated barrier-trap steps can otherwise carry c far beyond the model's
prediction. A pair that still fails at such a c is likely limited by something other than
its barrier, so further steps would raise c without producing crossings.

Per-Pair Offsets and Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each training pair's last ``offset_history`` (5) targets are stored in
``bandit_state.json`` under its original site and substituent numbers, per coefficient and,
for x and s, per direction. The next proposal combines them with the model's prediction
as a prior of sd ``offset_prior_sd``. A pair the model predicts well moves little, and one
it predicts poorly is corrected from its own data without distorting the shared model.
The shared model carries what generalizes, and the offsets hold what is specific to a
pair. Validation never applies offsets, so it measures the shared model alone.

With ``offset_sampling``, proposals draw each pair's offset rather than using its mean.

* An unmeasured coefficient is explored with sd ``offset_base_sd × factor`` on top of the
  Thompson sample, and a measured one is drawn from the pair's offset posterior, so
  exploration narrows as data accumulate.
* A failure with no direction for *any* coefficient doubles the pair's factors, up to
  ``explore_factor_max`` (8), and a success halves them. A failure that does give a
  direction leaves them alone, since the targets address it.
* ``explore_after_failures`` (3) failures in a row double all factors, measured
  coefficients included (capped by the offset sd). Because b gets a target whenever either
  substituent is sampled, this is how a stuck pair usually widens its search.

Diagnosable failures are corrected directly, and undiagnosable ones widen the search until
something measurable happens, which lets a new system converge even when its failure mode
was not anticipated.

Group End-Term Correction
~~~~~~~~~~~~~~~~~~~~~~~~~

In a group run, each substituent's endpoint collects the end terms (x, s) of every partner
at the site. Each pair's terms can sample well in its own pair run and still sum to a well
that holds the endpoint off in the group, which no pair run can see. Two patterns are
detected (``mllf.cb.group_correction``).

* **Endpoint**: one substituent dominates the site but almost never reaches λ = 0.99,
  while several partners hold the rest (``group_holdoff``, logged as ``[HOLDOFF]``).
* **Site**: no substituent at a site of three or more reaches λ = 0.99 and the site barely
  transitions, while another site in the same run samples normally.

A flagged run lowers those pairs' x and s for that group by ``step`` of their first value
per run, scaled by each partner's λ share for an endpoint, down to ``floor``. A corrected
site that then traps endpoints (some substituents unsampled, few transitions) is stepped
back by half. The corrections are stored per group in ``bandit_state.json``, the group's
later runs and its export use them, the model trains on the corrected values for those
edges, and c and b are left unchanged. A run that stepped does not count toward
``group_max_failures`` (:doc:`workflow`).

Selection
~~~~~~~~~

With ``bandit.selection: uncertainty`` (the hindsight default), combos are ranked by the
percentile of their shared posterior variance plus ``selection_attention_weight`` times
the pair's own state (+1 if its last run failed, up to +1 for widened exploration), so
stuck pairs keep coming back. With REINFORCE, the score is the policy's mean action
standard deviation, which measures how much the policy is exploring a combo rather than
how much it would learn from it, and the default is ``random`` (stratified) selection.

**Revisits** (``bandit.revisit``). The per-pair corrections above need *consecutive*
visits, and in a large pool, selection may not draw a pair again for many epochs. Pairs in
progress are therefore rerun each epoch, taking up to ``revisit.fraction`` (0.5) of each
system's ``min_combos_per_system`` slots. A pair leaves the revisit pool once *satisfied*
(``revisit.successes`` good runs, 2) or *exhausted* (``revisit.max_visits`` visits), so a
pair that cannot be fixed stops holding a slot.

Full-Network Regression
~~~~~~~~~~~~~~~~~~~~~~~

With ``shared_update: regression`` (:mod:`mllf.cb.hindsight_regression`), the epoch's
hindsight targets train the whole policy by gradient descent rather than only each head's
last layer (``shared_update: posterior``).

* Each prediction is pulled toward its target, weighted by the target's variance.
* Pretraining runs are replayed in the same loss (``regression.replay_weight``, 0.5, with
  the data-hygiene weights), so fitting new pairs does not overwrite what the model knows.
  Each source's error is measured against its own starting residual, so the noisier
  pretraining targets do not swamp the online signal.
* ``regression.replay_selection: coverage`` (:mod:`mllf.cb.replay_selection`) gives each
  system a share of ``replay_runs`` (2000) proportional to its substituents, at least
  ``replay_min_per_system`` (3), and fills it with the runs that cover the system's pairs
  best. A pair's score in a run is its population balance times its crossings relative to
  the run length's success test. Runs are picked greedily by the pair coverage they add (a
  pair counts as covered at score 0.5), so a group run that samples many pairs well wins
  over pair runs that each sample one slightly better. A system stops below its share once
  nothing adds coverage, and the share it leaves goes to systems that used all of theirs,
  in proportion to their uncovered pairs. Each run keeps weight only on the pairs it is
  the best chosen run for. Substituents are matched across ALF groups, combinations and
  online runs by fragment content against ``replay_systems_root/<system>/prep``, and a
  per-pair report (``<replay_cache>_pairs.csv``) lists each pair's run, source and score.
  ``random`` takes one random run per combination directory instead.
* ``regression.replay_pretraining_only: true`` skips online runs collected into the
  datasets (tagged with ``collection_tag``, :doc:`cb_pretraining`), for a model that was
  not pretrained on them.
* Targets from the last ``regression.online_history`` (5) epochs are kept.
* Each Bayesian head is then re-seeded with the trained weights and its calibrated prior
  precision. The precision is not updated from online runs, so the posterior variance
  reflects distance from the pretraining data, and the attention term is what brings back
  pairs that fail online.
* With ``regression.validation_gate`` (on by default), a validation pass worse than the
  last accepted one by more than ``gate_margin`` (2) restores that model, keeping the
  offsets.

Unlike REINFORCE, which learns only whether an action scored above or below average, each
run here supplies a direction and size for every coefficient it measured. The replay set
can be prebuilt with ``training/build_regression_replay.sh``.

See Also
--------

* :doc:`workflow`: stages, configuration and outputs of online training
* :doc:`unimol_representation`: node embeddings and environment consensus
* :doc:`cb_pretraining`: behavior cloning and data hygiene
* :doc:`file_handling`: ``variables.py`` storage conventions
