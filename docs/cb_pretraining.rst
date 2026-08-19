CB Behavior Cloning
===================

Overview
--------

Before reinforcement learning, the policy network can be **pretrained** using behavior cloning
on expert bias coefficients from existing optimized systems. This provides a warm start that:

* Accelerates CB training by starting near good solutions
* Reduces early exploration waste on poor bias values
* Stabilizes training by preventing catastrophic forgetting
* Transfers knowledge across similar chemical systems

Behavior cloning learns to imitate expert bias coefficients by supervised learning on
(graph structure → bias coefficients) pairs collected from pretraining systems.


Running Pretraining
--------------------

Pretraining is invoked via ``mllf.cb.pretrain_policy``'s CLI:

.. code-block:: bash

   python -m mllf.cb.pretrain_policy \
       --pretraining-dir pretraining/ \
       --output-dir models/my_pretrain_run \
       --config examples/workflow_pretrain.yaml \
       --epochs 50

``--pretraining-dir`` may be repeated to combine multiple collected-data directories into
one training set. Key flags (all optional, sensible defaults shown):

* ``--use-best-only`` — keep only the highest-reward run per system (default: use all valid runs)
* ``--no-per-pair-awr`` — disable per-pair AWR weighting and fall back to one scalar weight
  per run (see "Per-Pair AWR Weighting" above; per-pair is the default)
* ``--awr-temperature`` (default 0.5) — AWR temperature :math:`\beta`
* ``--min-transitions N`` / ``--min-reward-threshold`` / ``--stratified-negative-fraction`` —
  quality filters, see "Quality Filtering" below (mutually exclusive; ``--min-transitions``
  takes precedence)
* ``--no-filter-outliers`` / ``--outlier-std-threshold`` — statistical outlier filtering
  (enabled by default at ±3σ)
* ``--reward-weighted`` — additionally weight each run's loss by its (clamped, normalized)
  reward
* ``--bayesian-heads`` / ``--bayesian-prior-precision`` — after the normal behavior-cloning
  loop, also save a ``use_bayesian_heads=True`` sibling policy
  (``best_policy_bayesian.pt`` / ``final_policy_bayesian.pt``) for NeuralLinear + Thompson
  Sampling online training (see :doc:`cb_setup`), warm-started from the trained
  deterministic readout

The ``--config`` file (e.g. ``examples/workflow_pretrain.yaml``) controls the Uni-Mol
representation settings — ``unimol.structure_selection`` (minimized vs. unrelaxed
structures), ``unimol.environment_cutoff``, ``unimol.cache_embeddings`` — and the reward
function weights used to compute the per-run/per-pair AWR weights described above; it does
**not** control filtering or optimizer flags, which are CLI-only. See
``examples/pretrain_wUnimol.sh`` for a complete SLURM submission example (which sources
``examples/pretrain_with_filtering.sh`` and sets these flags via environment variables), and
:doc:`examples` for a walkthrough.

Behavior Cloning Training
--------------------------

The policy network learns to predict expert bias coefficients through supervised learning
on the filtered dataset. This process mimics how expert systems (ALF or converged CB) assign
coefficients to different molecular transition types.

.. _Training Objective:

**Training Objective**:

The model minimises an **Advantage-Weighted Regression (AWR)** loss: the negative
log-likelihood of the expert action under the current policy distribution, scaled by
an exponential weight derived from the run’s reward. Backpropagation runs only through
the *active* (non-zero) target for each edge via an ``active_mask``:

.. math::

   \mathcal{L}_{\text{AWR}} = -\frac{1}{|\mathcal{G}|}
   \sum_{r \in \mathcal{G}}
   \exp\!\left(\frac{R_r}{\beta}\right)
   \cdot
   \frac{1}{|\mathcal{A}_r|}
   \sum_{(i,j,k) \in \mathcal{A}_r}
   \log \mathcal{N}\!\left(a_{ij}^{(k)};
     \mu_{ij}^{(k)}, (\sigma_{ij}^{(k)})^2\right)

where:

- :math:`\mathcal{G}` is the set of runs sharing the same graph structure (gradient-accumulated
  before a single optimizer step)
- :math:`R_r` is the pre-computed BC reward for run :math:`r` (stored in ``run["_bc_reward"]``)
- :math:`\beta` is the AWR temperature (default: 1.0)
- :math:`\mathcal{A}_r = \{(i,j,k) : |a_{ij}^{(k)}| > \epsilon\}` is the set of active
  (non-zero) target entries for run :math:`r`
- :math:`k \in \{\text{linear, quadratic, skew, end}\}` are bias types
- :math:`a_{ij}^{(k)}` is the expert coefficient; :math:`\mu_{ij}^{(k)}, \sigma_{ij}^{(k)}`
  are the predicted mean and standard deviation
- :math:`\epsilon = 10^{-8}` is a small threshold to identify non-zero targets
- The exponential weight is capped at 20 to prevent a single excellent run from dominating

When ``_bc_reward`` is 0.0 (default), :math:`\exp(0/\beta) = 1.0` and AWR reduces to
unweighted NLL — pure behavior cloning with no weighting.

The AWR objective uses the Gaussian NLL rather than MSE, which trains the standard deviation
:math:`\sigma` alongside the mean. Larger :math:`\sigma` values allow the policy more
exploratory freedom in RL fine-tuning; the learned :math:`\sigma` is a useful prior for
initialising the exploration scale.


**Per-Pair AWR Weighting** (default):

By default (``--no-per-pair-awr`` to disable), the AWR weight above is not a single scalar
per run — it is computed **per edge, per dimension** via
``compute_pairwise_confidence_weights`` / ``compute_pair_reward`` (the same reward function
used by online REINFORCE, see :doc:`cb_setup`). Concretely, each pair's confidence weight
combines:

* **Population balance**: :math:`\min(p_i, p_j) / \max(p_i, p_j)` — zero if either
  substituent was never sampled
* **DDG reliability**: zero weight if the pair's ΔΔG is missing, NaN, or infinite (no usable
  λ-space crossing was observed for that specific pair)
* **Antisymmetric handling**: a missing forward-direction DDG is filled in from the reverse
  direction's negation when available, rather than being treated as unobserved

This means that within one run, well-resolved pairs (balanced populations, a clean DDG)
drive more gradient than poorly-resolved pairs, instead of every pair in the run sharing one
flat run-level weight. This keeps the pretraining signal consistent with what online
REINFORCE later optimizes against. The AWR temperature :math:`\beta` (``--awr-temperature``,
default 0.5) still controls how sharply confidence is weighted.

**Graph Caching**:

Before training begins, all pretraining graphs are built once and stored in an in-memory
cache. Each subsequent epoch iterates over the cached graphs instead of re-parsing RTF/PDB
files and recomputing Uni-Mol embeddings on every pass. This is the dominant source of
training speedup: for example, a dataset of ~25,000 runs may span only ~250 unique prep
directories (many runs share the same prep). Embedding computation then runs ~250 times
rather than ~25,000 times as it would without structure sharing. Rebuilding on every epoch
would cost many hours over a full training run. Set ``unimol.cache_embeddings: true`` in the
``--config`` file (see "Running Pretraining" above) to additionally persist computed
embeddings to disk across separate pretraining invocations.

**Learning Rate Schedule**:

A cosine annealing schedule decays the learning rate from the initial value down to
``lr / 100`` over the full number of training epochs. This allows the optimizer to make
large updates early in training while converging smoothly at the end, and helps avoid
escape from a good minimum once one is found. The current learning rate is printed
after each epoch and saved in the checkpoint.

**Training Outputs**:

The pretraining process produces several artifacts:

* **best_policy.pt:** Trained policy network weights ready for downstream use
* **training_log.txt:** Loss curves and convergence metrics across epochs
* **filtering_stats.json:** Record of which runs were excluded and why
* **checkpoint files:** Intermediate model states for recovery or analysis

**Using Pretrained Models**:

The pretrained policy provides a strong initialization for reinforcement learning on new
systems. Rather than starting with random weights, the CB agent begins with a policy that
already understands basic patterns in bias coefficient assignment.

Transfer learning is most effective when pretraining systems share structural or chemical
similarity with the target system. However, even diverse pretraining data improves learning
efficiency by teaching general principles of bias coefficient assignment.

Expert Coefficient Collection
------------------------------

Expert bias coefficients come from two sources:

1. **ALF-predicted coefficients**: Bias values predicted by the Adaptive Landscape Flattening (ALF) algorithm
2. **CB-optimized coefficients**: Bias values from converged CB training runs

Both are stored in ``variables.py`` files in prep directories:

.. code-block:: python

   # variables.py from an optimized system
   bias_string = '''
   b:  # Linear bias
   - 0.245
   - -0.132
   - 0.089
   c:  # Quadratic bias (symmetric, upper triangle stored)
   - [0.0, 2.34, -1.56]
   - [0.0, 0.0, 3.12]
   - [0.0, 0.0, 0.0]
   # ... skew and end biases ...
   '''

Pretraining Dataset Generation
-------------------------------

The pretraining system automatically discovers and collects expert demonstrations from
the ``pretraining/`` directory structure:

**Directory Structure**:

.. code-block:: text

   pretraining/
   ├── 14benz_solv/            # Per-run prep (each run carries its own prep/)
   │   ├── run1/
   │   │   ├── prep/
   │   │   │   ├── core.pdb
   │   │   │   ├── site1_sub1_pres.rtf
   │   │   │   └── ...
   │   │   └── variables.py  # Expert coefficients
   │   ├── run2/
   │   └── ...
   ├── 123benz_solvent_group1/ # Shared prep (all runs share a single prep/)
   │   ├── prep/
   │   │   ├── core.pdb
   │   │   ├── site1_sub1_pres.rtf
   │   │   └── ...
   │   ├── run1/
   │   │   └── variables.py  # Expert coefficients
   │   ├── run2/
   │   └── ...
   ├── abl_protein_mutant_group1/
   │   ├── run1/
   │   └── ...
   ├── 14benz_pair_combos/  # Multi-combo structure
   │   ├── comb_0063.../
   │   │   ├── run_001/
   │   │   │   ├── prep/
   │   │   │   └── variables.py
   │   │   └── ...
   │   └── ...
   └── ...

**Automatic Discovery**:

The ``pretrain_with_filtering.sh`` script scans all subdirectories and collects:

1. **System identification**: Detects all systems in ``pretraining/``
2. **Run enumeration**: Finds all run directories per system
3. **Variables extraction**: Parses ``variables.py`` for bias coefficients
4. **RTF parsing**: Builds graph structure from ``prep/*.rtf`` files
5. **Performance metrics**: Extracts rewards from simulation metadata

**Supported Structures**:

* **Standard**: ``pretraining/system_name/run*/`` (e.g., ``14benz_solv/run1/``)
* **Combo**: ``pretraining/system_name/comb_*/run_*/`` (e.g., ``14benz_pair_combos/comb_0063.../run_046/``)

Each run directory must contain:

* ``variables.py`` with bias coefficients (b, c, x, s matrices)
* ``prep/`` with RTF/PDB files — either as a subdirectory of the run directory
  (per-run prep) or as a shared ``prep/`` in the parent system directory (takes
  priority when both exist)
* Optional: ``metadata.json`` with reward/performance data

**Dataset Format** (internal):

.. code-block:: python

   # Collected dataset structure
   # [
   #   {
   #     'graph': Graph object,
   #     'bias_coefficients': {
   #       'linear': [...],
   #       'quadratic': [[...]],
   #       'skew': [[...]],
   #       'end': [[...]]
   #     },
   #     'system_name': 'abl_protein_mutant_group1',
   #     'run_name': 'run1',
   #     'reward': 0.89,
   #   },
   #   ...
   # ]

**Quality Filtering**:

The pretraining pipeline includes automatic filtering to exclude poorly-performing data.
These filters ensure the policy learns from successful, generalizable bias configurations
rather than unstable or failed simulation runs.

**1. Statistical Outlier Filtering**

Expert demonstrations with abnormally large bias coefficient values are excluded based on
statistical deviation from the dataset mean. By default, runs with coefficients beyond
±3 standard deviations (σ) are filtered out.

*Why this matters:* Unstable simulations can produce extreme bias values that reflect
numerical issues rather than effective sampling strategies. Including these outliers in
training would corrupt the learned policy, causing it to predict unrealistic coefficients
for new systems. The threshold can be adjusted to be more permissive (e.g., ±4σ) for
diverse datasets or stricter (e.g., ±2σ) when data quality is uncertain.

**2. Minimum Reward Threshold**

Demonstrations can be filtered based on their achieved sampling performance (reward).
Only runs that meet a minimum reward criterion are included in the training dataset.

*Why this matters:* Not all expert bias coefficients lead to adequate sampling. Some
ALF predictions or early CB attempts may stabilize the simulation but fail to achieve
sufficient transitions between states. By setting a reward threshold (e.g., ≥0.5),
the policy learns only from configurations that demonstrably improved sampling efficiency.
This prevents the model from imitating mediocre solutions.

**3. Best-Only Mode**

When multiple runs exist for the same system, only the highest-reward run is included
in training. All other runs from that system are excluded, regardless of their individual
quality.

*Why this matters:* Multiple runs per system often represent iterative refinement—early
attempts with suboptimal coefficients followed by improved solutions. Training on all
runs would give equal weight to both poor and excellent solutions from the same system.
Best-only mode focuses learning on proven successful configurations, which is particularly
valuable when pretraining data includes many exploratory runs.

If the number of systems is small, then behavior cloning may not be as effective due to
limited data diversity.

**4. Stratified Negative Sampling**

As an alternative to a hard reward threshold, stratified negative sampling keeps all
positive-reward runs and samples a fraction from each negative-reward bucket
(``(-inf,−50]``, ``(-50,−40]``, …, ``(-10,0)``) using a **quadratic ramp**: the
worst bucket retains 0% and the best negative bucket retains at most
``fraction_per_bucket`` (default: **55%**). Intermediate buckets scale as
:math:`f_i = f_{\max} \times (i / (N-1))^2`, concentrating sampling on near-zero runs
whose coefficients were almost correct.

*Why this matters:* Pure best-only cloning can give the policy an impoverished view of the
reward distribution, making it hard to distinguish near-success from complete failure.
Stratified sampling exposes it to the full reward landscape while still over-representing
higher-quality runs. Positive-reward runs are always retained in full. (Current training
has no separate Q-critic or value network — see :doc:`cb_setup` — so this filter shapes the
data the single behavior-cloned policy sees, not a critic's warmup data.)

**Combined Filtering Strategy**:

Filters can be combined to implement sophisticated data selection policies. A typical
production configuration uses **outlier filtering** (default, ±3σ) + **per-pair AWR**
(default) + either **best-only** (``--use-best-only``, narrow but clean) or **stratified
negative sampling** (``--stratified-negative-fraction``, broader reward coverage), depending
on how much exploratory/failed data the pretraining set contains.


See Also
--------

* :doc:`file_handling` - Bias coefficient file formats, ``parse_single_ddg`` reference
* :doc:`unimol_representation` - Pretrained Uni-Mol node embeddings pretraining trains on top of
* :doc:`cb_setup` - CB infrastructure and policy architecture
* :doc:`workflow` - Complete CB training workflow
* :doc:`examples` - Running pretraining and workflows
* ``examples/pretrain_wUnimol.sh`` / ``examples/pretrain_with_filtering.sh`` - Pretraining SLURM scripts
* ``examples/workflow_pretrain.yaml`` - Pretraining ``--config`` file template
* ``src/mllf/cb/pretrain_policy.py`` - Behavior cloning implementation
* ``src/mllf/cb/policy.py`` - Policy network architecture
* ``src/mllf/cb/bayesian_head.py`` - NeuralLinear + Thompson Sampling head, used by ``--bayesian-heads``
* ``src/mllf/cb/workflow_utils.py`` - ``compute_pair_reward``, ``parse_simulation_metrics``
