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


Behavior Cloning Training
--------------------------

The policy network learns to predict expert bias coefficients through supervised learning
on the filtered dataset. This process mimics how expert systems (ALF or converged CB) assign
coefficients to different molecular transition types.

.. _Training Objective:

**Training Objective**:

The model minimizes a **masked** mean squared error. Backpropagation runs only through
the *active* (non-zero) target for each edge. Because each directed edge carries exactly
one non-zero bias type (e.g., a ``linear_fwd`` edge has a non-zero linear target but zero
quadratic, skew, and end targets), averaging over all four outputs would produce a gradient
signal dominated by pushing inactive heads toward zero. The mask avoids this.

.. math::

   \mathcal{L}_{\text{BC}} = \frac{1}{|\mathcal{A}|} \sum_{(i,j,k) \in \mathcal{A}} \| a_{ij}^{(k)} - \hat{a}_{ij}^{(k)} \|^2

where:

- :math:`\mathcal{A} = \{(i,j,k) : |a_{ij}^{(k)}| > \epsilon\}` is the set of active (non-zero) target entries
- :math:`k \in \{\text{linear, quadratic, skew, end}\}` are bias types
- :math:`a_{ij}^{(k)}` is the expert coefficient
- :math:`\hat{a}_{ij}^{(k)}` is the predicted coefficient
- :math:`\epsilon = 10^{-8}` is a small threshold to identify non-zero targets

The DDG signal from λ-space crossings is instead used during reinforcement learning
via per-pair rewards (see :doc:`workflow` and :doc:`cb_setup`), which provides a
cleaner separation between supervised imitation (BC) and exploration feedback (RL).


**Graph Caching**:

Before training begins, all pretraining graphs are built once and stored in an in-memory
cache. Each subsequent epoch iterates over the cached graphs instead of re-parsing RTF
files and recomputing DeepSet embeddings on every pass. This is the dominant source of
training speedup: for example, the current dataset of ~25,000 runs spans only ~250
unique graph structures (many runs share the same prep directory). The AEV computation
runs ~250 times (~8 minutes) rather than ~25,000 times (~14 hours) as it would
without structure sharing. Rebuilding on every epoch would cost hundreds of hours over
a full training run.

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

The pretrained encoder (RGCN) can be frozen during initial CB training to preserve learned
graph representations, or allowed to fine-tune for task-specific adaptation. Fine-tuning
typically uses a lower learning rate (e.g., 0.0001) to prevent disrupting pretrained features.

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
   c:  # Quadratic bias (antisymmetric)
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

*Why this matters:* Pure best-only cloning can leave the Q-critic with an impoverished view
of the reward distribution, making it hard to distinguish near-success from complete failure.
Stratified sampling exposes the Q-critic to the full reward landscape while still
over-representing higher-quality runs, enabling better-calibrated value estimates during
RL warmup. Positive-reward runs are always retained in full.

**Combined Filtering Strategy**:

Filters can be combined to implement sophisticated data selection policies. The typical
production configuration uses **outlier filtering** + **best-only BC** + **stratified
negative sampling for Q-warmup** (controlled by ``--q-stratified-fraction``): BC clones
only the best-run behavior while the Q-critic sees the full reward distribution.


See Also
--------

* :doc:`file_handling` - Bias coefficient file formats, ``parse_single_ddg`` reference
* :doc:`deepset_pretraining` - Pretrained AtomBondGNN node embeddings
* :doc:`cb_setup` - CB infrastructure and policy architecture
* :doc:`workflow` - Complete CB training workflow
* :doc:`examples` - Running pretraining and workflows
* ``examples/pretrain_with_filtering.sh`` - Pretraining SLURM script
* ``src/mllf/cb/pretrain_policy.py`` - Behavior cloning implementation
* ``src/mllf/cb/policy.py`` - Policy network architecture
* ``src/mllf/cb/workflow_utils.py`` - ``compute_pair_reward``, ``parse_simulation_metrics``
