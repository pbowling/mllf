AtomBondGNN Pretraining
=======================

Overview
--------

AtomBondGNN pretraining provides learned physical representations of substituents that replace
manual feature engineering. Instead of hand-crafted features like atom counts and charges,
we use a pretrained graph neural network to compress atom-level physics (spatial arrangements,
charges, chemical composition, and bond topology) into compact 64-dimensional embeddings.

These embeddings are then used as input features for the ``SitePoolMLPPolicy`` in the
contextual bandit training (see :doc:`cb_setup`). The ``AtomBondGNN`` improves on earlier
DeepSet-style pooling by using a **dual-stream GINEConv architecture**: a substituent
stream propagates information along sub-graph bonds, and a parallel core stream processes
the shared scaffold, whose mean-pooled summary modulates the attention gate. This provides
scaffold-aware pooling while keeping the embedding a pure substituent representation.


4-Step Pretraining Pipeline
----------------------------

Step 1: Atom-Level Physical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each atom in a substituent PDB file, we compute:

* **ANI-2x AEV** (Atomic Environment Vector): 2288-dimensional vector encoding
  radial and angular spatial symmetry functions
* **Partial charge** from RTF file: 1-dimensional scalar
* **Atom-type one-hot**: 11-dimensional vector identifying element species (H, C, N, O, F, S, Cl, Br, I, P, X)
* **Concatenation**: ``[AEV (2288D), charge (1D), atom_id (11D)] → 2300D atom features``

**What are AEVs?**

Atomic Environment Vectors (AEVs) are rotationally and translationally invariant
representations of an atom's local chemical environment. They encode information about:

* **Radial symmetry functions**: Capture distances to neighboring atoms of each element type
* **Angular symmetry functions**: Capture angles formed by triplets of atoms (atom-center-atom)

AEVs are derived from the ANI neural network potential (`TorchANI <https://aiqm.github.io/torchani/>`_)
and provide a physics-informed, geometry-aware representation of atomic environments.

**ANI-2x Parameters Used**:

* **Radial cutoff (Rcr)**: 5.2 Å - maximum distance for pairwise interactions
* **Angular cutoff (Rca)**: 3.5 Å - maximum distance for angular triplet interactions  
* **Number of species**: 11 elements (H, C, N, O, F, S, Cl, Br, I, P, X)
* **AEV dimension**: 2288D total

  - Radial features: 16 radial basis functions × 11 element types = 176D
  - Angular features: 8 angular basis functions × 11 × (11+1)/2 pairs = 528D
  - Total per subAEV × 4 subAEVs = 2288D

The high dimensionality captures rich geometric and chemical information about each atom's
local environment, which is then compressed by the autoencoder into 64D embeddings.

**Spatial Cutoffs**:

The AEV computation is **context-aware**: atoms see neighboring atoms from:

- The shared core of the ligand (bonded neighbors within cutoff)
- Nearby substituents from other sites (multi-site spatial filtering within 5.1 Å)
- Environment atoms within 5.1 Å, sourced from ``minimized.pdb`` when available:

  * **Protein systems**: Post-minimization protein atoms within 5.1 Å of the substituent
  * **Solvent systems**: Post-minimization water molecules within 5.1 Å of the substituent
  * **Vacuum systems**: No additional environment atoms (core + other-site subs only)

Using **energy-minimized coordinates** is preferred over pre-simulation PDB files because
minimization resolves steric clashes and produces geometries representative of the sampled
ensemble, leading to more accurate AEV descriptors.


**Why AEVs for Chemistry?**

1. **Invariance**: Rotationally and translationally invariant (no dependence on molecular orientation)
2. **Locality**: Each atom's AEV depends only on its local environment (within cutoff)
3. **Differentiable**: Smooth functions of atomic positions, enabling gradient-based learning
4. **Physics-informed**: Derived from neural network potentials trained on quantum chemistry data
5. **Transferable**: Learned from diverse molecules, generalizes to new chemical structures

For more details on AEV computation, see the 
`TorchANI AEV documentation <https://aiqm.github.io/torchani/api_autogen/torchani.aev.html>`_.

Step 2-3: AtomBondGNN Autoencoder Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train an ``AtomBondGNNAutoencoder`` to learn bond-topology-aware atom representations:

**Encoder Architecture** (dual-stream):

.. code-block:: text

   Input: [AEV (2288D), charge (1D), atom_id (11D)] = 2300D per atom
   
   Substituent stream (sub atoms only):
       sub_input_proj: Linear(2300→256) + ReLU, Linear(256→256) + ReLU
       sub_gin_layers: 4× GINEConv(256→256, edge_dim=1) + ReLU
   
   Core stream (core + ref-sub atoms):
       core_input_proj: Linear(2300→256) + ReLU, Linear(256→256) + ReLU
       core_gin_layers: 4× GINEConv(256→256, edge_dim=1) + ReLU
       core_summary = mean-pool(core_gin output)
   
   Scaffold-aware attention pooling (over sub atoms only):
       gate_score = sigmoid(Linear(concat(sub_h, core_summary), 1))
       pool_nn(sub_h) weighted sum → 64D substituent embedding
   Output: 64D substituent embedding

The **GINEConv layers** pass messages along molecular bonds extracted from RDKit
bond topology (RTF BOND section as fallback), using edge features encoding bond type.
Each atom "sees" its bonded neighbors before pooling, capturing functional group
identity and local bonded context that independent per-atom processing cannot represent.

The **core stream** produces a single mean-pooled scaffold summary that is concatenated
into the gate of the attentional pooling step. This makes the attention weights
scaffold-aware (the same sub atom should be weighted differently in different scaffolds)
without contaminating the sub feature path — the final pooled embedding remains a
pure substituent representation.

**Decoder Architecture**:

The decoder is a lightweight per-atom linear layer applied to GINConv hidden states
**before pooling**:

.. code-block:: text

   GINConv hidden states [N, 256] → Linear(256 → 2300) → reconstructed atom features [N, 2300]

This reconstruction target forces the GINConv layers to maintain atom-level information
in their hidden states, even though the encoder ultimately produces a single pooled vector.

**Loss Function**: Combined reconstruction (MSE) + supervised contrastive (NT-Xent) loss:

.. math::

   \mathcal{L} = \mathcal{L}_{\text{MSE}} + \alpha \cdot \mathcal{L}_{\text{NT-Xent}}

where:

.. math::

   \mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2

and :math:`\mathcal{L}_{\text{NT-Xent}}` is the **supervised NT-Xent contrastive loss**
(Khosla et al., 2020) applied at the substituent embedding level.  Two substituents are
treated as **positive pairs** when their sets of distinct CGenFF atom types are identical
(i.e., they contain chemically equivalent functional groups).  This pulls chemically
equivalent substituents together in embedding space and pushes chemically distinct
substituents apart, regardless of scaffold:

.. math::

   \mathcal{L}_{\text{NT-Xent}} = -\frac{1}{|\mathcal{P}|}\sum_{i \in \mathcal{P}}
   \frac{1}{|P_i|} \sum_{j \in P_i}
   \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}
             {\sum_{k \ne i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}

where :math:`\mathbf{z}_i` is the L2-normalised 64D substituent embedding, :math:`\tau` is
the temperature (default: 0.07), :math:`P_i` is the set of positives for anchor :math:`i`,
and :math:`\mathcal{P}` is the set of anchors with at least one positive in the batch.

The default weighting is :math:`\alpha = 0.5`.

**Checkpoint Saving**:

After training, ``AtomBondGNNAutoencoder.save_encoder()`` saves only the encoder layers
(excluding the decoder) in a format compatible with ``AtomBondGNN.load_state_dict()``.
This checkpoint can then be loaded by ``load_pretrained_atombondgnn()`` for inference.


Step 4: AtomBondGNN Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The trained encoder produces a single 64-dimensional substituent embedding per molecule
via **scaffold-aware attentional pooling**. The gate score combines each substituent
atom’s hidden state with the core-stream summary vector, making the importance weight
scaffold-dependent:

.. code-block:: text

   gate_score(atom) = sigmoid(Linear(concat(sub_h, core_summary), 1))  # scaffold-aware
   embedding        = sum(gate_score × Linear(sub_h, 64)) / sum(gate_score)

This provides permutation invariance and handles variable-size substituents while
preferentially weighting atoms that carry the most predictive information.

The ``SitePoolMLPPolicy`` uses the 64D AtomBondGNN embeddings directly as node features
(``data.x``), without a subsequent graph convolution step. A site-level mean-pool of
these embeddings provides the system-context signal (see :doc:`cb_setup`).

Using Pretrained Models
-----------------------

Once trained, the AtomBondGNN encoder integrates into the CB workflow:

.. code-block:: python

   from mllf.cb import graph_utils
   from mllf.cb.deepset_autoencoder import load_pretrained_atombondgnn
   
   # Load pretrained AtomBondGNN encoder
   deepset = load_pretrained_atombondgnn('models/best_encoder.pt', freeze_weights=True)
   
   # Convert graph to PyG format with AtomBondGNN embeddings
   data, extras = graph_utils.build_pyg_graph_from_mllf_graph(
       graph,
       deepset_model=deepset,
       pdb_dir=prep_dir,
       rtf_results=rtf_data,
       prep_dir=prep_dir,  # For multi-site spatial filtering
       protein_pdb=protein_pdb,  # For protein systems
       solvent_state='protein',
       aev_cutoff=5.1
   )
   
   # data.x now contains [num_nodes, 64] AtomBondGNN embeddings

See Also
--------

* :doc:`file_handling` - PDB and RTF file parsing
* :doc:`cb_setup` - CB infrastructure and SitePoolMLPPolicy architecture
* :doc:`cb_pretraining` - Behavior cloning from expert coefficients
* :doc:`workflow` - Complete workflow from combo generation to training
* ``src/mllf/cb/deepset.py`` - ``AtomBondGNN`` class definition
* ``src/mllf/cb/deepset_autoencoder.py`` - ``AtomBondGNNAutoencoder``, ``load_pretrained_atombondgnn``
* ``src/mllf/cb/deepset_pretraining_dataset.py`` - Dataset generation (bond topology, distinct_atom_types)
* ``src/mllf/cb/aev_processor.py`` - AEV computation and sub_mask generation
  (``detect_minimized_pdb``, ``extract_environment_atoms_from_minimized``)
* ``src/mllf/cb/graph_utils.py`` - PyG graph construction with sub_mask and site_index
* ``examples/run_deepset_pretraining.py`` - Example pretraining workflow (contrastive + reconstruction)
