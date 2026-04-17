AtomBondGNN Pretraining
=======================

Overview
--------

AtomBondGNN pretraining provides learned physical representations of substituents that replace
manual feature engineering. Instead of hand-crafted features like atom counts and charges,
we use a pretrained graph neural network to compress atom-level physics (spatial arrangements,
charges, chemical composition, and bond topology) into compact 64-dimensional embeddings.

These embeddings are then used as input features for the RGCN policy network in the
contextual bandit training (see :doc:`cb_setup`). The ``AtomBondGNN`` improves on earlier
DeepSet-style pooling by propagating information along molecular bonds before aggregation,
capturing bonded chemical context that independent per-atom processing cannot represent.


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

**Encoder Architecture**:

.. code-block:: text

   Input: [AEV (2288D), charge (1D), atom_id (11D)] = 2300D per atom
       Linear(2300 → 256) + ReLU                    [input projection]
       GINConv layer 1 (bond-graph topology) + ReLU  [bond propagation]
       GINConv layer 2 (bond-graph topology) + ReLU  [bond propagation]
       GlobalAttentionPool(gate: 256→1, nn: 256→64)  [substituent pooling]
   Output: 64D substituent embedding

The **GINConv layers** pass messages along molecular bonds extracted from RDKit
bond topology (RTF BOND section as fallback). This lets each atom "see" its bonded
neighbors before pooling — capturing functional group identity and local bonded context
that independent per-atom processing cannot represent.

The **GlobalAttentionPool** learns to weight atoms by their importance to the substituent
prediction task, rather than applying equal weight to each atom as in max- or sum-pooling.

**Decoder Architecture**:

The decoder is a lightweight per-atom linear layer applied to GINConv hidden states
**before pooling**:

.. code-block:: text

   GINConv hidden states [N, 256] → Linear(256 → 2300) → reconstructed atom features [N, 2300]

This reconstruction target forces the GINConv layers to maintain atom-level information
in their hidden states, even though the encoder ultimately produces a single pooled vector.

**Loss Function**: Mean Squared Error (MSE) between input and reconstructed atom features

.. math::

   \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2

where :math:`\mathbf{x}_i` is the 2300D input feature for atom :math:`i` and
:math:`\hat{\mathbf{x}}_i` is the per-atom reconstruction from the GINConv hidden state.

**Checkpoint Saving**:

After training, ``AtomBondGNNAutoencoder.save_encoder()`` saves only the encoder layers
(excluding the decoder) in a format compatible with ``AtomBondGNN.load_state_dict()``.
This checkpoint can then be loaded by ``load_pretrained_atombondgnn()`` for inference.


Step 4: AtomBondGNN Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The trained encoder produces a single 64-dimensional substituent embedding per molecule
via **GlobalAttentionPool** — an attention mechanism that learns to weight atoms
by their relevance to the prediction task:

.. code-block:: text

   gate_score(atom) = sigmoid(Linear(hidden, 1))   # learned importance per atom
   embedding        = sum(gate_score × Linear(hidden, 64)) / sum(gate_score)

This provides permutation invariance and handles variable-size substituents while
preferentially weighting atoms that carry the most predictive information.

The RGCN encoder applies **LayerNorm** to the 64D AtomBondGNN embeddings before the first
graph convolution layer. This normalises the input distribution across features and
prevents gradient instability caused by embedding magnitudes varying with substituent size.
LayerNorm's learnable γ/β parameters preserve size-related information while removing
mean-shift and scale differences that would otherwise destabilise RGCN training.

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
* :doc:`cb_setup` - CB infrastructure and RGCN architecture
* :doc:`cb_pretraining` - Behavior cloning from expert coefficients
* :doc:`workflow` - Complete workflow from combo generation to training
* ``src/mllf/cb/deepset.py`` - ``AtomBondGNN`` class definition
* ``src/mllf/cb/deepset_autoencoder.py`` - ``AtomBondGNNAutoencoder``, ``load_pretrained_atombondgnn``
* ``src/mllf/cb/deepset_pretraining_dataset.py`` - Dataset generation (bond topology extraction, training pipeline)
* ``src/mllf/cb/train_deepset_autoencoder.py`` - Training script
* ``src/mllf/cb/aev_processor.py`` - AEV computation, minimized.pdb extraction
  (``detect_minimized_pdb``, ``extract_environment_atoms_from_minimized``)
* ``src/mllf/cb/graph_utils.py`` - RGCN training AEV pipeline (shares extraction logic)
* ``examples/run_deepset_pretraining.py`` - Example pretraining workflow
