Uni-Mol Representation
======================

The node features of the policy are **Uni-Mol** embeddings (``unimol_tools.UniMolRepr``,
the 84M-parameter ``unimolv1`` model, pretrained on 1.1B PubChem molecules). Uni-Mol is
used as is and never fine-tuned, so pretraining and online training only learn the bias
heads on top of fixed embeddings (:doc:`cb_setup`). A substituent is embedded in its 3D
context rather than as an isolated fragment, because its biases depend on that context:
the same group needs different biases in water, in vacuum and in a binding site.

Building the Input
------------------

``construct_full_ligand()`` assembles the atoms Uni-Mol sees, in a fixed order: the
substituent, the core, the reference substituent (``sub1``) of each other site, then
environment atoms. The reference substituents keep a multi-site molecule whole. The
environment depends on the system's ``solvent_state``:

* ``solv`` / ``solvent``: water within the cutoff
* ``protein``: protein atoms (and usually crystallographic waters) within the cutoff
* ``gas`` / ``vacuum``: none

Environment PDBs are found from the config's search paths, then ``protein*.pdb`` /
``solvent*.pdb``, then ``pre_min.pdb`` / ``minimized.pdb``. For periodic systems,
distances use the minimum-image convention with the box from the PDB (or the ``prep/``
script), so atoms are not lost or duplicated across the box edge. Uni-Mol accepts at
most 256 atoms; beyond that, the farthest environment atoms are dropped.

Environment Consensus
---------------------

Selecting the atoms within the cutoff of *each substituent* gives every substituent a
slightly different environment, because each has a different shape. Two substituents at
the same site would then disagree about which waters count, and that difference would
show up in the shared half of the edge input as noise.

The environment is therefore defined from the **core**, which is identical for every
substituent at a site: atoms within ``env_cutoff`` (default 8.0 Å) of the core form one
set (the closest 256 if there are more), reused for every substituent at that site
(``mllf.cb.environment_consensus``, ``build_environment_consensus``). Pretraining,
analysis and online training use the same consensus. It is skipped in gas phase.

Dual Embeddings
---------------

``get_substituent_dual_embeddings()`` computes two 512D embeddings per substituent:

1. **ligand-only** (core + substituent): the substituent's own chemistry
2. **full** (core + substituent + reference substituents + consensus environment): its
   context

The node feature is ``[ligand_only, full]`` (1024D). The policy uses the *difference* of
the ligand-only halves and the *mean* of the full halves for each edge (:doc:`cb_setup`),
so the two embeddings carry different information rather than repeating each other.

Caching
-------

Embeddings are cached on disk, keyed by a hash of the input structure and settings
(``save_embedding`` / ``load_embedding``), when ``unimol.cache_embeddings: true``
(``unimol.cache_dir``; by default ``<output.base_dir>/embeddings_cache`` in online
training). The same substituent in the same environment is then embedded once across
epochs and runs.
