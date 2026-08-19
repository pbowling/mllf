Uni-Mol Representation
=======================

Overview
--------

Node features for the CB policy come from **Uni-Mol**, a molecular
representation model pretrained on 1.1B molecules from PubChem
(``unimol_tools.UniMolRepr``, the ``unimolv1`` 84M-parameter checkpoint). For each
substituent, ``mllf.cb.unimol_representation`` assembles a complete 3D ligand system
(core scaffold + substituent + optional environment atoms) and runs it through the
frozen, pretrained Uni-Mol model to obtain a 512-dimensional embedding. These embeddings
are the node features consumed by ``UnimolPolicy`` (see :doc:`cb_setup`).

This replaces an earlier, project-specific ``AtomBondGNN``/DeepSet pipeline that trained
a dual-stream GINEConv autoencoder from scratch on AEV features. There is no longer a
separate "pretrain the encoder" step: Uni-Mol is used off-the-shelf and never fine-tuned,
so representation quality comes from the foundation model rather than from an in-repo
training run. The behavior-cloning stage described in :doc:`cb_pretraining` trains only
the downstream ``BiasHeadMLP`` weights on top of these fixed embeddings.

Ligand Construction
--------------------

``construct_full_ligand()`` builds the atom list and coordinates Uni-Mol needs, in a fixed
atom order: **substituent, then core, then reference substituents from other sites, then
environment atoms**. Reference substituents (typically each other site's ``sub1``) provide
full molecular context for multi-site systems; environment atoms (protein and/or solvent)
are optionally included within a radial cutoff of the ligand. The total atom count is
capped at Uni-Mol's 256-atom limit — see `Atom Limit Enforcement`_ below.

``get_unimol_representation()`` then formats the resulting ``{'atoms': [...], 'coordinates':
[...]}`` dict for ``UniMolRepr.get_repr()`` and returns the 512D molecular embedding (the
pretrained model is initialized once per process and cached in a module-level global).

Environment Context
--------------------

Which atoms count as "environment" depends on ``solvent_state``:

* **solv**: water molecules within the cutoff of the ligand
* **protein**: protein atoms (and typically crystallographic waters) within the cutoff
* **gas**/**vacuum**: no environment atoms — only core + substituent(+ ref subs)

Environment PDB files are located via a fallback chain (custom search paths from the
workflow config, then ``protein*.pdb``/``solvent*.pdb`` variants, then ``pre_min.pdb``/
``minimized.pdb``), and periodic systems are handled by applying the minimum-image
convention against the crystal box recorded in the PDB (or the ``prep/`` script) before
distance-filtering, so environment atoms don't spuriously wrap across the box edge.

Environment Consensus
~~~~~~~~~~~~~~~~~~~~~~

Naively selecting "atoms within *cutoff* of each substituent" gives a slightly different
environment atom set per substituent (since each substituent has a different shape), which
injects noise into the ``mean_full`` half of the edge input (see `Dual Embeddings`_) — two
substituents at the same site should share an environment signal, not disagree about which
water molecules count.

``mllf.cb.environment_consensus`` and ``build_environment_consensus()`` (in
``unimol_representation.py``) fix this by defining the environment from the **core
scaffold** instead: since the core is identical for every substituent at a site, atoms
within ``env_cutoff`` (default 8.0 Å) of the core form one consensus set, capped at 256
atoms (keeping the closest), that is reused for every substituent's embedding at that
site. This makes the environment component of the embedding consistent across pretraining,
analysis, and online training, and is enabled by default
(``build_site_consensus()`` / the ``site_consensus`` map in ``examples/run_workflow.py``'s
``build_graph_and_data``). It is skipped for gas-phase systems, which have no environment
atoms to filter.

Atom Limit Enforcement
~~~~~~~~~~~~~~~~~~~~~~~

Uni-Mol supports at most 256 atoms per input. When the ligand + environment atom count
would exceed this, environment atoms are truncated by distance — the closest atoms to the
ligand are kept, farther ones dropped — rather than failing outright.

Dual Embeddings
----------------

``get_substituent_dual_embeddings()`` computes **two** 512D embeddings per substituent
node instead of one:

1. **Ligand-only** (core + substituent, no environment, no reference subs) — captures
   substituent-specific chemistry.
2. **Ligand + environment** (core + substituent + reference subs + environment atoms,
   consensus-filtered) — captures the shared environmental context.

The policy concatenates these into a 1024D node feature ``[ligand_only, full]`` and, per
edge, builds ``[diff_ligand, mean_full]``: the *difference* of the ligand-only halves is
antisymmetric and carries the substituent-vs-substituent signal, while the *mean* of the
full halves is symmetric and carries the (substituent-independent) environmental signal.
See ``UnimolPolicy._build_edge_input`` and :doc:`cb_setup` for how this feeds the policy
heads.

Embedding Caching
-------------------

``save_embedding()`` / ``load_embedding()`` persist a computed embedding to disk keyed by
a hash of the input structure and parameters (``_generate_embedding_cache_key``), so
repeated pretraining epochs over the same runs don't recompute Uni-Mol inference every
pass. Caching is opt-in via the pretraining config's ``unimol.cache_embeddings`` /
``unimol.cache_dir`` (see :doc:`cb_pretraining`); online training in
``examples/run_workflow.py`` recomputes embeddings each epoch (no cache) since the pool of
active combos changes every epoch.

See Also
--------

* :doc:`cb_setup` - How Uni-Mol embeddings feed ``UnimolPolicy``
* :doc:`cb_pretraining` - Behavior cloning on top of frozen Uni-Mol embeddings
* :doc:`workflow` - ``solvent_state`` and environment configuration in the training workflow
* :doc:`api` - API reference for ``mllf.cb.unimol_representation`` and
  ``mllf.cb.environment_consensus``
