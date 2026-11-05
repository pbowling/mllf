Uni-Mol Representation
======================

The node features of the policy are **Uni-Mol** embeddings (``unimol_tools.UniMolRepr``,
the 84M-parameter ``unimolv1`` model pretrained on 1.1B PubChem molecules). Uni-Mol is
used as is and never fine-tuned, so pretraining and online training learn only the bias
heads on top of fixed embeddings (:doc:`cb_setup`). Each substituent is embedded in its 3D
context rather than as an isolated fragment, because the same group needs different
biases in different molecules.

Node Features
-------------

``get_substituent_dual_embeddings()`` returns two 512-D halves per substituent, and the
node feature is their concatenation (1024-D).

1. **Ligand**: the core and substituent, in that order after the substituent itself
   (``construct_full_ligand()``). With ``unimol.ligand_other_sites: true``, it also holds
   every other site's reference substituent (``sub1``), so a substituent is embedded in
   the molecule it belongs to.
2. **Second half**, set by ``unimol.env_input``:

   * ``phase`` (the default) and ``phase_ligmean``: two flags, [condensed, protein]
     (vacuum [0, 0], solvent [1, 0], protein [1, 1]), read from the system name and
     followed by zeros.
   * ``dual``: a second Uni-Mol embedding of the ligand together with its environment
     (below).

The policy combines the halves per edge (:doc:`cb_setup`), using the *difference* of the
ligand halves and the symmetric second half, so the two halves carry different
information. With ``unimol.site_size_input``, the slot after the phase flags holds each
substituent's site size for the current run. That value is set per run and is never
cached, since the same substituent appears in pairs and in groups.

The ``unimol`` settings change what the model sees, so online training and pretraining
must use the same ones (:doc:`cb_pretraining`). The defaults are the settings of the
current pretrained model (``pretraining_Nov02b``), so a config that leaves them out matches
it:

.. code-block:: yaml

   unimol:
     env_input: phase             # phase flags in the second half
     site_size_input: true        # edge input: 512 + 2 + 1 = 515-D
     ligand_other_sites: true     # other sites' reference substituents in the ligand
     ligand_from_minimized: false
     structure_selection: unrelaxed

A model pretrained with other values needs them written out in both its pretraining and
its online config. A different ``env_input`` or ``site_size_input`` changes the width of
the edge input, so loading the model fails with a shape mismatch; a different
``ligand_other_sites`` keeps the width and goes unnoticed, so its value is recorded in
``pretrain_metadata.json`` and should be checked against the online config.

Environment Embedding (``env_input: dual``)
-------------------------------------------

The environment embedding adds the atoms around the ligand to the second half. Which atoms
depends on the system's ``solvent_state``.

* ``solv`` / ``solvent``: water within the cutoff
* ``protein``: protein atoms (and usually crystallographic waters) within the cutoff
* ``gas`` / ``vacuum``: none

Environment PDBs are found from the config's search paths, then ``protein*.pdb`` and
``solvent*.pdb``, then ``pre_min.pdb`` and ``minimized.pdb``. A solvent leg takes its
environment from the solvent even when its ``prep/`` also holds a protein file. For
periodic systems, distances use the minimum-image convention with the box from the PDB (or
the ``prep/`` script), so atoms are neither lost nor duplicated across the box edge.
Uni-Mol accepts at most 256 atoms, and beyond that the farthest environment atoms are
dropped.

Selecting the atoms within the cutoff of *each substituent* would give every substituent a
slightly different environment, because each has a different shape. Two substituents at
the same site would then disagree about which waters count, and that difference would
appear as noise in the shared half of the edge input. The environment is therefore defined
from the **core**, which is identical for every substituent at a site: atoms within
``env_cutoff`` (8.0 Å) of the core form one set per site (the closest 256 if there are
more), reused for every substituent at that site (``build_environment_consensus``,
``mllf.cb.environment_consensus``). Pretraining writes these sets to
``environment_consensus.json`` so that analysis and replay use the same atoms. Runs with a
phase input neither build nor write them.

Caching
-------

With ``unimol.cache_embeddings: true``, embeddings are cached on disk, keyed by a hash of
the input structure and settings (``save_embedding`` / ``load_embedding``), in
``unimol.cache_dir`` (by default ``<output.base_dir>/embeddings_cache`` in online
training). The same substituent is then embedded once across epochs and runs. The ligand
half does not depend on the environment settings, so a ligand embedding cached by a run
with ``env_input: dual`` is reused by a run with a phase input, and the reverse.
