File Handling
=============

``mllf.file_handling`` reads and writes the files of an MSLD system: topology fragments
(RTF), coordinates (PDB), bias coefficients (``variables.py``) and simulation output. How
combinations of these files are generated is described in :doc:`workflow`.

Preparing Fragments
-------------------

The RTF and PDB fragments come from
`msld-py-prep <https://github.com/Vilseck-Lab/msld-py-prep>`_, which aligns the ligands,
finds their maximum common substructure, renormalizes charges and writes the core and
per-site substituent files. Charge renormalization gives every substituent at a site
identical atom types and similar charges for corresponding atoms, with the total equal to
the ligand's charge, so λ can move smoothly between chemically different substituents
(Vilseck et al., *J. Chem. Inf. Model.* 2022, 62, 1479–1488).

Before using the files, check:

* **Charge changes.** msld-py-prep flags renormalization changes above 5%. If one
  appears, move the problematic core atoms into the fragment definitions
  (``MCS_for_MSLD.txt``).
* **Total charge.** Core plus all site charges should equal the ligand's charge.
* **Chiral groups.** MCS can merge chiral atoms. Rename their atom types temporarily
  before the MCS search, then restore them.
* **Assignment.** Inspect the core/fragment split visually (``vis_check.py`` in PyMOL).

Lone pairs (``LP`` atom types) are virtual sites; the RTF parser drops them.

RTF Files
---------

Each substituent has a ``site#_sub#_pres.rtf`` patch. ``parse_rtf_file`` reads the atom
name, CGenFF atom type and partial charge (e) of each ``ATOM`` line:

.. code-block:: text

   PRES SUB1
   ATOM C1   CG2R61  -0.115
   ATOM H1   HGR61    0.115
   ...

PDB Files
---------

Coordinates are used to assemble the core, substituent, reference substituents and
environment for Uni-Mol (:doc:`unimol_representation`). ``parse_pdb_file`` tries RDKit
first, then whitespace-delimited parsing for CHARMM-style files, then fixed PDB columns.

Fragment PDBs (``site#_sub#_frag.pdb``) repeat the anchor atoms that bond them to the core,
so those atoms appear twice when core and fragment are combined.
``find_duplicate_atoms`` / ``remove_duplicate_atoms`` drop the copies.
``find_reference_subs_from_other_sites`` returns the reference substituent
(``site#_sub1``) of each other site within a distance cutoff, which is how a multi-site
molecule is kept whole around the substituent being embedded.

Bias Coefficients
-----------------

Biases are stored in ``variables.py`` as a YAML ``bias_string`` holding the matrices b, c,
x and s, one row and column per substituent, in the order of the λ blocks:

* **b (linear)**: a vector, one value per substituent.
* **c (quadratic)**: symmetric. Only the upper triangle is written, and the simulator uses
  that value for both directions (shared, not negated).
* **x (skew)** and **s (end)**: full matrices; the two directions are independent.

.. code-block:: python

   from mllf.file_handling.read_bias_coeff import read_bias_coeff
   from mllf.file_handling.write_bias_coeff import write_bias_inp_from_graph

   bias = read_bias_coeff('variables.py')       # {'b': [...], 'c': [[...]], 'x': ..., 's': ...}
   write_bias_inp_from_graph(graph, 'combo_dir/variables.py', sub_counts=[3, 2],
                             mapping={'cs': 'quadratic', 'ss': 'end', 'xs': 'skew'})

``read_bias_coeff`` also returns the flat CHARMM scalars (``lams``, ``cs``, ``xs``, ``ss``,
negated relative to the matrix convention) and, when it can find them, the substituents
per site (``_nsubs_per_site``). Older ``.inp`` files are read as well. Each bias position
is mapped to a substituent from what the run executed (:doc:`cb_pretraining`), not from
file order.

Simulation Output
-----------------

``mllf.file_handling.read_output`` reads the CHARMM/MSLD output. Block IDs start at 2
(block 1 holds everything not perturbed: the core and environment), so node ``i`` is
block ``i + 2``.

.. code-block:: python

   from mllf.file_handling.read_output import (
       terminated_normally, parse_single_population, parse_transitions_and_rates,
       parse_single_ddg, parse_pair_profile,
   )

   ok = terminated_normally(text)
   pops = parse_single_population(text)            # {block: {'counts': {0.95: n, 0.99: n}, 'site': s}}
   trans, rates = parse_transitions_and_rates(text) # {site: {λ: count}}, {site: {λ: rate}}
   ddg = parse_single_ddg(text)                     # {(i, j): ΔΔG or None}
   profile = parse_pair_profile(text)               # {(lo, hi): {'barrier', 'skew', 'end', 'transitions'}}

* **Populations, transitions and rates** come from the ``SINGLE POPULATION``,
  ``SINGLE TRANSITIONS`` and ``SINGLE TRANS RATES`` tables. The λ cutoffs are read from
  the table headers (``[0.95, 0.99]`` if none are found).
* **ΔΔG** is the last biased column of ``SINGLE DDG`` (the highest λ cutoff). ``NaN`` (no
  crossing) and ``±Infinity`` (crossings in one direction only) are returned as ``None``,
  since a ΔΔG needs at least one round trip.
* **Pair profiles** come from the per-frame λ trajectory (``parse_lambda_trajectory``).
  Each pair's frames are projected onto :math:`r = \lambda_{hi}/(\lambda_{lo}+\lambda_{hi})`,
  and :math:`G = -kT \ln` (density) gives the midpoint barrier, the skew and end
  asymmetries, and the pair's own transition count. A signal is ``None`` when the pair
  has too few frames or never reached an endpoint. These are the measurements behind the
  per-edge reward and the hindsight targets (:doc:`cb_setup`); ``parse_pair_barriers``
  returns only the barrier.

See Also
--------

* :doc:`workflow` - combination generation and directory layout
* :doc:`cb_setup` - how predictions become ``variables.py``
* `ParamChem <https://cgenff.umaryland.edu/>`_ - CGenFF parameterization
