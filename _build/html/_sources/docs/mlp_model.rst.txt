PyTorch MLP (PTMLP)
====================

This project contains a small PyTorch MLP used as an example model for predicting
per-pair bias coefficients. The implementation lives in `mllf.mlp.pt_model`.

Key points
----------

- The `PTMLP` class builds a fully-connected feed-forward network with configurable
  hidden layers and an output dimension matching the number of targets (e.g., 4 for
  lams, cs, ss, xs).
- Training utilities `train_one_epoch` and `evaluate` are provided to simplify
  training loops and evaluation in the examples.

Example: quick training loop
=============================

A minimal training loop (conceptual) looks like this:

.. code-block:: python

   from mllf.mlp.pt_model import PTMLP, train_one_epoch, evaluate
   import torch
   import torch.nn as nn

   model = PTMLP(in_dim, hidden=(128,64), out_dim=4)
   opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
   loss_fn = nn.MSELoss()

   for ep in range(50):
     train_loss = train_one_epoch(model, opt, loss_fn, X_train, y_train, batch_size=16, device='cpu')
     val_loss, _ = evaluate(model, loss_fn, X_val, y_val, device='cpu')
     print(ep, train_loss, val_loss)

See `examples/mlp/train_mlp_pt.py` and `examples/mlp/hp_search_pt.py` for full working
examples and hyperparameter search scripts.

Pair generation and bias setup
------------------------------

The training examples rely on a small utility that discovers per-run fragment
files and the associated variables (bias coefficient) files and assembles a
per-fragment mapping of ordered pairs with scalar bias coefficients. The
function used by the examples is ``mllf.mlp.setup_pairs.assemble_pairs``.

What ``assemble_pairs`` does

- Walk a root training directory and treat each subdirectory as a run. Run
  directories are expected to be named like ``system_solvent_fnex`` (for
  example ``14benz_vac_5.5``) though the parser tolerates fewer parts.
- For each run it parses RTF fragment files (via ``mllf.file_handling.read_rtf``)
  to discover fragment metadata (site index, substituent index, atom types, and total
  charge).
- It finds the matching variables file in the run directory (``variables*.py``
  or ``variables*.inp``) and parses bias coefficient groups using
  ``mllf.file_handling.read_bias_coeff``.
- Per-fragment entries include the raw per-substituent vectors (for example, a
  ``lams_vector`` of linear bias coefficients for each substituent) and computed ordered
  pair mappings (for example ``pair_1_2`` representing the scalar to use when
  the ordered pair is (1,2)).

Bias groups and precedence

- linear / fixed bias (``lams``): called the linear or fixed bias in the
  code, ``lams`` provides a per-sub linear term. When a per-sub ``lams``
  vector is available the assembler computes pairwise differences
  ``lams[b] - lams[a]`` for ordered pairs and stores them under
  ``biases['pairwise_lams']``; these values are preferred for the ``lams``
  component of the final per-pair mapping. With correct linear bias
  settings, perturbations at each site are equally populated.

- quadratic bias (``cs``): the ``cs`` group contains quadratic bias
  coefficients. The quadratic term is largely responsible for removing
  barriers in alchemical space that arise from electrostatic interactions.

- skew bias (``xs``): after the introduction of soft-core potentials the
  barriers in alchemical space became less symmetric. The ``xs`` (skew)
  bias captures residual asymmetry not modeled by the quadratic term; it
  is used to fit residuals beyond the quadratic and end biases.

- end bias (``ss``): the ``ss`` group is an end bias coefficient. The end
  bias accounts for the entropic and surface-tension cost of displacing
  solvent and nearby molecules to make space for a substituent to appear.

The assembler treats the ``cs``, ``xs``, and ``ss`` groups similarly: each
group may contain explicit per-pair keys (for example ``cs1s1s1s2``) or
per-substituent scalars (e.g., ``cs1`` entries that implicitly apply to a substituent). The
assembler first searches for explicit per-pair keys; if none are present it
derives per-substituent scalars (by averaging matching entries) and computes
pairwise differences. Finally, the assembler merges available group maps
into a unified per-pair mapping under ``biases['pairs']`` so each ordered
pair includes numeric values for whichever groups are available (``lams``,
``cs``, ``xs``, ``ss``). Only pairs involving the fragment's sub-index are
included for that fragment.

How to view the assembled pairs

Call::

  from mllf.mlp.setup_pairs import assemble_pairs

  out = assemble_pairs("/path/to/examples/mlp/training_files")

`out` is a dictionary keyed by run directory name. Each run maps fragment
identifiers (like ``site1_sub2``) to a dictionary containing a ``biases``
sub-dictionary. The ``biases['pairs']`` mapping is the easiest place to find
the final ordered-pair scalars used by the examples and model training.

Example (CLI smoke test)

The module's main guard includes a small CLI that runs against the example
training files and prints a readable per-fragment listing of `pairs` if you
want a quick check.
