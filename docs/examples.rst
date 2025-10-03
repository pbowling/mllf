Running examples
================

The repository includes small example scripts under ``examples/mlp/`` that
assemble training data and run quick experiments. Typical workflows:

- ``examples/mlp/train_mlp_pt.py`` — train a PyTorch MLP on assembled data and
  evaluate on held-out splits.
- ``examples/mlp/hp_search_pt.py`` — simple grid search for a 4-output PTMLP
  model using k-fold CV.
- ``examples/mlp/hp_search_four_models.py`` — perform per-bias group grid
  searches and save final models.

Example: running a quick grid search
------------------------------------

To run a quick grid search (the scripts can be edited to reduce epochs for a
smoke test)::

  python examples/mlp/hp_search_pt.py

If you need to run a single example and your environment is set up, install
the project in editable mode first::

  pip install -e .

Tips
----

- For quick debugging, reduce the ``epochs`` variable in the example scripts.
- The examples rely on ``mllf.mlp.setup_pairs.assemble_pairs`` to locate and
  load ``examples/mlp/training_files/`` data and variable files.
