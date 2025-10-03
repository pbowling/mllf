.. image:: mllf_logo.png
  :align: center
  :width: 300px

Machine Learned Landscape Flattening (mllf)
===========================================

Brief introduction
------------------

Machine Learned Landscape Flattening (mllf) provides tools and
documentation for applying adaptive biasing and multisite λ-dynamics to
accelerate alchemical free-energy calculations. The project includes an
MLP-based bias model, example workflows, and utilities for working with
simulation outputs.

What you'll find in these docs
------------------------------

- Background: conceptual and mathematical background for multisite
  λ-dynamics and Adaptive Landscape Flattening (ALF).
- Installation: how to install the package and any optional dependencies.
- PyTorch MLP (PTMLP): API and usage notes for the model used to predict
  bias corrections.
- Examples: runnable examples demonstrating typical workflows.
- References: bibliography for cited literature.
- API: generated API reference for the `mllf` package.

Quick start (recommended)
-------------------------

1. Read the Background section to understand the modeling approach.
2. Follow Installation to set up the environment.
3. Run the Examples to see a complete workflow, then inspect the API
   reference for integration points.

Contents (top-level)
--------------------

.. toctree::
   :maxdepth: 1

   background
   installation
   mlp_model
   examples
   references
   api
