.. image:: mllf_logo.png
  :align: center
  :width: 300px

Machine Learned Landscape Flattening (mllf)
===========================================

Brief introduction
------------------

Machine Learned Landscape Flattening (mllf) provides tools and
documentation for applying adaptive biasing and multisite λ-dynamics to
accelerate alchemical free-energy calculations. The project includes a
contextual bandit (CB) framework using graph neural networks to predict
optimal bias coefficients, complete training workflows, and utilities for
working with simulation outputs.

What you'll find in these docs
------------------------------

- **Background**: conceptual and mathematical background for multisite
  λ-dynamics and Adaptive Landscape Flattening (ALF)
- **Installation**: how to install the package and dependencies
- **Workflow System**: complete pipeline from combination generation to training
- **Contextual Bandit Setup**: graph neural network architecture for bias prediction
- **Examples**: runnable training workflows with SLURM integration
- **References**: bibliography for cited literature
- **API**: generated API reference for the ``mllf`` package

Quick start (recommended)
-------------------------

1. Read the **Background** section to understand the modeling approach
2. Follow **Installation** to set up the environment
3. Review **Workflow System** for the complete pipeline overview
4. Check **Examples** for running the training workflow
5. Inspect the **API** reference for integration details

Contents (top-level)
--------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   background
   installation
   workflow
   cb_setup
   examples
   references
   api
