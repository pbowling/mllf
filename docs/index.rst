.. image:: mllf_logo.png
  :align: center
  :width: 300px

Machine Learned Landscape Flattening (mllf)
===========================================

Multisite λ-dynamics samples many alchemical perturbations in one simulation, but only
after each system has bias coefficients that flatten its alchemical landscape. Finding
those biases iteratively costs many simulations per system, and nothing learned on one
system carries over to the next.

``mllf`` predicts the biases instead. A shared model, conditioned on Uni-Mol embeddings
of each substituent in its environment, proposes biases for systems it has never
simulated. Each simulation then corrects both the model and the pair of substituents it
sampled. The aim is converged biases in fewer simulated nanoseconds than iterative
flattening, including for systems absent from the training data.

Where to start
--------------

* New to MSLD biases: :doc:`background`.
* Running something: :doc:`installation`, then :doc:`examples`.
* Configuring online training: :doc:`workflow`.
* How the model learns: :doc:`cb_setup`, :doc:`cb_pretraining` and
  :doc:`unimol_representation`.
* Input and output formats: :doc:`file_handling`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   background
   installation
   examples
   workflow
   cb_setup
   cb_pretraining
   unimol_representation
   file_handling
   references
   api
