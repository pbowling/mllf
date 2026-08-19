mllf API
=========

File Handling
-------------

For detailed documentation on file formats and usage examples, see :doc:`file_handling`.

.. autosummary::
   :toctree: api

   mllf.file_handling.read_bias_coeff
   mllf.file_handling.read_rtf
   mllf.file_handling.read_pdb
   mllf.file_handling.write_bias_coeff
   mllf.file_handling.read_output
   mllf.file_handling.generate_combinations

Uni-Mol Representation
-----------------------

For detailed documentation on Uni-Mol embeddings and environment context, see
:doc:`unimol_representation`.

.. autosummary::
   :toctree: api

   mllf.cb.unimol_representation
   mllf.cb.environment_consensus
   mllf.cb.atom_vocab

Graph and CB Modules
--------------------

For detailed documentation on CB architecture, see :doc:`cb_setup`.

.. autosummary::
   :toctree: api

   mllf.cb.graph
   mllf.cb.graph_utils
   mllf.cb.policy
   mllf.cb.bayesian_head
   mllf.cb.train_improved

Pretraining (Behavior Cloning)
-------------------------------

For detailed documentation on behavior-cloning pretraining, see :doc:`cb_pretraining`.

.. autosummary::
   :toctree: api

   mllf.cb.pretrain_policy
   mllf.cli.collect_pretraining_data

Workflow Utilities
------------------

For detailed documentation on the workflow system, see :doc:`workflow`.

.. autosummary::
   :toctree: api

   mllf.cli.workflow
   mllf.cli.sim
   mllf.cb.workflow_utils
