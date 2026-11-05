Command Line
============

Installing the package provides one command, ``mllf``, with a subcommand for each step
of the pipeline (``python -m mllf`` is the same command):

.. code-block:: text

   mllf [-h] [--version] [-v | -q] COMMAND [options]

   combos     generate combination directories from site/sub files
   collect    collect finished MSLD runs into pretraining records
   pretrain   behavior-clone the policy on collected runs
   replay     prebuild the regression replay set of a workflow config
   train      online training from a workflow config
   export     export production biases from a trained policy
   cost       simulated ns per system: online run vs ALF

``mllf COMMAND -h`` lists a command's options. For example:

.. code-block:: bash

   mllf collect previous_runs/14benz_solv --output-dir pretraining/14benz_solv
   mllf pretrain --pretraining-dir pretraining/14benz_solv --output-dir models/14benz \
       --config examples/workflow_pretrain.yaml --bayesian-heads
   mllf replay online_1.yaml --cache replay_online_1.pt
   mllf train examples/workflow_14benz.yaml
   mllf export online_1.yaml --label checkpoints_online_1
   mllf cost checkpoints_online_1 --pretraining-root pretraining

Output and verbosity
--------------------

Progress, warnings and errors go to standard output as plain text (through Python's
``logging``, under the ``mllf`` logger). ``-q`` and ``-v`` may come before or after the
command:

* ``-q``: warnings and errors only.
* default: progress messages.
* ``-v``: verbose detail as well; for ``mllf train`` this is the config's
  ``verbose: true``.
* ``-vv``: debug messages, prefixed with the level and the module they come from.

When mllf is used as a library (a notebook, a job script), the same messages are shown
by default; ``mllf.log.setup_logging(-1)`` keeps only warnings, and the ``mllf`` logger
can be given other handlers like any Python logger.

Older entry points
------------------

The module paths used before the ``mllf`` command still work and run the same code:

* ``python -m mllf.cli.train_workflow CONFIG``: ``mllf train CONFIG``
* ``python -m mllf.cb.pretrain_policy``: ``mllf pretrain``
* ``python -m mllf.cli.collect_pretraining_data``: ``mllf collect``
* ``python -m mllf.cb.export_biases``: ``mllf export``
* ``python -m mllf.cb.cost_ledger``: ``mllf cost``
* ``python -m mllf.file_handling.generate_combinations``: ``mllf combos``

The inline Python in ``training/build_regression_replay.sh`` can become
``mllf replay "$CONFIG"`` with ``--cache``, ``--manifest``, ``--consensus`` and
``--keep-online-runs`` in place of the ``REPLAY_*`` variables.
