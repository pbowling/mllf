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
   scout      scouting runs before a stage, and their summary
   train      online training from a workflow config
   export     export production biases from a trained policy
   cost       simulated ns per system: online run vs ALF
   analyze    figures and a report: pretrained model or online run
   archive    archive a run's combo directories, one .tar.gz per system

``mllf COMMAND -h`` lists a command's options. For example:

.. code-block:: bash

   mllf collect previous_runs/14benz_solv --output-dir pretraining/14benz_solv
   mllf pretrain --pretraining-dir pretraining/14benz_solv --output-dir models/14benz \
       --config examples/workflow_pretrain.yaml --bayesian-heads
   mllf replay online_1.yaml --cache replay_online_1.pt
   mllf scout online_1.yaml --out scouting/online_1 --wait
   mllf train examples/minimal/workflow.yaml
   mllf export online_1.yaml --label checkpoints_online_1
   mllf cost checkpoints_online_1 --pretraining-root pretraining
   mllf analyze online online_1.yaml
   mllf analyze pretrain models/14benz
   mllf archive online_1.yaml --remove

Output and verbosity
--------------------

Each run writes two streams. The console (standard output, and so a SLURM job's output
file) shows progress at the level of phases and epochs: setup totals, the per-epoch
summary that also goes to ``epoch_summary.log``, stage changes, warnings and errors. A log
file in the run's output directory records every message with a time stamp, including the
per-combo, per-run and per-embedding detail (cache hits, job submissions, retries,
coverage lines):

* ``mllf train``: ``training.log`` in ``output.base_dir`` (appended to on resume)
* ``mllf pretrain``: ``pretrain.log`` in ``--output-dir``
* ``mllf collect``: ``collect.log`` in ``--output-dir``
* ``mllf replay``: the replay cache path with the suffix ``.log``

``-q`` and ``-v`` set the console level and may come before or after
the command:

* ``-q``: warnings and errors only.
* default: progress messages.
* ``-v``: verbose detail as well; for ``mllf train`` this is the config's
  ``verbose: true``.
* ``-vv``: every message, prefixed with its level and module.

The live job status line (``[running=... | queued=...]``) and Uni-Mol's progress bars
appear only in an interactive terminal. When mllf is used as a library (a notebook, a job
script), the console messages are shown by default; ``mllf.log.setup_logging(-1)`` keeps
only warnings, ``mllf.log.log_to_file(path)`` records everything in a file, and the
``mllf`` logger can be given other handlers like any Python logger.

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
