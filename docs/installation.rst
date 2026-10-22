Installation
============

Install the package in editable mode, with the documentation requirements, into a fresh
environment:

.. code-block:: bash

   conda create -n mllf python=3.10      # or: python -m venv .venv && source .venv/bin/activate
   conda activate mllf
   pip install -e .
   pip install -r docs/requirements.txt

Simulations need CHARMM with pyCHARMM on the cluster; the workflow loads it through the
``slurm.module`` setting (:doc:`workflow`).

Run the tests with:

.. code-block:: bash

   python -m pytest tests/ -v

To build the documentation locally, see ``docs/README.md``.
