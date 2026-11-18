Installation
============

``mllf`` needs Python 3.10 or newer, and we recommend a fresh conda or venv environment.
PyTorch is installed first, because the right build depends on the hardware (see
`pytorch.org <https://pytorch.org/get-started/locally/>`_); for example, the CPU-only
build:

.. code-block:: bash

   conda create -n mllf python=3.10      # or: python -m venv .venv && source .venv/bin/activate
   conda activate mllf
   pip install torch --index-url https://download.pytorch.org/whl/cpu

Then install ``mllf`` from a clone of the repository:

.. code-block:: bash

   git clone https://github.com/pbowling/mllf.git
   cd mllf
   pip install -e ".[analysis]"          # the extra adds the figures of mllf analyze

The ``analysis`` extra adds matplotlib, pandas, scipy and pillow for :doc:`analysis`, and
``dev`` adds the test and lint tools. Installing puts the ``mllf`` command on the path
(:doc:`cli`), and ``mllf -h`` lists its subcommands.

Simulations need CHARMM with pyCHARMM and a Slurm cluster, neither of which ``pip``
installs. The workflow loads CHARMM through the ``slurm:`` block's ``module`` and
``setup`` settings (:doc:`workflow`). Each system's simulation script also reads the
CHARMM36 topology and parameter files; the minimal example's script reads them from the
directory in ``CHARMM_TOPPAR`` (:doc:`examples`).

Run the tests with:

.. code-block:: bash

   pip install -e ".[analysis,dev]"
   python -m pytest tests/ -v

To build the documentation locally, see ``docs/README.md``.
