"""mllf: machine-learned bias coefficients for multisite lambda dynamics.

Subpackages:

* ``mllf.file_handling`` -- reading and writing simulation inputs/outputs
  (bias files, PDB/RTF fragments, combination directories, manifests).
* ``mllf.cb`` -- the contextual-bandit policy and online training.
* ``mllf.pretrain`` -- behavior-cloning pretraining on collected runs.
* ``mllf.cli`` -- the ``mllf`` command and its subcommands.
* ``mllf.log`` -- logging (messages go to stdout; ``mllf -q`` / ``-v``).

Import from the submodules directly, e.g.
``from mllf.file_handling.read_bias_coeff import read_bias_coeff``.
"""

from mllf.log import install_default_handler

__version__ = "0.1.0"
__author__ = "Paige E. Bowling"
__email__ = "pbowling@umich.edu"

install_default_handler()
