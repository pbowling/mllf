"""mllf: machine-learned bias coefficients for multisite lambda dynamics.

Subpackages:

* ``mllf.file_handling`` -- reading and writing simulation inputs/outputs
  (bias files, PDB/RTF fragments, combination directories, manifests).
* ``mllf.cb`` -- the contextual-bandit policy, its pretraining and the
  online-training epoch.
* ``mllf.cli`` -- command-line entry points.

Import from the submodules directly, e.g.
``from mllf.file_handling.read_bias_coeff import read_bias_coeff``.
"""

__version__ = "0.1.0"
__author__ = "Paige E. Bowling"
__email__ = "pbowling@umich.edu"
