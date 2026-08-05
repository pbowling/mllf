"""mllf package
"""

__all__ = [
	"__version__",
	"read_bias_coeff",
	"parse_old",
	"parse_new",
]

__version__ = "0.1.0"
__author__ = "Paige E. Bowling"
__email__ = "pbowling@umich.edu"

try:
	from .file_handling.read_bias_coeff import read_bias_coeff, parse_old, parse_new
except Exception:  # pragma: no cover - allow import-time failures when running static checks
	# avoid import errors during packaging or when the submodule isn't available
	read_bias_coeff = None
	parse_old = None
	parse_new = None
