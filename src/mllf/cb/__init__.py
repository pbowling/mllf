"""Contextual bandit package for MLLD: RGCN encoder + edge-value policy.

This package provides models and helpers to convert our existing Graph
representation into a PyTorch Geometric graph and a small REINFORCE-style
policy that predicts continuous bias coefficients per-edge.
"""

"""Contextual bandit package for MLLD: models and helpers.

Importing the full `rgcn` module requires `torch_geometric` to be
installed; avoid raising an ImportError at package import time so that
non-GNN code (for example, graph utilities and tests) can import
`mllf.cb.graph` without needing PyG installed.
"""
from importlib import import_module

# Always export graph utilities and training helpers
from .graph_utils import build_pyg_graph_from_mllf_graph
from .policy import EdgePolicy
from .train import reinforce_train_step

# Try to import RGCNEncoder lazily; if PyG is not available, expose None
try:
	_rgcn = import_module('.rgcn', __package__)
	RGCNEncoder = getattr(_rgcn, 'RGCNEncoder')
except Exception:
	RGCNEncoder = None

__all__ = ["RGCNEncoder", "EdgePolicy", "build_pyg_graph_from_mllf_graph", "reinforce_train_step"]
