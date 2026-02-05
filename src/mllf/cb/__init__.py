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

# Always export graph utilities, atom vocabulary, and training helpers
from .graph_utils import build_pyg_graph_from_mllf_graph
from .atom_vocab import get_atom_type_vocab
from .policy import EdgePolicy
from .train_improved import reinforce_train_step, compute_reward_from_raw_metrics

# Try to import RGCNEncoder and pretraining modules lazily
try:
	_rgcn = import_module('.rgcn', __package__)
	RGCNEncoder = getattr(_rgcn, 'RGCNEncoder')
	_load_pretrained = import_module('.load_pretrained', __package__)
	load_pretrained_encoder = getattr(_load_pretrained, 'load_pretrained_encoder')
	freeze_encoder = getattr(_load_pretrained, 'freeze_encoder')
	unfreeze_encoder = getattr(_load_pretrained, 'unfreeze_encoder')
except Exception:
	RGCNEncoder = None
	load_pretrained_encoder = None
	freeze_encoder = None
	unfreeze_encoder = None

__all__ = [
	"RGCNEncoder", 
	"EdgePolicy", 
	"build_pyg_graph_from_mllf_graph", 
	"reinforce_train_step",
	"compute_reward_from_raw_metrics",
	"load_pretrained_encoder",
	"freeze_encoder",
	"unfreeze_encoder",
]
