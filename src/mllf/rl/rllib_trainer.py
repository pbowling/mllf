"""RLlib trainer integration for GraphEnv and GNNPolicy.

This module registers a custom RLlib torch model that wraps the project's
`GNNPolicy` and provides a small `train` helper to start RLlib training using
the existing `GraphEnv` (supports observation_format='graph').

Note: Ray/RLlib is an optional dependency. Importing this module will attempt
to import `ray` and will raise if it's not available. Install with:

    pip install "ray[rllib]>=2.4.0,<3.0.0"

The trainer uses a simple config suitable for experiments; adapt as needed.
"""
from typing import Dict, Any, Optional

try:
    import ray
    from ray import tune
    from ray.rllib.models.modelv2 import ModelV2
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.models import ModelCatalog
    from ray.rllib.algorithms.ppo import PPO, PPOConfig
    from ray.rllib.utils.typing import ModelConfigDict
except Exception as e:  # pragma: no cover - optional dependency
    raise

import torch
import torch.nn as nn

from mllf.rl.gnn_policy import GNNPolicy
from mllf.rl.env import GraphEnv
from mllf.rl.graph_space import GraphInstance


class RLlibGNNModel(TorchModelV2, nn.Module):
    """Wrap GNNPolicy as an RLlib TorchModelV2.

    Expects the environment to provide observations as a dict with keys:
      - "nodes": shape (num_nodes, node_feat)
      - "edges": shape (num_edges, edge_feat) or None
      - "edge_links": shape (num_edges, 2) indices

    RLlib will pass observations as tensors; this model converts them to numpy
    arrays and calls the GNNPolicy forward. This is a lightweight bridge and may
    be optimized by making GNNPolicy fully torch-native.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # infer node feature dim from observation space if possible
        # obs_space is a gym.spaces.Dict with subspaces; handle missing keys defensively
        node_dim = 4
        action_dim = action_space.shape[0] if hasattr(action_space, "shape") else num_outputs
        self.gnn = GNNPolicy(node_feat_dim=node_dim, action_dim=action_dim, hidden_dim=64)

        # value head placeholder (GNNPolicy already provides critic)
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        # Expect obs to be a dict of tensors (nodes, edges, edge_links)
        obs = input_dict.get("obs")
        if not isinstance(obs, dict):
            # If obs is a single tensor (flat), fallback to default behavior
            # flatten and pass through
            flat = obs
            # treat flat as a single-batch action input
            action = flat
            self._value = torch.zeros(action.shape[0] if action.dim() > 0 else 1, dtype=torch.float32)
            return action, []

        nodes = obs.get("nodes")  # expected shape (B, N, F) or (N, F)
        edges = obs.get("edges")  # expected shape (B, E, Fe) or (E, Fe)
        edge_links = obs.get("edge_links")  # expected shape (E, 2)

        # ensure tensors are on the model device
        device = next(self.parameters()).device
        nodes_t = nodes.to(device) if nodes is not None else None
        edges_t = edges.to(device) if edges is not None else None
        edge_links_t = edge_links.to(device).long() if edge_links is not None else None

        # Call the torch-native GNNPolicy, which returns (action_tensor, value_tensor)
        action_t, value_t = self.gnn.forward(nodes_t, edges_t, edge_links_t)

        # store value for value_function()
        # value_t shape: (B,) ; make a single tensor for RLlib
        self._value = value_t.detach()
        return action_t, []

    def value_function(self):
        return self._value


def register_model():
    ModelCatalog.register_custom_model("rllib_gnn", RLlibGNNModel)


def train(num_iters: int = 100, config_overrides: Optional[Dict[str, Any]] = None, stop: Optional[Dict[str, Any]] = None):
    """Start a small RLlib PPO training run using the GraphEnv and the GNN model.

    Args:
        num_iters: number of training iterations
        config_overrides: dict to merge into the base config
        stop: stopping criteria for tune.run
    """
    register_model()

    base_config = {
        "env": GraphEnv,
        "env_config": {"num_nodes": 3, "max_steps": 50, "observation_format": "graph"},
        "framework": "torch",
        "num_workers": 0,
        "model": {"custom_model": "rllib_gnn"},
        # small batch sizes for quick tests
        "train_batch_size": 200,
        "sgd_minibatch_size": 64,
    }
    if config_overrides:
        base_config.update(config_overrides)

    ray.init(ignore_reinit_error=True)
    # Use the registered algorithm name with tune.run for compatibility across
    # RLlib versions ("PPO" refers to the PPO implementation).
    analysis = tune.run("PPO", config=base_config, stop=stop or {"training_iteration": num_iters}, verbose=1)
    ray.shutdown()
    return analysis
