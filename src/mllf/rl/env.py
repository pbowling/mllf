"""A minimal custom Gym environment for testing A2C training.

This environment uses a simple discrete action space and a small observation
vector. It's deterministic and intended only as a scaffold to wire up
Stable Baselines3 training scripts.
"""
from typing import Tuple, Optional

import numpy as np

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - allow environments with gymnasium
    import gymnasium as gym
    from gymnasium import spaces

from .graph import Graph


class GraphEnv(gym.Env):
    """Environment where the observation is the flattened edge coefficients of a graph.

    Observation: Box with length = n_edges * 4 (linear, quadratic, skew, end)
    Action: Box with same length representing additive updates to the edge coefficients.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_nodes: int = 3, max_steps: int = 50, coeff_limit: float = 10.0):
        super().__init__()
        self.max_steps = max_steps
        self.num_nodes = num_nodes
        self.graph = Graph(num_nodes)
        self._step_count = 0
        n_edges = num_nodes * (num_nodes - 1) // 2
        self.obs_dim = n_edges * 4
        self.coeff_limit = coeff_limit
        low = -coeff_limit * np.ones(self.obs_dim, dtype=np.float32)
        high = coeff_limit * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Actions are continuous additive updates to the coefficients
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        # reset graph to zero coefficients
        self.graph = Graph(self.num_nodes)
        return self.graph.as_vector()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=float).flatten()
        if action.size != self.obs_dim:
            raise ValueError(f"Action size {action.size} does not match expected {self.obs_dim}")
        # apply additive update scaled by a small factor
        current = self.graph.as_vector()
        new = np.clip(current + 0.1 * action, -self.coeff_limit, self.coeff_limit)
        self.graph.from_vector(new)
        self._step_count += 1
        done = self._step_count >= self.max_steps
        # define a placeholder reward: negative L2 distance from a target (e.g., all ones)
        target = np.ones_like(new)
        reward = -float(np.linalg.norm(new - target))
        info = {}
        return new, reward, done, info

    def render(self, mode="human"):
        print(f"Step {self._step_count}: graph_vector={self.graph.as_vector()}\n")

    def close(self):
        return None


# Keep the SimpleCustomEnv name for compatibility but make it a thin wrapper
class SimpleCustomEnv(GraphEnv):
    def __init__(self, max_steps: int = 50):
        # default to 3 nodes for the compatibility env (obs_dim = 3 edges * 4 = 12)
        super().__init__(num_nodes=3, max_steps=max_steps)

