"""A minimal custom Gym environment for testing A2C training.

This environment uses a simple discrete action space and a small observation
vector. It's deterministic and intended only as a scaffold to wire up
Stable Baselines3 training scripts.
"""
from typing import Tuple

import numpy as np

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - allow environments with gymnasium
    import gymnasium as gym
    from gymnasium import spaces


class SimpleCustomEnv(gym.Env):
    """Simple deterministic environment.

    Observation: Box(3,) floats in [-1, 1]
    Action: Discrete(2)
    Episode length is fixed (max_steps).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps: int = 50):
        super().__init__()
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._step_count = 0
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        self.state = np.zeros(3, dtype=np.float32)
        # Gym and Gymnasium have slightly different reset return signatures; return obs only
        return self.state

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        # apply a trivial dynamic: add or subtract 0.1 to the first state component
        if action == 1:
            self.state[0] = np.clip(self.state[0] + 0.1, -1.0, 1.0)
        else:
            self.state[0] = np.clip(self.state[0] - 0.1, -1.0, 1.0)

        self._step_count += 1
        done = self._step_count >= self.max_steps
        # Reward is higher when first state is near +1.0
        reward = float(self.state[0])
        info = {}
        return self.state, reward, done, info

    def render(self, mode="human"):
        print(f"Step {self._step_count}: state={self.state}")

    def close(self):
        return None
