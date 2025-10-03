"""Evaluation helper to run a trained SB3 model in the environment."""
from typing import Optional

import numpy as np

try:
    from stable_baselines3 import A2C
except Exception:  # pragma: no cover - optional dependency
    A2C = None

from .wrappers import make_env


def evaluate(model_path: str, n_episodes: int = 10, max_steps: int = 50):
    if A2C is None:
        raise RuntimeError("stable-baselines3 is not installed. Install the 'rl' extras: `pip install .[rl]` to run evaluation.")

    model = A2C.load(model_path)
    env = make_env(max_steps=max_steps)
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += reward
        rewards.append(total)
    return np.mean(rewards)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m mllf.rl.eval <model_path>")
    else:
        mean = evaluate(sys.argv[1])
        print(f"Mean reward: {mean}")
