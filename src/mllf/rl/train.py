"""Training script for A2C using Stable Baselines3.

This script is intentionally minimal and safe to import. Running it will
execute a short training loop when invoked as a script.
"""
from typing import Optional

import os

try:
    from stable_baselines3 import A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:  # pragma: no cover - optional dependency
    A2C = None
    DummyVecEnv = None

from .wrappers import make_env


def train(output_dir: str = "models", total_timesteps: int = 10000, seed: Optional[int] = None):
    if A2C is None:
        raise RuntimeError("stable-baselines3 is not installed. Install the 'rl' extras: `pip install .[rl]` to run training.")

    os.makedirs(output_dir, exist_ok=True)
    env = DummyVecEnv([lambda: make_env(max_steps=50)])

    model = A2C("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(output_dir, "a2c_simple.zip")
    model.save(model_path)
    return model_path


if __name__ == "__main__":
    # quick smoke training run when invoked directly
    model_file = train(total_timesteps=2000)
    print(f"Model saved to: {model_file}")
