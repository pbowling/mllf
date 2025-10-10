"""Evaluation helpers for trained policies.

This module no longer depends on Stable-Baselines3. Use RLlib evaluation tools or
custom evaluation loops that load policies from RLlib checkpoints or call the
policy directly in a PyTorch loop.

The utilities here provide small helpers to run a policy callable against
`make_env` instances. Trainers that use RLlib should use RLlib's evaluation
APIs instead.
"""
from typing import Callable

from .wrappers import make_env


def evaluate_callable_policy(policy_fn: Callable, n_episodes: int = 10, max_steps: int = 50):
    """Run a policy function (obs -> action) in the default env for n_episodes.

    This is a simple evaluation harness for custom policies.
    """
    env = make_env(max_steps=max_steps)
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        total = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, done, info = env.step(action)
            total += reward
        rewards.append(total)
    return sum(rewards) / len(rewards) if rewards else 0.0
