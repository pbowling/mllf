"""Utilities to create environments for training and evaluation."""
from typing import Callable

from gym import spaces
import numpy as np

from .env import SimpleCustomEnv


def make_env(max_steps: int = 50) -> SimpleCustomEnv:
    """Factory returning a new SimpleCustomEnv instance."""
    return SimpleCustomEnv(max_steps=max_steps)
