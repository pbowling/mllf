"""Utilities to create environments for training and evaluation."""
from typing import Callable, Optional

import numpy as np

from .env import GraphEnv, SimpleCustomEnv


def make_env(num_nodes: int = 3, max_steps: int = 50, initial_graph=None) -> GraphEnv:
    """Factory returning a new GraphEnv instance.

    Keeps a simple signature for backward compatibility (SimpleCustomEnv defaults).
    """
    return GraphEnv(num_nodes=num_nodes, max_steps=max_steps, initial_graph=initial_graph)
