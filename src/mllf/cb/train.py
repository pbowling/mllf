"""Simple REINFORCE training loop helper for the edge-policy.

This file provides a helper `reinforce_train_step` that performs one
policy-gradient update given a callable environment evaluator that maps
actions -> reward. The environment evaluator must accept a NumPy or torch
tensor of edge actions and return a scalar reward.
"""
from typing import Callable

import torch


def reinforce_train_step(policy, optimizer: torch.optim.Optimizer, data, env_reward_fn: Callable, baseline: float = 0.0, gamma: float = 1.0):
    """Perform one REINFORCE update.

    - policy: instance of EdgePolicy
    - optimizer: optimizer
    - data: PyG data object with x, edge_index, edge_type, edge_attr
    - env_reward_fn: callable(actions: torch.Tensor) -> float (or tensor)
    - baseline: scalar baseline to reduce variance
    Returns: (loss_value, reward)
    """
    policy.train()
    optimizer.zero_grad()

    x = data.x
    edge_index = data.edge_index
    edge_type = data.edge_type
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

    actions, logp, mean, std = policy.get_actions(x, edge_index, edge_type, edge_attr)

    # Evaluate reward from environment (user-provided function)
    # allow env_reward_fn to accept torch tensor or convert to numpy
    with torch.no_grad():
        r = env_reward_fn(actions)
        if isinstance(r, torch.Tensor):
            reward = float(r.item())
        else:
            reward = float(r)

    # policy loss: -E[logp * (reward - baseline)] ; sum over edges
    # here we treat reward as a scalar applied to all edge actions
    adv = reward - baseline
    loss = -(logp.sum() * adv)
    loss.backward()
    optimizer.step()

    return float(loss.item()), reward
