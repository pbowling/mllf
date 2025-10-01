"""PyTorch MLP for predicting bias coefficients."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class PTMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (64, 16), out_dim: int = 4):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_epoch(model: nn.Module, opt: torch.optim.Optimizer, loss_fn, X, y, batch_size=32, device='cpu'):
    model.train()
    n = X.shape[0]
    idx = torch.randperm(n)
    total_loss = 0.0
    for start in range(0, n, batch_size):
        batch = idx[start:start+batch_size]
        xb = X[batch].to(device)
        yb = y[batch].to(device)
        opt.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * xb.shape[0]
    return total_loss / n


def evaluate(model: nn.Module, loss_fn, X, y, device='cpu'):
    model.eval()
    with torch.no_grad():
        preds = model(X.to(device))
        loss = loss_fn(preds, y.to(device))
    return float(loss.item()), preds.cpu()
