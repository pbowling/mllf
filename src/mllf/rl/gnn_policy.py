"""Torch-native batched GNN actor-critic policy.

This policy accepts batched graph tensors and returns action tensors and value
estimates, all in torch tensors. Expected input shapes:

- nodes: Tensor shape (B, N, F) or (N, F) where B is batch size, N is number
  of nodes, and F is node feature dim.
- edges: Tensor shape (B, E, Fe) or (E, Fe) (optional)
- edge_links: LongTensor shape (E, 2) specifying (src, dst) for each edge. The
  same edge ordering is assumed across the batch.

The implementation uses scatter_add for message aggregation so it is efficient
for moderate graph sizes where edges are shared across the batch.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, nodes: torch.Tensor, edge_links: Optional[torch.LongTensor] = None, edges: Optional[torch.Tensor] = None):
        """Message passing convolution.

        nodes: (B, N, F)
        edge_links: (E, 2) long tensor (src, dst)
        edges: (B, E, Fe) or (E, Fe) unused currently but kept for extension.
        """
        # Ensure batch dim
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        B, N, F = nodes.shape

        if edge_links is None or edges is None:
            out = self.lin(nodes)
            return F.relu(out)

        # edge_links assumed shape (E,2)
        src = edge_links[:, 0]
        dst = edge_links[:, 1]

        # gather source node features for all edges -> (B, E, F)
        # torch indexing with a 1D index selects along dim=1
        msgs = nodes[:, src, :]

        # aggregate messages per destination using scatter_add along node dim
        out = torch.zeros_like(nodes)
        # dst_idx for broadcasting: shape (B, E, 1)
        dst_idx = dst.view(1, -1, 1).expand(B, -1, F)
        out = out.scatter_add(1, dst_idx, msgs)

        # combine with node features and apply linear
        h = nodes + out
        return F.relu(self.lin(h))


class GNNPolicy(nn.Module):
    def __init__(self, node_feat_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GraphConv(node_feat_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, nodes: torch.Tensor, edges: Optional[torch.Tensor] = None, edge_links: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns (action_tensor, value_tensor), where action_tensor has shape
        (B, action_dim) and value_tensor has shape (B,).
        """
        # normalize dims to (B, N, F)
        if nodes.dim() == 2:
            nodes = nodes.unsqueeze(0)
        B = nodes.shape[0]

        # ensure edge tensors have batch dim if present
        if edges is not None and edges.dim() == 2:
            edges = edges.unsqueeze(0).expand(B, -1, -1)

        # convs
        h = self.conv1(nodes, edge_links, edges)
        h = self.conv2(h, edge_links, edges)

        # global pooling (mean over nodes)
        hg = h.mean(dim=1)  # (B, hidden)

        action = self.actor(hg)  # (B, action_dim)
        value = self.critic(hg).squeeze(-1)  # (B,)

        return action, value
