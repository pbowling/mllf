"""Edge value policy built on top of a node encoder.

Policy outputs a mean value per edge (continuous) and uses a global learnable
log-std to create a Gaussian policy. It returns sampled actions and log
probabilities for use with REINFORCE.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeValueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class EdgePolicy(nn.Module):
    def __init__(self, encoder: nn.Module, emb_dim: int, edge_feat_dim: int = 0, mlp_hidden: int = 64):
        super().__init__()
        self.encoder = encoder
        # input to edge-mlp is concat([emb_u, emb_v, edge_feat])
        self.edge_mlp = EdgeValueMLP(2 * emb_dim + edge_feat_dim, mlp_hidden, 1)
        # global learnable log std (scalar)
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward_node_embeddings(self, x, edge_index, edge_type):
        return self.encoder(x, edge_index, edge_type)

    def edge_inputs(self, node_emb: torch.Tensor, edge_index: torch.LongTensor, edge_feat: Optional[torch.Tensor] = None):
        # edge_index: [2, E]
        src = edge_index[0]
        dst = edge_index[1]
        u = node_emb[src]
        v = node_emb[dst]
        if edge_feat is None:
            inp = torch.cat([u, v], dim=-1)
        else:
            inp = torch.cat([u, v, edge_feat], dim=-1)
        return inp

    def forward_edges(self, node_emb: torch.Tensor, edge_index: torch.LongTensor, edge_feat: Optional[torch.Tensor] = None):
        inp = self.edge_inputs(node_emb, edge_index, edge_feat)
        mean = self.edge_mlp(inp).squeeze(-1)  # [E]
        std = torch.exp(self.log_std)
        return mean, std

    def get_actions(self, x, edge_index, edge_type, edge_feat: Optional[torch.Tensor] = None, deterministic: bool = False):
        """Return actions and log_probs for every edge.

        actions: [E]
        log_probs: [E]
        """
        node_emb = self.forward_node_embeddings(x, edge_index, edge_type)
        mean, std = self.forward_edges(node_emb, edge_index, edge_feat)
        if deterministic:
            actions = mean
            # Gaussian log prob for deterministic treated as delta (approx)
            logp = torch.zeros_like(mean)
        else:
            dist = torch.distributions.Normal(mean, std)
            actions = dist.rsample()
            logp = dist.log_prob(actions)
        return actions, logp, mean, std
