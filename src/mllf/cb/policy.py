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
    def __init__(self, encoder: nn.Module, emb_dim: int, edge_feat_dim: int = 0, mlp_hidden: int = 64, mlp_out_dim: int = 1):
        super().__init__()
        self.encoder = encoder
        # input to edge-mlp is concat([emb_u, emb_v, edge_feat])
        # mlp_out_dim controls how many coefficients per directed edge
        self.edge_mlp = EdgeValueMLP(2 * emb_dim + edge_feat_dim, mlp_hidden, mlp_out_dim)
        # global learnable log std (scalar) applied to all output dims
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
        mean = self.edge_mlp(inp)  # [E, out_dim]
        std = torch.exp(self.log_std)
        return mean, std

    def get_actions(self, x, edge_index, edge_type, edge_feat: Optional[torch.Tensor] = None, deterministic: bool = False):
        """Return actions and log_probs for every edge.

        actions: [E]
        log_probs: [E]
        """
        node_emb = self.forward_node_embeddings(x, edge_index, edge_type)
        mean, std = self.forward_edges(node_emb, edge_index, edge_feat)
        # mean: [E, D] or [E] depending on out_dim; ensure 2D
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        if deterministic:
            actions = mean
            # deterministic: treat logp as zeros per-output then sum
            logp_per = torch.zeros_like(mean)
        else:
            # std is scalar; broadcast to mean shape
            std_b = std if isinstance(std, torch.Tensor) else torch.tensor(std, device=mean.device, dtype=mean.dtype)
            dist = torch.distributions.Normal(mean, std_b)
            actions = dist.rsample()
            logp_per = dist.log_prob(actions)  # [E, D]
        # sum logp across output dims to get per-edge scalar logp
        logp = logp_per.sum(dim=-1)
        # if single-dim, squeeze actions to [E]
        if actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        return actions, logp, mean, std

    @classmethod
    def from_pyg_data(cls, encoder: nn.Module, emb_dim: int, data, mlp_hidden: int = 64, mlp_out_dim: int = 1):
        """Construct an EdgePolicy using a PyG `data` object to infer edge_feat_dim.

        This helper reads `data.edge_attr` (if present) to determine the
        per-edge feature size so the internal MLP has the correct fixed input
        dimension: 2*emb_dim + edge_feat_dim.
        """
        edge_feat_dim = 0
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # edge_attr is [E, F]
            try:
                edge_feat_dim = int(data.edge_attr.shape[1])
            except Exception:
                edge_feat_dim = 0
        return cls(encoder, emb_dim, edge_feat_dim, mlp_hidden, mlp_out_dim)
