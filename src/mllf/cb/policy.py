"""Edge value policy built on top of a node encoder.

For each directed edge this policy produces a vector of means and a vector
of log-standard-deviations (one per predicted coefficient). The MLP outputs
concatenated [mu_1,...,mu_D, logsigma_1,...,logsigma_D] which are split
and used to parameterize a per-edge independent Gaussian distribution.
The agent samples continuous actions v_ij ~ N(mu_ij, sigma_ij^2) for every
directed edge and returns sampled values plus per-edge log-probabilities.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeValueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1):
        """Simple MLP producing `out_dim` outputs per input.

        Note: callers may request `out_dim = D*2` so the output can be split
        into means and log-stds for D predicted coefficients.
        """
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
        # The MLP produces D means and D log-stds concatenated -> 2*D outputs
        self.mlp_out_dim = int(mlp_out_dim)
        self.edge_mlp = EdgeValueMLP(2 * emb_dim + edge_feat_dim, mlp_hidden, self.mlp_out_dim * 2)

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
        out = self.edge_mlp(inp)  # [E, 2*D]
        # ensure 2D
        if out.dim() == 1:
            out = out.unsqueeze(0)
        E, C = out.shape
        D = self.mlp_out_dim
        if C != 2 * D:
            # fallback: if out dimension doesn't match expectation, treat as D=1
            D = C // 2 if C >= 2 else 1
        mean = out[:, :D]
        log_std = out[:, D: D + D]
        return mean, log_std

    def get_actions(self, x, edge_index, edge_type, edge_feat: Optional[torch.Tensor] = None, deterministic: bool = False):
        """Return sampled actions and log-probabilities for every directed edge.

        Returns:
            actions: Tensor of shape [E] or [E, D]
            logp: per-edge scalar log-prob Tensor [E]
            mean: Tensor of shape [E, D]
            log_std: Tensor of shape [E, D]
        """
        node_emb = self.forward_node_embeddings(x, edge_index, edge_type)
        mean, log_std = self.forward_edges(node_emb, edge_index, edge_feat)
        # ensure 2D
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(-1)

        if deterministic:
            actions = mean
            logp_per = torch.zeros_like(mean)
        else:
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            actions = dist.rsample()
            logp_per = dist.log_prob(actions)

        # sum logp across output dims to get per-edge scalar logp
        logp = logp_per.sum(dim=-1)
        # if single-dim, squeeze actions to [E]
        if actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        return actions, logp, mean, log_std

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
