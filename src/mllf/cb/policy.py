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
    def __init__(self, in_dim: int, hidden: int = 64, num_bias_types: int = 4):
        """MLP with shared trunk and separate heads per bias type.
        
        Architecture:
        - Shared trunk: processes concatenated node embeddings
        - Separate heads: one per bias type (quadratic, skew, end, linear)
        - Each head outputs (mean, log_std) for its bias type
        
        Args:
            in_dim: Input dimension (typically 2*emb_dim + edge_feat_dim)
            hidden: Hidden layer size
            num_bias_types: Number of bias types (default: 4)
        """
        super().__init__()
        self.num_bias_types = num_bias_types
        
        # Shared trunk - deeper network for better representation learning
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Separate heads for each bias type
        # Each head outputs 2 values: mean and log_std
        self.heads = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in range(num_bias_types)
        ])

    def forward(self, x):
        """Forward pass with separate heads.
        
        Returns:
            Tensor [E, 2*num_bias_types] with means and log_stds concatenated:
            [mean_1, mean_2, ..., mean_D, logstd_1, logstd_2, ..., logstd_D]
        """
        # Shared representation
        h = self.trunk(x)  # [E, hidden]
        
        # Apply each head
        head_outputs = [head(h) for head in self.heads]  # List of [E, 2] tensors
        
        # Stack and split into means and log_stds
        stacked = torch.stack(head_outputs, dim=1)  # [E, num_bias_types, 2]
        means = stacked[:, :, 0]  # [E, num_bias_types]
        log_stds = stacked[:, :, 1]  # [E, num_bias_types]
        
        # Concatenate to match expected output format [mean_1...mean_D, logstd_1...logstd_D]
        out = torch.cat([means, log_stds], dim=-1)  # [E, 2*num_bias_types]
        
        return out


class EdgePolicy(nn.Module):
    def __init__(self, encoder: nn.Module, emb_dim: int, edge_feat_dim: int = 0, mlp_hidden: int = 64, mlp_out_dim: int = 4):
        super().__init__()
        self.encoder = encoder
        # input to edge-mlp is concat([emb_u, emb_v, edge_feat])
        # mlp_out_dim controls how many coefficients per directed edge (bias types)
        # The MLP uses separate heads for each bias type
        self.mlp_out_dim = int(mlp_out_dim)
        self.edge_mlp = EdgeValueMLP(2 * emb_dim + edge_feat_dim, mlp_hidden, num_bias_types=self.mlp_out_dim)

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
        
        # Scale mean outputs to expected bias coefficient range using tanh
        # This constrains outputs to [-20, 20] which covers typical MSLD bias magnitudes
        mean = torch.tanh(mean) * 20.0
        
        # Clamp log_std to prevent extreme standard deviations that can cause NaN
        # exp(-20) ≈ 2e-9, exp(3.5) ≈ 33 — allows wider exploration than previous max=2.0
        log_std = torch.clamp(log_std, min=-20.0, max=3.5)
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
