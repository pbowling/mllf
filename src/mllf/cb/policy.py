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
    def __init__(self, in_dim: int, hidden: int = 64, num_bias_types: int = 4, bias_embed_dim: int = 16):
        """MLP with shared trunk and separate heads per bias type.
        
        Architecture:
        - Shared trunk: processes concatenated node embeddings
        - Bias-type embeddings: learnable vectors identifying each bias type
        - Separate heads: one per bias type, receives [trunk_output, bias_embedding]
        - Each head outputs (mean, log_std) for its bias type
        
        Key innovation: Bias-type embeddings provide unique context to each head,
        forcing specialization by giving heads different input signals.
        
        Args:
            in_dim: Input dimension (typically 2*emb_dim + edge_feat_dim)
            hidden: Hidden layer size
            num_bias_types: Number of bias types (default: 4)
            bias_embed_dim: Dimension of bias-type embeddings (default: 16)
        """
        super().__init__()
        self.num_bias_types = num_bias_types
        self.bias_embed_dim = bias_embed_dim
        
        # Shared trunk - lightweight for basic edge representation
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU()
        )
        
        # Learnable bias-type embeddings
        # Each bias type gets a unique embedding that will be learned during training
        # This provides bias-specific context to force head specialization
        self.bias_type_embeddings = nn.Embedding(num_bias_types, bias_embed_dim)
        
        # Separate heads - deep and specialized for each bias type
        # Input: [trunk_output (hidden), bias_type_embedding (bias_embed_dim)]
        # This concatenated input ensures each head receives unique information
        head_input_dim = hidden + bias_embed_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 2)
            ) for _ in range(num_bias_types)
        ])

    def forward(self, x):
        """Forward pass with bias-type embeddings.
        
        Returns:
            Tensor [E, 2*num_bias_types] with means and log_stds concatenated:
            [mean_1, mean_2, ..., mean_D, logstd_1, logstd_2, ..., logstd_D]
        """
        # Shared representation (lightweight)
        h = self.trunk(x)  # [E, hidden]
        E = h.shape[0]
        
        # Get bias-type embeddings for all types
        bias_type_ids = torch.arange(self.num_bias_types, device=h.device)  # [num_bias_types]
        bias_embeddings = self.bias_type_embeddings(bias_type_ids)  # [num_bias_types, bias_embed_dim]
        
        # Apply each head with its specific bias-type embedding
        head_outputs = []
        for i, head in enumerate(self.heads):
            # Expand bias embedding to match batch size and concatenate with trunk output
            bias_emb = bias_embeddings[i:i+1].expand(E, -1)  # [E, bias_embed_dim]
            head_input = torch.cat([h, bias_emb], dim=-1)  # [E, hidden + bias_embed_dim]
            head_outputs.append(head(head_input))  # [E, 2]
        
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
        
        # Scale mean outputs to expected bias coefficient range using tanh.
        # Factors set to cover the empirical maximum across ALL pretraining runs
        # (20K+ run full scan: linear max 277, quadratic max 470, skew max 77, end max 27)
        # with ~10% headroom.  Previous bounds (±72/±72/±16.5/±8) were derived from
        # a small biased sample and clipped 17.5% of linear and 34.4% of quadratic targets.
        scale_factors = torch.tensor([305.0, 520.0, 85.0, 30.0], device=mean.device)
        mean = torch.tanh(mean) * scale_factors.unsqueeze(0)
        
        # Clamp log_std to prevent extreme standard deviations that can cause NaN
        # exp(-20) ≈ 2e-9, exp(2.0) ≈ 7.4 — provides exploration while preventing extreme outliers
        # Note: Higher values (e.g., 3.5 → std≈33) can produce samples far beyond intended ranges
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
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
            
            # CRITICAL: Clip sampled actions to intended ranges per bias type
            # Without this, exploration noise (std up to ~7.4) can produce extreme outliers
            # Scale factors: [linear=61.4, quadratic=70.5, skew=6.6, end=3.6]
            scale_factors = torch.tensor([61.4, 70.5, 6.6, 3.6], device=actions.device)
            # Add small margin (5%) to allow slight overshoot during exploration
            clip_limits = scale_factors * 1.05
            actions = torch.clamp(actions, -clip_limits.unsqueeze(0), clip_limits.unsqueeze(0))

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
