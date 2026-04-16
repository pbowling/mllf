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
    def __init__(self, encoder: nn.Module, emb_dim: int, edge_feat_dim: int = 0,
                 mlp_hidden: int = 64, mlp_out_dim: int = 4, p1_dim: int = 0):
        super().__init__()
        self.encoder = encoder
        self.mlp_out_dim = int(mlp_out_dim)
        self.p1_dim = int(p1_dim)
        # Input to edge-mlp: [P1_src, P2_src, P1_dst, P2_dst, edge_feat] when p1_dim>0
        # otherwise: [P2_src, P2_dst, edge_feat] (backward-compatible)
        in_dim = 2 * (emb_dim + p1_dim) + edge_feat_dim
        self.edge_mlp = EdgeValueMLP(in_dim, mlp_hidden, num_bias_types=self.mlp_out_dim)
        # Scale factors for the 4 MSLD bias types: [linear, quadratic, skew, end].
        # Empirical max across 20K+ pretraining runs with ~10% headroom.
        # Registered as non-persistent so they are NOT saved in state_dict
        # (they are constants, not learned state).
        self.register_buffer(
            'scale_factors',
            torch.tensor([305.0, 520.0, 85.0, 30.0]),
            persistent=False,
        )

    def forward_node_embeddings(self, x, edge_index, edge_type):
        return self.encoder(x, edge_index, edge_type)

    def edge_inputs(self, node_emb: torch.Tensor, edge_index: torch.LongTensor,
                    edge_feat: Optional[torch.Tensor] = None,
                    p1_emb: Optional[torch.Tensor] = None):
        """Build per-edge input tensor.

        When p1_emb is provided (skip connection), concatenates
        [P1_src, P2_src, P1_dst, P2_dst, edge_feat]; otherwise [P2_src, P2_dst, edge_feat].
        """
        # edge_index: [2, E]
        src = edge_index[0]
        dst = edge_index[1]
        if p1_emb is not None:
            parts = [p1_emb[src], node_emb[src], p1_emb[dst], node_emb[dst]]
        else:
            parts = [node_emb[src], node_emb[dst]]
        if edge_feat is not None:
            parts.append(edge_feat)
        return torch.cat(parts, dim=-1)

    def forward_edges(self, node_emb: torch.Tensor, edge_index: torch.LongTensor,
                      edge_feat: Optional[torch.Tensor] = None,
                      p1_emb: Optional[torch.Tensor] = None):
        inp = self.edge_inputs(node_emb, edge_index, edge_feat, p1_emb)
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
        # with ~10% headroom.
        mean = torch.tanh(mean) * self.scale_factors.unsqueeze(0)
        
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
        p1_for_skip = x if self.p1_dim > 0 else None
        mean, log_std = self.forward_edges(node_emb, edge_index, edge_feat, p1_for_skip)
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
            
            # Clip sampled actions to the scale-factor range (+5% margin).
            # Matches the tanh scale factors in forward_edges.
            clip_limits = self.scale_factors * 1.05
            actions = torch.clamp(actions, -clip_limits.unsqueeze(0), clip_limits.unsqueeze(0))

        # sum logp across output dims to get per-edge scalar logp
        logp = logp_per.sum(dim=-1)
        # if single-dim, squeeze actions to [E]
        if actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        return actions, logp, mean, log_std

    def evaluate_logp(
        self,
        x,
        edge_index,
        edge_type,
        edge_feat: Optional[torch.Tensor],
        saved_actions: torch.Tensor,
    ):
        """Evaluate log π_θ(saved_actions | x) under current policy parameters.

        Used for on-policy REINFORCE: the actions that were actually submitted to
        the simulation are treated as fixed (no gradient through them), while the
        distribution parameters (mean, log_std) retain gradients so that the policy
        update correctly reinforces or suppresses the actions that led to the reward.

        Args:
            saved_actions: Tensor [E] or [E, D] — actions from the simulation run
                (i.e., the clipped actions stored in epoch_results.pt).

        Returns:
            logp: [E] scalar log-prob per edge summed over output dims.
            log_std: [E, D] current log_std (for entropy regularisation caller).
        """
        node_emb = self.forward_node_embeddings(x, edge_index, edge_type)
        p1_for_skip = x if self.p1_dim > 0 else None
        mean, log_std = self.forward_edges(node_emb, edge_index, edge_feat, p1_for_skip)
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(-1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        # saved_actions treated as a fixed constant — no gradient through them
        a = saved_actions.detach().to(mean.device)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        logp = dist.log_prob(a).sum(dim=-1)
        return logp, log_std

    @classmethod
    def from_pyg_data(cls, encoder: nn.Module, emb_dim: int, data, mlp_hidden: int = 64,
                      mlp_out_dim: int = 1, p1_dim: int = 0):
        """Construct an EdgePolicy using a PyG ``data`` object to infer edge_feat_dim.

        Reads ``data.edge_attr`` (if present) for per-edge feature size.
        Pass ``p1_dim=data.x.size(1)`` to enable the skip-connection path where
        pre-RGCN node embeddings are concatenated alongside RGCN embeddings.

        Input dimension: 2*(emb_dim + p1_dim) + edge_feat_dim.
        """
        edge_feat_dim = 0
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            try:
                edge_feat_dim = int(data.edge_attr.shape[1])
            except Exception:
                edge_feat_dim = 0
        return cls(encoder, emb_dim, edge_feat_dim, mlp_hidden, mlp_out_dim, p1_dim)
