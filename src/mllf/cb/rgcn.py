"""RGCN encoder using PyTorch Geometric.

Implements a small multi-layer RGCN that returns node embeddings of fixed
dimension. Requires `torch` and `torch_geometric` to be installed in the runtime
where training occurs.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv



class RGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, out_dim: int, num_relations: int, num_bases: Optional[int]=None):
        super().__init__()

        # Normalise input node features before the first convolution.
        # This stabilises sum-pool embeddings whose L2 magnitude grows with
        # substituent size (atom count), preventing gradient explosion while
        # allowing the learnable γ/β parameters to recover useful scale signal.
        self.input_norm = nn.LayerNorm(in_dim)

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(RGCNConv(prev, h, num_relations, num_bases=num_bases))
            prev = h
        # final layer to out_dim
        layers.append(RGCNConv(prev, out_dim, num_relations, num_bases=num_bases))

        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_type: torch.LongTensor) -> torch.Tensor:
        """Forward pass.

        x: [N, in_dim]
        edge_index: [2, E]
        edge_type: [E] (relation ids in [0, num_relations-1])
        Returns: node embeddings [N, out_dim]
        """
        h = self.input_norm(x)
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index, edge_type)
            if i != len(self.layers) - 1:
                h = self.act(h)
        return h
