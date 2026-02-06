"""Value network for baseline estimation in REINFORCE.

The value network learns to predict the expected reward for a given combination,
providing a state-dependent baseline that reduces variance in policy gradient updates.
This is a standard component in Actor-Critic methods (A2C, PPO, etc.).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ValueNetwork(nn.Module):
    """Value network that predicts expected reward from graph encoding.
    
    Architecture:
    - Takes node embeddings from encoder
    - Global pooling (mean) to get graph-level representation
    - MLP to predict scalar value (expected reward)
    
    This provides a state-dependent baseline V(s) for REINFORCE:
        advantage = R - V(s)
    instead of a fixed or moving average baseline.
    """
    
    def __init__(self, emb_dim: int, hidden_dims: list = None):
        """Initialize value network.
        
        Args:
            emb_dim: Dimension of node embeddings from encoder
            hidden_dims: Hidden layer dimensions (default: [64, 32])
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Build MLP layers
        layers = []
        in_dim = emb_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        
        # Final layer outputs single scalar (value prediction)
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, node_embeddings: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict expected reward from node embeddings.
        
        Args:
            node_embeddings: Node embeddings from encoder [num_nodes, emb_dim]
            batch: Batch assignment tensor [num_nodes] (for batched graphs)
                   If None, assumes single graph
        
        Returns:
            value: Predicted expected reward (scalar or [batch_size])
        """
        # Global mean pooling to get graph-level representation
        if batch is None:
            # Single graph: mean over all nodes
            graph_embedding = node_embeddings.mean(dim=0, keepdim=True)  # [1, emb_dim]
        else:
            # Batched graphs: mean per graph
            from torch_geometric.nn import global_mean_pool
            graph_embedding = global_mean_pool(node_embeddings, batch)  # [batch_size, emb_dim]
        
        # Predict value
        value = self.mlp(graph_embedding).squeeze(-1)  # [batch_size] or scalar
        
        return value
    
    def predict(self, node_embeddings: torch.Tensor, batch: Optional[torch.Tensor] = None) -> float:
        """Convenience method for getting scalar prediction (no gradients).
        
        Args:
            node_embeddings: Node embeddings from encoder
            batch: Batch assignment tensor
        
        Returns:
            Scalar value prediction
        """
        with torch.no_grad():
            value = self.forward(node_embeddings, batch)
            return value.item() if value.numel() == 1 else value.cpu().numpy()
