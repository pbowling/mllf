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


class QNetwork(nn.Module):
    """Per-edge Q(s, a) critic for per-pair credit assignment.

    Estimates Q(s, a) from the edge state representation *and* the actions
    (bias coefficients) sampled by the actor for that edge.  Concatenating
    the action lets the critic directly grade what the actor did, rather than
    only evaluating the structural context.

        Q_input = [edge_state (in_dim), action (action_dim)]
        A_pair  = R_pair - Q(edge_state, action).detach()

    Args:
        in_dim: Edge state dimension (typically 2*(p2_dim + p1_dim) + edge_feat_dim).
        action_dim: Number of bias coefficient dimensions output by the actor
            (default: 4 — linear, quadratic, skew, end).
        hidden_dims: Hidden layer sizes (default: [64, 32]).
    """

    def __init__(self, in_dim: int, action_dim: int = 4, hidden_dims: list = None):
        super().__init__()

        self.action_dim = action_dim

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        in_d = in_dim + action_dim          # state + action concatenated
        for h in hidden_dims:
            layers.extend([nn.Linear(in_d, h), nn.ReLU()])
            in_d = h
        layers.append(nn.Linear(in_d, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_inputs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict per-edge Q(s, a) values.

        Args:
            edge_inputs: [E, in_dim] edge state representations.
            actions: [E] or [E, action_dim] bias coefficients sampled by the actor.

        Returns:
            q_values: [E] per-edge Q-value predictions.
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)          # [E, 1] if single-dim action
        qa = torch.cat([edge_inputs, actions], dim=-1)   # [E, in_dim + action_dim]
        return self.mlp(qa).squeeze(-1)
