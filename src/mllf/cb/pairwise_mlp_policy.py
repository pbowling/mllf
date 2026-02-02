"""Pairwise MLP policy that predicts bias coefficients directly from substituent features.

This policy skips the graph construction and node embedding steps, instead using
substituent features (atom types, elements, environment) directly as input to an MLP.

**Prediction Structure:**
- Only upper-triangular pairs (i<j) within each site - prevents cancellation
- No cross-site pairs - all bias terms are within-site only
- Linear biases made antisymmetric in conversion: b[j,i] = -b[i,j]
- Nonlinear biases stay upper-triangular: lower triangle remains 0

Architecture:
- Input: concatenated features for two substituents [feat_i, feat_j]
- Trunk: Shared MLP layers
- Bias embeddings: Learned per-bias-type context
- Separate heads: One per bias type [linear, quadratic, skew, end]
- Output: [mean, log_std] for each bias type, scaled to physical ranges
"""
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseBiasMLP(nn.Module):
    """MLP that predicts bias coefficients for a directed pair of substituents.
    
    Instead of using graph structure and node embeddings, this directly takes
    substituent features (atom types, elements, charge, environment) and
    predicts the 4 bias coefficients for the directed edge (sub_i -> sub_j).
    
    Feature modes:
    - 'difference' (default): feat_i - feat_j (preserves directionality i->j)
      Input dimension: (3 + 14 + 161) = 178 dims for CGenFF
      Note: Environment flags are preserved (not differenced) since both subs in same env
      Structure: [charge_diff, is_solvent, is_protein, element_diffs, atom_type_diffs]
    - 'concat': [feat_i, feat_j] (absolute features)
      Input dimension: 2 * 178 = 356 dims
    - 'both': [feat_i, feat_j, feat_i - feat_j] (absolute + difference)
      Input dimension: 3 * 178 = 534 dims
    
    Each feature vector contains:
    - charge (1 dim)
    - environment (2 dims: is_solvent, is_protein)
    - element counts (~14 dims)
    - atom type counts (~161 dims for CGenFF)
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128],
        num_bias_types: int = 4,
        bias_embed_dim: int = 16,
        dropout: float = 0.1,
        feature_mode: str = 'difference'
    ):
        """Initialize pairwise bias MLP.
        
        Args:
            feature_dim: Dimension of a single substituent's features
            hidden_dims: List of hidden layer sizes
            num_bias_types: Number of bias coefficients to predict (default: 4)
            bias_embed_dim: Dimension of bias-type embeddings
            dropout: Dropout probability for regularization
            feature_mode: How to combine pair features ('difference', 'concat', or 'both')
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bias_types = num_bias_types
        self.bias_embed_dim = bias_embed_dim
        self.feature_mode = feature_mode
        
        # Determine input dimension based on feature mode
        if feature_mode == 'difference':
            pair_input_dim = feature_dim  # feat_i - feat_j
        elif feature_mode == 'concat':
            pair_input_dim = 2 * feature_dim  # [feat_i, feat_j]
        elif feature_mode == 'both':
            pair_input_dim = 3 * feature_dim  # [feat_i, feat_j, feat_i - feat_j]
        else:
            raise ValueError(f"Invalid feature_mode: {feature_mode}. Must be 'difference', 'concat', or 'both'")
        
        # Shared trunk to process pair features
        layers = []
        prev_dim = pair_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # Bias-type embeddings for specialization
        self.bias_type_embeddings = nn.Embedding(num_bias_types, bias_embed_dim)
        
        # Separate heads for each bias type
        head_input_dim = prev_dim + bias_embed_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, prev_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(prev_dim // 2, 2)  # mean and log_std
            ) for _ in range(num_bias_types)
        ])
    
    def forward(self, pair_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to predict bias coefficients.
        
        Args:
            pair_features: [N_pairs, pair_input_dim] pair features based on feature_mode:
                - 'difference': feat_i - feat_j
                - 'concat': [feat_i, feat_j]
                - 'both': [feat_i, feat_j, feat_i - feat_j]
        
        Returns:
            Tuple of (means, log_stds):
            - means: [N_pairs, num_bias_types] predicted means
            - log_stds: [N_pairs, num_bias_types] predicted log standard deviations
        """
        # Process pair through shared trunk
        h = self.trunk(pair_features)  # [N_pairs, hidden_dim]
        N = h.shape[0]
        
        # Get bias-type embeddings
        bias_type_ids = torch.arange(self.num_bias_types, device=h.device)
        bias_embeddings = self.bias_type_embeddings(bias_type_ids)  # [num_bias_types, bias_embed_dim]
        
        # Apply each head with its bias-type embedding
        means_list = []
        log_stds_list = []
        
        for i, head in enumerate(self.heads):
            # Concatenate trunk output with bias embedding
            bias_emb = bias_embeddings[i:i+1].expand(N, -1)  # [N, bias_embed_dim]
            head_input = torch.cat([h, bias_emb], dim=-1)  # [N, hidden_dim + bias_embed_dim]
            
            # Head outputs [mean, log_std]
            out = head(head_input)  # [N, 2]
            means_list.append(out[:, 0:1])
            log_stds_list.append(out[:, 1:2])
        
        means = torch.cat(means_list, dim=1)  # [N, num_bias_types]
        log_stds = torch.cat(log_stds_list, dim=1)  # [N, num_bias_types]
        
        # Apply scaling and clamping
        # Scale factors based on 52-system analysis: [linear, quadratic, skew, end]
        scale_factors = torch.tensor([79.0, 163.0, 11.0, 7.0], device=means.device)
        means = torch.tanh(means) * scale_factors.unsqueeze(0)
        
        # Clamp log_std to prevent extreme standard deviations
        log_stds = torch.clamp(log_stds, min=-20.0, max=2.0)
        
        return means, log_stds


class PairwiseMLPPolicy(nn.Module):
    """Policy that predicts bias coefficients for all directed pairs in a combination.
    
    This is a simpler alternative to the graph-based policy. Instead of:
    1. Building a graph
    2. Computing node embeddings via RGCN
    3. Predicting edge values from embeddings
    
    We directly:
    1. Extract substituent features (atom types, elements, etc.)
    2. For each directed pair, compute difference features (i->j)
    3. Predict bias coefficients via MLP
    
    By default, uses difference features (feat_i - feat_j) which:
    - Preserves directionality (i->j has opposite sign from j->i)
    - Focuses on relative properties (key for interactions)
    - Reduces input dimension (178 vs 356)
    
    This approach is conceptually simpler and may work better when:
    - Graph structure doesn't provide much signal
    - Pairwise interactions dominate
    - Feature-based similarity/difference is more important than topology
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128],
        num_bias_types: int = 4,
        bias_embed_dim: int = 16,
        dropout: float = 0.1,
        feature_mode: str = 'difference'
    ):
        """Initialize pairwise MLP policy.
        
        Args:
            feature_dim: Dimension of single substituent features
            hidden_dims: Hidden layer sizes for the MLP
            num_bias_types: Number of bias types (default: 4)
            bias_embed_dim: Bias-type embedding dimension
            dropout: Dropout probability
            feature_mode: How to combine pair features ('difference', 'concat', or 'both')
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_mode = feature_mode
        self.mlp = PairwiseBiasMLP(
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            num_bias_types=num_bias_types,
            bias_embed_dim=bias_embed_dim,
            dropout=dropout,
            feature_mode=feature_mode
        )
    
    def get_actions(
        self,
        substituent_features: torch.Tensor,
        pairs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get bias coefficient predictions for directed pairs.
        
        Args:
            substituent_features: [N_subs, feature_dim] features for all substituents
            pairs: [N_pairs, 2] indices of directed pairs (i, j)
            deterministic: If True, return means; if False, sample from distribution
        
        Returns:
            Tuple of (actions, logp, means, log_stds):
            - actions: [N_pairs, num_bias_types] sampled or mean actions
            - logp: [N_pairs] log probabilities of actions
            - means: [N_pairs, num_bias_types] predicted means
            - log_stds: [N_pairs, num_bias_types] predicted log stds
        """
        # Extract features for each pair
        feat_i = substituent_features[pairs[:, 0]]  # [N_pairs, feature_dim]
        feat_j = substituent_features[pairs[:, 1]]  # [N_pairs, feature_dim]
        
        # Check for environment mismatches (should never happen in practice)
        if (feat_i[:, 1:3] != feat_j[:, 1:3]).any():
            import warnings
            warnings.warn(
                "Environment mismatch detected between paired substituents! "
                "All substituents in a combination should be in the same environment. "
                "This may indicate a data loading error.",
                UserWarning
            )
        
        # Compute pair features based on mode
        if self.feature_mode == 'difference':
            # Compute difference features but preserve environment flags
            # Feature structure: [charge, is_solvent, is_protein, element_counts..., atom_type_counts...]
            # We want: [charge_diff, is_solvent_i, is_protein_i, element_diffs..., atom_type_diffs...]
            feat_diff = feat_i - feat_j  # [N_pairs, feature_dim]
            # Replace environment flags (indices 1-2) with feat_i's values (actual environment)
            feat_diff[:, 1] = feat_i[:, 1]  # is_solvent from substituent i
            feat_diff[:, 2] = feat_i[:, 2]  # is_protein from substituent i
            pair_features = feat_diff
        elif self.feature_mode == 'concat':
            pair_features = torch.cat([feat_i, feat_j], dim=-1)  # [N_pairs, 2*feature_dim]
        elif self.feature_mode == 'both':
            feat_diff = feat_i - feat_j
            # Also preserve environment flags in difference features for 'both' mode
            feat_diff[:, 1] = feat_i[:, 1]
            feat_diff[:, 2] = feat_i[:, 2]
            pair_features = torch.cat([feat_i, feat_j, feat_diff], dim=-1)  # [N_pairs, 3*feature_dim]
        
        # Forward pass
        means, log_stds = self.mlp(pair_features)  # Each [N_pairs, num_bias_types]
        
        if deterministic:
            actions = means
            logp = torch.zeros(means.shape[0], device=means.device)
        else:
            # Sample from normal distribution
            std = torch.exp(log_stds)
            dist = torch.distributions.Normal(means, std)
            actions = dist.rsample()
            logp_per = dist.log_prob(actions)  # [N_pairs, num_bias_types]
            
            # Clip actions to intended ranges
            scale_factors = torch.tensor([79.0, 163.0, 11.0, 7.0], device=actions.device)
            clip_limits = scale_factors * 1.05
            actions = torch.clamp(actions, -clip_limits.unsqueeze(0), clip_limits.unsqueeze(0))
            
            # Sum log probabilities across bias types
            logp = logp_per.sum(dim=-1)  # [N_pairs]
        
        return actions, logp, means, log_stds
    
    def forward(self, substituent_features: torch.Tensor, pairs: torch.Tensor):
        """Forward pass (for compatibility).
        
        Args:
            substituent_features: [N_subs, feature_dim]
            pairs: [N_pairs, 2] directed pair indices
        
        Returns:
            Tuple of (means, log_stds)
        """
        feat_i = substituent_features[pairs[:, 0]]
        feat_j = substituent_features[pairs[:, 1]]
        
        # Check for environment mismatches
        if (feat_i[:, 1:3] != feat_j[:, 1:3]).any():
            import warnings
            warnings.warn(
                "Environment mismatch detected between paired substituents! "
                "All substituents in a combination should be in the same environment.",
                UserWarning
            )
        
        # Compute pair features based on mode
        if self.feature_mode == 'difference':
            # Compute difference features but preserve environment flags
            feat_diff = feat_i - feat_j
            feat_diff[:, 1] = feat_i[:, 1]  # is_solvent from substituent i
            feat_diff[:, 2] = feat_i[:, 2]  # is_protein from substituent i
            pair_features = feat_diff
        elif self.feature_mode == 'concat':
            pair_features = torch.cat([feat_i, feat_j], dim=-1)
        elif self.feature_mode == 'both':
            feat_diff = feat_i - feat_j
            feat_diff[:, 1] = feat_i[:, 1]
            feat_diff[:, 2] = feat_i[:, 2]
            pair_features = torch.cat([feat_i, feat_j, feat_diff], dim=-1)
        
        return self.mlp(pair_features)
