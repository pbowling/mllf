import torch
import torch.nn as nn


class _AttentionPool(nn.Module):
    """Simple wrapper for attention pooling gate network."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        with torch.no_grad():
            self.gate_nn[-1].bias.zero_()

    def forward(self, x):
        gate_logits = self.gate_nn(x)
        gate_weights = torch.softmax(gate_logits, dim=0)
        return (x * gate_weights).sum(dim=0)


class DeepSetFeatureExtractor(nn.Module):
    """DeepSet feature extractor for substituents using 3D atomic information.
    
    Implements the 4-step pipeline:
    1. Atom-Level Physical Representation: AEV + charge + atom identity
    2. Shared MLP: Compresses each atom's features independently
    3. Permutation-Invariant Pooling: Max-pool across atoms
    4. Returns fixed-size substituent embedding
    
    Args:
        aev_length: Dimension of AEV vectors (default: 2288 for ANI-2x with 11 species)
        num_atom_types: Number of distinct atom types/elements (default: 11 = 10 common + 1 unknown)
        embedding_dim: Output dimension for substituent embedding (default: 64)
        hidden_dim: Hidden layer dimension in the MLP (default: 256)
        include_charge: Whether to include atomic charge in features (default: True)
        include_atom_id: Whether to include atom type one-hot in features (default: True)
    """
    def __init__(self, aev_length=2288, num_atom_types=11, embedding_dim=64, 
                 hidden_dim=256, include_charge=True, include_atom_id=True):
        super().__init__()
        
        self.include_charge = include_charge
        self.include_atom_id = include_atom_id
        self.num_atom_types = num_atom_types
        
        # Calculate input dimension: AEV + optional charge (1D) + optional atom_id (one-hot)
        input_dim = aev_length
        if include_charge:
            input_dim += 1
        if include_atom_id:
            input_dim += num_atom_types
        
        # Shared MLP that processes each atom independently
        self.atom_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, aev_tensor, charges=None, atom_ids=None):
        """Forward pass through DeepSet encoder.
        
        Args:
            aev_tensor: [num_atoms, aev_length] - AEV vectors for each atom
            charges: [num_atoms, 1] or [num_atoms] - Partial charges (optional)
            atom_ids: [num_atoms] - Integer atom type IDs (optional)
            
        Returns:
            substituent_embedding: [embedding_dim] - Fixed-size substituent representation
        """
        features = [aev_tensor]
        
        # Concatenate charge if provided
        if self.include_charge:
            if charges is None:
                raise ValueError("charges must be provided when include_charge=True")
            if charges.dim() == 1:
                charges = charges.unsqueeze(1)
            features.append(charges)
        
        # Concatenate one-hot atom identity if provided
        if self.include_atom_id:
            if atom_ids is None:
                raise ValueError("atom_ids must be provided when include_atom_id=True")
            # Convert to one-hot encoding
            atom_one_hot = torch.nn.functional.one_hot(atom_ids, num_classes=self.num_atom_types).float()
            features.append(atom_one_hot)
        
        # Concatenate all features
        atom_features = torch.cat(features, dim=-1)
        
        # Pass through shared MLP
        atom_embeddings = self.atom_mlp(atom_features)
        
        # Max-Pool across the atoms to summarize the substituent
        # This is permutation-invariant and handles variable-sized substituents
        substituent_embedding, _ = torch.max(atom_embeddings, dim=0) 
        
        return substituent_embedding


class AtomBondGNN(nn.Module):
    """Bond-topology-aware substituent encoder using unified graph neural networks.

    Drop-in replacement for DeepSetFeatureExtractor that uses GINEConv message
    passing over the entire molecular bond graph (not split by sub/core boundaries),
    followed by GlobalAttentionPool to discover salient atoms.

    Same interface attributes as DeepSetFeatureExtractor:
    - ``include_charge``, ``include_atom_id`` flags
    - ``atom_mlp[-1].out_features`` for downstream embedding-dim queries

    Args:
        aev_length: Dimension of AEV vectors (default: 2288)
        num_atom_types: Number of distinct atom types/elements (default: 11)
        embedding_dim: Output dimension for substituent embedding (default: 64)
        hidden_dim: Hidden layer dimension in GINEConv MLPs (default: 256)
        include_charge: Whether to include atomic charge in features (default: True)
        include_atom_id: Whether to include atom type one-hot in features (default: True)
        bond_attr_dim: Dimension of bond attributes (default: 1)
        num_gin_layers: Number of GINEConv layers (default: 4)
    """

    def __init__(self, aev_length=2288, num_atom_types=11, embedding_dim=64,
                 hidden_dim=256, include_charge=True, include_atom_id=True,
                 bond_attr_dim=1, num_gin_layers=4):
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.include_charge = include_charge
        self.include_atom_id = include_atom_id
        self.num_atom_types = num_atom_types
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bond_attr_dim = bond_attr_dim
        self.num_gin_layers = num_gin_layers

        input_dim = aev_length
        if include_charge:
            input_dim += 1
        if include_atom_id:
            input_dim += num_atom_types

        def _gin_mlp(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.ReLU(),
                nn.Linear(out_d, out_d),
            )

        # Unified architecture: single input projection + GINEConv stack
        self.unified_input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.unified_gin_layers = nn.ModuleList([
            GINEConv(_gin_mlp(hidden_dim, hidden_dim), edge_dim=bond_attr_dim)
            for _ in range(num_gin_layers)
        ])

        # Global attention pooling: learns to downweight static core, upweight active sites
        self.attention_pool = _AttentionPool(hidden_dim)

        # Bottleneck projection: hidden_dim -> embedding_dim
        self.bottleneck = nn.Linear(hidden_dim, embedding_dim)

        # atom_mlp[-1] compatibility shim for graph_utils fallback embedding-dim query
        self.atom_mlp = nn.Sequential(nn.Linear(hidden_dim, embedding_dim))

    def forward(self, aev_tensor, charges=None, atom_ids=None,
                bond_edge_index=None, bond_edge_attr=None, sub_mask=None):
        """Forward pass through unified AtomBondGNN architecture.

        Processes entire molecule through single GINEConv stack (not split by sub/core).
        GlobalAttentionPool learns to downweight static core and upweight active sites.
        The sub_mask parameter is accepted for backward compatibility but ignored.

        Args:
            aev_tensor: [N, aev_length]
            charges: [N] or [N, 1] partial charges (required when include_charge=True)
            atom_ids: [N] integer atom-type IDs (required when include_atom_id=True)
            bond_edge_index: [2, 2E] bidirectional bond edges (optional, falls back to self-loops)
            bond_edge_attr: [2E, bond_attr_dim] bond-type weights (optional)
            sub_mask: [N] boolean tensor (ACCEPTED FOR COMPATIBILITY, IGNORED in unified model)

        Returns:
            substituent_embedding: [embedding_dim]
        """
        dev = aev_tensor.device

        def _build_input(aev, chg, ids):
            parts = [aev]
            if self.include_charge:
                if chg is None:
                    raise ValueError("charges must be provided when include_charge=True")
                parts.append(chg.unsqueeze(1) if chg.dim() == 1 else chg)
            if self.include_atom_id:
                if ids is None:
                    raise ValueError("atom_ids must be provided when include_atom_id=True")
                parts.append(torch.nn.functional.one_hot(ids, self.num_atom_types).float())
            return torch.cat(parts, dim=-1)

        # Project input to hidden dimension
        x = self.unified_input_proj(_build_input(aev_tensor, charges, atom_ids))  # [N, H]

        # Prepare edge index and attributes (fall back to self-loops if no edges)
        if bond_edge_index is not None and bond_edge_index.size(1) > 0:
            ei = bond_edge_index.to(dev)
            ea = (bond_edge_attr.to(dev).unsqueeze(1) if bond_edge_attr.dim() == 1
                  else bond_edge_attr.to(dev)) if bond_edge_attr is not None else torch.zeros(
                      bond_edge_index.size(1), self.bond_attr_dim, device=dev)
        else:
            n = x.size(0)
            idx = torch.arange(n, device=dev)
            ei = torch.stack([idx, idx])
            ea = torch.zeros(n, self.bond_attr_dim, device=dev)

        # Message passing through GINEConv layers with residual connections
        for gin in self.unified_gin_layers:
            x = torch.relu(gin(x, ei, ea) + x)

        # Global attention pooling: learn importance weights per atom
        aggregated = self.attention_pool(x)  # [H]

        # Bottleneck projection to embedding dimension
        embedding = self.bottleneck(aggregated)  # [embedding_dim]

        return embedding