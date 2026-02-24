import torch
import torch.nn as nn

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