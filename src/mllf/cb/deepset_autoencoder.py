"""
DeepSet Autoencoder for pretraining atom-level embeddings.

This implements Step 2 of the 4-step pretraining process:
Build an autoencoder with symmetric encoder/decoder networks to learn
compressed representations of atomic environments (AEV + charge).
"""

import torch
import torch.nn as nn


class DeepSetEncoder(nn.Module):
    """Encoder network that compresses atom features to embedding space.
    
    This becomes the final DeepSet MLP after training.
    
    Architecture:
        Input: 2289D (AEV 2288D + charge 1D)
        Hidden: 256D + ReLU
        Output: 64D (embedding bottleneck)
    """
    
    def __init__(self, input_dim=2289, hidden_dim=256, embedding_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [num_atoms, input_dim] atom features (AEV + charge)
            
        Returns:
            [num_atoms, embedding_dim] compressed atom embeddings
        """
        return self.network(x)


class DeepSetDecoder(nn.Module):
    """Decoder network that reconstructs atom features from embeddings.
    
    This is discarded after training - only used for autoencoder loss.
    
    Architecture:
        Input: 64D (embedding)
        Hidden: 256D + ReLU
        Output: 2289D (reconstructed AEV + charge)
    """
    
    def __init__(self, embedding_dim=64, hidden_dim=256, output_dim=2289):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: [num_atoms, embedding_dim] compressed embeddings
            
        Returns:
            [num_atoms, output_dim] reconstructed atom features
        """
        return self.network(x)


class DeepSetAutoencoder(nn.Module):
    """Complete autoencoder for pretraining DeepSet embeddings.
    
    This trains the encoder to compress atom-level physics (AEV + charge)
    into a compact 64D representation that captures steric crowding,
    electronegativity, and Van der Waals radius.
    
    After training:
    1. Sever the decoder
    2. Add max-pooling to the encoder
    3. Use as node embeddings in RGCN
    """
    
    def __init__(self, input_dim=2289, hidden_dim=256, embedding_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.encoder = DeepSetEncoder(input_dim, hidden_dim, embedding_dim)
        self.decoder = DeepSetDecoder(embedding_dim, hidden_dim, input_dim)
    
    def forward(self, x):
        """
        Args:
            x: [num_atoms, input_dim] atom features (AEV + charge)
            
        Returns:
            dict with keys:
                - 'embedding': [num_atoms, embedding_dim] compressed representation
                - 'reconstruction': [num_atoms, input_dim] reconstructed features
        """
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        
        return {
            'embedding': embedding,
            'reconstruction': reconstruction
        }
    
    def encode(self, x):
        """Encode atom features to embeddings (inference only)."""
        return self.encoder(x)
    
    def save_encoder(self, path):
        """Save only the encoder for deployment."""
        torch.save({
            'state_dict': self.encoder.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'embedding_dim': self.embedding_dim,
        }, path)
        print(f"Encoder saved to {path}")


class PretrainedDeepSet(nn.Module):
    """Pretrained DeepSet with max-pooling for RGCN integration.
    
    This is Step 4 of the pretraining process:
    - Loads the trained encoder
    - Adds torch.max(dim=0) pooling
    - Optionally freezes weights
    - Ready to plug into RGCN as node feature generator
    """
    
    def __init__(self, encoder_path, freeze_weights=True):
        super().__init__()
        
        # Load pretrained encoder
        checkpoint = torch.load(encoder_path, map_location='cpu')
        
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.embedding_dim = checkpoint['embedding_dim']
        
        # Recreate encoder
        self.encoder = DeepSetEncoder(
            self.input_dim, 
            self.hidden_dim, 
            self.embedding_dim
        )
        self.encoder.load_state_dict(checkpoint['state_dict'])
        
        # Freeze weights if requested
        if freeze_weights:
            self.encoder.requires_grad_(False)
            self.frozen = True
        else:
            self.frozen = False
        
        print(f"Loaded pretrained DeepSet from {encoder_path}")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Frozen: {self.frozen}")
    
    def forward(self, atom_features):
        """
        Args:
            atom_features: [num_atoms, input_dim] atom features (AEV + charge)
            
        Returns:
            [embedding_dim] pooled node embedding
        """
        # Encode individual atoms
        atom_embeddings = self.encoder(atom_features)  # [num_atoms, embedding_dim]
        
        # Max-pool across atoms
        pooled_embedding, _ = torch.max(atom_embeddings, dim=0)  # [embedding_dim]
        
        return pooled_embedding
    
    def unfreeze(self):
        """Allow fine-tuning of the encoder."""
        self.encoder.requires_grad_(True)
        self.frozen = False
        print("Encoder weights unfrozen for fine-tuning")
    
    def freeze(self):
        """Freeze encoder weights."""
        self.encoder.requires_grad_(False)
        self.frozen = True
        print("Encoder weights frozen")


def create_autoencoder(input_dim=2289, hidden_dim=256, embedding_dim=64):
    """Factory function to create a DeepSet autoencoder."""
    return DeepSetAutoencoder(input_dim, hidden_dim, embedding_dim)


def load_pretrained_deepset(encoder_path, freeze_weights=True):
    """Factory function to load a pretrained DeepSet for inference."""
    return PretrainedDeepSet(encoder_path, freeze_weights)
