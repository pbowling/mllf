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
    2. Add sum-pooling to the encoder
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


class _LastLayerProxy:
    """Minimal proxy exposing .out_features for compatibility with DeepSetFeatureExtractor."""
    def __init__(self, out_features: int):
        self.out_features = out_features


class _AtomMlpProxy:
    """Proxy for atom_mlp[-1] access pattern used by graph_utils embedding-dim detection."""
    def __init__(self, out_features: int):
        self._last = _LastLayerProxy(out_features)

    def __getitem__(self, idx):
        return self._last


class PretrainedDeepSet(nn.Module):
    """Pretrained DeepSet with sum-pooling for RGCN integration.
    
    This is Step 4 of the pretraining process:
    - Loads the trained encoder
    - Adds torch.sum(dim=0) pooling
    - Optionally freezes weights
    - Ready to plug into RGCN as node feature generator

    The class implements the same interface as DeepSetFeatureExtractor so it can
    be used as a drop-in replacement in graph_utils.build_pyg_graph_from_mllf_graph.
    The autoencoder was trained with include_charges=True and include_atom_ids=False,
    so input_dim = aev_length + 1 (charge).
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

        # DeepSetFeatureExtractor compatibility attributes
        # The autoencoder was trained with charges included and no atom-type one-hots.
        self.include_charge = True
        self.include_atom_id = False
        # atom_mlp[-1].out_features used by graph_utils for fallback zero embeddings
        self.atom_mlp = _AtomMlpProxy(self.embedding_dim)
        
        print(f"Loaded pretrained DeepSet from {encoder_path}")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Frozen: {self.frozen}")
    
    def forward(self, atom_features=None, *, aev_tensor=None, charges=None, atom_ids=None):
        """Forward pass — accepts both native and DeepSetFeatureExtractor calling conventions.

        Native convention (original):
            forward(atom_features)  — [num_atoms, input_dim] pre-concatenated tensor

        DeepSetFeatureExtractor convention (for graph_utils compatibility):
            forward(aev_tensor=..., charges=...)  — separate AEV and charge tensors

        Returns:
            [embedding_dim] pooled node embedding
        """
        if atom_features is None:
            if aev_tensor is None:
                raise ValueError("Either atom_features or aev_tensor must be provided")
            parts = [aev_tensor]
            if self.include_charge and charges is not None:
                if charges.dim() == 1:
                    charges = charges.unsqueeze(1)
                parts.append(charges)
            atom_features = torch.cat(parts, dim=-1)

        # Encode individual atoms then sum-pool
        atom_embeddings = self.encoder(atom_features)      # [num_atoms, embedding_dim]
        pooled_embedding = torch.sum(atom_embeddings, dim=0)  # [embedding_dim]
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


# ---------------------------------------------------------------------------
# Attention pooling for unified architecture
# ---------------------------------------------------------------------------

class GlobalAttentionPool(nn.Module):
    """Learnable attention pooling that discovers salient nodes.
    
    Computes a gating network that assigns attention weights to each node,
    then returns the weighted sum of node embeddings. This allows the model
    to automatically learn which atoms (active sites) are important and which
    (core scaffold) can be downweighted.
    
    Args:
        hidden_dim: Dimension of input node embeddings
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Initialize gate biases toward uniform attention to prevent collapse
        with torch.no_grad():
            self.gate_nn[-1].bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate node embeddings with learned attention weights.
        
        Args:
            x: [num_atoms, hidden_dim] node embeddings
            
        Returns:
            [hidden_dim] weighted-sum aggregated embedding
        """
        if x.size(0) == 0:
            return torch.zeros(x.size(1), device=x.device)
        
        gate_logits = self.gate_nn(x)  # [num_atoms, 1]
        gate_weights = torch.softmax(gate_logits, dim=0)  # [num_atoms, 1]
        aggregated = (x * gate_weights).sum(dim=0)  # [hidden_dim]
        return aggregated


# ---------------------------------------------------------------------------
# AtomBondGNN autoencoder (for bond-topology-aware pretraining)
# ---------------------------------------------------------------------------

class AtomBondGNNAutoencoder(nn.Module):
    """Autoencoder for pretraining AtomBondGNN atom-level embeddings.

    Uses a unified graph neural network architecture: all atoms (focus sub + core + 
    other subs) are processed through a single GINEConv stack. A learnable 
    GlobalAttentionPool then discovers which atoms (typically the active site) are 
    most important and downweights the static core scaffold.

    This design is boundary-invariant: identical final ligands with different 
    sub/core file splits get identical embeddings (up to the learned attention).
    The contrastive loss uses full-ligand atom composition to enable this invariance,
    and uniformity loss separates different cores.

    Training loss: per-atom MSE reconstruction of input features
    (AEV + charge + atom-type one-hot) applied to hidden states *before* pooling.

    After training call :meth:`save_encoder` to persist an
    ``AtomBondGNN``-compatible checkpoint.  The decoder weights are
    intentionally excluded from the saved file.

    Args:
        aev_length: AEV feature dimension (default: 2288).
        num_atom_types: Number of element species (default: 11).
        embedding_dim: Substituent embedding bottleneck dimension (default: 64).
        hidden_dim: Hidden dimension for GINEConv MLPs (default: 256).
        include_charge: Include partial charge in input features (default: True).
        include_atom_id: Include atom-type one-hot in input features (default: True).
        bond_attr_dim: Dimension of bond-type edge features (default: 1).
        num_gin_layers: Number of GINEConv layers (default: 4).
    """

    def __init__(
        self,
        aev_length: int = 2288,
        num_atom_types: int = 11,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        include_charge: bool = True,
        include_atom_id: bool = True,
        bond_attr_dim: int = 1,
        num_gin_layers: int = 4,
    ):
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.aev_length = aev_length
        self.num_atom_types = num_atom_types
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.include_charge = include_charge
        self.include_atom_id = include_atom_id
        self.bond_attr_dim = bond_attr_dim
        self.num_gin_layers = num_gin_layers

        input_dim = aev_length
        if include_charge:
            input_dim += 1
        if include_atom_id:
            input_dim += num_atom_types
        self.input_dim = input_dim

        # ── Unified encoder layers (single stream for entire molecule) ──
        def _gin_mlp(in_d: int, out_d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.ReLU(),
                nn.Linear(out_d, out_d),
            )

        # Single unified input projection for entire ligand (sub + core + ref subs)
        self.unified_input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Single unified GINEConv stack (messages propagate across entire topology)
        self.unified_gin_layers = nn.ModuleList([
            GINEConv(_gin_mlp(hidden_dim, hidden_dim), edge_dim=bond_attr_dim)
            for _ in range(num_gin_layers)
        ])

        # Learnable attention pooling (discovers important atoms automatically)
        self.attention_pool = GlobalAttentionPool(hidden_dim)

        # Final bottleneck projection (hidden → embedding)
        self.bottleneck = nn.Linear(hidden_dim, embedding_dim)

        # Compatibility shim (matches AtomBondGNN attribute)
        self.atom_mlp = nn.Sequential(nn.Linear(hidden_dim, embedding_dim))

        # ── Decoder: per-atom reconstruction from hidden states ──
        self.decoder = nn.Linear(hidden_dim, input_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input(
        self,
        aev_tensor: torch.Tensor,
        charges: torch.Tensor,
        atom_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate AEV + charges + atom-type one-hot → [N, input_dim]."""
        parts = [aev_tensor]
        if self.include_charge:
            if charges.dim() == 1:
                charges = charges.unsqueeze(1)
            parts.append(charges)
        if self.include_atom_id:
            one_hot = torch.nn.functional.one_hot(
                atom_ids, num_classes=self.num_atom_types
            ).float()
            parts.append(one_hot)
        return torch.cat(parts, dim=-1)

    def _unified_forward(
        self,
        aev_tensor: torch.Tensor,
        charges: torch.Tensor,
        atom_ids: torch.Tensor,
        bond_edge_index: torch.Tensor,
        bond_edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process entire molecule through unified GINEConv stack.
        
        Returns:
            [N, hidden_dim] hidden node embeddings after message passing
        """
        x = self.unified_input_proj(self._build_input(aev_tensor, charges, atom_ids))
        dev = aev_tensor.device
        
        # Ensure bond edge index is valid; use self-loops if needed
        if bond_edge_index is not None and bond_edge_index.size(1) > 0:
            ei = bond_edge_index.to(dev)
            if bond_edge_attr is not None:
                ea = bond_edge_attr.to(dev)
                if ea.dim() == 1:
                    ea = ea.unsqueeze(1)
            else:
                ea = torch.zeros(ei.size(1), self.bond_attr_dim, device=dev)
        else:
            # No bonds: use self-loops only
            n = x.size(0)
            idx = torch.arange(n, device=dev)
            ei = torch.stack([idx, idx])
            ea = torch.zeros(n, self.bond_attr_dim, device=dev)
        
        # Message passing with residual connections
        for gin in self.unified_gin_layers:
            x = torch.relu(gin(x, ei, ea) + x)
        
        return x

    # ------------------------------------------------------------------
    # Forward / encode
    # ------------------------------------------------------------------

    def forward(
        self,
        aev_tensor: torch.Tensor,
        charges: torch.Tensor,
        atom_ids: torch.Tensor,
        bond_edge_index: torch.Tensor,
        bond_edge_attr: torch.Tensor = None,
        sub_mask: torch.Tensor = None,
    ) -> dict:
        """Autoencoder forward pass (training mode).

        Uses unified GINEConv processing over the entire molecular graph
        (focus sub + core + other reference subs). The sub_mask parameter
        is accepted for backward compatibility but ignored; all atoms are 
        processed together.

        Returns:
            dict with keys:
                'input': [N, input_dim] original input features
                'reconstruction': [N, input_dim] reconstructed features
                'embedding': [embedding_dim] final pooled embedding
        """
        x_in = self._build_input(aev_tensor, charges, atom_ids)  # [N, input_dim]

        # Unified forward: process entire molecule
        x_hidden = self._unified_forward(
            aev_tensor, charges, atom_ids, bond_edge_index, bond_edge_attr
        )  # [N, hidden_dim]

        # Reconstruct all atoms from hidden representation
        reconstruction = self.decoder(x_hidden)  # [N, input_dim]

        # Pool to final embedding via attention
        aggregated = self.attention_pool(x_hidden)  # [hidden_dim]
        embedding = self.bottleneck(aggregated)  # [embedding_dim]

        return {
            'input': x_in,
            'reconstruction': reconstruction,
            'sub_mask': sub_mask,  # kept for compatibility, not used
            'embedding': embedding,
        }

    def encode(
        self,
        aev_tensor: torch.Tensor,
        charges: torch.Tensor,
        atom_ids: torch.Tensor,
        bond_edge_index: torch.Tensor,
        bond_edge_attr: torch.Tensor = None,
        sub_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode a molecule to a pooled [embedding_dim] vector (inference).
        
        Args:
            aev_tensor: [N, 2288] Atomic Environment Vectors
            charges: [N] or [N, 1] partial charges
            atom_ids: [N] atom type IDs
            bond_edge_index: [2, num_bonds] edge indices
            bond_edge_attr: [num_bonds] or [num_bonds, 1] bond attributes
            sub_mask: [N] bool mask (accepted for compatibility, ignored)
            
        Returns:
            [embedding_dim] final embedding vector
        """
        # Unified forward
        x_hidden = self._unified_forward(
            aev_tensor, charges, atom_ids, bond_edge_index, bond_edge_attr
        )  # [N, hidden_dim]

        # Pool and project to embedding
        aggregated = self.attention_pool(x_hidden)  # [hidden_dim]
        embedding = self.bottleneck(aggregated)  # [embedding_dim]
        
        return embedding

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_encoder(self, path: str) -> None:
        """Save the encoder layers as an AtomBondGNN-compatible checkpoint.

        The decoder (``self.decoder``) is excluded.  The saved file can be
        loaded directly into :class:`~mllf.cb.deepset.AtomBondGNN` via::

            model = AtomBondGNN(aev_length=..., ...)
            ckpt  = torch.load(path, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])

        Args:
            path: Destination file path.
        """
        # Filter out decoder weights — remaining keys match AtomBondGNN exactly.
        encoder_state = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith('decoder.')
        }
        torch.save(
            {
                'model_class': 'AtomBondGNN',
                'state_dict': encoder_state,
                'aev_length': self.aev_length,
                'num_atom_types': self.num_atom_types,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'include_charge': self.include_charge,
                'include_atom_id': self.include_atom_id,
                'bond_attr_dim': self.bond_attr_dim,
                'num_gin_layers': self.num_gin_layers,
            },
            path,
        )
        print(f"AtomBondGNN encoder saved to {path}")


def load_pretrained_atombondgnn(encoder_path: str, freeze_weights: bool = True):
    """Load a pretrained AtomBondGNN encoder from a checkpoint.

    The returned model is a :class:`~mllf.cb.deepset.AtomBondGNN` instance
    and can be used directly as ``deepset_model`` in any call to
    :func:`~mllf.cb.graph_utils.build_pyg_graph_from_mllf_graph`.

    Args:
        encoder_path: Path to checkpoint saved by
            :meth:`AtomBondGNNAutoencoder.save_encoder`.
        freeze_weights: If True, calls ``requires_grad_(False)`` (default: True).

    Returns:
        Frozen (or unfrozen) :class:`~mllf.cb.deepset.AtomBondGNN`.
    """
    from mllf.cb.deepset import AtomBondGNN

    ckpt = torch.load(encoder_path, weights_only=False, map_location='cpu')
    if ckpt.get('model_class') != 'AtomBondGNN':
        raise ValueError(
            f"Checkpoint at {encoder_path!r} was not saved by "
            "AtomBondGNNAutoencoder.save_encoder(). "
            f"model_class={ckpt.get('model_class')!r}"
        )

    model = AtomBondGNN(
        aev_length=ckpt['aev_length'],
        num_atom_types=ckpt['num_atom_types'],
        embedding_dim=ckpt['embedding_dim'],
        hidden_dim=ckpt['hidden_dim'],
        include_charge=ckpt['include_charge'],
        include_atom_id=ckpt['include_atom_id'],
        bond_attr_dim=ckpt.get('bond_attr_dim', 1),
        num_gin_layers=ckpt.get('num_gin_layers', 2),
    )
    model.load_state_dict(ckpt['state_dict'])
    if freeze_weights:
        model.requires_grad_(False)

    print(f"Loaded pretrained AtomBondGNN from {encoder_path}")
    print(f"  aev_length={ckpt['aev_length']}, hidden_dim={ckpt['hidden_dim']}, "
          f"embedding_dim={ckpt['embedding_dim']}")
    print(f"  Frozen: {freeze_weights}")
    return model
