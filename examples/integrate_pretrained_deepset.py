"""
Example integration of pretrained DeepSet into the RGCN workflow.

This shows how to modify compute_deepset_embedding_for_node() in graph_utils.py
to use a pretrained autoencoder instead of the randomly initialized DeepSet.
"""

import torch
from pathlib import Path
from typing import Optional

from mllf.cb.deepset_autoencoder import load_pretrained_deepset
from mllf.cb.aev_processor import get_atom_features_with_context


# Global cache for pretrained models (load once, use many times)
_PRETRAINED_MODELS = {}


def load_or_get_pretrained_deepset(
    encoder_path: str,
    freeze_weights: bool = True
):
    """Load a pretrained DeepSet model (cached).
    
    Args:
        encoder_path: Path to pretrained encoder checkpoint
        freeze_weights: Whether to freeze the encoder weights
        
    Returns:
        PretrainedDeepSet model
    """
    if encoder_path not in _PRETRAINED_MODELS:
        print(f"Loading pretrained DeepSet from {encoder_path}")
        _PRETRAINED_MODELS[encoder_path] = load_pretrained_deepset(
            encoder_path, 
            freeze_weights=freeze_weights
        )
    return _PRETRAINED_MODELS[encoder_path]


def compute_pretrained_deepset_embedding(
    pdb_path: str,
    rtf_entry: dict,
    system_name: Optional[str] = None,
    pretrained_models_dir: str = '/home/pbowling/mllf/pretraining_output/trained_models',
    freeze_weights: bool = True,
    prep_dir: Optional[str] = None,
    protein_pdb: Optional[str] = None,
    solvent_state: str = 'solv',
    aev_cutoff: float = 5.1,
):
    """Compute node embedding using a pretrained DeepSet encoder.
    
    This is a drop-in replacement for compute_deepset_embedding_for_node()
    that uses pretrained weights instead of random initialization.
    
    Args:
        pdb_path: Path to substituent PDB file
        rtf_entry: RTF metadata dict
        system_name: Name of system (e.g., '1benz_solvent_group1')
                     If None, uses a default pretrained model
        pretrained_models_dir: Root directory for pretrained models
        freeze_weights: Whether to freeze encoder weights
        prep_dir: Prep directory for spatial filtering
        protein_pdb: Protein PDB path (if applicable)
        solvent_state: Solvent state ('solv', 'gas', 'protein')
        aev_cutoff: AEV cutoff distance
        
    Returns:
        torch.Tensor: [embedding_dim] node embedding (default: 64D)
    """
    # Determine which pretrained model to use
    if system_name:
        encoder_path = Path(pretrained_models_dir) / system_name / 'best_encoder.pt'
    else:
        # Fall back to a default model (e.g., trained on combined data)
        encoder_path = Path(pretrained_models_dir) / 'default' / 'best_encoder.pt'
    
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Pretrained encoder not found: {encoder_path}\n"
            f"Run pretraining first: python examples/run_deepset_pretraining.py"
        )
    
    # Load pretrained model (cached)
    pretrained_deepset = load_or_get_pretrained_deepset(
        str(encoder_path),
        freeze_weights=freeze_weights
    )
    
    # Extract atom features (AEV + charges + context)
    atom_feats = get_atom_features_with_context(
        pdb_path=pdb_path,
        rtf_entry=rtf_entry,
        prep_dir=prep_dir,
        protein_pdb=protein_pdb,
        solvent_state=solvent_state,
        aev_cutoff=aev_cutoff,
        include_charges=True,
        include_atom_ids=False,  # Not needed for pretrained model
    )
    
    # Concatenate AEV + charges
    aevs = atom_feats['aevs']  # [num_atoms, 2288]
    charges = atom_feats.get('charges')
    
    if charges is None:
        # Fall back to zeros if no charges
        charges = torch.zeros(len(aevs), dtype=torch.float32)
    
    # Ensure charges are 2D: [num_atoms, 1]
    if charges.dim() == 1:
        charges = charges.unsqueeze(1)
    
    # Concatenate: [num_atoms, 2289]
    atom_features = torch.cat([aevs, charges], dim=1)
    
    # Get pretrained embedding with max-pooling
    with torch.no_grad() if freeze_weights else torch.enable_grad():
        node_embedding = pretrained_deepset(atom_features)
    
    return node_embedding


# Example usage in graph_utils.py:
"""
def compute_deepset_embedding_for_node(
    pdb_path: str,
    rtf_entry: dict,
    deepset_model: Optional[nn.Module] = None,
    prep_dir: Optional[str] = None,
    protein_pdb: Optional[str] = None,
    solvent_state: str = 'solv',
    aev_cutoff: float = 5.1,
    use_pretrained: bool = True,           # NEW
    pretrained_system: Optional[str] = None,  # NEW
):
    if use_pretrained:
        # Use pretrained encoder
        return compute_pretrained_deepset_embedding(
            pdb_path=pdb_path,
            rtf_entry=rtf_entry,
            system_name=pretrained_system,
            prep_dir=prep_dir,
            protein_pdb=protein_pdb,
            solvent_state=solvent_state,
            aev_cutoff=aev_cutoff,
        )
    else:
        # Original randomly initialized DeepSet
        if deepset_model is None:
            from mllf.cb.deepset import DeepSet
            deepset_model = DeepSet()
        
        atom_feats = get_atom_features_with_context(...)
        # ... existing code
"""


if __name__ == '__main__':
    # Example: Compute embedding for a substituent
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python integrate_pretrained.py <pdb_path> <rtf_path> [system_name]")
        sys.exit(1)
    
    pdb_path = sys.argv[1]
    rtf_path = sys.argv[2]
    system_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Parse RTF
    from mllf.file_handling.read_rtf import parse_rtf_file
    rtf_entry = parse_rtf_file(rtf_path)
    
    # Compute embedding
    embedding = compute_pretrained_deepset_embedding(
        pdb_path=pdb_path,
        rtf_entry=rtf_entry,
        system_name=system_name,
    )
    
    print(f"Generated embedding: shape={embedding.shape}")
    print(f"First 10 values: {embedding[:10].tolist()}")
