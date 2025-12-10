"""Helpers to convert existing Graph objects to PyTorch Geometric Data.

This module is intentionally defensive: it will try to import the project's
Graph class and fall back to handling networkx-like graph objects.
"""
from typing import Tuple, Optional
import torch
from torch_geometric.data import Data
from .atom_vocab import get_atom_type_vocab



def _node_feature_from_meta(meta: dict, atom_type_vocab: dict = None):
    """Create a numeric vector from node metadata.

    Expected keys in meta: 
    - 'total_charge' (float)
    - 'solvent' (str: 'vacuum'/'gas', 'solvent'/'solv', or 'protein')
    - 'distinct_atom_types' (list of atom type strings, e.g., ['CG2R61', 'HGR61'])
    
    The function returns a 1-D torch.float tensor with features:
    [charge, is_vacuum, is_solvent, is_protein, <multi-hot atom types>]
    
    If atom_type_vocab is provided, distinct_atom_types are encoded as a multi-hot vector.
    Otherwise, they are omitted from the feature vector.
    """
    charge = float(meta.get('total_charge', 0.0))
    
    # Handle solvent as categorical: vacuum/gas, solvent/solv, or protein
    solvent_str = (meta.get('solvent', '') or '').lower()
    is_vacuum = 1.0 if solvent_str in ('vacuum', 'gas', 'vac') else 0.0
    is_solvent = 1.0 if solvent_str in ('solvent', 'solv', 'water', 'aq', 'sol') else 0.0
    is_protein = 1.0 if solvent_str in ('protein', 'prot') else 0.0
    
    base_features = [charge, is_vacuum, is_solvent, is_protein]
    
    # Encode distinct atom types as multi-hot if vocabulary provided
    if atom_type_vocab is not None:
        distinct_types = meta.get('distinct_atom_types', [])
        if not isinstance(distinct_types, (list, tuple)):
            distinct_types = []
        
        # Create multi-hot encoding
        vocab_size = len(atom_type_vocab)
        atom_encoding = [0.0] * vocab_size
        for atom_type in distinct_types:
            if atom_type in atom_type_vocab:
                idx = atom_type_vocab[atom_type]
                atom_encoding[idx] = 1.0
        
        base_features.extend(atom_encoding)
    
    return torch.tensor(base_features, dtype=torch.get_default_dtype())


def build_pyg_graph_from_mllf_graph(g, relation_names: list = None, toppar_dir: Optional[str] = None, toppar_files: list = None, warn_missing_types: bool = True) -> Tuple[object, dict]:
    """Convert a Graph-like object `g` into a PyG Data object and metadata.

    We expand each undirected graph edge into up to four directed relation edges,
    one per bias type. The default `relation_names` is ['linear','quadratic','skew','end'].
    
    Node features are constructed from metadata and include:
    - Charge, environment type (vacuum/solvent/protein)
    - Multi-hot encoding of distinct atom types
    
    The atom type vocabulary is loaded from CHARMM toppar files (default: package toppar/ directory).
    This ensures consistent feature dimensions across different graphs and training runs.

    Args:
        g: Graph object with node metadata
        relation_names: List of base relation types (default: ['linear', 'quadratic', 'skew', 'end'])
        toppar_dir: Path to toppar directory (None uses package default)
        toppar_files: List of specific toppar filenames to include (e.g., ['top_all36_cgenff.rtf'])
        warn_missing_types: If True, warn when sub RTF files contain atom types not in vocabulary

    Returns (pyg_data, extras) where extras contain:
        - relation_names: List of all relation type names
        - relation_map: Dict mapping relation names to indices
        - base_relation_map: Dict mapping base types to (fwd, bwd) relation names
        - atom_type_vocab: Dict mapping atom type strings to feature indices
    """

    if relation_names is None:
        base_relation_names = ['linear', 'quadratic', 'skew', 'end']
    else:
        base_relation_names = list(relation_names)

    # Expand base relations into directed relation types: e.g. 'linear_fwd', 'linear_bwd'
    relation_names = []
    base_relation_map = {}
    for r in base_relation_names:
        fwd = f"{r}_fwd"
        bwd = f"{r}_bwd"
        base_relation_map[r] = (fwd, bwd)
        relation_names.append(fwd)
        relation_names.append(bwd)

    rel_to_idx = {r: i for i, r in enumerate(relation_names)}

    # Load atom type vocabulary from toppar files
    atom_type_vocab = get_atom_type_vocab(toppar_dir, toppar_files)
    
    # Check for missing atom types in graph if requested
    if warn_missing_types and atom_type_vocab:
        missing_types = set()
        for i in range(g.num_nodes):
            meta = g.get_node_info(i) if hasattr(g, 'get_node_info') else {}
            distinct_types = meta.get('distinct_atom_types', [])
            for atom_type in distinct_types:
                if atom_type not in atom_type_vocab:
                    missing_types.add(atom_type)
        
        if missing_types:
            import warnings
            sorted_missing = sorted(missing_types)
            warnings.warn(
                f"Found {len(sorted_missing)} atom type(s) in substituent RTF files that are not in the vocabulary: "
                f"{sorted_missing}. These atom types will not be encoded in node features. "
                f"Consider adding the corresponding toppar file(s) to the vocabulary.",
                UserWarning
            )
    
    # collect node features with atom type encoding
    node_feats = []
    for i in range(g.num_nodes):
        meta = g.get_node_info(i) if hasattr(g, 'get_node_info') else {}
        node_feats.append(_node_feature_from_meta(meta, atom_type_vocab))
    x = torch.stack(node_feats, dim=0)

    # expand edges: for each undirected (i,j) and for each bias that is allowed.
    # For each base bias we create two directed relation types so that A->B and B->A
    # are represented by distinct relation ids and can be learned separately.
    src = []
    dst = []
    edge_type_list = []
    edge_attr_list = []

    # Graph.edges is stored as dict keyed by (i,j) -> EdgeCoeffs
    for (i, j), coeffs in getattr(g, 'edges', {}).items():
        # determine which bias types are allowed from edge_mask (if present)
        mask = None
        if hasattr(g, 'edge_mask'):
            mask = g.edge_mask.get((i, j))
        for bias in base_relation_names:
            allowed = True if mask is None else bool(mask.get(bias, False))
            if not allowed:
                continue
            fwd_name, bwd_name = base_relation_map[bias]
            fwd_idx = rel_to_idx[fwd_name]
            bwd_idx = rel_to_idx[bwd_name]
            # add directed edge i->j as the forward relation for this bias
            src.append(int(i))
            dst.append(int(j))
            edge_type_list.append(fwd_idx)
            # edge_attr: only include one-hot over directed relation types
            k = len(relation_names)
            one_hot = torch.zeros((k,), dtype=torch.get_default_dtype())
            one_hot[fwd_idx] = 1.0
            edge_attr_list.append(one_hot)
            # add reverse direction j->i as the backward relation type
            src.append(int(j))
            dst.append(int(i))
            edge_type_list.append(bwd_idx)
            one_hot_r = torch.zeros((k,), dtype=torch.get_default_dtype())
            one_hot_r[bwd_idx] = 1.0
            edge_attr_list.append(one_hot_r)

    k = len(relation_names)
    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_attr = torch.zeros((0, k), dtype=torch.get_default_dtype())
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        edge_attr = torch.stack(edge_attr_list, dim=0)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr)
    extras = {
        'relation_names': relation_names,
        'relation_map': rel_to_idx,
        'base_relation_map': base_relation_map,
        'atom_type_vocab': atom_type_vocab,
    }
    return data, extras
