"""Helpers to convert existing Graph objects to PyTorch Geometric Data.

This module is intentionally defensive: it will try to import the project's
Graph class and fall back to handling networkx-like graph objects.
"""
from typing import Tuple, Optional, List, Dict
import torch
import os
import warnings
from torch_geometric.data import Data
from .atom_vocab import get_atom_type_vocab
from pathlib import Path as _Path


def _find_protein_pdb(prep_path) -> Optional[str]:
    """Return path string to protein PDB in prep_path, or None.

    Checks (in order): protein.pdb, proa.pdb, proa_*.pdb (first match).
    """
    prep = _Path(prep_path)
    for name in ('protein.pdb', 'proa.pdb'):
        p = prep / name
        if p.exists():
            return str(p)
    # proa variants like proa_i315.pdb
    for p in sorted(prep.glob('proa_*.pdb')):
        return str(p)
    return None


def build_directed_pairs(nsubs_per_site: List[int]) -> List[Tuple[int, int]]:
    """Build list of directed pairs for all substituents within each site.
    
    Generates BOTH directions (i→j and j→i) within each site to allow
    independent predictions for skew and end biases.
    No cross-site pairs are generated.
    
    This is useful for both graph-based and pairwise approaches when you need
    all directed edges within sites for training or prediction.
    
    Args:
        nsubs_per_site: List of substituent counts per site
    
    Returns:
        List of (i, j) pairs including both directions within the same site
        
    Example:
        >>> build_directed_pairs([2, 3])  # Site 1: 2 subs, Site 2: 3 subs
        [(0, 1), (1, 0), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3)]
    """
    pairs = []
    offset = 0
    
    for nsubs in nsubs_per_site:
        # Generate both directions for all pairs within site
        for i in range(nsubs):
            for j in range(nsubs):
                if i != j:  # Skip diagonal (self-pairs)
                    pairs.append((offset + i, offset + j))
        offset += nsubs
    
    return pairs



def _node_feature_from_meta(meta: dict, atom_type_vocab: dict = None, element_vocab: dict = None, atom_to_element: dict = None):
    """Create a numeric vector from node metadata.

    Expected keys in meta: 
    - 'total_charge' (float) - from RTF file, summed over all atoms
    - 'solvent' (str: 'solvent'/'solv' or 'protein') - environmental context
    - 'distinct_atom_types' (list of atom type strings, e.g., ['CG2R61', 'HGR61', 'CG2R61'])
    
    The function returns a 1-D torch.float tensor with features:
    [charge, is_solvent, is_protein, <element counts>, <atom type counts>]
    
    If atom_type_vocab, element_vocab, and atom_to_element mapping are provided, distinct_atom_types
    are encoded as two separate count vectors:
    - Element counts: how many atoms of each element (e.g., 6 C, 4 H, 1 O)
    - Atom type counts: how many of each specific atom type (e.g., 3 CG2R61, 2 HGR61)
    
    Elements are extracted from atom types using the atom_to_element mapping.
    
    This encoding captures both composition (counts) and chemical diversity:
    - Coarse counts: total atoms per element
    - Fine counts: atoms per specific CHARMM type
    - More informative than binary presence/absence
    """
    charge = float(meta.get('total_charge', 0.0))
    
    # Handle solvent as categorical: solvent/solv or protein
    solvent_str = (meta.get('solvent', '') or '').lower()
    is_solvent = 1.0 if solvent_str in ('solvent', 'solv', 'water', 'aq', 'sol') else 0.0
    is_protein = 1.0 if solvent_str in ('protein', 'prot') else 0.0
    
    base_features = [charge, is_solvent, is_protein]
    
    # Encode elements and atom types with counts if vocabularies provided
    if element_vocab is not None and atom_type_vocab is not None and atom_to_element is not None:
        distinct_types = meta.get('distinct_atom_types', [])
        if not isinstance(distinct_types, (list, tuple)):
            distinct_types = []
        
        # Create element count encoding (how many atoms of each element)
        element_encoding = [0.0] * len(element_vocab)
        # Create atom type count encoding (how many of each specific type)
        atom_type_encoding = [0.0] * len(atom_type_vocab)
        
        # Count occurrences of each atom type and element
        for atom_type in distinct_types:
            # Increment atom type count
            if atom_type in atom_type_vocab:
                idx = atom_type_vocab[atom_type]
                atom_type_encoding[idx] += 1.0
                
                # Increment element count using the mapping
                element = atom_to_element.get(atom_type)
                if element and element in element_vocab:
                    elem_idx = element_vocab[element]
                    element_encoding[elem_idx] += 1.0
        
        base_features.extend(element_encoding)
        base_features.extend(atom_type_encoding)
    
    return torch.tensor(base_features, dtype=torch.get_default_dtype())


def build_pyg_graph_from_mllf_graph(
    g, 
    relation_names: list = None, 
    toppar_dir: Optional[str] = None, 
    toppar_files: list = None, 
    warn_missing_types: bool = True,
    pdb_dir: Optional[str] = None,
    pdb_pattern: str = "site{site}_sub{sub}.pdb",
    rtf_results: Optional[Dict] = None,
    prep_dir: Optional[str] = None,
    protein_pdb: Optional[str] = None,
    solvent_state: Optional[str] = None,
    solvent_pdb: Optional[str] = None
) -> Tuple[object, dict]:
    """Convert a Graph-like object `g` into a PyG Data object and metadata.

    We expand each undirected graph edge into up to four directed relation edges,
    one per bias type. The default `relation_names` is ['linear','quadratic','skw','end'].
    
    Standard mode:
       - Charge, environment type (solvent/protein)
       - Element count encoding (coarse chemical composition)
       - Atom type count encoding (fine CHARMM type composition)
    
    The vocabularies are loaded from CHARMM CGenFF toppar file by default.

    Args:
        g: Graph object with node metadata
        relation_names: List of base relation types (default: ['linear', 'quadratic', 'skew', 'end'])
        toppar_dir: Path to toppar directory (None uses package default)
        toppar_files: List of specific toppar filenames to include.
                     Default: ['top_all36_cgenff.rtf'] (CGenFF only)
        warn_missing_types: If True, warn when sub RTF files contain atom types not in vocabulary
        pdb_dir: Directory containing PDB files
        pdb_pattern: Pattern for PDB filenames (default: "site{site}_sub{sub}.pdb")
        rtf_results: Optional dict of RTF parsed data for charge extraction
        prep_dir: Optional prep directory for multi-site spatial filtering
        protein_pdb: Optional protein PDB file path (for protein phase systems)
        solvent_state: Optional solvent state ('solv', 'gas', or 'protein')
        solvent_pdb: Optional solvent PDB file path

    Returns (pyg_data, extras) where extras contain:
        - relation_names: List of all relation type names
        - relation_map: Dict mapping relation names to indices
        - base_relation_map: Dict mapping base types to (fwd, bwd) relation names
        - atom_type_vocab: Dict mapping atom type strings to feature indices
        - element_vocab: Dict mapping element symbols to feature indices
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

    # Load atom type and element vocabularies from toppar files.
    # Pass toppar_files=None when unspecified so all .rtf/.str files in the
    # toppar directory are parsed (including custom_ligand_types.rtf).
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(toppar_dir, toppar_files)
    
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
    
    # collect node features
    node_feats = []
    site_ids = []  # 0-indexed site assignment per node
    
    
    # Build node features
    for i in range(g.num_nodes):
        meta = g.get_node_info(i) if hasattr(g, 'get_node_info') else {}
        # site is 1-indexed in node metadata; store as 0-indexed for indexing
        site_ids.append(max(0, meta.get('site', 1) - 1))
        
        node_feats.append(_node_feature_from_meta(meta, atom_type_vocab, element_vocab, atom_to_element))
    
    x = torch.stack(node_feats, dim=0)
    site_index = torch.tensor(site_ids, dtype=torch.long)

    # expand edges: for each undirected (i,j) and for each bias that is allowed.
    # Different bias types have different directionality rules:
    # - Linear: Only FROM reference sub (sub 1) TO other subs (one direction only)
    # - Quadratic: Only upper triangle (one direction: i->j where i<j)
    # - Skew/End: Both directions (i->j and j->i)
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
        
        # Get node metadata for directionality rules
        node_i_info = g.get_node_info(i) if hasattr(g, 'get_node_info') else {}
        node_j_info = g.get_node_info(j) if hasattr(g, 'get_node_info') else {}
        sub_i = node_i_info.get('sub')
        sub_j = node_j_info.get('sub')
        
        for bias in base_relation_names:
            allowed = True if mask is None else bool(mask.get(bias, False))
            if not allowed:
                continue
            
            fwd_name, bwd_name = base_relation_map[bias]
            fwd_idx = rel_to_idx[fwd_name]
            bwd_idx = rel_to_idx[bwd_name]
            k = len(relation_names)
            
            # Determine directionality based on bias type
            create_forward = True
            create_backward = True
            
            if bias == 'linear':
                # Linear: Only from reference sub (sub 1) to other subs
                # If i is sub 1: create i->j only
                # If j is sub 1: create j->i only
                # If neither is sub 1: skip (edge_mask should have disabled this)
                if sub_i == 1 and sub_j != 1:
                    create_forward = True
                    create_backward = False
                elif sub_j == 1 and sub_i != 1:
                    create_forward = False
                    create_backward = True
                else:
                    # Both are sub 1 or neither is sub 1 - shouldn't happen with proper edge_mask
                    continue
                    
            elif bias == 'quadratic':
                # Quadratic: Only upper triangle (i < j)
                if i < j:
                    create_forward = True
                    create_backward = False
                else:
                    create_forward = False
                    create_backward = True
                    
            # Skew and end: both directions (default behavior)
            
            # Create forward edge (i->j)
            if create_forward:
                src.append(int(i))
                dst.append(int(j))
                edge_type_list.append(fwd_idx)
                one_hot = torch.zeros((k,), dtype=torch.get_default_dtype())
                one_hot[fwd_idx] = 1.0
                edge_attr_list.append(one_hot)
            
            # Create backward edge (j->i)
            if create_backward:
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

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr,
                site_index=site_index)
    extras = {
        'relation_names': relation_names,
        'relation_map': rel_to_idx,
        'base_relation_map': base_relation_map,
        'atom_type_vocab': atom_type_vocab,
        'element_vocab': element_vocab,
        'atom_to_element': atom_to_element,
    }
    
    return data, extras
