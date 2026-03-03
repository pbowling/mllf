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


def compute_deepset_embedding_for_node(
    node_idx: int,
    g,
    deepset_model,
    pdb_dir: str,
    pdb_pattern: str = "site{site}_sub{sub}.pdb",
    rtf_results: Optional[Dict] = None,
    prep_dir: Optional[str] = None,
    protein_pdb: Optional[str] = None,
    solvent_state: Optional[str] = None,
    aev_cutoff: float = 5.1
):
    """Compute DeepSet embedding for a single node/substituent.
    
    This implements Steps 1-3 of the 4-step pipeline:
    1. Extract atom-level features (AEV + charge + atom_id) from PDB
    2. Pass through shared MLP
    3. Max-pool to get fixed-size substituent embedding
    
    For multi-site systems, when prep_dir is provided, this uses spatial filtering
    to include reference substituents from other sites and protein atoms within
    the AEV cutoff distance for accurate molecular context.
    
    Args:
        node_idx: Node index in graph
        g: Graph object with node metadata
        deepset_model: Trained DeepSetFeatureExtractor model
        pdb_dir: Directory containing PDB files
        pdb_pattern: Pattern for PDB filenames (default: "site{site}_sub{sub}.pdb")
        rtf_results: Optional dict of RTF parsed data for extracting charges
        prep_dir: Optional prep directory for multi-site spatial filtering
        protein_pdb: Optional protein PDB file path (for protein phase systems)
        solvent_state: Optional solvent state ('solv', 'gas', or 'protein')
        aev_cutoff: Distance cutoff in Angstroms for spatial filtering (default: 5.1 Å)
        
    Returns:
        torch.Tensor: [embedding_dim] substituent embedding
    """
    from .aev_processor import (
        get_atom_features, get_atom_features_with_context,
        detect_minimized_pdb, extract_environment_atoms_from_minimized,
    )
    from pathlib import Path

    # Get node metadata
    node_info = g.get_node_info(node_idx) if hasattr(g, 'get_node_info') else {}
    site = node_info.get('site')
    sub  = node_info.get('sub')

    if site is None or sub is None:
        raise ValueError(f"Node {node_idx} missing site or sub metadata")

    # Construct PDB paths
    pdb_filename = pdb_pattern.format(site=site, sub=sub)
    pdb_path = os.path.join(pdb_dir, pdb_filename)

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Get RTF entry for charges if available
    rtf_entry = None
    if rtf_results is not None:
        rtf_key = f"site{site}_sub{sub}"
        rtf_entry = rtf_results.get(rtf_key)

    # Determine if we should use context-aware AEV computation
    use_context = prep_dir is not None

    if use_context:
        prep_path = Path(prep_dir)
        core_pdb = prep_path / 'core.pdb'

        if not core_pdb.exists():
            warnings.warn(f"core.pdb not found in {prep_dir}, falling back to single-PDB AEV computation")
            use_context = False
        else:
            # ------------------------------------------------------------------
            # Try to build environment context from minimized.pdb first.
            # This gives the most accurate AEV context for both protein and
            # solvent phase systems because it uses the post-minimization
            # coordinates of the entire simulated system.
            # ------------------------------------------------------------------
            sub_frag_pdb = prep_path / f'site{site}_sub{sub}_frag.pdb'
            min_pdb = detect_minimized_pdb(prep_path)

            protein_context   = None  # pre-parsed tuple for protein_pdb arg
            solvent_ctx       = None  # pre-parsed tuple for solvent_context arg
            effective_protein = protein_pdb  # fallback path string

            if min_pdb and sub_frag_pdb.exists():
                env_ctx = extract_environment_atoms_from_minimized(
                    minimized_pdb=min_pdb,
                    sub_pdb=sub_frag_pdb,
                    core_pdb=core_pdb,
                    aev_cutoff=aev_cutoff,
                    prep_dir=prep_path,
                )
                if env_ctx is not None:
                    if solvent_state == 'protein':
                        protein_context   = env_ctx
                        effective_protein = protein_context
                    elif solvent_state in ('solvent', 'solv', 'water'):
                        solvent_ctx = env_ctx
                    # vacuum/gas: env_ctx not used (no extra environment atoms)
                elif solvent_state == 'protein' and protein_pdb is None:
                    # minimized.pdb gave nothing — look for standalone protein PDB
                    default_protein = prep_path / 'protein.pdb'
                    if default_protein.exists():
                        effective_protein = str(default_protein)
                        warnings.warn("Using protein.pdb from prep directory for AEV spatial filtering")
                    else:
                        warnings.warn(
                            f"solvent_state is 'protein' but no environment atoms found in "
                            f"minimized.pdb and no protein.pdb in prep directory."
                        )
            elif solvent_state == 'protein' and protein_pdb is None:
                # No minimized.pdb and no sub_frag — try standalone protein PDB
                default_protein = prep_path / 'protein.pdb'
                if default_protein.exists():
                    effective_protein = str(default_protein)
                    warnings.warn("Using protein.pdb from prep directory for AEV spatial filtering")
                else:
                    warnings.warn(
                        f"solvent_state is 'protein' but no protein PDB found. "
                        f"Specify protein_pdb in config or add protein.pdb to prep directory."
                    )

            atom_feats = get_atom_features_with_context(
                substituent_pdb=pdb_path,
                core_pdb=str(core_pdb),
                protein_pdb=effective_protein,
                solvent_context=solvent_ctx,
                rtf_entry=rtf_entry,
                include_charges=deepset_model.include_charge,
                include_atom_ids=deepset_model.include_atom_id,
                prep_dir=prep_dir,
                aev_cutoff=aev_cutoff,
            )
    
    if not use_context:
        # Extract atom-level features (Step 1) - single PDB
        atom_feats = get_atom_features(
            pdb_path,
            rtf_entry=rtf_entry,
            include_charges=deepset_model.include_charge,
            include_atom_ids=deepset_model.include_atom_id
        )
    
    # Pass through DeepSet model (Steps 2-3: MLP + max-pool)
    with torch.no_grad():
        embedding = deepset_model(
            aev_tensor=atom_feats['aevs'],
            charges=atom_feats.get('charges'),
            atom_ids=atom_feats.get('atom_ids')
        )
    
    return embedding


def compute_deepset_embeddings_for_graph(
    g,
    deepset_model,
    pdb_dir: str,
    pdb_pattern: str = "site{site}_sub{sub}.pdb",
    rtf_results: Optional[Dict] = None,
    prep_dir: Optional[str] = None,
    protein_pdb: Optional[str] = None,
    solvent_state: Optional[str] = None,
    aev_cutoff: float = 5.1
) -> torch.Tensor:
    """Compute DeepSet embeddings for all nodes in a graph.
    
    Args:
        g: Graph object with node metadata
        deepset_model: Trained DeepSetFeatureExtractor model
        pdb_dir: Directory containing PDB files
        pdb_pattern: Pattern for PDB filenames
        rtf_results: Optional dict of RTF parsed data
        prep_dir: Optional prep directory for multi-site spatial filtering
        protein_pdb: Optional protein PDB file path (for protein phase systems)
        solvent_state: Optional solvent state ('solv', 'gas', or 'protein')
        aev_cutoff: Distance cutoff in Angstroms for spatial filtering (default: 5.1 Å)
        
    Returns:
        torch.Tensor: [num_nodes, embedding_dim] embeddings for all substituents
    """
    embeddings = []
    
    for node_idx in range(g.num_nodes):
        try:
            embedding = compute_deepset_embedding_for_node(
                node_idx, g, deepset_model, pdb_dir, pdb_pattern, rtf_results,
                prep_dir=prep_dir,
                protein_pdb=protein_pdb,
                solvent_state=solvent_state,
                aev_cutoff=aev_cutoff
            )
            embeddings.append(embedding)
        except Exception as e:
            warnings.warn(f"Failed to compute DeepSet embedding for node {node_idx}: {e}")
            # Use zero embedding as fallback
            embedding_dim = deepset_model.atom_mlp[-1].out_features
            embeddings.append(torch.zeros(embedding_dim))
    
    return torch.stack(embeddings, dim=0)


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
    deepset_model = None,
    pdb_dir: Optional[str] = None,
    pdb_pattern: str = "site{site}_sub{sub}.pdb",
    rtf_results: Optional[Dict] = None,
    use_deepset_only: bool = False,
    prep_dir: Optional[str] = None,
    protein_pdb: Optional[str] = None,
    solvent_state: Optional[str] = None,
    aev_cutoff: float = 5.1
) -> Tuple[object, dict]:
    """Convert a Graph-like object `g` into a PyG Data object and metadata.

    We expand each undirected graph edge into up to four directed relation edges,
    one per bias type. The default `relation_names` is ['linear','quadratic','skw','end'].
    
    Node features are constructed from metadata. There are two modes:
    
    1. Standard mode (deepset_model=None):
       - Charge, environment type (solvent/protein)
       - Element count encoding (coarse chemical composition)
       - Atom type count encoding (fine CHARMM type composition)
       
    2. DeepSet mode (deepset_model provided):
       - DeepSet substituent embedding (from 3D atomic structure + charges)
       - Implements Step 4 of the 4-step pipeline
       - Spatial context (protein atoms) already encoded in AEVs
       - Charge already included in DeepSet atom features
       - Solvent state NOT included (constant within graph, provides no differentiation)
       
    The use_deepset_only parameter is deprecated and has no effect (DeepSet mode
    always uses only DeepSet embeddings).
    
    The vocabularies are loaded from CHARMM CGenFF toppar file by default.

    Args:
        g: Graph object with node metadata
        relation_names: List of base relation types (default: ['linear', 'quadratic', 'skew', 'end'])
        toppar_dir: Path to toppar directory (None uses package default)
        toppar_files: List of specific toppar filenames to include.
                     Default: ['top_all36_cgenff.rtf'] (CGenFF only)
        warn_missing_types: If True, warn when sub RTF files contain atom types not in vocabulary
        deepset_model: Optional DeepSetFeatureExtractor model for 3D structural features
        pdb_dir: Directory containing PDB files (required if deepset_model provided)
        pdb_pattern: Pattern for PDB filenames (default: "site{site}_sub{sub}.pdb")
        rtf_results: Optional dict of RTF parsed data for charge extraction
        use_deepset_only: DEPRECATED - has no effect (DeepSet mode always uses only embeddings)
        prep_dir: Optional prep directory for multi-site spatial filtering
        protein_pdb: Optional protein PDB file path (for protein phase systems)
        solvent_state: Optional solvent state ('solv', 'gas', or 'protein')
        aev_cutoff: Distance cutoff in Angstroms for spatial filtering (default: 5.1 Å)

    Returns (pyg_data, extras) where extras contain:
        - relation_names: List of all relation type names
        - relation_map: Dict mapping relation names to indices
        - base_relation_map: Dict mapping base types to (fwd, bwd) relation names
        - atom_type_vocab: Dict mapping atom type strings to feature indices
        - element_vocab: Dict mapping element symbols to feature indices
        - deepset_dim: Dimension of DeepSet embeddings (if used)
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

    # Load atom type and element vocabularies from toppar files
    # Default to CGenFF only if no files specified
    if toppar_files is None:
        toppar_files = ['top_all36_cgenff.rtf']
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
    deepset_embeddings = None
    
    # Compute DeepSet embeddings if model provided
    if deepset_model is not None:
        if pdb_dir is None:
            raise ValueError("pdb_dir must be provided when using deepset_model")
        
        deepset_embeddings = compute_deepset_embeddings_for_graph(
            g, deepset_model, pdb_dir, pdb_pattern, rtf_results,
            prep_dir=prep_dir,
            protein_pdb=protein_pdb,
            solvent_state=solvent_state,
            aev_cutoff=aev_cutoff
        )
    
    # Build node features
    for i in range(g.num_nodes):
        meta = g.get_node_info(i) if hasattr(g, 'get_node_info') else {}
        
        if deepset_model is not None:
            # DeepSet mode: Use only molecular embeddings
            # Spatial context (including protein environment) is already encoded in AEVs
            # Charge is already included in DeepSet atom features
            # Solvent state is constant within a graph (provides no node differentiation)
            node_feats.append(deepset_embeddings[i])
        else:
            # Standard mode: Use count-based compositional encoding
            node_feats.append(_node_feature_from_meta(meta, atom_type_vocab, element_vocab, atom_to_element))
    
    x = torch.stack(node_feats, dim=0)

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

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr)
    extras = {
        'relation_names': relation_names,
        'relation_map': rel_to_idx,
        'base_relation_map': base_relation_map,
        'atom_type_vocab': atom_type_vocab,
        'element_vocab': element_vocab,
        'atom_to_element': atom_to_element,
    }
    
    # Add DeepSet info if used
    if deepset_model is not None:
        deepset_dim = deepset_model.atom_mlp[-1].out_features
        extras['deepset_dim'] = deepset_dim
        extras['use_deepset_only'] = True  # Always True now (parameter deprecated)
        extras['node_feature_dim'] = deepset_dim  # Only DeepSet embeddings
    
    return data, extras
