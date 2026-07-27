"""Uni-Mol 512-dimensional molecular representations with environment context.

Constructs complete ligand systems (core + substituent(s) + environment) and computes
Uni-Mol representations trained on 1.1B molecules from PubChem.

Key Features:
- Automatic ligand assembly: combine core + substituent(s) into unified 3D structure
- Multi-site support: include reference subs (site#_sub1) from other sites
- Environment context: add protein/solvent atoms within spatial cutoff (~5 Å)
- Atom limit enforcement: Uni-Mol supports max 256 atoms; enforce via radial filtering
- Format flexibility: handle PDB files and CHARMM CRD coordinates seamlessly
- Embedding caching: save/load computed embeddings to disk for efficient reuse across runs

Functions:
- construct_full_ligand(): Build complete system with optional environment
- get_unimol_representation(): Compute 512D molecular embedding
- get_substituent_unimol_representation(): One-shot API for substituent + context
- get_substituent_unimol_with_environment(): Full API with environment + optional caching
- save_embedding(): Cache embedding to disk with metadata
- load_embedding(): Load cached embedding if available
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import json
import hashlib

from unimol_tools import UniMolRepr

# Import existing infrastructure for coordinate extraction
from mllf.file_handling.read_pdb import (
    parse_pdb_file,
    find_reference_subs_from_other_sites,
    extract_site_number,
    calculate_min_distance,
)
from mllf.cb.aev_processor import (
    get_full_ligand_atom_features,
    _convert_crd_to_tmp_pdb,
)


# Global Uni-Mol representation model (initialized once for efficiency)
_unimol_model = None


def _get_unimol_model(use_cuda: bool = False) -> UniMolRepr:
    """Initialize or retrieve the global Uni-Mol model instance.
    
    The model is initialized once and cached to avoid repeated downloads/loading.
    The Uni-Mol v1 84M model is pretrained on 1.1B molecules from PubChem.
    
    Args:
        use_cuda: Whether to use GPU (default: False for CPU inference)
        
    Returns:
        UniMolRepr: Initialized Uni-Mol representation model
    """
    global _unimol_model
    if _unimol_model is None:
        _unimol_model = UniMolRepr(
            remove_hs=False,  # Keep hydrogen atoms for accurate 3D geometry
            use_cuda=use_cuda,
            batch_size=32,
            model_name='unimolv1',
            model_size='84m',
            max_atoms=256,  # Uni-Mol max atom capacity
        )
    return _unimol_model


def _extract_atom_identifiers_from_pdb(
    pdb_path: str,
    ligand_coords: Optional[np.ndarray] = None,
    ligand_cutoff: float = 8.0,
    return_distances: bool = False,
) -> set:
    """Extract unique atom identifiers from PDB file, optionally filtered by distance.
    
    Each atom is identified by its (file_index, residue_number, chain_id, atom_name)
    tuple, where file_index is the atom's 0-based position among ATOM/HETATM records
    in the file. The file_index is required for uniqueness: (resnum, chain, atomname)
    alone is NOT guaranteed unique — CHARMM PDB output wraps residue numbers for large
    solvent boxes (fixed-width column limit), so many distinct water molecules can
    share the same (resnum, chain, atomname). Since consensus atoms are always built
    from and matched against the SAME physical environment file, the file index is a
    safe, unambiguous identifier.
    
    Args:
        pdb_path: Path to PDB file
        ligand_coords: Optional [N_ligand, 3] coordinates. If provided, only returns
            identifiers for atoms within ligand_cutoff distance.
        ligand_cutoff: Distance cutoff (Å) for filtering atoms (only if ligand_coords provided)
        return_distances: If True and ligand_coords provided, return dict mapping atom_ids to min distances
        
    Returns:
        Set of (file_index, resnum, chain, atomname) tuples for atoms in the PDB
        If return_distances=True: Dict of {atom_id: min_distance_to_ligand}
    """
    atom_ids = set()
    atom_distances = {}  # atom_id -> min distance to ligand
    
    try:
        with open(pdb_path, 'r') as f:
            coords_list = []
            atom_lines = []
            file_idx = 0
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    resnum = int(line[22:26].strip())
                    chain = line[21].strip() or 'A'  # Default to 'A' if blank
                    atomname = line[12:16].strip().upper()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    atom_id = (file_idx, resnum, chain, atomname)
                    atom_ids.add(atom_id)
                    coords_list.append([x, y, z])
                    atom_lines.append(atom_id)
                    file_idx += 1
            
            # If ligand coordinates provided, filter by distance and track distances
            if ligand_coords is not None and len(coords_list) > 0:
                coords_arr = np.array(coords_list, dtype=np.float32)
                ligand_arr = np.array(ligand_coords, dtype=np.float32)
                
                # Calculate minimum distance to any ligand atom
                distances = np.min(
                    np.linalg.norm(coords_arr[:, None, :] - ligand_arr[None, :, :], axis=2),
                    axis=1
                )
                
                # Build filtered set and distances dict
                atom_ids_filtered = set()
                for idx, atom_id in enumerate(atom_lines):
                    if distances[idx] <= ligand_cutoff:
                        atom_ids_filtered.add(atom_id)
                        if return_distances:
                            atom_distances[atom_id] = float(distances[idx])
                
                if return_distances:
                    return atom_distances
                else:
                    return atom_ids_filtered
    
    except Exception as e:
        warnings.warn(f"Could not extract atom identifiers from {pdb_path}: {e}")
    
    if return_distances:
        return {}
    return atom_ids


def _default_protein_pdb_candidates(base_dir: Path) -> List[Path]:
    """Ordered list of default protein environment PDB candidates for a directory.

    Shared by build_environment_consensus() and _load_environment_pdb() so both
    always resolve to the SAME physical protein file for a given prep directory.
    This matters because consensus atom identifiers are (file_index, resnum,
    chain, atomname) tuples that are only meaningful when built from and matched
    against the identical file. Includes the ``proa_*.pdb`` glob pattern used for
    mutant/variant-labeled protein files (e.g. ``proa_i315.pdb``) — previously
    only ``_load_environment_pdb`` searched for this pattern, while
    ``build_environment_consensus`` only checked the exact name ``proa.pdb``,
    causing consensus to be built from an unrelated (usually much smaller)
    solvent file whenever no plain ``proa.pdb``/``protein.pdb`` existed.
    """
    base_dir = Path(base_dir)
    candidates = [
        base_dir / "protein.pdb",
        base_dir / "prot.pdb",
        base_dir / "proa.pdb",
        base_dir / "pdb" / "protein.pdb",  # Handle nested pdb/protein.pdb (e.g., indolizine_prot)
    ]
    # Mutant/variant-labeled protein files (e.g. proa_i315.pdb for point mutants)
    candidates.extend(sorted(base_dir.glob("proa_*.pdb")))
    return candidates


def _default_solvent_pdb_candidates(base_dir: Path) -> List[Path]:
    """Ordered list of default solvent/water environment PDB candidates.

    See _default_protein_pdb_candidates() for why this list is shared between
    build_environment_consensus() and _load_environment_pdb().
    """
    base_dir = Path(base_dir)
    return [
        base_dir / "solvent.pdb",
        base_dir / "solv.pdb",
        base_dir / "waterbox.pdb",
        base_dir / "water.pdb",
        base_dir / "wata.pdb",
        base_dir / "solvent_box.pdb",
        base_dir / "watbox.pdb",
        base_dir / "environment.pdb",
    ]


def build_environment_consensus(
    core_pdb: str,
    prep_dir: Path,
    env_cutoff: float = 8.0,
    custom_search_paths: Optional[List[str]] = None,
) -> Optional[set]:
    """Build environment atom set for the ligand core.
    
    Identifies all environment atoms within env_cutoff distance of the core scaffold.
    This provides a consistent environment representation across pretraining, analysis,
    and online training.
    
    **Strategy:**
    1. Parse core scaffold to get its coordinates
    2. Find environment file in prep_dir
    3. Extract atoms within env_cutoff of the core
    4. Return this set (capped at 256 atoms, keeping closest)
    
    Atoms are identified by (file_index, resnum, chain, atomname) tuples. The
    file_index disambiguates atoms that share (resnum, chain, atomname) — e.g.
    CHARMM's fixed-width PDB columns cause residue numbers to wrap for large
    solvent boxes, so many distinct water molecules can otherwise collide.
    
    Args:
        core_pdb: Path to core scaffold PDB
        prep_dir: Prep directory containing environment PDB files
        env_cutoff: Distance cutoff (Å) for identifying relevant environment (default: 8.0)
        custom_search_paths: Optional custom PDB search paths (from yaml config)
        
    Returns:
        Set of (file_index, resnum, chain, atomname) tuples for environment atoms,
        or None if no environment found
    """
    try:
        # Get core coordinates
        core_coords, _ = parse_pdb_file(str(core_pdb))
        if not core_coords:
            print(f"      [CONSENSUS] No coordinates found in core: {Path(core_pdb).name}")
            return None
        
        core_coords = np.array(core_coords, dtype=np.float32)
        
        # Try to find environment file (custom paths first, then defaults)
        env_pdb_path = None
        if custom_search_paths:
            for pdb_name in custom_search_paths:
                test_path = prep_dir / pdb_name
                if test_path.exists():
                    env_pdb_path = test_path
                    break
        
        if env_pdb_path is None:
            # Try default locations in prep_dir first
            # Priority: protein variants → solvent variants → minimized (fallback)
            # Uses the SAME candidate helpers as _load_environment_pdb() to
            # guarantee both functions resolve to the identical physical file.
            env_candidates = (
                _default_protein_pdb_candidates(prep_dir)
                + _default_solvent_pdb_candidates(prep_dir)
                + [prep_dir / "pre_min.pdb", prep_dir / "minimized.pdb"]
            )
            # Also try parent directory (for combo structures)
            core_parent = Path(core_pdb).parent
            if core_parent != prep_dir:
                env_candidates.extend(
                    _default_protein_pdb_candidates(core_parent)
                    + _default_solvent_pdb_candidates(core_parent)
                    + [core_parent / "pre_min.pdb", core_parent / "minimized.pdb"]
                )
            
            for candidate in env_candidates:
                if candidate.exists():
                    env_pdb_path = candidate
                    break
        
        if env_pdb_path is None:
            # Debug: print available files in prep_dir
            available = list(prep_dir.glob("*.pdb"))
            print(f"      [CONSENSUS] No environment file found in {prep_dir.name} (available: {[p.name for p in available[:3]]}...)")
            return None
        
        # Extract atoms within env_cutoff of the core, tracking distances
        atom_dist_dict = _extract_atom_identifiers_from_pdb(
            str(env_pdb_path),
            ligand_coords=core_coords,
            ligand_cutoff=env_cutoff,
            return_distances=True,
        )
        
        if not atom_dist_dict:
            print(f"      [CONSENSUS] No atoms within {env_cutoff}Å of core in {env_pdb_path.name}")
            return None
        
        # Convert dict to set of atom identifiers
        consensus_atoms = set(atom_dist_dict.keys())
        
        # Cap at 256 atoms - keep closest ones
        if len(consensus_atoms) > 256:
            # Sort by distance and keep 256 closest
            sorted_atoms = sorted(consensus_atoms, key=lambda a: atom_dist_dict[a])
            consensus_atoms = set(sorted_atoms[:256])
            print(f"      [CONSENSUS] Built core consensus: {len(consensus_atoms)} atoms (capped at 256)")
        else:
            print(f"      [CONSENSUS] Built core consensus: {len(consensus_atoms)} atoms")
        
        return consensus_atoms
    
    except Exception as e:
        print(f"      [CONSENSUS] Exception building core consensus: {e}")
        return None


def _filter_environment_by_consensus(
    coords: List[List[float]],
    elements: List[str],
    pdb_path: str,
    consensus_atoms: set,
) -> Optional[tuple]:
    """Filter environment atoms to only include those in consensus set.
    
    Args:
        coords: List of [x,y,z] coordinates from parsed PDB
        elements: List of element symbols matching coords
        pdb_path: Path to original PDB file (to extract atom identifiers)
        consensus_atoms: Set of (file_index, resnum, chain, atomname) tuples to keep
            (see _extract_atom_identifiers_from_pdb for why file_index is required)
        
    Returns:
        (filtered_coords, filtered_elements) tuple, or None if no atoms match consensus
    """
    filtered_coords = []
    filtered_elements = []
    
    try:
        idx = 0
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    if idx < len(elements):
                        resnum = int(line[22:26].strip())
                        chain = line[21].strip() or 'A'
                        atomname = line[12:16].strip().upper()
                        
                        # Check if this atom is in consensus (file_index disambiguates
                        # atoms that share (resnum, chain, atomname), e.g. wrapped
                        # water residue numbers in large solvent boxes)
                        if (idx, resnum, chain, atomname) in consensus_atoms:
                            filtered_coords.append(coords[idx])
                            filtered_elements.append(elements[idx])
                    
                    idx += 1
    
    except Exception as e:
        warnings.warn(f"Could not filter by consensus: {e}")
        return None
    
    if filtered_coords:
        return (filtered_coords, filtered_elements)
    
    return None


def _generate_embedding_cache_key(
    sub_pdb: str,
    core_pdb: str,
    prep_dir: str,
    env_cutoff: float,
    include_other_sites: bool,
    custom_search_paths: Optional[List[str]] = None,
) -> str:
    """Generate a unique cache key for an embedding based on input parameters.
    
    Creates a deterministic hash from the system configuration, allowing embeddings
    to be cached and reused across runs. The hash includes:
    - Paths to core and substituent PDBs
    - Prep directory path
    - Environment cutoff and search configuration
    - Whether reference subs from other sites are included
    
    Args:
        sub_pdb: Path to substituent fragment PDB
        core_pdb: Path to core scaffold PDB
        prep_dir: Prep directory containing environment files
        env_cutoff: Environment distance cutoff in Angstroms
        include_other_sites: Whether reference subs from other sites are included
        custom_search_paths: Optional custom PDB search paths for environment
        
    Returns:
        str: Unique hash key for this embedding configuration
    """
    # Normalize paths to handle variations
    key_components = {
        'sub_pdb': str(Path(sub_pdb).resolve()),
        'core_pdb': str(Path(core_pdb).resolve()),
        'prep_dir': str(Path(prep_dir).resolve()),
        'env_cutoff': float(env_cutoff),
        'include_other_sites': bool(include_other_sites),
        'custom_search_paths': sorted(custom_search_paths) if custom_search_paths else None,
    }
    
    # Create deterministic JSON string and hash it
    key_str = json.dumps(key_components, sort_keys=True)
    cache_key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    return cache_key


def save_embedding(
    embedding: np.ndarray,
    cache_dir: Path,
    sub_pdb: str,
    core_pdb: str,
    prep_dir: str,
    env_cutoff: float = 5.0,
    include_other_sites: bool = False,
    custom_search_paths: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Path:
    """Save a computed Uni-Mol embedding to disk with metadata.
    
    Embeddings are stored in a structured directory with metadata JSON files
    tracking the configuration used to compute them. This allows efficient
    reloading during subsequent pretraining or analysis runs.
    
    Directory structure:
        cache_dir/
            embeddings/
                {sub_name}/
                    {cache_key}.npy          # embedding array
                    {cache_key}_metadata.json  # configuration metadata
    
    Args:
        embedding: [512] numpy array of Uni-Mol embedding
        cache_dir: Base cache directory (will create embeddings/ subdirectory)
        sub_pdb: Path to substituent PDB
        core_pdb: Path to core scaffold PDB
        prep_dir: Prep directory path
        env_cutoff: Environment cutoff used for computation
        include_other_sites: Whether ref subs from other sites were included
        custom_search_paths: Custom search paths used for environment
        overwrite: If True, overwrite existing cache (default: False)
        
    Returns:
        Path: Path where embedding was saved
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache key
    cache_key = _generate_embedding_cache_key(
        sub_pdb, core_pdb, prep_dir, env_cutoff, include_other_sites, custom_search_paths
    )
    
    # Extract substituent name from path
    sub_name = Path(sub_pdb).stem.replace('_frag', '')
    
    # Create subdirectory for this substituent
    sub_cache_dir = cache_dir / 'embeddings' / sub_name
    sub_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths for embedding and metadata
    embedding_path = sub_cache_dir / f'{cache_key}.npy'
    metadata_path = sub_cache_dir / f'{cache_key}_metadata.json'
    
    # Check if already exists
    if embedding_path.exists() and not overwrite:
        warnings.warn(f"Embedding cache already exists at {embedding_path}, skipping (use overwrite=True to replace)")
        return embedding_path
    
    # Save embedding
    np.save(embedding_path, embedding)
    
    # Save metadata
    metadata = {
        'sub_pdb': str(Path(sub_pdb).resolve()),
        'core_pdb': str(Path(core_pdb).resolve()),
        'prep_dir': str(Path(prep_dir).resolve()),
        'env_cutoff': float(env_cutoff),
        'include_other_sites': bool(include_other_sites),
        'custom_search_paths': sorted(custom_search_paths) if custom_search_paths else None,
        'embedding_shape': list(embedding.shape),
        'embedding_dtype': str(embedding.dtype),
        'cache_key': cache_key,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"      [CACHE] Saved embedding to {embedding_path}")
    
    return embedding_path


def load_embedding(
    cache_dir: Path,
    sub_pdb: str,
    core_pdb: str,
    prep_dir: str,
    env_cutoff: float = 5.0,
    include_other_sites: bool = False,
    custom_search_paths: Optional[List[str]] = None,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """Load a cached Uni-Mol embedding from disk if available.
    
    Checks if a matching embedding exists in the cache directory based on
    the input configuration. Returns the embedding if found, None otherwise.
    
    Args:
        cache_dir: Base cache directory to check
        sub_pdb: Path to substituent PDB
        core_pdb: Path to core scaffold PDB
        prep_dir: Prep directory path
        env_cutoff: Environment cutoff parameter
        include_other_sites: Whether ref subs from other sites are included
        custom_search_paths: Custom search paths for environment
        verbose: If True, print cache hit/miss messages (default: True)
        
    Returns:
        [512] numpy array if found, None otherwise
    """
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return None
    
    # Generate cache key to look up
    cache_key = _generate_embedding_cache_key(
        sub_pdb, core_pdb, prep_dir, env_cutoff, include_other_sites, custom_search_paths
    )
    
    # Extract substituent name
    sub_name = Path(sub_pdb).stem.replace('_frag', '')
    
    # Build paths
    embedding_path = cache_dir / 'embeddings' / sub_name / f'{cache_key}.npy'
    
    if embedding_path.exists():
        embedding = np.load(embedding_path)
        if verbose:
            print(f"      [CACHE] Loaded embedding from {embedding_path}")
        return embedding
    
    if verbose:
        print(f"      [CACHE] No cached embedding found for {sub_name}")
    
    return None


def _extract_ligand_from_pdb(
    pdb_path: str,
    core_pdb: str,
    sub_pdb: str,
    ref_sub_pdbs: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """Extract ligand atom coordinates from minimized.pdb using atom name matching.
    
    This is critical for systems like cmet_solvent_group1 where minimized.pdb contains
    the FULL system (core + all substituents from all sites + ref_subs + solvent). 
    We need to extract only the atoms corresponding to the specific core + sub + ref_subs.
    
    Strategy:
    1. Read atom names from core.pdb, sub_pdb, and all ref_sub PDBs
    2. Build a name -> coordinate map from minimized.pdb
    3. Extract only those atoms whose names match core + sub + ref_subs
    4. Return their coordinates in the minimized frame
    
    This is robust to coordinate frame differences and PBC wrapping.
    
    Args:
        pdb_path: Path to minimized.pdb (full system)
        core_pdb: Path to core.pdb (for getting core atom names)
        sub_pdb: Path to sub_pdb (for getting sub atom names)
        ref_sub_pdbs: Optional list of ref_sub PDB paths (for getting their atom names)
        
    Returns:
        [N_lig, 3] numpy array of ligand atom coordinates from minimized.pdb,
        matched by atom name, or None if not found
    """
    try:
        # Read atom names from core, sub, and ref_subs (uppercase for matching)
        def _read_atom_names(pdb_file: str) -> List[str]:
            names = []
            try:
                with open(pdb_file, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            atom_name = line[12:16].strip().upper()
                            names.append(atom_name)
            except Exception:
                pass
            return names
        
        ligand_names = set()
        ligand_names.update(_read_atom_names(core_pdb))
        ligand_names.update(_read_atom_names(sub_pdb))
        
        # Add ref_sub atom names
        if ref_sub_pdbs:
            for ref_pdb in ref_sub_pdbs:
                ligand_names.update(_read_atom_names(ref_pdb))
        
        if not ligand_names:
            return None
        
        # Build name -> coordinate map from minimized.pdb
        name_to_coord = {}
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_name = line[12:16].strip().upper()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    name_to_coord[atom_name] = [x, y, z]
        
        # Extract ligand coordinates by matching names
        ligand_coords = []
        for name in ligand_names:
            if name in name_to_coord:
                ligand_coords.append(name_to_coord[name])
        
        if ligand_coords:
            return np.array(ligand_coords, dtype=np.float32)
    except Exception:
        pass
    
    return None


def _get_all_ligand_atom_names(
    core_pdb: str,
    sub_pdb: str,
    ref_sub_pdbs: Optional[List[str]] = None,
) -> set:
    """Get all ligand atom names (core + sub + ref_subs) for excluding from environment.
    
    Args:
        core_pdb: Path to core PDB
        sub_pdb: Path to sub PDB
        ref_sub_pdbs: Optional list of ref_sub PDB paths
        
    Returns:
        Set of atom names (uppercase) to exclude from environment
    """
    def _read_atom_names(pdb_file: str) -> List[str]:
        names = []
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        atom_name = line[12:16].strip().upper()
                        names.append(atom_name)
        except Exception:
            pass
        return names
    
    atom_names = set()
    atom_names.update(_read_atom_names(core_pdb))
    atom_names.update(_read_atom_names(sub_pdb))
    if ref_sub_pdbs:
        for ref_pdb in ref_sub_pdbs:
            atom_names.update(_read_atom_names(ref_pdb))
    
    return atom_names


def _filter_ref_sub_by_distance(
    ref_sub_pdb: str,
    sub_coords: np.ndarray,
    cutoff: float = 5.0,
) -> Optional[tuple]:
    """Filter a reference substituent PDB by distance to main substituent.
    
    Loads a ref_sub PDB and keeps only atoms within cutoff distance of the
    main substituent. If no atoms pass the filter, returns None.
    
    Args:
        ref_sub_pdb: Path to ref_sub PDB file
        sub_coords: [N_sub, 3] array of main substituent coordinates
        cutoff: Distance cutoff in Ångströms
        
    Returns:
        (coords, elements) tuple for filtered ref_sub atoms, or None if no atoms pass filter
    """
    try:
        if not Path(ref_sub_pdb).exists():
            return None
        
        # Load ref_sub coordinates
        ref_coords, ref_elements = parse_pdb_file(str(ref_sub_pdb))
        
        if not ref_coords or len(ref_coords) == 0:
            return None
        
        ref_coords = np.array(ref_coords, dtype=np.float32)
        sub_coords = np.array(sub_coords, dtype=np.float32)
        
        # Calculate distances from each ref atom to closest sub atom
        # Shape: [N_ref] - minimum distance from each ref atom to any sub atom
        distances = np.min(np.linalg.norm(ref_coords[:, None, :] - sub_coords[None, :, :], axis=2), axis=1)
        
        # Keep only atoms within cutoff
        mask = distances <= cutoff
        
        if not np.any(mask):
            # No atoms within cutoff
            return None
        
        # Filter atoms and sort by distance (closest first)
        filtered_indices = np.argsort(distances[mask])
        filtered_coords = ref_coords[mask][filtered_indices]
        filtered_elements = [ref_elements[i] for i in np.where(mask)[0][filtered_indices]]
        
        return (filtered_coords.tolist(), filtered_elements)
    except Exception as e:
        warnings.warn(f"Could not filter ref_sub {Path(ref_sub_pdb).name}: {e}")
        return None


def _parse_crystal_box_from_pdb(pdb_path: str) -> Optional[np.ndarray]:
    """Parse CRYST1 record from PDB to get box dimensions.
    
    Reads the CRYST1 line to extract lattice parameters (A, B, C) and angles.
    For cubic systems (A=B=C, THETA=90), returns isotropic box vector.
    
    CRYST1 format: columns 7-15 (A), 16-24 (B), 25-33 (C), 34-40 (α), 41-47 (β), 48-54 (γ)
    
    Args:
        pdb_path: Path to PDB file containing CRYST1 record
        
    Returns:
        [3] numpy array [box_x, box_y, box_z] in Ångströms, or None if not found
    """
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('CRYST1'):
                    # Extract lattice parameters (in Ångströms)
                    a = float(line[6:15])   # A
                    b = float(line[15:24])  # B
                    c = float(line[24:33])  # C
                    
                    # Verify cubic system (A=B=C with 90° angles)
                    if abs(a - b) < 0.01 and abs(b - c) < 0.01:
                        return np.array([a, b, c], dtype=np.float32)
                    else:
                        # Non-cubic; return as-is
                        return np.array([a, b, c], dtype=np.float32)
    except Exception as e:
        pass
    
    return None


def _minimum_image_convention(
    coords_env: np.ndarray,
    coords_lig: np.ndarray,
    box_lengths: np.ndarray
) -> np.ndarray:
    """Compute minimum-image distances for cubic/orthorhombic PBC.
    
    For a cubic or orthorhombic cell, applies the minimum image convention:
    for each environment atom, computes displacement vectors to all ligand atoms,
    applies periodic wrapping, and returns the minimum distance to any ligand atom.
    
    Vectorized implementation (no Python loops).
    
    Args:
        coords_env: [N_env, 3] environment atom coordinates
        coords_lig: [N_lig, 3] ligand atom coordinates
        box_lengths: [3] box dimensions [Lx, Ly, Lz]
        
    Returns:
        [N_env] array of minimum distances to any ligand atom
    """
    # Displacement vectors: [N_env, N_lig, 3]
    delta = coords_env[:, None, :] - coords_lig[None, :, :]
    
    # Apply minimum image convention
    # For each component, wrap to [-L/2, L/2]
    delta = delta - box_lengths[None, None, :] * np.round(delta / box_lengths[None, None, :])
    
    # Compute distances: [N_env, N_lig]
    distances = np.linalg.norm(delta, axis=2)
    
    # Minimum distance to any ligand atom: [N_env]
    min_distances = np.min(distances, axis=1)
    
    return min_distances


def _wrap_coords_to_nearest_image(
    coords: np.ndarray,
    ligand_coords: np.ndarray,
    box_lengths: np.ndarray
) -> np.ndarray:
    """Wrap coordinates to nearest periodic image relative to ligand centroid.
    
    Translates coordinates so they're in the nearest periodic image of the
    ligand centroid. Useful after PBC distance calculations to ensure returned
    coordinates are in the correct periodic image.
    
    Args:
        coords: [N, 3] coordinates to wrap
        ligand_coords: [N_lig, 3] ligand atom coordinates
        box_lengths: [3] box dimensions [Lx, Ly, Lz]
        
    Returns:
        [N, 3] wrapped coordinates in nearest image
    """
    # Ligand centroid
    ligand_centroid = np.mean(ligand_coords, axis=0)
    
    # Displacement from centroid
    delta = coords - ligand_centroid[None, :]
    
    # Apply minimum image convention
    delta = delta - box_lengths[None, :] * np.round(delta / box_lengths[None, :])
    
    # Wrap back to nearest image
    wrapped = ligand_centroid[None, :] + delta
    
    return wrapped


def _filter_and_truncate_environment_atoms(
    coords: List[List[float]],
    elements: List[str],
    ligand_coords: Optional[np.ndarray] = None,
    ligand_cutoff: float = 5.0,
    atom_limit: int = 256,
    source_name: str = "environment",
) -> Optional[tuple]:
    """Filter environment atoms by distance to ligand and truncate to atom limit.
    
    Applied consistently to all environment sources (minimized.pdb, protein.pdb, etc.)
    to ensure uniform behavior whether using minimized or unrelaxed structures.
    
    Args:
        coords: List of [x,y,z] coordinates from parsed PDB
        elements: List of element symbols matching coords
        ligand_coords: [N_ligand, 3] coordinates of core+sub atoms
        ligand_cutoff: Distance cutoff (Å) for filtering atoms
        atom_limit: Maximum atoms to keep (256 for Uni-Mol)
        source_name: Name of source PDB for logging (e.g., "minimized.pdb", "protein.pdb")
    
    Returns:
        (coords, elements) tuple with filtered/truncated atoms, or None if no atoms within cutoff
    """
    if ligand_coords is None or len(ligand_coords) == 0:
        # No ligand info, return all atoms capped at atom_limit
        if len(coords) > atom_limit:
            return (coords[:atom_limit], elements[:atom_limit])
        return (coords, elements)
    
    coords_arr = np.array(coords)
    ligand_arr = np.array(ligand_coords)
    
    # Calculate distances from each environment atom to nearest ligand atom
    distances = np.min(
        np.linalg.norm(coords_arr[:, None, :] - ligand_arr[None, :, :], axis=2),
        axis=1
    )
    
    # Keep atoms within the ligand cutoff (environment context)
    env_mask = distances <= ligand_cutoff
    initial_atoms = np.sum(env_mask) if env_mask.any() else 0
    
    # Try PBC wrapping if not enough atoms found (heuristic: < 10 suggests ligand near box edge)
    if not env_mask.any() or np.sum(env_mask) < 10:
        # If exactly 0 atoms initially, this is suspicious - print diagnostic
        if not env_mask.any():
            print(f"        [ENV] {source_name}: 0 atoms within {ligand_cutoff} Å (ligand may be outside box), attempting PBC wrapping...")
        
        try:
            # Try to estimate box from coordinates (cubic approximation)
            all_coords = np.vstack([coords_arr, ligand_arr])
            min_coords = np.min(all_coords, axis=0)
            max_coords = np.max(all_coords, axis=0)
            estimated_box = max_coords - min_coords
            box_length = np.max(estimated_box)
            box_lengths = np.array([box_length, box_length, box_length], dtype=np.float32)
            
            pbc_distances = _minimum_image_convention(coords_arr, ligand_arr, box_lengths)
            env_mask = pbc_distances <= ligand_cutoff
            distances = pbc_distances
            
            if env_mask.any():
                pbc_atoms = np.sum(env_mask)
                if initial_atoms == 0:
                    print(f"        [ENV] PBC wrapping recovered {pbc_atoms} atoms within {ligand_cutoff} Å")
                else:
                    print(f"        [ENV] PBC (estimated cubic) found {pbc_atoms} atoms within {ligand_cutoff} Å")
        except Exception as e:
            print(f"        [ENV] PBC wrapping failed: {e}, using direct distance filtering")
    
    if not env_mask.any():
        # 0 atoms even after PBC - this is a structural issue
        if initial_atoms == 0:
            print(f"        [ENV] ERROR: {source_name} has 0 atoms within {ligand_cutoff} Å even after PBC wrapping")
            print(f"        [ENV]   This suggests a structure preparation issue (ligand/box mismatch)")
            return None
        else:
            print(f"        [ENV] {source_name}: no atoms within {ligand_cutoff} Å of ligand")
            return None
    
    # Filter by distance mask
    env_indices = np.where(env_mask)[0]
    env_distances = distances[env_indices]
    
    # Sort by distance (closest first) for proper truncation priority
    sorted_order = np.argsort(env_distances)
    env_indices = env_indices[sorted_order]
    
    # Apply atom limit
    if len(env_indices) > atom_limit:
        env_indices = env_indices[:atom_limit]
    
    # Get filtered coordinates and elements
    filtered_coords = [coords[i] for i in env_indices]
    filtered_elements = [elements[i] for i in env_indices]
    
    n_total = len(coords_arr)
    n_kept = len(filtered_coords)
    print(f"        [ENV] {source_name}: filtered to {n_kept}/{n_total} atoms (distance <= {ligand_cutoff} Å, sorted closest-first, capped at {atom_limit})")
    
    return (filtered_coords, filtered_elements)


def _select_environment_atoms(
    coords: List[List[float]],
    elements: List[str],
    pdb_file_path: str,
    sub_coords: Optional[np.ndarray],
    ligand_cutoff: float,
    atom_limit: int,
    source_name: str,
    consensus_atoms: Optional[set],
) -> Optional[tuple]:
    """Select environment atoms for a substituent's embedding.

    When ``consensus_atoms`` is provided, atoms are selected purely by
    consensus-set membership (identifier match) — NOT intersected with a
    separate distance-to-substituent filter. The consensus set is already
    built from distance-to-CORE and capped at ``atom_limit`` atoms (see
    ``build_environment_consensus``), and is intentionally the SAME set for
    every substituent at a site (the whole point of consensus filtering is
    that swapping substituents doesn't change the environment used).
    Combining it with an independent substituent-distance filter would
    intersect two separately-capped nearest-atom windows that frequently
    share no atoms at all for substituents extending away from the core,
    incorrectly zeroing out the environment entirely.

    When no consensus is available (vacuum systems, or consensus build
    failed for this site), falls back to the original behavior: filter atoms
    by distance to the substituent, capped at ``atom_limit``.

    Returns:
        (coords, elements) tuple, or None if no atoms were selected.
    """
    if consensus_atoms is not None:
        filtered = _filter_environment_by_consensus(coords, elements, pdb_file_path, consensus_atoms)
        if filtered is None:
            print(f"        [CONSENSUS] {source_name}: no atoms matched consensus set")
            return None
        filtered_coords, filtered_elements = filtered
        print(f"        [CONSENSUS] {source_name}: selected {len(filtered_coords)} consensus atoms")
        return (filtered_coords, filtered_elements)

    print(f"        [CONSENSUS] {source_name}: NOT using core consensus (none built/provided for this site) — "
          f"falling back to distance-to-substituent filtering")
    return _filter_and_truncate_environment_atoms(
        coords=coords,
        elements=elements,
        ligand_coords=sub_coords,
        ligand_cutoff=ligand_cutoff,
        atom_limit=atom_limit,
        source_name=source_name,
    )


def _load_environment_pdb(
    prep_dir: Path,
    sub_coords: Optional[np.ndarray] = None,
    ligand_cutoff: float = 5.0,
    custom_search_paths: Optional[List[str]] = None,
    core_pdb: Optional[str] = None,
    sub_pdb: Optional[str] = None,
    ref_sub_pdbs: Optional[List[str]] = None,
    skip_minimized: bool = False,
    consensus_atoms: Optional[set] = None,
) -> Optional[tuple]:
    """Load environment PDB with intelligent fallback chain and consensus filtering.
    
    **Filtering:** Environment atoms are filtered by distance to SUBSTITUENT only,
    not the full ligand (core+sub+ref_subs). This focuses the environment context
    on the site of substitution.
    
    **Consensus:** If consensus_atoms is provided (set of (resnum, chain, atomname) tuples),
    environment atoms are further filtered to only include atoms in the consensus set.
    This prevents odd environment noise by only including atoms that appear for all
    substituents at the site.
    
    Priority order (when skip_minimized=False):
    1. Custom paths (if provided via yaml config)
    2. minimized.pdb (full system; extract core+sub+ref_subs, rest is environment)
    3. minimized.crd (CHARMM extended format; auto-converted to PDB)
    4. protein.pdb / proa.pdb / nested pdb/protein.pdb (auto-detect variants)
    5. solvent.pdb / waterbox.pdb / environment.pdb (auto-detect variants)
    
    When skip_minimized=True (for unrelaxed structures), minimized.pdb and .crd are skipped:
    1. Custom paths (if provided via yaml config)
    2. protein.pdb / proa.pdb / nested pdb/protein.pdb (auto-detect variants)
    3. solvent.pdb / waterbox.pdb / environment.pdb (auto-detect variants)
    
    Args:
        prep_dir: Path to prep directory containing PDB files
        sub_coords: [N_sub, 3] numpy array of substituent atom coordinates.
            Used to filter environment atoms by distance (keeps only atoms within
            ligand_cutoff of any substituent atom).
        ligand_cutoff: Distance cutoff (Ångströms) for filtering atoms by distance to sub
        custom_search_paths: Optional list of custom PDB file paths (from yaml config)
            e.g., ['pdb/protein.pdb', 'proa.pdb', 'prob.pdb']. Checked before defaults.
        core_pdb: Path to core.pdb (for extracting core atoms from minimized.pdb)
        sub_pdb: Path to sub_pdb (for extracting sub atoms from minimized.pdb)
        ref_sub_pdbs: Optional list of paths to ref_sub PDB files (for extracting from minimized.pdb)
        skip_minimized: If True, skip minimized.pdb and minimized.crd (for unrelaxed structures)
        consensus_atoms: Optional set of (resnum, chain, atomname) tuples for consensus filtering.
            If provided, environment is further filtered to only include these atoms.
        
    Returns:
        (coords [M, 3], elements [M]) tuple if environment found, else None
    """
    prep_dir = Path(prep_dir)
    
    def _apply_consensus_if_needed(coords_result, elements_result, pdb_file_path):
        """Apply consensus filtering to environment if consensus_atoms provided."""
        if consensus_atoms is not None and coords_result is not None and elements_result is not None:
            filtered = _filter_environment_by_consensus(
                coords_result, elements_result, pdb_file_path, consensus_atoms
            )
            if filtered is not None:
                filtered_coords, filtered_elements = filtered
                print(f"        [CONSENSUS] Filtered to {len(filtered_coords)}/{len(coords_result)} atoms by consensus")
                return (filtered_coords, filtered_elements)
            else:
                # No atoms pass consensus filter
                print(f"        [CONSENSUS] No atoms matched consensus set, skipping this source")
                return None
        print(f"        [CONSENSUS] {Path(pdb_file_path).name}: NOT using core consensus (none built/provided "
              f"for this site) — using unfiltered atoms from this source")
        return (coords_result, elements_result)
    
    # Try custom search paths first (from yaml config)
    if custom_search_paths:
        for pdb_name in custom_search_paths:
            pdb_path = prep_dir / pdb_name
            if pdb_path.exists():
                try:
                    coords, elements = parse_pdb_file(str(pdb_path))
                    print(f"        [ENV] {pdb_path.relative_to(prep_dir)}: loaded {len(coords)} atoms (custom/configured)")
                    return _apply_consensus_if_needed(coords, elements, str(pdb_path)) or (coords, elements)
                except Exception as e:
                    warnings.warn(f"Could not load custom PDB {pdb_path}: {e}, trying next source")
    
    # Try minimized.pdb first (usually contains ligand + protein + water)
    # Skip if skip_minimized=True (for unrelaxed structures)
    minimized_pdb = prep_dir / "minimized.pdb"
    if minimized_pdb.exists() and not skip_minimized:
        try:
            coords, elements = parse_pdb_file(str(minimized_pdb))
            
            # Extract core + sub + ref_subs from minimized.pdb using atom name matching
            # These are NOT part of the environment (they're the ligand itself)
            ligand_from_min = None
            if core_pdb is not None and sub_pdb is not None:
                ligand_from_min = _extract_ligand_from_pdb(
                    str(minimized_pdb), core_pdb, sub_pdb, ref_sub_pdbs
                )
            
            # Get all ligand atom names to exclude from environment
            ligand_atom_names = _get_all_ligand_atom_names(core_pdb, sub_pdb, ref_sub_pdbs)
            
            # Extract environment atoms (everything in minimized.pdb NOT in the ligand)
            env_coords = []
            env_elements = []
            with open(minimized_pdb, 'r') as f:
                for i, line in enumerate(f):
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        atom_name = line[12:16].strip().upper()
                        # Skip if this atom is part of the ligand
                        if atom_name not in ligand_atom_names:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            env_coords.append([x, y, z])
                            # Try to get element from atom name (last 1-2 chars)
                            element = ''.join([c for c in atom_name if c.isalpha()])[:2]
                            env_elements.append(element if element else 'X')
            
            # Select environment atoms: consensus membership if available,
            # else fall back to distance-to-substituent filtering.
            if sub_coords is not None and len(sub_coords) > 0:
                result = _select_environment_atoms(
                    env_coords, env_elements, str(minimized_pdb),
                    sub_coords=sub_coords, ligand_cutoff=ligand_cutoff, atom_limit=256,
                    source_name="minimized.pdb (environment)", consensus_atoms=consensus_atoms,
                )
                return result
            else:
                # No sub coords provided, cap at 256 atoms
                if len(env_coords) > 256:
                    env_coords = env_coords[:256]
                    env_elements = env_elements[:256]
                print(f"        [ENV] minimized.pdb (environment): loaded {len(env_coords)} atoms (no distance filtering, capped at 256)")
                result = _apply_consensus_if_needed(env_coords, env_elements, str(minimized_pdb))
                return result if result[0] is not None else None
        except Exception as e:
            warnings.warn(f"Could not load minimized.pdb: {e}, falling back to other sources")
    
    # Fallback: Try to convert minimized.crd (CHARMM extended CRD format) if minimized.pdb doesn't exist
    # This handles cases like luis_p38_protein_group1 where we have CRD but not PDB
    # Skip if skip_minimized=True (for unrelaxed structures)
    minimized_crd = prep_dir / "minimized.crd"
    if minimized_crd.exists() and not skip_minimized:
        try:
            # Convert CRD to temporary PDB for consistent coordinate handling
            crd_pdb = _convert_crd_to_tmp_pdb(minimized_crd)
            coords, elements = parse_pdb_file(str(crd_pdb))
            
            # Extract core + sub + ref_subs from converted PDB using atom name matching
            ligand_from_crd = None
            if core_pdb is not None and sub_pdb is not None:
                ligand_from_crd = _extract_ligand_from_pdb(
                    str(crd_pdb), core_pdb, sub_pdb, ref_sub_pdbs
                )
            
            # Get all ligand atom names to exclude from environment
            ligand_atom_names = _get_all_ligand_atom_names(core_pdb, sub_pdb, ref_sub_pdbs)
            
            # Extract environment atoms (everything NOT in the ligand)
            env_coords = []
            env_elements = []
            with open(crd_pdb, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        atom_name = line[12:16].strip().upper()
                        if atom_name not in ligand_atom_names:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            env_coords.append([x, y, z])
                            element = ''.join([c for c in atom_name if c.isalpha()])[:2]
                            env_elements.append(element if element else 'X')
            
            # Select environment atoms: consensus membership if available,
            # else fall back to distance-to-substituent filtering.
            if sub_coords is not None and len(sub_coords) > 0:
                result = _select_environment_atoms(
                    env_coords, env_elements, str(crd_pdb),
                    sub_coords=sub_coords, ligand_cutoff=ligand_cutoff, atom_limit=256,
                    source_name="minimized.crd (environment)", consensus_atoms=consensus_atoms,
                )
                return result
            else:
                # No sub coords, cap at 256 atoms
                if len(env_coords) > 256:
                    env_coords = env_coords[:256]
                    env_elements = env_elements[:256]
                print(f"        [ENV] minimized.crd (environment): loaded {len(env_coords)} atoms (no distance filtering, capped at 256)")
                result = _apply_consensus_if_needed(env_coords, env_elements, str(crd_pdb))
                return result if result[0] is not None else None
        except Exception as e:
            warnings.warn(f"Could not load/convert minimized.crd: {e}, falling back to other sources")
    
    # Fallback: Try individual protein PDB files (for unrelaxed structures)
    # Uses the same candidate list as build_environment_consensus() to ensure
    # both resolve to the identical file (see _default_protein_pdb_candidates).
    protein_candidates = _default_protein_pdb_candidates(prep_dir)
    
    for protein_pdb in protein_candidates:
        if protein_pdb.exists():
            try:
                coords, elements = parse_pdb_file(str(protein_pdb))
                rel_path = protein_pdb.relative_to(prep_dir)
                
                # Select environment atoms: consensus membership if available,
                # else fall back to distance-to-substituent filtering.
                if sub_coords is not None and len(sub_coords) > 0:
                    result = _select_environment_atoms(
                        coords, elements, str(protein_pdb),
                        sub_coords=sub_coords, ligand_cutoff=ligand_cutoff, atom_limit=256,
                        source_name=str(rel_path), consensus_atoms=consensus_atoms,
                    )
                    if result is not None:
                        return result
                    # If no atoms selected, continue to next source
                else:
                    # No sub coords, cap at 256 atoms
                    if len(coords) > 256:
                        coords = coords[:256]
                        elements = elements[:256]
                    print(f"        [ENV] {rel_path}: loaded {len(coords)} atoms (no distance filtering, capped at 256)")
                    return (coords, elements)
            except Exception as e:
                warnings.warn(f"Could not load {protein_pdb.name}: {e}, trying next source")
    
    # Fallback: Try solvent/water PDB files (for unrelaxed structures)
    # Uses the same candidate list as build_environment_consensus() to ensure
    # both resolve to the identical file (see _default_solvent_pdb_candidates).
    solvent_candidates = _default_solvent_pdb_candidates(prep_dir)
    
    solvent_files_found = False
    for solvent_pdb in solvent_candidates:
        if solvent_pdb.exists():
            solvent_files_found = True
            try:
                coords, elements = parse_pdb_file(str(solvent_pdb))
                
                # Select environment atoms: consensus membership if available,
                # else fall back to distance-to-substituent filtering.
                if sub_coords is not None and len(sub_coords) > 0:
                    result = _select_environment_atoms(
                        coords, elements, str(solvent_pdb),
                        sub_coords=sub_coords, ligand_cutoff=ligand_cutoff, atom_limit=256,
                        source_name=solvent_pdb.name, consensus_atoms=consensus_atoms,
                    )
                    if result is not None:
                        return result
                    # If no atoms selected, continue to next source
                else:
                    # No sub coords, cap at 256 atoms
                    if len(coords) > 256:
                        coords = coords[:256]
                        elements = elements[:256]
                    print(f"        [ENV] {solvent_pdb.name}: loaded {len(coords)} atoms (no distance filtering, capped at 256)")
                    result = _apply_consensus_if_needed(coords, elements, str(solvent_pdb))
                    if result and result[0] is not None:
                        return result
                    # If no consensus match, continue to next source
            except Exception as e:
                warnings.warn(f"Could not load {solvent_pdb.name}: {e}, trying next source")
    
    # Fallback: If solvent files exist but all failed (0 atoms), try minimized structure
    if solvent_files_found and not skip_minimized:
        minimized_pdb = prep_dir / "minimized.pdb"
        if minimized_pdb.exists():
            try:
                print(f"        [ENV] Solvent files found but returned 0 atoms, falling back to minimized structure")
                coords, elements = parse_pdb_file(str(minimized_pdb))
                
                if sub_coords is not None and len(sub_coords) > 0:
                    result = _select_environment_atoms(
                        coords, elements, str(minimized_pdb),
                        sub_coords=sub_coords, ligand_cutoff=ligand_cutoff, atom_limit=256,
                        source_name="minimized.pdb (fallback)", consensus_atoms=consensus_atoms,
                    )
                    if result is not None:
                        return result
                else:
                    # No sub coords, cap at 256 atoms
                    if len(coords) > 256:
                        coords = coords[:256]
                        elements = elements[:256]
                    print(f"        [ENV] minimized.pdb: loaded {len(coords)} atoms (no distance filtering, capped at 256)")
                    result = _apply_consensus_if_needed(coords, elements, str(minimized_pdb))
                    if result and result[0] is not None:
                        return result
                    return (coords, elements)
            except Exception as e:
                warnings.warn(f"Fallback to minimized.pdb failed: {e}")
    
    # No environment found from solvent or fallback
    if solvent_files_found:
        print(f"        [ENV] WARNING: Environment files exist but no atoms within {ligand_cutoff} Å found")
    else:
        print(f"        [ENV] no environment file found in {prep_dir.name}")
    return None


def construct_full_ligand(
    sub_pdb: str,
    core_pdb: str,
    sub_rtf_data: Optional[dict] = None,
    core_rtf_data: Optional[dict] = None,
    ref_sub_info: Optional[List[Tuple[str, Optional[dict]]]] = None,
    protein_pdb: Optional[str] = None,
    solvent_coords: Optional[np.ndarray] = None,
    solvent_elements: Optional[List[str]] = None,
    cutoff: float = 5.0,
    enforce_atom_limit: int = 256,
) -> Dict:
    """Construct complete ligand with environment atoms for Uni-Mol representation.

    Combines core scaffold + substituent(s) + reference subs at other sites (multi-site
    systems) + optional protein/solvent environment atoms within radial cutoff.

    Atom ordering in output: substituent first, then core, then ref-subs, then environment.

    For multi-site systems, the ref_sub_info should contain reference substituents from
    other sites (typically site#_sub1). These provide full molecular context for the
    substitution pattern at non-target sites.
    
    **Important:** When ref_subs are pre-filtered by distance (from main substituent),
    they should be passed as (pdb_path, (coords, elements), rtf_data) tuples. When not
    pre-filtered, use (pdb_path, rtf_data) tuples and they will be loaded in full.

    Args:
        sub_pdb: Path to substituent fragment PDB file (e.g., site1_sub2_frag.pdb)
        core_pdb: Path to core scaffold PDB file
        sub_rtf_data: Optional RTF metadata dict for substituent (for charges/types)
        core_rtf_data: Optional RTF metadata dict for core
        ref_sub_info: Optional list of reference substituent information tuples:
            - (pdb_path, rtf_data): Load ref_sub PDB in full
            - (pdb_path, (coords, elements), rtf_data): Use pre-filtered coords/elements
            For pretraining, typically uses pre-filtered format with atoms within 5Å
            of the main substituent.
        protein_pdb: Optional path to protein PDB file (or tuple of (coords, elements))
        solvent_coords: Optional [N, 3] solvent atom coordinates (numpy array)
        solvent_elements: Optional list of solvent element symbols
        cutoff: Radial cutoff for including environment atoms (default: 5.0 Å).
            Environment atoms beyond this distance from ligand are excluded.
        enforce_atom_limit: Maximum total atoms to include (default: 256, Uni-Mol max).
            If exceeded, environment atoms are truncated.

    Returns:
        dict with keys:
            'atoms': list of element symbols for all atoms
            'coordinates': [N, 3] numpy array of atomic coordinates (float32)
            'n_sub': int, number of substituent atoms
            'n_core': int, number of core scaffold atoms
            'n_ref_per_site': list[int], atom counts for each reference substituent
            'n_env': int, number of environment atoms included
            'metadata': dict with source information (paths, atom counts, etc.)
    """
    # Parse ligand atom coordinates directly (we need coords, not AEVs)
    sub_coords, sub_elements = parse_pdb_file(sub_pdb, rtf_data=sub_rtf_data)
    core_coords, core_elements = parse_pdb_file(core_pdb, rtf_data=core_rtf_data)

    n_sub = len(sub_coords)
    n_core = len(core_coords)

    # Parse reference subs if provided (multi-site systems)
    # Can be either (pdb_path, rtf) or (pdb_path, (coords, elements), rtf) with pre-filtered coords
    ref_coords_list = []
    ref_elements_list = []
    n_ref_per_site = []
    for ref_entry in (ref_sub_info or []):
        if len(ref_entry) == 2:
            # Old format: (pdb_path, rtf)
            ref_pdb, ref_rtf = ref_entry
            ref_coords, ref_elements = parse_pdb_file(ref_pdb, rtf_data=ref_rtf)
        elif len(ref_entry) == 3:
            # New format: (pdb_path, (coords, elements), rtf) with pre-filtered coords
            ref_pdb, (ref_coords, ref_elements), ref_rtf = ref_entry
        else:
            warnings.warn(f"Unexpected ref_sub_info format: {ref_entry}, skipping")
            continue
        
        ref_coords_list.append(ref_coords)
        ref_elements_list.append(ref_elements)
        n_ref_per_site.append(len(ref_coords))

    # Combine ligand atoms (sub + core + ref subs)
    all_coords = [sub_coords, core_coords] + ref_coords_list
    all_elements = [sub_elements, core_elements] + ref_elements_list

    n_ligand = sum(len(e) for e in all_elements)

    # Add environment atoms if provided, filtered by radial cutoff from full ligand
    n_env = 0
    env_coords_to_add = []
    env_elements_to_add = []
    
    # Build full ligand coordinate array for filtering
    all_ligand_coords_arr = np.vstack([np.array(coords) for coords in all_coords])

    if solvent_coords is not None and solvent_elements is not None:
        # Filter solvent atoms by distance to ALL ligand atoms (core + sub + refs)
        solvent_arr = np.array(solvent_coords)
        
        # Compute minimum distance from each solvent atom to any ligand atom
        distances = np.min(
            np.linalg.norm(solvent_arr[:, None, :] - all_ligand_coords_arr[None, :, :], axis=2),
            axis=1
        )
        nearby_mask = distances <= cutoff
        nearby_indices = np.where(nearby_mask)[0]
        nearby_distances = distances[nearby_mask]
        
        # Sort by distance (closest first) - will be helpful if truncation is needed
        if len(nearby_indices) > 0:
            sorted_order = np.argsort(nearby_distances)
            nearby_indices = nearby_indices[sorted_order]
        
        env_coords_to_add = solvent_arr[nearby_indices].tolist()
        env_elements_to_add = [solvent_elements[i] for i in nearby_indices]
        n_env = len(env_coords_to_add)

    elif protein_pdb is not None:
        # Handle protein_pdb - can be either a raw file path (string) or pre-filtered tuple
        if isinstance(protein_pdb, str):
            # Raw file path: parse and filter by distance to ALL ligand atoms (core + sub + refs)
            prot_coords, prot_elements = parse_pdb_file(protein_pdb)
            prot_coords_arr = np.array(prot_coords)
            
            # Compute minimum distance from each protein atom to any ligand atom
            distances = np.min(
                np.linalg.norm(prot_coords_arr[:, None, :] - all_ligand_coords_arr[None, :, :], axis=2),
                axis=1
            )
            nearby_mask = distances <= cutoff
            nearby_indices = np.where(nearby_mask)[0]
            nearby_distances = distances[nearby_mask]
            
            # Sort by distance (closest first) - will be helpful if truncation is needed
            if len(nearby_indices) > 0:
                sorted_order = np.argsort(nearby_distances)
                nearby_indices = nearby_indices[sorted_order]
            
            env_coords_to_add = prot_coords_arr[nearby_indices].tolist()
            env_elements_to_add = [prot_elements[i] for i in nearby_indices]
            n_env = len(env_coords_to_add)
        else:
            # Pre-filtered tuple from _load_environment_pdb(): already filtered by distance to full ligand
            # Just use directly without additional filtering (atoms are already sorted by distance)
            prot_coords, prot_elements = protein_pdb
            env_coords_to_add = prot_coords
            env_elements_to_add = prot_elements
            n_env = len(env_coords_to_add)

    # Enforce atom limit by truncating environment if necessary
    total_atoms = n_ligand + n_env
    if total_atoms > enforce_atom_limit:
        # Remove environment atoms to stay within limit
        max_env = enforce_atom_limit - n_ligand
        if max_env < 0:
            # Ligand itself exceeds limit - this shouldn't happen in practice
            warnings.warn(
                f"Ligand has {n_ligand} atoms, exceeds limit {enforce_atom_limit}. "
                f"Truncating ligand to fit (removing reference subs and environment)."
            )
            # Truncate ligand to keep only sub + core
            all_coords = all_coords[:2]
            all_elements = all_elements[:2]
            n_ref_per_site = []
            n_env = 0
        else:
            # Truncate environment only
            env_coords_to_add = env_coords_to_add[:max_env]
            env_elements_to_add = env_elements_to_add[:max_env]
            n_env = len(env_coords_to_add)

    # Concatenate all atom groups in order
    final_coords = []
    final_elements = []
    for coords, elements in zip(all_coords, all_elements):
        final_coords.extend(coords)
        final_elements.extend(elements)

    final_coords.extend(env_coords_to_add)
    final_elements.extend(env_elements_to_add)

    return {
        'atoms': final_elements,
        'coordinates': np.array(final_coords, dtype=np.float32),
        'n_sub': n_sub,
        'n_core': n_core,
        'n_ref_per_site': n_ref_per_site,
        'n_env': n_env,
        'metadata': {
            'sub_pdb': str(sub_pdb),
            'core_pdb': str(core_pdb),
            'total_atoms': len(final_elements),
            'ligand_atoms': n_ligand,
            'env_atoms': n_env,
        }
    }


def get_unimol_representation(
    ligand_dict: Dict,
    use_cuda: bool = False,
    return_atomic_reprs: bool = False,
) -> np.ndarray:
    """Compute Uni-Mol 512-dimensional representation for a ligand system.

    Takes a ligand dictionary (from construct_full_ligand) and computes the
    pretrained Uni-Mol embedding. The model was trained on 1.1B molecules from
    PubChem and provides robust molecular representations.

    Args:
        ligand_dict: Dict from construct_full_ligand() with 'atoms' and 'coordinates'
        use_cuda: Whether to use GPU for inference (default: False for CPU)
        return_atomic_reprs: If True, return per-atom representations instead of
            molecular representation. Returns dict with 'atoms' and 'representations'.

    Returns:
        If return_atomic_reprs=False: [512] numpy array of molecular embedding
        If return_atomic_reprs=True: dict with per-atom representations
    """
    model = _get_unimol_model(use_cuda=use_cuda)

    # Format for Uni-Mol: dict with 'atoms' (list of element symbols) and
    # 'coordinates' (numpy array of shape [N, 3])
    unimol_input = {
        'atoms': ligand_dict['atoms'],
        'coordinates': ligand_dict['coordinates'],
    }

    # Get representation from pretrained model
    result = model.get_repr(
        data=unimol_input,
        return_atomic_reprs=return_atomic_reprs,
        return_tensor=False
    )

    # Extract molecular representation (first item in result list)
    if return_atomic_reprs:
        return result[0]  # Dict with atomic representations
    else:
        return np.array(result[0], dtype=np.float32)  # [512] molecular embedding


def get_substituent_unimol_representation(
    sub_pdb: str,
    core_pdb: str,
    prep_dir: Optional[str] = None,
    sub_rtf_data: Optional[dict] = None,
    core_rtf_data: Optional[dict] = None,
    include_other_sites: bool = True,
    use_cuda: bool = False,
) -> np.ndarray:
    """Compute Uni-Mol representation for a single substituent with context.

    For multi-site systems, includes reference substituents (sub1) from other sites,
    providing full molecular context for the substitution pattern.

    This is a convenience API that combines construct_full_ligand() and
    get_unimol_representation() for typical use cases.

    Args:
        sub_pdb: Path to substituent fragment PDB (e.g., site1_sub2_frag.pdb)
        core_pdb: Path to core scaffold PDB
        prep_dir: Optional prep directory path. If provided and system is multi-site,
            automatically locates reference subs from other sites.
        sub_rtf_data: Optional RTF metadata dict for substituent
        core_rtf_data: Optional RTF metadata dict for core
        include_other_sites: If True and prep_dir provided, include ref subs from
            other sites in multi-site systems (default: True)
        use_cuda: Whether to use GPU (default: False for CPU)

    Returns:
        [512] numpy array of Uni-Mol molecular embedding
    """
    # Find reference subs from other sites if multi-site system
    ref_sub_info = None
    if include_other_sites and prep_dir:
        try:
            ref_pdb_list = find_reference_subs_from_other_sites(sub_pdb, prep_dir, cutoff=5.0)
            if ref_pdb_list:
                ref_sub_info = [(p, None) for p in ref_pdb_list]  # No RTF data for refs
        except Exception as e:
            warnings.warn(f"Could not find reference subs: {e}")

    # Construct full ligand (sub + core + ref subs, no environment)
    ligand_dict = construct_full_ligand(
        sub_pdb=sub_pdb,
        core_pdb=core_pdb,
        sub_rtf_data=sub_rtf_data,
        core_rtf_data=core_rtf_data,
        ref_sub_info=ref_sub_info,
        protein_pdb=None,
        solvent_coords=None,
        solvent_elements=None,
    )

    # Compute and return Uni-Mol representation
    return get_unimol_representation(ligand_dict, use_cuda=use_cuda)


def get_substituent_unimol_with_environment(
    sub_pdb: str,
    core_pdb: str,
    prep_dir: Path,
    sub_rtf_data: Optional[dict] = None,
    core_rtf_data: Optional[dict] = None,
    include_other_sites: bool = False,
    env_cutoff: float = 8.0,
    atom_limit: int = 256,
    use_cuda: bool = False,
    custom_search_paths: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    save_cache: bool = False,
    skip_minimized: bool = False,
    consensus_atoms: Optional[set] = None,
) -> np.ndarray:
    """Compute Uni-Mol 512D representation for substituent with full environment context.
    
    Specifically designed for pretraining and online training. Constructs the complete
    system with intelligent fallback chain for environment PDB detection.
    
    **Result Composition:**
    core + sub + filtered_ref_subs + filtered_environment
    
    - core: All atoms from core.pdb
    - sub: All atoms from sub_pdb
    - filtered_ref_subs: Reference substituents from other sites, filtered to keep only
      atoms within env_cutoff (5.0 Å) distance from the main substituent
    - filtered_environment: Protein, solvent, or minimized system atoms filtered to keep
      only those within env_cutoff distance from the main substituent
      (For vacuum systems, environment is always empty)
    
    **Consensus Filtering:**
    If consensus_atoms is provided (set of (resnum, chain, atomname) tuples), environment
    atoms are further filtered to only include atoms that appear for ALL substituents at
    the site. This prevents odd environment noise in the mean embedding used by the policy.
    
    **Vacuum System Handling:**
    Systems with "vac" or "vacuum" in their name are automatically detected and skip
    environment loading entirely. This ensures no environment atoms are accidentally
    included even if solvent.pdb or protein.pdb files exist in the prep directory.
    
    **Environment Priority Chain (when skip_minimized=False, non-vacuum only):**
    1. Custom paths (if provided, e.g., from yaml config like pdb/protein.pdb)
    2. minimized.pdb (preferred; ligand atoms automatically excluded)
    3. protein.pdb / proa.pdb / proa_*.pdb / nested pdb/protein.pdb
    4. solvent.pdb / waterbox.pdb / water.pdb / etc. variants
    
    **When skip_minimized=True (unrelaxed structures, non-vacuum only):**
    1. Custom paths
    2. protein.pdb / proa.pdb / variants (loads unrelaxed protein structure)
    3. solvent.pdb / waterbox.pdb / variants (loads unrelaxed solvent structure)
    
    **Filtering:**
    All environment atoms are filtered by distance to the SUBSTITUENT ONLY, ensuring
    consistent filtering regardless of structure type (minimized/unrelaxed).
    Reference substituents are also filtered by the same distance criterion.
    
    **Caching:**
    If cache_dir is provided, embeddings are automatically cached on disk with metadata.
    On subsequent calls with identical parameters, cached embeddings are loaded instead
    of recomputed, significantly speeding up pretraining and analysis.
    
    Args:
        sub_pdb: Path to substituent fragment PDB (e.g., site1_sub2_frag.pdb)
        core_pdb: Path to core scaffold PDB
        prep_dir: Prep directory containing environment PDB files (minimized.pdb, etc.)
        sub_rtf_data: Optional RTF metadata dict for substituent
        core_rtf_data: Optional RTF metadata dict for core
        include_other_sites: If True, include reference subs from other sites (default: False)
            Each ref_sub is filtered to keep only atoms within env_cutoff of the main sub.
        env_cutoff: Distance cutoff (Ångströms) for filtering all environment/ref_sub atoms
            (default: 8.0). Used for both reference substituents and environment atoms.
        atom_limit: Maximum atoms to include in representation (default: 256, Uni-Mol max)
        use_cuda: Whether to use GPU (default: False for CPU)
        custom_search_paths: Optional list of custom PDB paths to search before defaults.
            Used for yaml-configured paths like ['pdb/protein.pdb', 'proa.pdb', 'prob.pdb']
        cache_dir: Optional directory to cache embeddings. If provided, cached embeddings
            are automatically loaded if available, and newly computed ones are saved.
        save_cache: If True and cache_dir provided, save computed embedding to cache (default: False)
        skip_minimized: If True, skip minimized.pdb and minimized.crd files (for unrelaxed).
            Has no effect for vacuum systems (which skip all environment loading).
        consensus_atoms: Optional set of (resnum, chain, atomname) tuples for consensus filtering.
            If provided, environment is further filtered to only include these atoms.
        
    Returns:
        [512] numpy array of Uni-Mol molecular embedding
        
    Raises:
        FileNotFoundError: If core_pdb not found
        Exception: If Uni-Mol computation fails (caught and re-raised with context)
    """
    core_pdb_path = Path(core_pdb)
    prep_dir = Path(prep_dir)
    if not core_pdb_path.exists():
        raise FileNotFoundError(f"Core PDB not found: {core_pdb}")
    
    # Try to load from cache first if cache_dir provided
    sub_name = Path(sub_pdb).name
    if cache_dir is not None:
        cached_embedding = load_embedding(
            cache_dir=cache_dir,
            sub_pdb=sub_pdb,
            core_pdb=core_pdb,
            prep_dir=str(prep_dir),
            env_cutoff=env_cutoff,
            include_other_sites=include_other_sites,
            custom_search_paths=custom_search_paths,
            verbose=True,
        )
        if cached_embedding is not None:
            return cached_embedding
    
    print(f"      [UNIMOL] Computing for {sub_name} (cutoff={env_cutoff}Å)")
    
    # Parse core and sub to get coordinates
    core_coords, core_elements = parse_pdb_file(str(core_pdb))
    sub_coords, sub_elements = parse_pdb_file(str(sub_pdb))
    
    # Find reference subs BEFORE loading environment
    # For multi-site systems, ref_subs provide context for substituents at other sites
    # Each ref_sub is filtered by distance to the main substituent (keep only atoms within 5Å)
    ref_sub_info = None
    ref_sub_pdbs_list = []
    if include_other_sites:
        try:
            ref_pdb_list = find_reference_subs_from_other_sites(sub_pdb, str(prep_dir), cutoff=5.0)
            if ref_pdb_list:
                # Filter each ref_sub by distance to main substituent
                # Only include ref_subs that have atoms within 5Å of the substituent
                filtered_ref_info = []
                for ref_pdb_path in ref_pdb_list:
                    filtered_ref = _filter_ref_sub_by_distance(
                        ref_pdb_path,
                        sub_coords=sub_coords,
                        cutoff=env_cutoff,  # Use same cutoff as environment
                    )
                    if filtered_ref is not None:
                        # Store filtered coords/elements for later use
                        filtered_ref_info.append((ref_pdb_path, filtered_ref, None))  # (path, (coords, elements), rtf)
                        ref_sub_pdbs_list.append(ref_pdb_path)
                
                # Create ref_sub_info for construct_full_ligand
                if filtered_ref_info:
                    # For construct_full_ligand, we need (pdb_path, rtf_data) tuples
                    # But we want to use the filtered coordinates instead
                    # We'll need to modify construct_full_ligand to accept pre-filtered coords
                    ref_sub_info = filtered_ref_info
        except Exception as e:
            warnings.warn(f"Could not find/filter reference subs: {e}")
    
    # Check if this is a vacuum system - skip environment loading entirely for vacuum
    # Vacuum systems should have NO environment atoms, even if PDB files exist by accident
    # For nested combos (pretraining/SYSTEM/comb_*/prep), need to look TWO levels up
    # For direct systems (pretraining/SYSTEM/prep), look ONE level up
    potential_system = prep_dir.parent  # First level up: comb_* or direct system
    system_name = potential_system.name.lower()
    
    # If this looks like a combo directory, go up another level
    if system_name.startswith('comb_'):
        system_name = potential_system.parent.name.lower()
    
    is_vacuum_system = 'vac' in system_name or 'vacuum' in system_name
    
    if is_vacuum_system:
        print(f"      [ENV] Vacuum system detected ({system_name}), skipping environment loading")
        env_data = None
    else:
        # Load environment with intelligent fallback chain
        # Filters atoms by distance to SUBSTITUENT ONLY (not full ligand)
        # This ensures consistent filtering across minimized/unrelaxed/protein/solvent structures
        env_data = _load_environment_pdb(
            prep_dir,
            sub_coords=sub_coords,  # Filter by distance to SUB only
            ligand_cutoff=env_cutoff,
            custom_search_paths=custom_search_paths,
            core_pdb=core_pdb,
            sub_pdb=sub_pdb,
            ref_sub_pdbs=ref_sub_pdbs_list if ref_sub_pdbs_list else None,
            skip_minimized=skip_minimized,
            consensus_atoms=consensus_atoms,  # Further filter by consensus if provided
        )
    
    # Construct full ligand (sub + core + ref subs + environment)
    ligand_dict = construct_full_ligand(
        sub_pdb=sub_pdb,
        core_pdb=core_pdb,
        sub_rtf_data=sub_rtf_data,
        core_rtf_data=core_rtf_data,
        ref_sub_info=ref_sub_info,
        protein_pdb=env_data if env_data else None,  # Pass (coords, elements) tuple
        solvent_coords=None,
        solvent_elements=None,
        cutoff=env_cutoff,
        enforce_atom_limit=atom_limit,
    )
    
    # Compute and return Uni-Mol representation
    try:
        embedding = get_unimol_representation(ligand_dict, use_cuda=use_cuda)
        
        # Save to cache if requested
        if save_cache and cache_dir is not None:
            save_embedding(
                embedding=embedding,
                cache_dir=cache_dir,
                sub_pdb=sub_pdb,
                core_pdb=core_pdb,
                prep_dir=str(prep_dir),
                env_cutoff=env_cutoff,
                include_other_sites=include_other_sites,
                custom_search_paths=custom_search_paths,
            )
        
        return embedding
    except Exception as e:
        raise Exception(
            f"Failed to compute Uni-Mol representation for {Path(sub_pdb).name}: {e}"
        )


def get_substituent_dual_embeddings(
    sub_pdb: str,
    core_pdb: str,
    prep_dir: Path,
    sub_rtf_data: Optional[dict] = None,
    core_rtf_data: Optional[dict] = None,
    include_other_sites: bool = False,
    env_cutoff: float = 8.0,
    atom_limit: int = 256,
    use_cuda: bool = False,
    custom_search_paths: Optional[List[str]] = None,
    skip_minimized: bool = False,
    consensus_atoms: Optional[set] = None,
) -> tuple:
    """Compute two Uni-Mol embeddings: ligand-only and ligand+environment.
    
    This function computes two separate representations:
    1. **Ligand-only** [512]: core + sub (no environment, no ref_subs)
       - Captures substituent-specific information
    2. **Ligand+environment** [512]: core + sub + distance-filtered ref_subs + distance-filtered environment
       - Captures combined ligand and environment information
    
    The policy learns from both embeddings, using:
    - Difference of ligand-only (antisymmetric): captures substituent variation between nodes
    - Mean of ligand+environment (symmetric): captures environment effects
    
    This dual-embedding approach allows MLPs to simultaneously learn substituent-dependent
    and environment-dependent information.
    
    **Consensus Filtering:**
    If consensus_atoms is provided (set of (resnum, chain, atomname) tuples), environment
    atoms in the ligand+environment embedding are further filtered to only include atoms
    that appear for ALL substituents at the site. This prevents odd environment noise in
    the mean embedding used by the policy.
    
    Args:
        sub_pdb: Path to substituent fragment PDB (e.g., site1_sub2_frag.pdb)
        core_pdb: Path to core scaffold PDB
        prep_dir: Prep directory containing environment PDB files (minimized.pdb, etc.)
        sub_rtf_data: Optional RTF metadata dict for substituent
        core_rtf_data: Optional RTF metadata dict for core
        include_other_sites: If True, include reference subs from other sites (default: False)
        env_cutoff: Distance cutoff (Ångströms) for filtering environment/ref_sub atoms (default: 8.0)
        atom_limit: Maximum atoms to include in representation (default: 256, Uni-Mol max)
        use_cuda: Whether to use GPU (default: False for CPU)
        custom_search_paths: Optional list of custom PDB paths to search before defaults
        skip_minimized: If True, skip minimized.pdb (for unrelaxed structures)
        consensus_atoms: Optional set of (resnum, chain, atomname) tuples for consensus filtering.
            If provided, environment in ligand+environment embedding is filtered to only include
            these atoms, preventing noise in the mean embedding.
        
    Returns:
        (embedding_ligand_only [512], embedding_with_env [512])
    """
    core_pdb_path = Path(core_pdb)
    prep_dir = Path(prep_dir)
    if not core_pdb_path.exists():
        raise FileNotFoundError(f"Core PDB not found: {core_pdb}")
    
    # Construct ligand-only (no environment)
    ligand_only_dict = construct_full_ligand(
        sub_pdb=sub_pdb,
        core_pdb=core_pdb,
        sub_rtf_data=sub_rtf_data,
        core_rtf_data=core_rtf_data,
        ref_sub_info=None,  # No ref_subs
        protein_pdb=None,    # No environment
        solvent_coords=None,
        solvent_elements=None,
    )
    
    # Compute ligand-only embedding
    embedding_ligand_only = get_unimol_representation(ligand_only_dict, use_cuda=use_cuda)
    
    # Compute ligand+environment embedding (full version with environment)
    embedding_with_env = get_substituent_unimol_with_environment(
        sub_pdb=sub_pdb,
        core_pdb=core_pdb,
        prep_dir=prep_dir,
        sub_rtf_data=sub_rtf_data,
        core_rtf_data=core_rtf_data,
        include_other_sites=include_other_sites,
        env_cutoff=env_cutoff,
        atom_limit=atom_limit,
        use_cuda=use_cuda,
        custom_search_paths=custom_search_paths,
        skip_minimized=skip_minimized,
        consensus_atoms=consensus_atoms,  # Pass consensus if provided
    )
    
    return embedding_ligand_only, embedding_with_env
