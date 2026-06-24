"""Uni-Mol 512-dimensional molecular representations with environment context.

Constructs complete ligand systems (core + substituent(s) + environment) and computes
Uni-Mol representations trained on 1.1B molecules from PubChem.

Key Features:
- Automatic ligand assembly: combine core + substituent(s) into unified 3D structure
- Multi-site support: include reference subs (site#_sub1) from other sites
- Environment context: add protein/solvent atoms within spatial cutoff (~5 Å)
- Atom limit enforcement: Uni-Mol supports max 256 atoms; enforce via radial filtering
- Format flexibility: handle PDB files and CHARMM CRD coordinates seamlessly

Functions:
- construct_full_ligand(): Build complete system with optional environment
- get_unimol_representation(): Compute 512D molecular embedding
- get_substituent_unimol_representation(): One-shot API for substituent + context
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

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


def _extract_ligand_from_pdb(
    pdb_path: str,
    core_pdb: str,
    sub_pdb: str
) -> Optional[np.ndarray]:
    """Extract ligand atom coordinates from minimized.pdb using atom name matching.
    
    This is critical for systems like cmet_solvent_group1 where minimized.pdb contains
    the FULL system (core + all substituents from all sites + solvent). We need to extract
    only the atoms corresponding to the specific core + sub we care about.
    
    Strategy:
    1. Read atom names from core.pdb and sub_pdb
    2. Build a name -> coordinate map from minimized.pdb
    3. Extract only those atoms whose names match core + sub
    4. Return their coordinates in the minimized frame
    
    This is robust to coordinate frame differences and PBC wrapping.
    
    Args:
        pdb_path: Path to minimized.pdb (full system)
        core_pdb: Path to core.pdb (for getting core atom names)
        sub_pdb: Path to sub_pdb (for getting sub atom names)
        
    Returns:
        [N_lig, 3] numpy array of ligand atom coordinates from minimized.pdb,
        matched by atom name, or None if not found
    """
    try:
        # Read atom names from core and sub (uppercase for matching)
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
        
        core_names = _read_atom_names(core_pdb)
        sub_names = _read_atom_names(sub_pdb)
        ligand_names = set(core_names + sub_names)
        
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


def _load_environment_pdb(
    prep_dir: Path,
    ligand_coords: Optional[np.ndarray] = None,
    ligand_cutoff: float = 5.0,
    custom_search_paths: Optional[List[str]] = None,
    core_pdb: Optional[str] = None,
    sub_pdb: Optional[str] = None,
) -> Optional[tuple]:
    """Load environment PDB with intelligent fallback chain.
    
    Priority order:
    1. Custom paths (if provided via yaml config)
    2. minimized.pdb (preferred; filter to only environment atoms within cutoff from ligand)
    3. protein.pdb / proa.pdb / nested pdb/protein.pdb (auto-detect variants)
    4. solvent.pdb / waterbox.pdb / environment.pdb (auto-detect variants)
    
    When loading from minimized.pdb, atoms are filtered to keep only those within
    ligand_cutoff of any ligand atom (environment context). Atoms are sorted by distance
    (closest first) to enable proper 256-atom truncation via Uni-Mol atom limit.
    
    Args:
        prep_dir: Path to prep directory containing PDB files
        ligand_coords: [N_ligand, 3] numpy array of ligand atom coordinates (core+sub+ref).
            If provided and minimized.pdb is used, filters environment atoms
            by distance (keeps only atoms <= ligand_cutoff away from ligand).
        ligand_cutoff: Distance cutoff (Ångströms) for filtering minimized.pdb atoms
        custom_search_paths: Optional list of custom PDB file paths (from yaml config)
            e.g., ['pdb/protein.pdb', 'proa.pdb', 'prob.pdb']. Checked before defaults.
        
    Returns:
        (coords [M, 3], elements [M]) tuple if environment found, else None
    """
    prep_dir = Path(prep_dir)
    
    # Try custom search paths first (from yaml config)
    if custom_search_paths:
        for pdb_name in custom_search_paths:
            pdb_path = prep_dir / pdb_name
            if pdb_path.exists():
                try:
                    coords, elements = parse_pdb_file(str(pdb_path))
                    print(f"        [ENV] {pdb_path.relative_to(prep_dir)}: loaded {len(coords)} atoms (custom/configured)")
                    return (coords, elements)
                except Exception as e:
                    warnings.warn(f"Could not load custom PDB {pdb_path}: {e}, trying next source")
    
    # Try minimized.pdb first (usually contains ligand + protein + water)
    minimized_pdb = prep_dir / "minimized.pdb"
    if minimized_pdb.exists():
        try:
            coords, elements = parse_pdb_file(str(minimized_pdb))
            coords_arr = np.array(coords)
            
            # Try to extract ligand from minimized.pdb using atom name matching FIRST
            # This handles systems where coordinates are in different frames (e.g., thrombin_solvent_group2)
            ligand_arr = None
            if core_pdb is not None and sub_pdb is not None:
                ligand_in_pdb_coords = _extract_ligand_from_pdb(str(minimized_pdb), core_pdb, sub_pdb)
                if ligand_in_pdb_coords is not None and len(ligand_in_pdb_coords) > 0:
                    ligand_arr = np.array(ligand_in_pdb_coords)
            
            # Fall back to provided ligand_coords if extraction didn't work
            if ligand_arr is None and ligand_coords is not None and len(ligand_coords) > 0:
                ligand_arr = np.array(ligand_coords)
            
            # Now filter environment atoms by distance to ligand (whichever we got)
            if ligand_arr is not None:
                distances = np.min(
                    np.linalg.norm(coords_arr[:, None, :] - ligand_arr[None, :, :], axis=2),
                    axis=1
                )
                
                # Keep atoms within the ligand cutoff (environment context)
                env_mask = distances <= ligand_cutoff
                
                # Secondary check: if still no atoms found, try periodic boundary condition wrapping
                # This handles systems where ligand is at box edge and solvent wraps around
                if not env_mask.any() or np.sum(env_mask) < 10:
                    # Try to parse CRYST1 record for proper box dimensions
                    box_lengths = _parse_crystal_box_from_pdb(str(minimized_pdb))
                    
                    if box_lengths is not None:
                        # Use minimum image convention with parsed box
                        pbc_distances = _minimum_image_convention(coords_arr, ligand_arr, box_lengths)
                        env_mask = pbc_distances <= ligand_cutoff
                        distances = pbc_distances  # Use PBC distances for sorting
                        
                        if env_mask.any():
                            print(f"        [ENV] PBC (CRYST1) found {np.sum(env_mask)} atoms within {ligand_cutoff} Å")
                    else:
                        # Fallback: try to estimate box from coordinates
                        all_coords = np.vstack([coords_arr, ligand_arr])
                        min_coords = np.min(all_coords, axis=0)
                        max_coords = np.max(all_coords, axis=0)
                        estimated_box = max_coords - min_coords
                        
                        # For cubic system, enforce constraint A=B=C
                        # Use the maximum dimension for all three axes
                        box_length = np.max(estimated_box)
                        box_lengths = np.array([box_length, box_length, box_length], dtype=np.float32)
                        
                        pbc_distances = _minimum_image_convention(coords_arr, ligand_arr, box_lengths)
                        env_mask = pbc_distances <= ligand_cutoff
                        distances = pbc_distances
                        
                        if env_mask.any():
                            print(f"        [ENV] PBC (estimated cubic) found {np.sum(env_mask)} atoms within {ligand_cutoff} Å")
                
                if env_mask.any():
                    env_indices = np.where(env_mask)[0]
                    env_distances = distances[env_indices]
                    
                    # Sort by distance (closest first) for proper 256-atom truncation
                    sorted_order = np.argsort(env_distances)
                    env_indices = env_indices[sorted_order]
                    
                    # Get filtered environment coordinates and elements
                    env_coords_filtered = coords_arr[env_indices]
                    
                    # Wrap to nearest image if PBC was used
                    if 'pbc_distances' in locals() and (not env_mask[env_indices].all()):
                        # We used PBC, so wrap coordinates to nearest image
                        env_coords_filtered = _wrap_coords_to_nearest_image(
                            env_coords_filtered, ligand_arr, box_lengths
                        )
                    
                    coords = [env_coords_filtered[i].tolist() for i in range(len(env_coords_filtered))]
                    elements = [elements[i] for i in env_indices]
                    
                    # Enforce Uni-Mol atom limit (256 max)
                    if len(coords) > 256:
                        coords = coords[:256]
                        elements = elements[:256]
                    
                    n_total = len(coords_arr)
                    n_kept = len(coords)
                    print(f"        [ENV] minimized.pdb: filtered to {n_kept}/{n_total} atoms (distance <= {ligand_cutoff} Å, sorted closest-first, capped at 256)")
                    return (coords, elements)
                else:
                    # No atoms within cutoff of ligand even with PBC
                    print(f"        [ENV] minimized.pdb: no atoms within {ligand_cutoff} Å of ligand (even with PBC wrapping)")
                    return None
            else:
                # No ligand filtering, return all atoms from minimized.pdb
                print(f"        [DEBUG] ligand_coords is None or empty: len(ligand_coords)={len(ligand_coords) if ligand_coords is not None else 'None'}")
                print(f"        [ENV] minimized.pdb: loaded {len(coords)} atoms (no filtering)")
                return (coords, elements)
        except Exception as e:
            warnings.warn(f"Could not load minimized.pdb: {e}, falling back to other sources")
    
    # Fallback: Try individual protein PDB files
    protein_candidates = [
        prep_dir / "protein.pdb",
        prep_dir / "prot.pdb",
        prep_dir / "proa.pdb",
        prep_dir / "pdb" / "protein.pdb",  # Handle nested pdb/protein.pdb (e.g., indolizine_prot)
    ]
    # Also check for proa_*.pdb variants
    protein_candidates.extend(sorted(prep_dir.glob("proa_*.pdb")))
    
    for protein_pdb in protein_candidates:
        if protein_pdb.exists():
            try:
                coords, elements = parse_pdb_file(str(protein_pdb))
                rel_path = protein_pdb.relative_to(prep_dir)
                print(f"        [ENV] {rel_path}: loaded {len(coords)} atoms (protein/structure)")
                return (coords, elements)
            except Exception as e:
                warnings.warn(f"Could not load {protein_pdb.name}: {e}, trying next source")
    
    # Fallback: Try solvent/water PDB files
    solvent_candidates = [
        prep_dir / "solvent.pdb",
        prep_dir / "solv.pdb",
        prep_dir / "waterbox.pdb",
        prep_dir / "water.pdb",
        prep_dir / "solvent_box.pdb",
        prep_dir / "watbox.pdb",
        prep_dir / "environment.pdb",
    ]
    
    for solvent_pdb in solvent_candidates:
        if solvent_pdb.exists():
            try:
                coords, elements = parse_pdb_file(str(solvent_pdb))
                print(f"        [ENV] {solvent_pdb.name}: loaded {len(coords)} atoms (solvent/water)")
                return (coords, elements)
            except Exception as e:
                warnings.warn(f"Could not load {solvent_pdb.name}: {e}, trying next source")
    
    # No environment file found
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

    Args:
        sub_pdb: Path to substituent fragment PDB file (e.g., site1_sub2_frag.pdb)
        core_pdb: Path to core scaffold PDB file
        sub_rtf_data: Optional RTF metadata dict for substituent (for charges/types)
        core_rtf_data: Optional RTF metadata dict for core
        ref_sub_info: Optional list of (pdb_path, rtf_data) tuples for reference subs
            at other sites. For multi-site ligands, should include site#_sub1 from
            each non-target site.
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
    ref_coords_list = []
    ref_elements_list = []
    n_ref_per_site = []
    for ref_pdb, ref_rtf in (ref_sub_info or []):
        ref_coords, ref_elements = parse_pdb_file(ref_pdb, rtf_data=ref_rtf)
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
    env_cutoff: float = 5.0,
    atom_limit: int = 256,
    use_cuda: bool = False,
    custom_search_paths: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute Uni-Mol 512D representation for substituent with full environment context.
    
    Specifically designed for pretraining and online training. Constructs the complete
    system (core + sub + environment) with intelligent fallback chain for environment PDB
    detection and automatic filtering of ligand atoms from minimized.pdb.
    
    **Environment Priority Chain:**
    1. Custom paths (if provided, e.g., from yaml config like pdb/protein.pdb)
    2. minimized.pdb (preferred; ligand atoms automatically filtered by distance)
    3. protein.pdb / proa.pdb / proa_*.pdb / nested pdb/protein.pdb auto-detected variants
    4. solvent.pdb / waterbox.pdb / water.pdb / etc. auto-detected variants
    
    **Ligand Filtering:**
    When loading from minimized.pdb, atoms within env_cutoff (5.0 Å) of any core atom
    are excluded (assumed to be part of the ligand). This ensures correct environment
    context for Uni-Mol representation.
    
    Args:
        sub_pdb: Path to substituent fragment PDB (e.g., site1_sub2_frag.pdb)
        core_pdb: Path to core scaffold PDB
        prep_dir: Prep directory containing environment PDB files (minimized.pdb, etc.)
        sub_rtf_data: Optional RTF metadata dict for substituent
        core_rtf_data: Optional RTF metadata dict for core
        include_other_sites: If True, include reference subs from other sites (default: False)
            For pretraining, usually False (simplicity); for online training might be True
        env_cutoff: Distance cutoff (Ångströms) for environment atoms (default: 5.0)
        atom_limit: Maximum atoms to include in representation (default: 256, Uni-Mol max)
        use_cuda: Whether to use GPU (default: False for CPU)
        custom_search_paths: Optional list of custom PDB paths to search before defaults.
            Used for yaml-configured paths like ['pdb/protein.pdb', 'proa.pdb', 'prob.pdb']
        
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
    
    # Parse core and sub to get full ligand coordinates for environment filtering
    core_coords, core_elements = parse_pdb_file(str(core_pdb))
    sub_coords, sub_elements = parse_pdb_file(str(sub_pdb))
    
    # Combine core+sub coordinates (and ref if requested) for complete ligand distance filtering
    all_ligand_coords = [core_coords, sub_coords]
    all_ligand_elements = [core_elements, sub_elements]
    
    # Debug output: show what's being processed
    sub_name = Path(sub_pdb).name
    print(f"      [UNIMOL] Computing for {sub_name} (cutoff={env_cutoff}Å)")
    
    # Load environment with intelligent fallback chain
    # Automatically filters to keep only atoms within cutoff from FULL ligand (core+sub)
    # Checks custom paths first (from yaml config), then defaults
    # Atoms are sorted by distance (closest first) for proper 256-atom truncation
    try:
        ligand_coords_for_filtering = np.vstack([np.array(c) for c in all_ligand_coords])
    except Exception as e:
        print(f"        [ERROR] Failed to vstack ligand coordinates: {e}")
        ligand_coords_for_filtering = np.array([])
    env_data = _load_environment_pdb(
        prep_dir,
        ligand_coords=ligand_coords_for_filtering,
        ligand_cutoff=env_cutoff,
        custom_search_paths=custom_search_paths,
        core_pdb=core_pdb,
        sub_pdb=sub_pdb,
    )
    
    # Find reference subs if requested (multi-site systems)
    ref_sub_info = None
    if include_other_sites:
        try:
            ref_pdb_list = find_reference_subs_from_other_sites(sub_pdb, str(prep_dir), cutoff=5.0)
            if ref_pdb_list:
                ref_sub_info = [(p, None) for p in ref_pdb_list]  # No RTF data for refs
        except Exception as e:
            warnings.warn(f"Could not find reference subs: {e}")
    
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
        return get_unimol_representation(ligand_dict, use_cuda=use_cuda)
    except Exception as e:
        raise Exception(
            f"Failed to compute Uni-Mol representation for {Path(sub_pdb).name}: {e}"
        )
