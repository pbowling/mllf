"""PDB file parser

Provides helpers to parse PDB files and extract atomic coordinates and element symbols.
Handles non-standard PDB formats where atom names encode element symbols (e.g., C001, H002, CL01).

Functions:
- parse_pdb_file(path) -> tuple: (coordinates, elements)
- extract_site_number(pdb_path) -> int: extract site number from filename
- find_duplicate_atoms(coords1, coords2, tolerance) -> list: indices of duplicates in coords2
- remove_duplicate_atoms(coords, elements, core_coords) -> tuple: filtered (coords, elements)
- combine_pdb_files(pdb_files) -> tuple: (coordinates, elements, atom_counts)
- calculate_min_distance(coords1, coords2) -> float: minimum distance between two structures
- find_nearby_pdbs(target_pdb, candidate_pdbs, cutoff) -> list: PDBs within cutoff distance
- find_reference_subs_from_other_sites(target_pdb, prep_dir, cutoff) -> list: reference subs from other sites
- parse_pdb_dir(directory, pattern='*.pdb') -> dict mapping filename -> parsed data

Spatial Filtering for Multi-Site Systems:
The calculate_min_distance() and find_nearby_pdbs() functions enable automatic detection
of which PDB files should be included in AEV computation based on spatial proximity.

For multi-site systems, use find_reference_subs_from_other_sites() to get only the
reference substituent (site#_sub1) from other sites, excluding the current site.

Duplicate atom detection (find_duplicate_atoms/remove_duplicate_atoms) prevents double
counting when reference substituents share atoms with the core structure.

For ANI-2x, use cutoff=5.1 Å (radial) or 3.5 Å (angular) to match the AEV function cutoffs.

"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import glob
import numpy as np
import re


def parse_pdb_file(pdb_path: str, rtf_data: Optional[dict] = None) -> Tuple[List[List[float]], List[str]]:
    """Parse PDB file to extract coordinates and elements.
    
    Uses a three-tier parsing strategy:
    1. RDKit PDB parser (most robust for standard PDB formats)
    2. Flexible whitespace parsing (for CHARMM-style non-standard formats)
    3. Fixed-column PDB format (last resort)
    
    Optionally validates element identification against CGenFF atom types from RTF.
    
    Args:
        pdb_path: Path to PDB file
        rtf_data: Optional dict from parse_rtf_file() with 'atom_types' key.
                  If provided, elements are validated against CGenFF atom types.
        
    Returns:
        tuple: (coordinates, elements) where:
            - coordinates: list of [x, y, z] coordinate lists
            - elements: list of element symbols (str)
    """
    coordinates = []
    elements = []
    
    # Get CGenFF atom types if RTF data provided
    cgenff_atom_types = None
    if rtf_data is not None:
        cgenff_atom_types = rtf_data.get('atom_types', [])
    
    # TIER 1: Try RDKit parser first (handles standard PDB formats best)
    try:
        from rdkit import Chem
        from rdkit.Chem.rdBase import BlockLogs
        with BlockLogs():
            mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        
        if mol is not None and mol.GetNumAtoms() > 0:
            # Successfully parsed with RDKit - extract coordinates and elements
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                pos = conf.GetAtomPosition(atom_idx)
                coordinates.append([pos.x, pos.y, pos.z])
                elements.append(atom.GetSymbol())
            
            # Cross-validate with CGenFF if provided
            if cgenff_atom_types is not None:
                from mllf.cb.atom_vocab import infer_element_from_cgenff_type
                for atom_index, (element, cgenff_type) in enumerate(zip(elements, cgenff_atom_types)):
                    if atom_index >= len(cgenff_atom_types):
                        break
                    cgenff_element = infer_element_from_cgenff_type(cgenff_type)
                    if element.upper() != cgenff_element.upper():
                        warnings.warn(
                            f"Element mismatch in {pdb_path} at atom {atom_index}: "
                            f"RDKit -> '{element}' but CGenFF type '{cgenff_type}' -> '{cgenff_element}'. "
                            f"Using CGenFF element '{cgenff_element}'."
                        )
                        elements[atom_index] = cgenff_element
            
            return coordinates, elements
    except Exception as e:
        # RDKit failed - likely CHARMM-style non-standard format
        # Fall through to manual parsing
        pass
    
    # TIER 2 & 3: Manual line-by-line parsing for non-standard formats
    atom_index = 0
    parse_errors = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            
            atom_name = None
            x = y = z = None
            
            # TIER 2: Try flexible parsing (for CHARMM-style formats)
            try:
                parts = line.split()
                if len(parts) >= 6:  # Need at least: ATOM, serial, name, res, x, y, z
                    # Find atom name (typically 3rd field after ATOM and serial)
                    atom_name = parts[2]
                    
                    # Find coordinates - look for 3 consecutive floats with decimal points
                    # This distinguishes coordinates from integer residue numbers or chain IDs
                    # PDB coordinates always have decimal points (e.g., "10.000")
                    coords_found = None
                    for start_idx in range(3, len(parts) - 2):  # Need room for 3 values
                        try:
                            # Check that all 3 values have decimal points (are true coordinates)
                            if '.' in parts[start_idx] and '.' in parts[start_idx + 1] and '.' in parts[start_idx + 2]:
                                x_cand = float(parts[start_idx])
                                y_cand = float(parts[start_idx + 1])
                                z_cand = float(parts[start_idx + 2])
                                # Found 3 consecutive decimal floats - these are coordinates
                                coords_found = (x_cand, y_cand, z_cand)
                                break
                        except (ValueError, IndexError):
                            continue
                    
                    if coords_found:
                        x, y, z = coords_found
            except (ValueError, IndexError):
                pass
            
            # TIER 3: Fall back to fixed-column PDB format
            if atom_name is None or x is None:
                try:
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    pass
            
            # If all tiers failed, log error and skip
            if atom_name is None or x is None:
                error_msg = f"Failed to parse PDB line in {pdb_path}: {line.strip()}"
                parse_errors.append(error_msg)
                warnings.warn(error_msg)
                continue

            # Skip CHARMM pseudo-atoms (lone pairs and dummy atoms).
            # These are not real atoms and must not contribute to AEV computation.
            #   LP*  – lone pair virtual sites (e.g. LP0A, LP0B, LP0Q)
            #   DM*  – dummy atoms used in alchemical transformations (e.g. DM01, DM02)
            # They have no physical element and would otherwise be defaulted to 'H',
            # introducing spurious radial/angular terms into the AEV.
            if atom_name.upper().startswith(('LP', 'DM')):
                atom_index += 1
                continue

            # Extract element from atom name (e.g., "C001", "H002", "CL01", "B085")
            # For RCSB format, also check explicit element symbol in columns 76-77
            element_from_column = None
            if len(line) >= 78:
                raw_col = line[76:78].strip()
                if raw_col and raw_col.replace(' ', '').isalpha():
                    # Normalize capitalization: 'BR' -> 'Br', 'CL' -> 'Cl', 'C' -> 'C', etc.
                    norm = raw_col.capitalize()
                    # Validate against the known chemistry vocabulary to reject
                    # stray characters from segment-name fields that overlap this column
                    # (e.g. the trailing letter of a CHARMM segment such as 'LIGB').
                    _VALID_ELEMENTS = {
                        'H', 'C', 'N', 'O', 'F', 'S', 'P', 'B', 'I', 'K',
                        'Cl', 'Br', 'Al', 'Se', 'Na', 'Ca', 'Mg', 'Zn', 'Fe',
                    }
                    if norm in _VALID_ELEMENTS:
                        element_from_column = norm
                        element = norm
            
            # Fall back to extracting from atom name if no explicit element column
            if not element_from_column:
                element = None
                for two_letter in ['Cl', 'Br', 'Al', 'Se']:
                    if atom_name.upper().startswith(two_letter.upper()):
                        element = two_letter
                        break
                
                if element is None:
                    first_char = atom_name[0].upper()
                    if first_char in ['H', 'C', 'N', 'O', 'F', 'S', 'P', 'B', 'I']:
                        element = first_char
                
                if element is None:
                    warnings.warn(f"Could not determine element for atom {atom_name} in {pdb_path}")
                    element = 'H'  # Default fallback
            
            # Cross-validate with CGenFF atom type if available
            if cgenff_atom_types is not None and atom_index < len(cgenff_atom_types):
                cgenff_type = cgenff_atom_types[atom_index]
                # Import here to avoid circular dependency
                from mllf.cb.atom_vocab import infer_element_from_cgenff_type
                cgenff_element = infer_element_from_cgenff_type(cgenff_type)
                
                # Check for mismatch
                if element.upper() != cgenff_element.upper():
                    warnings.warn(
                        f"Element mismatch in {pdb_path}: "
                        f"PDB atom '{atom_name}' -> '{element}' but CGenFF type '{cgenff_type}' -> '{cgenff_element}'. "
                        f"Using CGenFF element '{cgenff_element}'."
                    )
                    element = cgenff_element  # Trust CGenFF over PDB parsing
            
            coordinates.append([x, y, z])
            elements.append(element)
            atom_index += 1
    
    # Check for critical parsing errors
    if parse_errors and not coordinates:
        error_msg = f"CRITICAL: Failed to parse any atoms from {pdb_path}. Errors:\n" + "\n".join(parse_errors[:5])
        raise ValueError(error_msg)
    
    return coordinates, elements


def extract_site_number(pdb_path: str) -> Optional[int]:
    """Extract site number from PDB filename.
    
    Args:
        pdb_path: Path to PDB file (e.g., "site1_sub2_frag.pdb")
        
    Returns:
        Site number as integer, or None if not found
    """
    filename = Path(pdb_path).name
    match = re.search(r'site(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def find_duplicate_atoms(coords1: List[List[float]], 
                        coords2: List[List[float]], 
                        tolerance: float = 1e-4) -> List[int]:
    """Find indices in coords2 that duplicate atoms in coords1.
    
    Two atoms are considered duplicates if their coordinates match within tolerance.
    
    Args:
        coords1: List of [x, y, z] coordinates (reference structure)
        coords2: List of [x, y, z] coordinates (structure to check)
        tolerance: Distance tolerance in Angstroms for matching (default: 0.0001 Å)
        
    Returns:
        List of indices in coords2 that are duplicates of atoms in coords1
    """
    if not coords1 or not coords2:
        return []
    
    arr1 = np.array(coords1)  # [n1, 3]
    arr2 = np.array(coords2)  # [n2, 3]
    
    duplicate_indices = []
    
    # For each atom in coords2, check if it matches any atom in coords1
    for i, coord2 in enumerate(arr2):
        distances = np.sqrt(np.sum((arr1 - coord2)**2, axis=1))
        if np.min(distances) < tolerance:
            duplicate_indices.append(i)
    
    return duplicate_indices


def remove_duplicate_atoms(coords: List[List[float]], 
                          elements: List[str],
                          core_coords: List[List[float]],
                          tolerance: float = 1e-4) -> Tuple[List[List[float]], List[str]]:
    """Remove atoms from coords/elements that duplicate atoms in core_coords.
    
    Args:
        coords: Coordinate list to filter
        elements: Element list to filter (same length as coords)
        core_coords: Reference coordinates (atoms to avoid duplicating)
        tolerance: Distance tolerance for duplicate detection (default: 0.0001 Å)
        
    Returns:
        Filtered (coords, elements) with duplicates removed
    """
    duplicate_indices = find_duplicate_atoms(core_coords, coords, tolerance)
    
    if not duplicate_indices:
        return coords, elements
    
    # Remove duplicates by keeping only non-duplicate indices
    duplicate_set = set(duplicate_indices)
    filtered_coords = [c for i, c in enumerate(coords) if i not in duplicate_set]
    filtered_elements = [e for i, e in enumerate(elements) if i not in duplicate_set]
    
    return filtered_coords, filtered_elements


def combine_pdb_files(pdb_files: List[str]) -> Tuple[List[List[float]], List[str], List[int]]:
    """Combine multiple PDB files into coordinate and element lists.
    
    Args:
        pdb_files: List of PDB file paths to combine
        
    Returns:
        tuple: (coordinates, elements, atom_counts) where:
            - coordinates: list of [x,y,z] for all atoms
            - elements: list of element symbols for all atoms
            - atom_counts: list of atom counts from each file
    """
    all_coords = []
    all_elements = []
    atom_counts = []
    
    for pdb_path in pdb_files:
        coords, elements = parse_pdb_file(pdb_path)
        all_coords.extend(coords)
        all_elements.extend(elements)
        atom_counts.append(len(coords))
    
    return all_coords, all_elements, atom_counts


def calculate_min_distance(coords1: List[List[float]], coords2: List[List[float]]) -> float:
    """Calculate minimum distance between two sets of coordinates.
    
    Args:
        coords1: List of [x, y, z] coordinates for first structure
        coords2: List of [x, y, z] coordinates for second structure
        
    Returns:
        Minimum distance in Angstroms between any two atoms
    """
    if not coords1 or not coords2:
        return float('inf')
    
    # Convert to numpy arrays for efficient computation
    arr1 = np.array(coords1)  # [n1, 3]
    arr2 = np.array(coords2)  # [n2, 3]
    
    # Calculate all pairwise distances
    # Broadcasting: arr1[:, None, :] has shape [n1, 1, 3]
    #               arr2[None, :, :] has shape [1, n2, 3]
    # Difference has shape [n1, n2, 3]
    diff = arr1[:, None, :] - arr2[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))  # [n1, n2]
    
    return float(np.min(distances))


def find_nearby_pdbs(target_pdb: str, 
                     candidate_pdbs: List[str], 
                     cutoff: float = 5.1) -> List[str]:
    """Find PDB files with atoms within cutoff distance of target PDB.
    
    This function determines which additional PDB files should be included
    in the molecular context for AEV computation based on spatial proximity.
    
    Args:
        target_pdb: Path to target PDB file
        candidate_pdbs: List of candidate PDB file paths to check
        cutoff: Distance cutoff in Angstroms (default: 5.1 Å for ANI-2x radial cutoff)
        
    Returns:
        List of PDB file paths that are within cutoff distance of target
    """
    nearby = []
    target_coords, _ = parse_pdb_file(target_pdb)
    
    for candidate_pdb in candidate_pdbs:
        if candidate_pdb == target_pdb:
            continue
            
        candidate_coords, _ = parse_pdb_file(candidate_pdb)
        min_dist = calculate_min_distance(target_coords, candidate_coords)
        
        if min_dist <= cutoff:
            nearby.append(candidate_pdb)
    
    return nearby


def find_reference_subs_from_other_sites(target_pdb: str,
                                         prep_dir: str,
                                         cutoff: float = 5.1) -> List[str]:
    """Find reference substituents (sub1) from other sites within cutoff distance.
    
    For multi-site systems, this function identifies which OTHER sites have their
    reference substituent (site#_sub1) within the cutoff distance. Only sub1 is 
    used from each site to provide a consistent reference structure.
    
    Substituents from the SAME site are excluded (those are what we're comparing).
    
    Args:
        target_pdb: Path to target substituent PDB file (e.g., "site1_sub2_frag.pdb")
        prep_dir: Directory containing all PDB files
        cutoff: Distance cutoff in Angstroms (default: 5.1 Å)
        
    Returns:
        List of reference substituent PDB paths from other sites within cutoff
    """
    target_site = extract_site_number(target_pdb)
    if target_site is None:
        return []
    
    prep_path = Path(prep_dir)
    
    # Find all site#_sub1 reference files
    reference_subs = sorted(prep_path.glob("site*_sub1_frag.pdb"))
    
    # Filter to only other sites (not the same site as target)
    other_site_refs = []
    for ref_pdb in reference_subs:
        ref_site = extract_site_number(str(ref_pdb))
        if ref_site is not None and ref_site != target_site:
            other_site_refs.append(str(ref_pdb))
    
    # Check which are within cutoff distance
    nearby_refs = find_nearby_pdbs(target_pdb, other_site_refs, cutoff)
    
    return nearby_refs


def parse_pdb_dir(directory: str, pattern: str = '*.pdb') -> Dict[str, Dict[str, object]]:
    """Parse all PDB files in a directory and return a mapping keyed by filename.
    
    Args:
        directory: Directory containing PDB files
        pattern: Glob pattern for matching PDB files (default: '*.pdb')
        
    Returns:
        dict mapping filename (without extension) to parsed data:
            {
                'coordinates': list of [x, y, z] lists,
                'elements': list of element symbols,
                'num_atoms': int,
                'filename': str,
                'filepath': str
            }
    """
    results: Dict[str, Dict[str, object]] = {}
    
    # Use pathlib for cleaner path handling
    dir_path = Path(directory)
    pdb_files = sorted(dir_path.glob(pattern))
    
    for pdb_file in pdb_files:
        coords, elements = parse_pdb_file(str(pdb_file))
        key = pdb_file.stem  # filename without extension
        
        results[key] = {
            'coordinates': coords,
            'elements': elements,
            'num_atoms': len(coords),
            'filename': pdb_file.name,
            'filepath': str(pdb_file.absolute())
        }
    
    return results


if __name__ == '__main__':
    # Quick smoke test when run directly
    import json
    import sys
    
    # Try to find example PDB files
    example_dirs = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', '14benz', 'prep'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pretraining', '14benz_solv'),
    ]
    
    for example_dir in example_dirs:
        example_dir = os.path.abspath(example_dir)
        if os.path.exists(example_dir):
            print(f"Parsing PDB files in: {example_dir}")
            results = parse_pdb_dir(example_dir)
            
            # Print summary
            print(f"Found {len(results)} PDB files:")
            for key, data in list(results.items())[:3]:  # Show first 3
                print(f"  {key}: {data['num_atoms']} atoms, elements: {set(data['elements'])}")
            
            if len(results) > 3:
                print(f"  ... and {len(results) - 3} more files")
            
            sys.exit(0)
    
    print("No example PDB directories found for smoke test.")
