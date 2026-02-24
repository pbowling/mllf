import torch
from torchani import AEVComputer
from rdkit import Chem
import warnings
from pathlib import Path

from mllf.file_handling.read_pdb import (
    parse_pdb_file, 
    find_reference_subs_from_other_sites,
    remove_duplicate_atoms
)

# Element to species ID mapping for common elements + unknown
# Rare CGenFF elements (B, Se, Al) mapped to 'X' (unknown) to reduce AEV dimensions
# This reduces AEV from 3120D (13 species) to 1920D (11 species)
ELEMENT_TO_ID = {
    # Common organic and drug-like elements (IDs 0-9)
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 
    'S': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'P': 9,
    # Unknown/rare element (ID 10)
    'X': 10,
    # Rare CGenFF elements mapped to unknown with warning
    'Al': 10,  # Aluminum (rare: only AlF4- compounds)
    'B': 10,   # Boron (rare: only boronic acids)
    'Se': 10,  # Selenium (rare: only selenocarbonyl)
}
NUM_SPECIES = 11  # 10 common elements + 1 unknown


# Initialize the computer with ANI-2x spatial grids, but with 13 species (all CGenFF elements)
aev_computer = AEVComputer.like_2x(num_species=NUM_SPECIES)


def extract_charges_from_pdb(pdb_path):
    """Extract partial charges from PDB file.
    
    Charges are typically stored in the temperature factor (B-factor) column or
    in the occupancy column of PDB files. This function tries both.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        torch.Tensor: [num_atoms] partial charges
    """
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not read molecule from {pdb_path}")
    
    num_atoms = mol.GetNumAtoms()
    charges = torch.zeros(num_atoms, dtype=torch.float32)
    
    # Try to extract charges from PDB file
    # Common convention: charges stored in B-factor column
    conformer = mol.GetConformer()
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        # RDKit doesn't directly expose B-factor, so we'll need to parse the file
        # For now, check if formal charge is available
        if atom.HasProp('_GasteigerCharge'):
            charges[i] = float(atom.GetProp('_GasteigerCharge'))
        elif atom.HasProp('_TriposPartialCharge'):
            charges[i] = float(atom.GetProp('_TriposPartialCharge'))
        else:
            # Default to formal charge if no partial charge available
            charges[i] = float(atom.GetFormalCharge())
    
    return charges


def extract_charges_from_rtf_metadata(rtf_entry):
    """Extract atomic charges from RTF metadata parsed by parse_rtf_dir.
    
    Args:
        rtf_entry: Dict from RTF parser containing 'charges' key, or None
        
    Returns:
        torch.Tensor: [num_atoms] partial charges, or None if not available
    """
    if rtf_entry is None:
        return None
    charges = rtf_entry.get('charges')
    if charges is None:
        return None
    return torch.tensor(charges, dtype=torch.float32)


def get_substituent_aevs(pdb_path):
    """Compute AEVs for all atoms in a substituent PDB file.
    
    Args:
        pdb_path: Path to substituent PDB file
        
    Returns:
        torch.Tensor: [num_atoms, aev_length] AEV vectors
    """
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not read molecule from {pdb_path}")
    
    # 1. Map string elements to integer IDs
    element_ids = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ELEMENT_TO_ID:
            warnings.warn(f"Element {symbol} not in ELEMENT_TO_ID mapping, using H as fallback")
            element_ids.append(ELEMENT_TO_ID['H'])
        else:
            # Warn if rare element is being mapped to unknown
            if symbol in ['Al', 'B', 'Se']:
                warnings.warn(f"Rare element {symbol} mapped to 'X' (unknown) - AEV may be less accurate")
            element_ids.append(ELEMENT_TO_ID[symbol])
    
    species_tensor = torch.tensor(element_ids, dtype=torch.long).unsqueeze(0)
    
    # 2. Extract Coordinates
    coords = [mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coordinates_tensor = torch.tensor([[pos.x, pos.y, pos.z] for pos in coords], dtype=torch.float32).unsqueeze(0)
    
    # 3. Compute AEVs
    with torch.no_grad():
        # Notice we pass the arguments directly as expected by the new forward() method
        aevs = aev_computer(species_tensor, coordinates_tensor)
        
    return aevs.squeeze(0) # Shape: [Num_Atoms, AEV_Length]


def get_atom_features(pdb_path, rtf_entry=None, include_charges=True, include_atom_ids=True):
    """Compute complete atom-level features for DeepSet input.
    
    This is the Step 1 of the 4-step pipeline: generating atom-level physical representations.
    
    Args:
        pdb_path: Path to substituent PDB file
        rtf_entry: Optional RTF metadata dict containing charges
        include_charges: Whether to extract/include charges
        include_atom_ids: Whether to include atom type IDs
        
    Returns:
        dict with keys:
            - 'aevs': [num_atoms, aev_length] AEV vectors
            - 'charges': [num_atoms] partial charges (if include_charges=True) 
            - 'atom_ids': [num_atoms] integer atom type IDs (if include_atom_ids=True)
    """
    result = {}
    
    # Compute AEVs
    aevs = get_substituent_aevs(pdb_path)
    result['aevs'] = aevs
    num_atoms = aevs.shape[0]
    
    # Extract charges
    if include_charges:
        charges = None
        # Try RTF metadata first
        if rtf_entry is not None:
            charges = extract_charges_from_rtf_metadata(rtf_entry)
        # Fall back to PDB file
        if charges is None:
            try:
                charges = extract_charges_from_pdb(pdb_path)
            except Exception as e:
                warnings.warn(f"Could not extract charges from {pdb_path}: {e}. Using zeros.")
                charges = torch.zeros(num_atoms, dtype=torch.float32)
        result['charges'] = charges
    
    # Extract atom IDs
    if include_atom_ids:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        element_ids = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in ELEMENT_TO_ID:
                warnings.warn(f"Element {symbol} not in ELEMENT_TO_ID mapping, using H as fallback")
                element_ids.append(ELEMENT_TO_ID['H'])
            else:
                # Warn if rare element is being mapped to unknown
                if symbol in ['Al', 'B', 'Se']:
                    warnings.warn(f"Rare element {symbol} mapped to 'X' (unknown) - AEV may be less accurate")
                element_ids.append(ELEMENT_TO_ID[symbol])
        
        result['atom_ids'] = torch.tensor(element_ids, dtype=torch.long)
    
    return result


def parse_pdb_coordinates_and_elements(pdb_path):
    """Parse PDB file to extract coordinates and elements.
    
    Wrapper around read_pdb.parse_pdb_file() that converts coordinates to tensor.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        tuple: (coordinates, elements) where:
            - coordinates: torch.Tensor [num_atoms, 3]
            - elements: list of element symbols
    """
    coords_list, elements = parse_pdb_file(pdb_path)
    coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
    return coords_tensor, elements


def get_atom_features_with_context(substituent_pdb, core_pdb, protein_pdb=None, 
                                   rtf_entry=None, include_charges=True, include_atom_ids=True,
                                   prep_dir=None, aev_cutoff=5.1):
    """Compute atom-level features with full molecular context.
    
    This function properly combines core + substituent (+ protein + nearby sites) to compute
    accurate AEVs where each atom sees its full molecular environment.
    
    Critical: AEVs are environment-dependent. Substituent atoms must see their
    bonded neighbors in the core to get accurate atomic environments.
    
    For multi-site systems, this function automatically detects and includes substituents 
    from other sites if they are within the AEV spatial cutoff (default 5.1 Å for ANI-2x).
    Similarly, protein atoms within the cutoff are included when available.
    
    Args:
        substituent_pdb: Path to substituent PDB file
        core_pdb: Path to core PDB file  
        protein_pdb: Optional path to protein PDB file
        rtf_entry: Optional RTF metadata dict containing charges
        include_charges: Whether to extract/include charges
        include_atom_ids: Whether to include atom type IDs
        prep_dir: Optional path to prep directory for multi-site spatial filtering
        aev_cutoff: Distance cutoff in Angstroms for including nearby atoms (default: 5.1 Å)
        
    Returns:
        dict with keys:
            - 'aevs': [num_sub_atoms, aev_length] AEV vectors for substituent atoms
            - 'charges': [num_sub_atoms] partial charges (if include_charges=True)
            - 'atom_ids': [num_sub_atoms] atom type IDs (if include_atom_ids=True)
            - 'num_atoms': Number of substituent atoms
            - 'context_info': Dict with information about included context atoms (if prep_dir provided)
    """
    # Start with core
    pdb_files = [core_pdb]
    context_labels = ['core']
    
    # Parse core coordinates (needed for duplicate detection)
    core_coords_list, core_elements = parse_pdb_file(str(core_pdb))
    
    # Check for reference substituents from other sites (multi-site filtering)
    nearby_refs = []
    nearby_ref_info = []
    if prep_dir is not None:
        # Find reference subs (sub1) from other sites within cutoff
        nearby_refs = find_reference_subs_from_other_sites(
            target_pdb=str(substituent_pdb),
            prep_dir=prep_dir,
            cutoff=aev_cutoff
        )
        
        if nearby_refs:
            warnings.warn(
                f"Multi-site system: Found {len(nearby_refs)} reference substituent(s) "
                f"from other sites within {aev_cutoff} Å cutoff. Including for accurate "
                f"AEV computation (duplicates with core will be removed)."
            )
            
            # Process each reference sub: remove duplicates with core
            for ref_sub_path in nearby_refs:
                ref_coords, ref_elements = parse_pdb_file(ref_sub_path)
                
                # Remove atoms that duplicate core atoms
                filtered_coords, filtered_elements = remove_duplicate_atoms(
                    coords=ref_coords,
                    elements=ref_elements,
                    core_coords=core_coords_list,
                    tolerance=1e-4
                )
                
                removed_count = len(ref_coords) - len(filtered_coords)
                if removed_count > 0:
                    ref_name = Path(ref_sub_path).name
                    nearby_ref_info.append({
                        'pdb': ref_name,
                        'original_atoms': len(ref_coords),
                        'filtered_atoms': len(filtered_elements),
                        'removed_duplicates': removed_count
                    })
                else:
                    ref_name = Path(ref_sub_path).name
                    nearby_ref_info.append({
                        'pdb': ref_name,
                        'original_atoms': len(ref_coords),
                        'filtered_atoms': len(filtered_elements),
                        'removed_duplicates': 0
                    })
                
                # Add filtered coordinates as tensors
                if filtered_coords:
                    pdb_files.append((filtered_coords, filtered_elements))
                    context_labels.append(f'other_site_ref_{len(nearby_ref_info)}')
    
    # Check if protein is nearby (for protein phase systems)
    include_protein = False
    if protein_pdb is not None:
        if prep_dir is not None:
            # Use spatial filtering for protein
            from mllf.file_handling.read_pdb import find_nearby_pdbs
            protein_nearby = find_nearby_pdbs(str(substituent_pdb), [str(protein_pdb)], cutoff=aev_cutoff)
            if protein_nearby:
                include_protein = True
                warnings.warn(
                    f"Including protein atoms within {aev_cutoff} Å cutoff for accurate AEV computation."
                )
        else:
            # No prep_dir: include all protein (backward compatibility)
            include_protein = True
    
    if include_protein:
        pdb_files.append(protein_pdb)
        context_labels.append('protein')
    
    # Add substituent last so we can track its indices
    pdb_files.append(substituent_pdb)
    context_labels.append('substituent')
    
    # Parse all PDB files
    all_coords = []
    all_elements = []
    atom_counts = []
    
    for item in pdb_files:
        if isinstance(item, tuple):
            # Pre-parsed coordinates and elements (from filtered reference subs)
            coords_list, elements = item
            coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
        else:
            # PDB path - parse it
            coords_tensor, elements = parse_pdb_coordinates_and_elements(item)
        
        all_coords.append(coords_tensor)
        all_elements.extend(elements)
        atom_counts.append(len(elements))
    
    # Combine coordinates
    combined_coords = torch.cat(all_coords, dim=0)  # [total_atoms, 3]
    
    # Determine substituent atom indices (always last in the list)
    sub_start = sum(atom_counts[:-1])
    sub_end = sum(atom_counts)
    
    # Convert elements to species IDs
    species_ids = []
    for element in all_elements:
        if element not in ELEMENT_TO_ID:
            warnings.warn(f"Element {element} not in ELEMENT_TO_ID mapping, using H")
            species_ids.append(ELEMENT_TO_ID['H'])
        else:
            # Warn if rare element is being mapped to unknown
            if element in ['Al', 'B', 'Se']:
                warnings.warn(f"Rare element {element} mapped to 'X' (unknown) - AEV may be less accurate")
            species_ids.append(ELEMENT_TO_ID[element])
    
    species_tensor = torch.tensor(species_ids, dtype=torch.long).unsqueeze(0)
    coords_tensor = combined_coords.unsqueeze(0)
    
    # Compute AEVs for ALL atoms (gives substituent atoms proper context)
    with torch.no_grad():
        all_aevs = aev_computer(species_tensor, coords_tensor)
        all_aevs = all_aevs.squeeze(0)  # [total_atoms, aev_length]
    
    # Extract substituent features
    result = {
        'aevs': all_aevs[sub_start:sub_end],
        'num_atoms': sub_end - sub_start
    }
    
    # Add context information if prep_dir was provided
    if prep_dir is not None:
        result['context_info'] = {
            'total_context_atoms': sum(atom_counts),
            'context_sources': context_labels,
            'atom_counts_per_source': atom_counts,
            'reference_subs_from_other_sites': nearby_ref_info,
            'protein_included': include_protein,
            'cutoff_used': aev_cutoff
        }
    
    # Add charges if requested
    if include_charges:
        charges = None
        if rtf_entry is not None:
            charges = extract_charges_from_rtf_metadata(rtf_entry)
        if charges is None:
            warnings.warn(f"No charges available for {substituent_pdb}. Using zeros.")
            charges = torch.zeros(result['num_atoms'], dtype=torch.float32)
        result['charges'] = charges
    
    # Add atom IDs if requested
    if include_atom_ids:
        sub_species = species_tensor.squeeze(0)[sub_start:sub_end]
        result['atom_ids'] = sub_species
    
    return result