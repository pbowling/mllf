import re
import tempfile
import torch
import numpy as np
from torchani import AEVComputer
from rdkit import Chem
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from mllf.file_handling.read_pdb import (
    parse_pdb_file, 
    find_reference_subs_from_other_sites,
    remove_duplicate_atoms
)

# Element to species ID mapping for common elements + unknown
# Rare CGenFF elements (B, Se, Al) mapped to 'X' (unknown) to reduce AEV dimensions
# This reduces AEV from 3120D (13 species) to 2288D (11 species)
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


# Initialize the computer with ANI-2x spatial grids, but with 11 species
aev_computer = AEVComputer.like_2x(num_species=NUM_SPECIES)


def map_element_to_species_id(element: str, warn_rare: bool = True) -> int:
    """Map element symbol to species ID with appropriate warnings.
    
    This is a helper function to centralize the element-to-ID mapping logic
    and avoid code duplication throughout the module.
    
    Args:
        element: Element symbol (e.g., 'C', 'H', 'Cl')
        warn_rare: Whether to warn about rare elements (default: True)
        
    Returns:
        Species ID (integer 0-10)
    """
    if element not in ELEMENT_TO_ID:
        warnings.warn(f"Element {element} not in ELEMENT_TO_ID mapping, using H as fallback")
        return ELEMENT_TO_ID['H']
    
    # Warn if rare element is being mapped to unknown
    if warn_rare and element in ['Al', 'B', 'Se']:
        warnings.warn(f"Rare element {element} mapped to 'X' (unknown) - AEV may be less accurate")
    
    return ELEMENT_TO_ID[element]


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


def get_substituent_aevs(pdb_path, validate_with_rtf=True):
    """Compute AEVs for all atoms in a substituent PDB file.
    
    Uses CHARMM-format PDB parser as primary method (handles PDB files without
    element columns), with RDKit as fallback for standard PDB files.
    
    Args:
        pdb_path: Path to substituent PDB file
        validate_with_rtf: If True, cross-validate element identification with CGenFF
                          atom types from RTF file (default: True)
        
    Returns:
        torch.Tensor: [num_atoms, aev_length] AEV vectors
    """
    # Try CHARMM-format parsing first (handles MSLD prep directory PDB files)
    try:
        # Try to load RTF data for validation if requested
        rtf_data = None
        if validate_with_rtf:
            from mllf.file_handling.read_rtf import parse_rtf_file
            pdb_path_obj = Path(pdb_path)
            rtf_path = pdb_path_obj.parent / pdb_path_obj.name.replace('_frag.pdb', '_pres.rtf')
            if rtf_path.exists():
                try:
                    rtf_data = parse_rtf_file(str(rtf_path))
                except Exception as e:
                    warnings.warn(f"Could not parse RTF file {rtf_path} for validation: {e}")
        
        coords_list, elements_list = parse_pdb_file(str(pdb_path), rtf_data=rtf_data)
        
        if not coords_list or not elements_list:
            raise ValueError(f"No atoms found in {pdb_path}")
        
        # Map elements to species IDs
        element_ids = [map_element_to_species_id(element) for element in elements_list]
        
        # Convert to tensors
        species_tensor = torch.tensor(element_ids, dtype=torch.long).unsqueeze(0)
        coordinates_tensor = torch.tensor(coords_list, dtype=torch.float32).unsqueeze(0)
        
    except Exception as e:
        # Fall back to RDKit for standard PDB files
        warnings.warn(f"CHARMM parser failed for {pdb_path}: {e}. Trying RDKit fallback.")
        
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is None:
            raise ValueError(f"Could not read molecule from {pdb_path} with either CHARMM or RDKit parser")
        
        # Map string elements to integer IDs
        element_ids = [map_element_to_species_id(atom.GetSymbol()) for atom in mol.GetAtoms()]
        
        species_tensor = torch.tensor(element_ids, dtype=torch.long).unsqueeze(0)
        
        # Extract Coordinates
        coords = [mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        coordinates_tensor = torch.tensor([[pos.x, pos.y, pos.z] for pos in coords], dtype=torch.float32).unsqueeze(0)
    
    # Compute AEVs
    with torch.no_grad():
        aevs = aev_computer(species_tensor, coordinates_tensor)
        
    return aevs.squeeze(0)  # Shape: [Num_Atoms, AEV_Length]


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
    
    # Extract atom IDs (and atom names for bond-topology-aware models)
    if include_atom_ids:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        element_ids = []
        atom_names_list = []
        for atom in mol.GetAtoms():
            element_ids.append(map_element_to_species_id(atom.GetSymbol()))
            info = atom.GetMonomerInfo()
            atom_names_list.append(info.GetName().strip() if info else f'X{atom.GetIdx()}')
        result['atom_ids'] = torch.tensor(element_ids, dtype=torch.long)
        result['atom_names'] = atom_names_list

    return result


def get_bond_edge_index_from_pdb(
    pdb_path: str,
    rtf_bonds: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build bidirectional bond edge index and bond-type edge attributes.

    Priority: RTF BOND section (when rtf_bonds provided) → RDKit Chem.MolFromPDBFile.
    CHARMM-format PDB files lack a filled element column, so RDKit emits per-atom
    warnings and frequently returns a mol with no bonds; RTF bonds avoid this entirely.

    Bond-type weights: SINGLE=1.0, DOUBLE=2.0, TRIPLE=3.0, AROMATIC=1.5 (RDKit path only).

    Args:
        pdb_path: Path to substituent PDB file.
        rtf_bonds: Optional list of (atom_name1, atom_name2) tuples from the RTF
                   BOND section, used as fallback when RDKit returns None or has
                   no bonds.

    Returns:
        edge_index: [2, 2E] bidirectional edge indices (both directions per bond)
        edge_attr:  [2E, 1] bond-type weights (one per directed edge)
    """
    BOND_TYPE_WEIGHT = {
        Chem.rdchem.BondType.SINGLE: 1.0,
        Chem.rdchem.BondType.DOUBLE: 2.0,
        Chem.rdchem.BondType.TRIPLE: 3.0,
        Chem.rdchem.BondType.AROMATIC: 1.5,
    }

    # When RTF bonds are available, prefer them: CHARMM-format PDB files rarely
    # have a filled element column, so RDKit warnings flood the log and the mol
    # frequently has no bonds.  RTF BOND sections are always correct for these
    # systems, so skip the RDKit attempt entirely when rtf_bonds are provided.
    if not rtf_bonds:
        from rdkit.rdBase import BlockLogs
        with BlockLogs():
            mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        if mol is not None and mol.GetNumBonds() > 0:
            src_list, dst_list, weights = [], [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                w = BOND_TYPE_WEIGHT.get(bond.GetBondType(), 1.0)
                src_list += [i, j]
                dst_list += [j, i]
                weights += [w, w]
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
            return edge_index, edge_attr

    # RTF BOND section (preferred for CHARMM systems; fallback otherwise)
    if rtf_bonds:
        # Build atom name → index mapping from PDB ATOM/HETATM records
        name_to_idx: dict = {}
        try:
            with open(str(pdb_path)) as fh:
                idx = 0
                for line in fh:
                    if line.startswith(('ATOM', 'HETATM')):
                        aname = line[12:16].strip().upper()
                        name_to_idx[aname] = idx
                        idx += 1
        except Exception:
            pass

        src_list, dst_list = [], []
        for a1, a2 in rtf_bonds:
            i = name_to_idx.get(a1.upper())
            j = name_to_idx.get(a2.upper())
            if i is not None and j is not None:
                src_list += [i, j]
                dst_list += [j, i]

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.ones(len(src_list), 1, dtype=torch.float32)
            return edge_index, edge_attr

    # Nothing found — return empty graph
    return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float32)


def build_full_ligand_bond_graph(
    sub_pdb: str,
    core_pdb: str,
    sub_rtf_bonds: List[Tuple[str, str]],
    core_rtf_bonds: List[Tuple[str, str]],
    ref_sub_info: Optional[List[Tuple[str, List[Tuple[str, str]]]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, List[int]]:
    """Build full-ligand bidirectional bond graph spanning sub + core + ref subs.

    Atom ordering matches :func:`get_full_ligand_atom_features`: substituent atoms
    first (indices 0..n_sub-1), then core atoms (n_sub..n_sub+n_core-1), then ref-sub
    atoms in ``ref_sub_info`` order.

    Each pres.rtf BOND section already encodes the attachment bond(s) between the
    substituent and the core scaffold (e.g. ``BOND C014 C022`` where C014 is a core
    atom).  These cross-group bonds are resolved correctly because we build a unified
    atom-name → global-index map that spans all three atom groups.

    Args:
        sub_pdb: Path to substituent fragment PDB file.
        core_pdb: Path to core PDB file.
        sub_rtf_bonds: Bond list from sub pres.rtf (may include cross-group attachment bonds).
        core_rtf_bonds: Bond list from core.rtf.
        ref_sub_info: Optional list of ``(pdb_path, rtf_bonds)`` for reference substituents
            at other sites (typically sub1 at each non-active site).

    Returns:
        edge_index: [2, 2E] bidirectional bond edge indices over the full ligand.
        edge_attr: [2E, 1] bond-type weights (all 1.0 for RTF-sourced bonds).
        n_sub: Number of substituent atoms.
        n_core: Number of core atoms.
        n_ref_per_site: List of atom counts for each reference substituent.
    """
    ref_sub_info_list = ref_sub_info or []

    # Read atom names from each PDB group using fixed-column parsing (reliable for CHARMM)
    sub_names = _read_atom_names_from_pdb(Path(sub_pdb))
    core_names = _read_atom_names_from_pdb(Path(core_pdb))
    ref_names_list: List[List[str]] = [
        _read_atom_names_from_pdb(Path(ref_pdb))
        for ref_pdb, _ in ref_sub_info_list
    ]

    n_sub = len(sub_names)
    n_core = len(core_names)
    n_ref_per_site = [len(names) for names in ref_names_list]

    # Build unified name → global index map
    # Sub atoms: indices 0..n_sub-1
    # Core atoms: indices n_sub..n_sub+n_core-1
    # Ref sub atoms: contiguous after core
    name_to_idx: dict = {}
    for i, name in enumerate(sub_names):
        name_to_idx[name] = i
    offset = n_sub
    for i, name in enumerate(core_names):
        name_to_idx[name] = offset + i
    offset += n_core
    for ref_names in ref_names_list:
        for i, name in enumerate(ref_names):
            if name not in name_to_idx:   # sub/core names take precedence
                name_to_idx[name] = offset + i
        offset += len(ref_names)

    # Warn when bond information is missing for any ligand component
    if not sub_rtf_bonds:
        warnings.warn(
            f"build_full_ligand_bond_graph: no bond information for substituent PDB "
            f"'{Path(sub_pdb).name}'. Sub atoms will be isolated (no intra-sub or "
            f"attachment bonds). Provide a pres.rtf with BOND entries."
        )
    if not core_rtf_bonds:
        warnings.warn(
            f"build_full_ligand_bond_graph: no bond information for core PDB "
            f"'{Path(core_pdb).name}'. Core atoms will be isolated. "
            f"Provide a core.rtf with BOND entries."
        )
    for ref_idx, (ref_pdb, ref_bonds) in enumerate(ref_sub_info_list):
        if not ref_bonds:
            warnings.warn(
                f"build_full_ligand_bond_graph: no bond information for reference "
                f"substituent '{Path(ref_pdb).name}' (index {ref_idx}). "
                f"Ref-sub atoms will be isolated."
            )

    # Collect all bonds from all RTF sources and resolve to global indices
    all_bond_lists: List[List[Tuple[str, str]]] = [sub_rtf_bonds, core_rtf_bonds]
    for _, ref_bonds in ref_sub_info_list:
        all_bond_lists.append(ref_bonds)

    src_list: List[int] = []
    dst_list: List[int] = []
    for bond_list in all_bond_lists:
        for a1, a2 in bond_list:
            i = name_to_idx.get(a1.upper())
            j = name_to_idx.get(a2.upper())
            if i is not None and j is not None:
                src_list += [i, j]
                dst_list += [j, i]
            # Silently skip bonds referencing atoms outside the current graph.
            # MSLD pres.rtf files include attachment bonds for all substituents
            # at a site; atoms from other subs (in different pretraining groups)
            # will not be in name_to_idx and should simply be ignored.

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.ones(len(src_list), 1, dtype=torch.float32)
    else:
        warnings.warn(
            f"build_full_ligand_bond_graph (sub='{Path(sub_pdb).name}'): "
            f"no bonds resolved for the full ligand graph. GINEConv will operate "
            f"on isolated nodes. Verify that sub_rtf_bonds and core_rtf_bonds are "
            f"non-empty and atom names match."
        )
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)

    return edge_index, edge_attr, n_sub, n_core, n_ref_per_site


def get_full_ligand_atom_features(
    sub_pdb: str,
    core_pdb: str,
    sub_rtf_data: Optional[dict] = None,
    core_rtf_data: Optional[dict] = None,
    ref_sub_info: Optional[List[Tuple[str, Optional[dict]]]] = None,
    protein_pdb=None,
    solvent_context=None,
    aev_cutoff: float = 5.1,
) -> dict:
    """Compute atom features for all ligand atoms (sub + core + ref subs).

    Unlike :func:`get_atom_features_with_context` which returns only substituent atom
    features, this function returns AEVs, charges, and atom_ids for the **entire
    ligand**: substituent + core scaffold + reference substituents at other sites.

    AEVs are computed with full molecular context (protein/solvent atoms are included
    in the AEV neighbourhood computation where available) but protein/solvent atoms are
    **not** returned — only ligand atoms are included in the output.

    Atom ordering: substituent first (indices 0..n_sub-1), then core
    (n_sub..n_sub+n_core-1), then ref subs in ``ref_sub_info`` order.  This matches
    the ordering expected by :func:`build_full_ligand_bond_graph`.

    Args:
        sub_pdb: Path to substituent fragment PDB file.
        core_pdb: Path to core PDB file.
        sub_rtf_data: Parsed RTF dict for the substituent (used for charges).
        core_rtf_data: Parsed RTF dict for the core (used for charges).
        ref_sub_info: List of ``(pdb_path, rtf_data)`` for reference substituents at
            other sites (typically sub1 at each non-active site).
        protein_pdb: Optional protein PDB path or pre-parsed ``(coords, elements)`` tuple
            for AEV context (protein atoms NOT returned in output).
        solvent_context: Optional pre-parsed ``(coords, elements)`` for solvent atoms
            (solvent atoms NOT returned in output).
        aev_cutoff: AEV spatial cutoff in Angstroms (default: 5.1).

    Returns:
        dict with:
            ``'aevs'``: [N_ligand, 2288] AEVs for all ligand atoms.
            ``'charges'``: [N_ligand] partial charges.
            ``'atom_ids'``: [N_ligand] int64 element species IDs.
            ``'n_sub'``: int — substituent atom count (aevs[0:n_sub] are sub atoms).
            ``'n_core'``: int — core atom count.
            ``'n_ref_per_site'``: list[int] — atom counts per reference substituent.
    """
    ref_sub_info_list = ref_sub_info or []

    # Parse coords/elements for each ligand group
    sub_coords, sub_elements = parse_pdb_file(str(sub_pdb))
    core_coords, core_elements = parse_pdb_file(str(core_pdb))
    ref_coords_list: List[list] = []
    ref_elements_list: List[list] = []
    for ref_pdb, _ in ref_sub_info_list:
        rc, re = parse_pdb_file(str(ref_pdb))
        ref_coords_list.append(rc)
        ref_elements_list.append(re)

    # Parse context atoms (protein or solvent) for AEV accuracy — NOT returned
    ctx_coords: list = []
    ctx_elements: list = []
    if protein_pdb is not None:
        if isinstance(protein_pdb, tuple):
            ctx_coords, ctx_elements = protein_pdb
        else:
            ctx_coords, ctx_elements = parse_pdb_file(str(protein_pdb))
    elif solvent_context is not None and isinstance(solvent_context, tuple) and solvent_context[0]:
        ctx_coords, ctx_elements = solvent_context

    # Concatenate in order: sub, core, ref subs, context
    all_groups = (
        [sub_coords, core_coords]
        + ref_coords_list
        + [ctx_coords]
    )
    all_elems = (
        [sub_elements, core_elements]
        + ref_elements_list
        + [ctx_elements]
    )
    group_counts = [len(e) for e in all_elems]

    n_sub = group_counts[0]
    n_core = group_counts[1]
    n_ref_per_site = group_counts[2:-1]      # ref sub groups (context group is last)
    n_ligand = n_sub + n_core + sum(n_ref_per_site)

    flat_coords = [c for g in all_groups for c in g]
    flat_elements = [e for g in all_elems for e in g]

    if not flat_coords:
        raise ValueError(f"No atoms found for full ligand of {sub_pdb}")

    # Compute AEVs for all atoms together (gives each atom proper environment context)
    species_ids = [map_element_to_species_id(e, warn_rare=False) for e in flat_elements]
    species_tensor = torch.tensor(species_ids, dtype=torch.long).unsqueeze(0)
    coords_tensor = torch.tensor(flat_coords, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        all_aevs = aev_computer(species_tensor, coords_tensor).squeeze(0)   # [N_total, 2288]

    # Extract only ligand atoms (exclude context)
    ligand_aevs = all_aevs[:n_ligand]                                        # [N_ligand, 2288]
    ligand_atom_ids = torch.tensor(species_ids[:n_ligand], dtype=torch.long) # [N_ligand]

    # Warn when charge information is missing for any ligand component
    if sub_rtf_data is None:
        warnings.warn(
            f"get_full_ligand_atom_features: no RTF data for substituent "
            f"'{Path(sub_pdb).name}'. Charges for {n_sub} sub atoms will be zero."
        )
    elif sub_rtf_data.get('charges') is None:
        warnings.warn(
            f"get_full_ligand_atom_features: RTF data for '{Path(sub_pdb).name}' "
            f"contains no 'charges' key. Charges for {n_sub} sub atoms will be zero."
        )
    if core_rtf_data is None:
        warnings.warn(
            f"get_full_ligand_atom_features: no RTF data for core "
            f"'{Path(core_pdb).name}'. Charges for {n_core} core atoms will be zero."
        )
    elif core_rtf_data.get('charges') is None:
        warnings.warn(
            f"get_full_ligand_atom_features: RTF data for core '{Path(core_pdb).name}' "
            f"contains no 'charges' key. Charges for {n_core} core atoms will be zero."
        )
    for ref_idx, (ref_pdb, ref_rtf) in enumerate(ref_sub_info_list):
        n_ref = len(ref_elements_list[ref_idx])
        if ref_rtf is None:
            warnings.warn(
                f"get_full_ligand_atom_features: no RTF data for reference substituent "
                f"'{Path(ref_pdb).name}' (index {ref_idx}). "
                f"Charges for {n_ref} ref atoms will be zero."
            )
        elif ref_rtf.get('charges') is None:
            warnings.warn(
                f"get_full_ligand_atom_features: RTF data for ref sub "
                f"'{Path(ref_pdb).name}' contains no 'charges' key. "
                f"Charges for {n_ref} ref atoms will be zero."
            )

    # Read atom names from PDB files for name-based charge matching
    sub_pdb_names = _read_atom_names_from_pdb(Path(sub_pdb))
    core_pdb_names = _read_atom_names_from_pdb(Path(core_pdb))
    ref_pdb_names_list = [
        _read_atom_names_from_pdb(Path(rp)) for rp, _ in ref_sub_info_list
    ]

    # Assemble charges from RTF data for each group
    def _charges(rtf_data: Optional[dict], n: int, label: str = '',
                 pdb_names: Optional[List[str]] = None) -> torch.Tensor:
        if rtf_data is not None:
            c = rtf_data.get('charges')
            if c is not None:
                t = torch.tensor(c, dtype=torch.float32)
                if len(t) == n:
                    return t
                # Count mismatch: try name-based matching when atom name lists available
                rtf_names = rtf_data.get('atom_names')
                if (pdb_names is not None and rtf_names is not None
                        and len(rtf_names) == len(c)):
                    name_to_charge = {name.upper(): q for name, q in zip(rtf_names, c)}
                    matched = [name_to_charge.get(name.upper()) for name in pdb_names]
                    if all(q is not None for q in matched):
                        return torch.tensor(matched, dtype=torch.float32)
                warnings.warn(
                    f"get_full_ligand_atom_features: charge count mismatch for "
                    f"{label or 'component'}: RTF has {len(t)} charges but PDB has "
                    f"{n} atoms. Using zero charges."
                )
        return torch.zeros(n, dtype=torch.float32)

    parts = (
        [_charges(sub_rtf_data, n_sub, Path(sub_pdb).name, sub_pdb_names),
         _charges(core_rtf_data, n_core, Path(core_pdb).name, core_pdb_names)]
        + [_charges(rrtf, len(ref_elements_list[i]), Path(rp).name,
                    ref_pdb_names_list[i])
           for i, (rp, rrtf) in enumerate(ref_sub_info_list)]
    )
    ligand_charges = torch.cat(parts, dim=0)   # [N_ligand]

    return {
        'aevs': ligand_aevs,
        'charges': ligand_charges,
        'atom_ids': ligand_atom_ids,
        'n_sub': n_sub,
        'n_core': n_core,
        'n_ref_per_site': n_ref_per_site,
    }


def parse_pdb_coordinates_and_elements(pdb_path):
    """Parse PDB file to extract coordinates and elements.
    
    Wrapper around read_pdb.parse_pdb_file() that converts coordinates to tensor.
    For substituent fragment PDB files (``*_frag.pdb``), automatically locates the
    companion ``*_pres.rtf`` in the same directory and passes it as ``rtf_data`` so
    that element symbols are cross-validated against CGenFF atom types.  This
    corrects cases where CHARMM uses a 1-char+index atom-name scheme that is
    ambiguous (e.g. ``B085`` for Bromine whose CGenFF type is ``BRGR1``).
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        tuple: (coordinates, elements) where:
            - coordinates: torch.Tensor [num_atoms, 3]
            - elements: list of element symbols
    """
    rtf_data = None
    pdb_path_obj = Path(str(pdb_path))
    if '_frag.pdb' in pdb_path_obj.name:
        rtf_path = pdb_path_obj.parent / pdb_path_obj.name.replace('_frag.pdb', '_pres.rtf')
        if rtf_path.exists():
            try:
                from mllf.file_handling.read_rtf import parse_rtf_file
                rtf_data = parse_rtf_file(str(rtf_path))
            except Exception:
                pass

    coords_list, elements = parse_pdb_file(str(pdb_path), rtf_data=rtf_data)
    coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
    return coords_tensor, elements


def _read_atom_names_from_pdb(pdb_path: Path) -> List[str]:
    """Return ordered list of atom names from ATOM/HETATM records in a PDB file.

    Uses fixed-column parsing (PDB cols 13–16, 0-indexed 12:16) which is
    reliable for CHARMM-generated PDB files.  Names are normalised to
    **uppercase** so that mixed-case frag.pdb names (e.g. ``Br0B``) match the
    all-uppercase names written by CHARMM into minimized.pdb (``BR0B``).
    Returns an empty list on any error so callers can fall back gracefully.
    """
    names: List[str] = []
    try:
        with open(str(pdb_path)) as fh:
            for line in fh:
                if line.startswith(('ATOM', 'HETATM')):
                    names.append(line[12:16].strip().upper())
    except Exception:
        pass
    return names


_BOX_RE = re.compile(r'^\s*box\s*=\s*([0-9]+(?:\.[0-9]+)?)')


def _read_box_from_prep_script(prep_dir: Path) -> Optional[float]:
    """Return the cubic box length from the system's prep Python script.

    Searches for a ``box = <number>`` assignment in the main prep ``.py`` file
    (any ``.py`` other than ``alf_info.py``) in *prep_dir*.  Returns ``None``
    if no such file or line is found, so the caller can skip MIC gracefully.
    """
    for py_file in sorted(prep_dir.glob('*.py')):
        if py_file.name == 'alf_info.py':
            continue
        try:
            for line in py_file.read_text().splitlines():
                m = _BOX_RE.match(line)
                if m:
                    return float(m.group(1))
        except OSError:
            continue
    return None


def _convert_crd_to_tmp_pdb(crd_path: Path) -> Path:
    """Convert a CHARMM extended CRD coordinate file to a temporary PDB file.

    Parses CHARMM EXT format and writes minimal ATOM-record PDB to a temp
    file so it can be consumed by parse_pdb_file.

    CHARMM EXT CRD column layout (1-indexed, Fortran):
      1-10:   atom serial (I10)
      11-20:  residue sequence number (I10)
      21-22:  spaces (2X)
      23-30:  residue name (A8)
      31-32:  spaces (2X)
      33-40:  atom name (A8)
      41-60:  X coordinate (F20.10)
      61-80:  Y coordinate (F20.10)
      81-100: Z coordinate (F20.10)
      101-102: spaces (2X)
      103-110: segment ID (A8)

    Args:
        crd_path: Path to the CHARMM EXT .crd file.

    Returns:
        Path to a newly-created temporary PDB file.

    Raises:
        ValueError: If no atoms were successfully parsed from the file.
    """
    atoms = []
    with open(crd_path, 'r') as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            # Skip CHARMM title / comment lines
            if line.startswith('*'):
                continue
            # Skip the atom-count line (e.g. "     73018  EXT")
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) <= 2 and parts[0].isdigit():
                continue
            # Need at least 100 chars to reach end of Z column
            if len(line) < 100:
                continue
            try:
                atom_no  = int(line[0:10])
                res_no   = int(line[10:20])
                resname  = line[22:30].strip()
                atomname = line[32:40].strip()
                x        = float(line[40:60])
                y        = float(line[60:80])
                z        = float(line[80:100])
                segid    = line[102:110].strip() if len(line) > 110 else ''
                # Derive single-char PDB chain from segid (PROA→A, PROB→B …)
                chain    = segid[-1] if segid and segid[-1].isalpha() else 'A'
                atoms.append((atom_no, res_no, resname, atomname, x, y, z, chain))
            except (ValueError, IndexError):
                continue

    if not atoms:
        raise ValueError(f"No atoms parsed from CRD file {crd_path}")

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.pdb', prefix='minimized_crd_', delete=False
    )
    try:
        _TWO_CHAR_ELEMENTS = {'CL', 'BR', 'FE', 'MG', 'ZN', 'MN', 'NA', 'CU', 'CO', 'NI', 'SE', 'SI'}
        for atom_no, res_no, resname, atomname, x, y, z, chain in atoms:
            # PDB atom name: 4-char names fill cols 13-16; shorter get a leading space
            aname_col = atomname[:4] if len(atomname) >= 4 else f' {atomname:<3}'
            # Infer element symbol from alphabetic prefix of atom name
            alpha = ''.join(c for c in atomname if c.isalpha())
            if len(alpha) >= 2 and alpha[:2].upper() in _TWO_CHAR_ELEMENTS:
                elem = alpha[:2].capitalize()
            else:
                elem = alpha[:1] if alpha else 'X'
            serial  = atom_no % 100000
            res_seq = res_no  % 10000
            tmp.write(
                f"ATOM  {serial:5d} {aname_col} {resname[:3]:3s} {chain}{res_seq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
            )
        tmp.write("END\n")
    finally:
        tmp.close()

    return Path(tmp.name)


def detect_minimized_pdb(prep_dir: Path) -> Optional[Path]:
    """Detect the minimized system structure in a prep directory.

    Checks for ``minimized.pdb`` first.  If not found, falls back to
    ``minimized.crd`` (CHARMM EXT format) and converts it to a temporary
    PDB file on-the-fly.

    Args:
        prep_dir: Path to prep directory

    Returns:
        Path to minimized.pdb (or a temporary PDB converted from
        minimized.crd), or None if neither file is found.
    """
    prep = Path(prep_dir)
    minimized_pdb = prep / 'minimized.pdb'
    if minimized_pdb.exists():
        return minimized_pdb

    minimized_crd = prep / 'minimized.crd'
    if minimized_crd.exists():
        try:
            tmp_pdb = _convert_crd_to_tmp_pdb(minimized_crd)
            return tmp_pdb
        except Exception as exc:
            warnings.warn(f"Could not convert {minimized_crd} to PDB: {exc}")

    return None


def extract_environment_atoms_from_minimized(
    minimized_pdb: Path,
    sub_pdb: Path,
    core_pdb: Path,
    aev_cutoff: float = 5.1,
    duplicate_tolerance: float = 0.5,
    prep_dir: Optional[Path] = None,
) -> Optional[Tuple[List, List]]:
    """Extract environment atoms (protein or solvent) from minimized.pdb.

    Removes core, target substituent, and (when ``prep_dir`` is given) all
    other ``site*_sub*_frag.pdb`` atoms from ``minimized.pdb`` via coordinate
    matching.  The remaining atoms are filtered to those within ``aev_cutoff``
    of the target substituent and returned as a pre-parsed tuple ready to be
    passed directly to :func:`get_atom_features_with_context` as
    ``protein_pdb`` or ``solvent_context``.

    This is the core extraction primitive used by both the RGCN training
    pipeline (via :func:`graph_utils.compute_deepset_embedding_for_node`) and
    the DeepSet pretraining pipeline.

    **Why exclude ALL substituents?**  ``minimized.pdb`` contains every
    substituent from the combination that was simulated.  Leaving other-site
    sub atoms in the returned tuple would double-count them because
    :func:`get_atom_features_with_context` adds those same atoms independently
    via ``find_reference_subs_from_other_sites``.

    **Why use atom-name-based exclusion?**  CHARMM energy minimization can
    shift atoms by more than the ``duplicate_tolerance`` (0.5 Å) or even
    wrap them into a different PBC image, so coordinate-based duplicate removal
    may silently fail for some systems. Matching by atom name (e.g. ``C102``,
    ``N001``) is robust regardless of the coordinate frame.  The same name
    lookup is used to resolve the substituent's reference position in the
    minimized frame when computing the distance filter, so water/protein atoms
    within ``aev_cutoff`` of the *actual* minimized position are always found.

    **Why 0.5 Å tolerance?**  Retained as the fallback when atom-name
    counts cannot be reconciled with ``parse_pdb_file`` output.

    Args:
        minimized_pdb: Path to minimized.pdb (full system with minimized coordinates)
        sub_pdb: Path to the substituent PDB file whose AEVs are being computed
        core_pdb: Path to core PDB file
        aev_cutoff: AEV spatial cutoff distance in Angstroms (default: 5.1 Å)
        duplicate_tolerance: Max distance (Å) to consider a minimized atom as
            matching a ligand fragment atom (default: 0.5 Å)
        prep_dir: Optional prep directory.  When provided, ALL
            ``site*_sub*_frag.pdb`` files found here are added to the
            exclusion set so other-site substituent atoms are removed from the
            returned context before it is used.

    Returns:
        ``(coords_list, elements_list)`` for environment atoms within cutoff,
        or ``None`` if no environment atoms are found after filtering.
    """
    min_coords, min_elements = parse_pdb_file(str(minimized_pdb))
    sub_coords, _ = parse_pdb_file(str(sub_pdb))
    core_coords, _ = parse_pdb_file(str(core_pdb))

    if not min_coords:
        return None

    # ------------------------------------------------------------------
    # Atom-name-based ligand exclusion and coordinate resolution.
    #
    # frag.pdb files may carry pre-simulation coordinates that differ from
    # minimized.pdb by > 0.5 Å (e.g. after PBC image wrapping shifts atoms
    # by box-length increments).  Coordinate-based duplicate removal then
    # silently fails, leaving all ligand atoms in the environment set and
    # placing the distance-cutoff filter at the wrong position.
    #
    # Strategy:
    #   1. Read atom names from minimized.pdb via fast fixed-column parsing.
    #   2. Build a {name -> coord} map for every atom in minimized.pdb.
    #   3. Collect all ligand atom names from core + all sub frag.pdbs.
    #   4. Remove minimized atoms whose name is in the ligand name set.
    #   5. Resolve the target substituent's coordinates in the minimized
    #      frame by looking up each atom's name in the map (falling back to
    #      the frag.pdb coordinate when the name is absent).
    # ------------------------------------------------------------------
    min_atom_names = _read_atom_names_from_pdb(minimized_pdb)
    sub_atom_names = _read_atom_names_from_pdb(sub_pdb)
    core_atom_names = _read_atom_names_from_pdb(core_pdb)

    use_name_based = (
        len(min_atom_names) == len(min_coords)
        and len(sub_atom_names) == len(sub_coords)
    )

    if use_name_based:
        # Build name → minimized-coord lookup
        min_name_to_coord = {name: coord for name, coord in zip(min_atom_names, min_coords)}

        # Collect all ligand atom names (core + all subs in prep_dir)
        ligand_names: set = set(sub_atom_names) | set(core_atom_names)
        if prep_dir is not None:
            for other_sub in sorted(Path(prep_dir).glob('site*_sub*_frag.pdb')):
                if other_sub.resolve() == Path(sub_pdb).resolve():
                    continue
                ligand_names.update(_read_atom_names_from_pdb(other_sub))

        # Remove ligand atoms by name → clean environment set
        env_coords = [c for name, c in zip(min_atom_names, min_coords) if name not in ligand_names]
        env_elements = [e for name, e in zip(min_atom_names, min_elements) if name not in ligand_names]

        # Resolve target sub positions in the minimized coordinate frame
        sub_coords_ref = [
            min_name_to_coord.get(name, frag_c)
            for name, frag_c in zip(sub_atom_names, sub_coords)
        ]
    else:
        # Fall back to original coordinate-matching approach
        ligand_coords = core_coords + sub_coords
        if prep_dir is not None:
            for other_sub in sorted(Path(prep_dir).glob('site*_sub*_frag.pdb')):
                if other_sub.resolve() == Path(sub_pdb).resolve():
                    continue
                other_coords, _ = parse_pdb_file(str(other_sub))
                ligand_coords.extend(other_coords)

        env_coords, env_elements = remove_duplicate_atoms(
            coords=min_coords,
            elements=min_elements,
            core_coords=ligand_coords,
            tolerance=duplicate_tolerance,
        )
        sub_coords_ref = sub_coords

    if not env_coords:
        return None

    # Spatial filter: keep only atoms within aev_cutoff of the substituent.
    # Apply the minimum image convention (MIC) when a cubic box length is
    # available so that atoms in adjacent PBC images are found correctly.
    box_length = _read_box_from_prep_script(Path(prep_dir)) if prep_dir is not None else None

    sub_arr  = np.array(sub_coords_ref)  # [n_sub,  3]
    env_arr  = np.array(env_coords)      # [n_env,  3]
    diff     = env_arr[:, None, :] - sub_arr[None, :, :]  # [n_env, n_sub, 3]
    if box_length is not None:
        diff = diff - box_length * np.round(diff / box_length)
    min_dists = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)  # [n_env]

    mask = min_dists <= aev_cutoff
    if not mask.any():
        return None

    return (
        [c for c, keep in zip(env_coords, mask) if keep],
        [e for e, keep in zip(env_elements, mask) if keep],
    )


def get_atom_features_with_context(substituent_pdb, core_pdb, protein_pdb=None,
                                   solvent_context=None,
                                   rtf_entry=None, include_charges=True, include_atom_ids=True,
                                   prep_dir=None, aev_cutoff=5.1):
    """Compute atom-level features with full molecular context.

    This function properly combines core + substituent (+ protein/solvent + nearby sites)
    to compute accurate AEVs where each atom sees its full molecular environment.

    Critical: AEVs are environment-dependent. Substituent atoms must see their
    bonded neighbors in the core to get accurate atomic environments.

    For multi-site systems, this function automatically detects and includes substituents
    from other sites if they are within the AEV spatial cutoff (default 5.1 Å for ANI-2x).
    Similarly, protein or solvent atoms within the cutoff are included when provided.

    Args:
        substituent_pdb: Path to substituent PDB file
        core_pdb: Path to core PDB file
        protein_pdb: Optional path to protein PDB file, or a pre-parsed
            ``(coords_list, elements_list)`` tuple already filtered to atoms within
            the AEV cutoff (e.g. extracted from minimized.pdb)
        solvent_context: Optional pre-parsed ``(coords_list, elements_list)`` tuple of
            solvent/water atoms within the AEV cutoff (e.g. extracted from minimized.pdb
            for a solvent-phase system).  These atoms are appended to the AEV context
            with label ``'solvent'``.
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
        if isinstance(protein_pdb, tuple):
            # Pre-parsed (coords, elements) tuple: caller has already extracted and
            # spatially filtered the protein atoms (e.g. from minimized.pdb).
            # Skip the find_nearby_pdbs check and include directly.
            include_protein = True
        elif prep_dir is not None:
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

    # Include pre-parsed solvent context (water molecules within cutoff)
    include_solvent = False
    if solvent_context is not None and isinstance(solvent_context, tuple) and solvent_context[0]:
        include_solvent = True
        pdb_files.append(solvent_context)
        context_labels.append('solvent')

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
    species_ids = [map_element_to_species_id(element) for element in all_elements]
    
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
            'solvent_included': include_solvent,
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