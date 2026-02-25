"""
Dataset generation for DeepSet autoencoder pretraining.

This module handles Step 1 of the 4-step pretraining process:
Iterate through substituent PDB files, calculate AEVs, concatenate with
partial charges, and generate training tensors.
"""

import torch
import json
from pathlib import Path
import warnings
from typing import List, Tuple, Dict, Optional

from mllf.cb.aev_processor import (
    get_substituent_aevs, 
    extract_charges_from_rtf_metadata,
    get_atom_features_with_context
)
from mllf.file_handling.read_rtf import parse_rtf_file
from mllf.file_handling.read_pdb import parse_pdb_file


def detect_core_pdb(prep_dir: Path) -> Optional[Path]:
    """Detect the core PDB file in a prep directory.
    
    Args:
        prep_dir: Path to prep directory
        
    Returns:
        Path to core.pdb, or None if not found
    """
    core_pdb = prep_dir / 'core.pdb'
    return core_pdb if core_pdb.exists() else None


def detect_protein_pdb(prep_dir: Path) -> Optional[Path]:
    """Detect the protein PDB file in a prep directory.
    
    Args:
        prep_dir: Path to prep directory
        
    Returns:
        Path to protein PDB file, or None if not found
    """
    # Common protein PDB filename patterns
    exclude_patterns = {'core.pdb', 'ions.pdb', 'solv.pdb', 'minimized.pdb', 
                        'cubic.pdb', 'water.pdb'}
    
    # Look for PDB files that match protein patterns
    protein_candidates = []
    for pdb_file in prep_dir.glob('*.pdb'):
        # Skip known non-protein files and substituent files
        if pdb_file.name in exclude_patterns:
            continue
        if pdb_file.name.startswith('site') and '_sub' in pdb_file.name:
            continue
            
        protein_candidates.append(pdb_file)
    
    # If we found exactly one candidate, that's likely the protein
    if len(protein_candidates) == 1:
        return protein_candidates[0]
    # If multiple candidates, look for common protein naming patterns
    elif len(protein_candidates) > 1:
        for candidate in protein_candidates:
            name_lower = candidate.name.lower()
            if any(pattern in name_lower for pattern in ['prot', 'pro', 'protein', 'receptor']):
                return candidate
        # Just return the first one if no clear match
        return protein_candidates[0]
    
    return None


def extract_charges_from_rtf(rtf_path: Path, pdb_name: str) -> Optional[torch.Tensor]:
    """Extract partial charges for a substituent from RTF file.
    
    Uses the shared extract_charges_from_rtf_metadata() function from aev_processor.
    
    Args:
        rtf_path: Path to RTF file (e.g., site1_sub1_pres.rtf)
        pdb_name: Name of PDB file to match atoms
        
    Returns:
        Tensor of partial charges [num_atoms], or None if not found
    """
    try:
        rtf_data = parse_rtf_file(str(rtf_path))
        return extract_charges_from_rtf_metadata(rtf_data)
    except Exception as e:
        warnings.warn(f"Could not extract charges from {rtf_path}: {e}")
        return None


def load_system_metadata(system_dir: Path) -> Dict:
    """Load metadata from a pretraining system directory.
    
    Args:
        system_dir: Path to system directory (e.g., abl_protein_mutant_group1)
        
    Returns:
        dict with keys:
            - 'system_name': Name of the system
            - 'solvent_state': 'protein', 'solvent', 'gas', or 'water'
            - 'prep_dir': Path to the prep directory
            - 'protein_pdb': Path to protein PDB if applicable
            - 'num_sites': Number of sites
            - 'num_substituents': Number of substituents per site
    """
    # Find first run directory with valid metadata
    run_dirs = sorted(system_dir.glob('run*'))
    if not run_dirs:
        raise ValueError(f"No run directories found in {system_dir}")
    
    for run_dir in run_dirs[:5]:  # Check first 5 runs
        metadata_path = run_dir / 'metadata.json'
        if not metadata_path.exists():
            continue
            
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Get prep directory from source_run_dir
            source_run_dir = Path(metadata['source_run_dir'])
            prep_dir = source_run_dir / 'prep'
            
            if not prep_dir.exists():
                continue
            
            # Determine solvent state
            solvent_state = metadata.get('solvent_state', 'unknown')
            if solvent_state == 'unknown':
                # Infer from directory name
                name_lower = system_dir.name.lower()
                if 'protein' in name_lower:
                    solvent_state = 'protein'
                elif 'water' in name_lower:
                    solvent_state = 'water'
                elif 'solv' in name_lower:
                    solvent_state = 'solvent'
                else:
                    solvent_state = 'gas'
            
            # Detect protein PDB if protein phase
            protein_pdb = None
            if solvent_state == 'protein':
                protein_pdb = detect_protein_pdb(prep_dir)
                if protein_pdb:
                    print(f"  Detected protein PDB: {protein_pdb.name}")
            
            return {
                'system_name': system_dir.name,
                'solvent_state': solvent_state,
                'prep_dir': prep_dir,
                'protein_pdb': protein_pdb,
                'num_sites': metadata.get('num_sites', 1),
                'num_substituents': metadata.get('num_substituents', 0),
            }
        except Exception as e:
            warnings.warn(f"Error reading metadata from {run_dir}: {e}")
            continue
    
    raise ValueError(f"Could not load valid metadata from {system_dir}")


def generate_training_data_for_system(
    system_dir: Path,
    output_path: Path,
    aev_cutoff: float = 5.1,
    verbose: bool = False
) -> Dict:
    """Generate training data for one pretraining system.
    
    Args:
        system_dir: Path to system directory
        output_path: Where to save the training tensor
        aev_cutoff: AEV cutoff distance in Angstroms
        verbose: If True, print detailed context information for first substituent
        
    Returns:
        dict with statistics about the generated data
    """
    system_name = system_dir.name
    print(f"\nProcessing {system_name}...")
    
    # Load system metadata
    metadata = load_system_metadata(system_dir)
    prep_dir = metadata['prep_dir']
    solvent_state = metadata['solvent_state']
    print(f"  Prep directory: {prep_dir}")
    print(f"  Solvent state: {solvent_state}")
    print(f"  Sites: {metadata['num_sites']}, Substituents: {metadata['num_substituents']}")
    
    # Detect core PDB (needed for context-aware AEV computation)
    core_pdb = detect_core_pdb(prep_dir)
    if core_pdb:
        print(f"  Core PDB: {core_pdb.name}")
    
    # For protein systems, detect protein PDB for context-aware AEVs
    protein_pdb = None
    if solvent_state == 'protein':
        protein_pdb = detect_protein_pdb(prep_dir)
        if protein_pdb:
            print(f"  Protein PDB: {protein_pdb.name}")
            print(f"  → Using protein context for accurate AEVs")
        else:
            warnings.warn(f"[{system_name}] Protein state but no protein PDB found")
    
    # Collect all substituent PDB files
    sub_pdbs = sorted(prep_dir.glob('site*_sub*_frag.pdb'))
    if not sub_pdbs:
        raise ValueError(f"No substituent PDB files found in {prep_dir}")
    
    print(f"  Found {len(sub_pdbs)} substituent PDB files")
    
    # Generate features for each substituent
    all_features = []
    total_atoms = 0
    first_sub_processed = False  # Track if we've shown verbose info
    
    for pdb_path in sub_pdbs:
        try:
            # Extract charges from RTF
            rtf_path = pdb_path.parent / pdb_path.name.replace('_frag.pdb', '_pres.rtf')
            rtf_data = None
            if rtf_path.exists():
                try:
                    rtf_data = parse_rtf_file(str(rtf_path))
                except Exception as e:
                    warnings.warn(f"[{system_name}] Could not parse RTF {rtf_path.name}: {e}")
            
            # Compute AEVs with appropriate context
            if solvent_state == 'protein' and protein_pdb and core_pdb:
                # Use context-aware AEV computation for protein systems
                # This includes protein atoms within cutoff for accurate environments
                features_dict = get_atom_features_with_context(
                    substituent_pdb=str(pdb_path),
                    core_pdb=str(core_pdb),
                    protein_pdb=str(protein_pdb),
                    rtf_entry=rtf_data,
                    include_charges=True,
                    include_atom_ids=False,
                    prep_dir=str(prep_dir),
                    aev_cutoff=aev_cutoff
                )
                aevs = features_dict['aevs']
                charges = features_dict.get('charges')
                
                # Print verbose context information for first substituent
                if verbose and not first_sub_processed:
                    context_info = features_dict.get('context_info', {})
                    print(f"\n  [VERBOSE] AEV Context for {pdb_path.name}:")
                    print(f"    Cutoff distance: {context_info.get('cutoff_used', aev_cutoff)} Å")
                    print(f"    Context sources: {', '.join(context_info.get('context_sources', []))}")
                    atom_counts = context_info.get('atom_counts_per_source', [])
                    sources = context_info.get('context_sources', [])
                    for source, count in zip(sources, atom_counts):
                        print(f"      - {source}: {count} atoms")
                    print(f"    Total context atoms: {context_info.get('total_context_atoms', 0)}")
                    print(f"    Protein included: {context_info.get('protein_included', False)}")
                    if context_info.get('reference_subs_from_other_sites'):
                        print(f"    Reference subs from other sites:")
                        for ref_info in context_info['reference_subs_from_other_sites']:
                            print(f"      - {ref_info['pdb']}: {ref_info['filtered_atoms']} atoms "
                                  f"(removed {ref_info['removed_duplicates']} duplicates)")
                    first_sub_processed = True
            else:
                # For non-protein systems, use substituent-only AEVs
                aevs = get_substituent_aevs(str(pdb_path))
                charges = extract_charges_from_rtf(rtf_path, pdb_path.name) if rtf_data else None
                
                # Print verbose info for non-protein systems
                if verbose and not first_sub_processed:
                    print(f"\n  [VERBOSE] AEV Context for {pdb_path.name}:")
                    print(f"    Mode: Substituent-only (no protein context)")
                    print(f"    Substituent atoms: {aevs.shape[0]}")
                    first_sub_processed = True
            
            num_atoms = aevs.shape[0]
            
            if charges is None:
                # Fall back to zeros if no charges available
                warnings.warn(f"[{system_name}] No charges found for {pdb_path.name}, using zeros")
                charges = torch.zeros(num_atoms, dtype=torch.float32)
            
            # Ensure charges match number of atoms
            if len(charges) != num_atoms:
                warnings.warn(f"[{system_name}] Charge count mismatch for {pdb_path.name}: "
                            f"{len(charges)} charges vs {num_atoms} atoms. Using zeros.")
                charges = torch.zeros(num_atoms, dtype=torch.float32)
            
            # Concatenate AEV + charges: [num_atoms, 2288 + 1] = [num_atoms, 2289]
            charges_expanded = charges.unsqueeze(1)  # [num_atoms, 1]
            features = torch.cat([aevs, charges_expanded], dim=1)
            
            all_features.append(features)
            total_atoms += num_atoms
            
        except Exception as e:
            warnings.warn(f"[{system_name}] Error processing {pdb_path.name}: {e}")
            continue
    
    if not all_features:
        raise ValueError(f"No valid features generated for {system_dir.name}")
    
    # Concatenate all features into one big tensor
    training_tensor = torch.cat(all_features, dim=0)  # [total_atoms, 2289]
    
    print(f"  Generated training tensor: {training_tensor.shape}")
    print(f"  Total atoms: {total_atoms}")
    
    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'features': training_tensor,
        'system_name': metadata['system_name'],
        'solvent_state': metadata['solvent_state'],
        'num_substituents': len(sub_pdbs),
        'total_atoms': total_atoms,
        'feature_dim': training_tensor.shape[1],
    }, output_path)
    
    print(f"  Saved to {output_path}")
    
    return {
        'system_name': metadata['system_name'],
        'num_substituents': len(sub_pdbs),
        'total_atoms': total_atoms,
        'feature_dim': training_tensor.shape[1],
        'output_path': str(output_path),
    }


def generate_all_pretraining_datasets(
    pretraining_root: Path,
    output_root: Path,
    skip_systems: Optional[List[str]] = None,
    verbose: bool = False
) -> List[Dict]:
    """Generate training datasets for all pretraining systems.
    
    Args:
        pretraining_root: Root directory containing all pretraining systems
        output_root: Root directory for output files
        skip_systems: List of system names to skip (default: ['14benz_pair_combos'])
        verbose: If True, print detailed context information for each system
        
    Returns:
        List of statistics dicts for each system
    """
    if skip_systems is None:
        skip_systems = ['14benz_pair_combos', '1_analysis_scripts']
    
    # Find all system directories
    system_dirs = [d for d in pretraining_root.iterdir() 
                   if d.is_dir() and d.name not in skip_systems]
    
    print(f"Found {len(system_dirs)} pretraining systems to process")
    
    all_stats = []
    errors = []
    
    for system_dir in sorted(system_dirs):
        output_path = output_root / f"{system_dir.name}_training_data.pt"
        
        try:
            stats = generate_training_data_for_system(system_dir, output_path, verbose=verbose)
            all_stats.append(stats)
        except Exception as e:
            error_msg = f"Failed to process {system_dir.name}: {e}"
            warnings.warn(error_msg)
            errors.append(error_msg)
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Successfully processed {len(all_stats)}/{len(system_dirs)} systems")
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    print(f"\nTotal atoms across all systems: {sum(s['total_atoms'] for s in all_stats):,}")
    
    return all_stats
