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
    get_atom_features_with_context,
    detect_minimized_pdb,
    extract_environment_atoms_from_minimized,
    get_bond_edge_index_from_pdb,
)
from mllf.file_handling.read_rtf import parse_rtf_file


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
    # Find first run directory with valid metadata.
    # Accept both 'run*' (most systems) and 'n_run*' (combo directories).
    run_dirs = sorted(list(system_dir.glob('run*')) + list(system_dir.glob('n_run*')))
    if not run_dirs:
        raise ValueError(f"No run directories found in {system_dir}")
    
    for run_dir in run_dirs[:5]:  # Check first 5 runs
        metadata_path = run_dir / 'metadata.json'
        if not metadata_path.exists():
            continue
            
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Get prep directory: try source_run_dir first, fall back to local copy
            source_run_dir = Path(metadata['source_run_dir'])
            prep_dir = source_run_dir / 'prep'

            if not prep_dir.exists():
                # Fall back to locally copied prep (populated by copy_prep_to_local.py
                # or collect_new_systems.sh) when the source tree is unavailable.
                local_prep = system_dir / 'prep'
                if local_prep.exists():
                    prep_dir = local_prep
                else:
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
            
            # Load active_subs_ordered from graph_info.json (present for systems
            # whose prep is a master prep shared across groups, e.g. luis_systems).
            active_subs_ordered = None
            graph_info_path = run_dir / 'graph_info.json'
            if graph_info_path.exists():
                try:
                    with open(graph_info_path) as _gf:
                        _gi = json.load(_gf)
                    active_subs_ordered = _gi.get('active_subs_ordered')
                except Exception:
                    pass

            return {
                'system_name': system_dir.name,
                'solvent_state': solvent_state,
                'prep_dir': prep_dir,
                'protein_pdb': protein_pdb,
                'num_sites': metadata.get('num_sites', 1),
                'num_substituents': metadata.get('num_substituents', 0),
                'active_subs_ordered': active_subs_ordered,
            }
        except Exception as e:
            warnings.warn(f"Error reading metadata from {run_dir}: {e}")
            continue
    
    raise ValueError(f"Could not load valid metadata from {system_dir}")


def _collect_leaf_system_dirs(top_dirs: List[Path]) -> List[Tuple[Path, str]]:
    """Expand collection directories into (system_dir, unique_output_name) pairs.

    A *collection* directory is one that contains no ``run*`` / ``n_run*``
    entries itself but whose children each have a ``prep/`` directory or their
    own ``run*`` / ``n_run*`` entries.  Examples: ``14benz_combos``,
    ``14benz_triplet_combos``, ``14benz_quad_combos_v2``.

    Leaf systems are returned as-is.  Collection directories are expanded so
    each child becomes a (path, unique_name) entry where ``unique_name`` is
    ``<parent>__<child>`` to avoid output-file collisions.

    Args:
        top_dirs: Top-level candidate directories (after skip-list filtering).

    Returns:
        List of (system_dir, unique_name) tuples ready for dataset generation.
    """
    result: List[Tuple[Path, str]] = []
    for d in top_dirs:
        run_entries = list(d.glob('run*')) + list(d.glob('n_run*'))
        if run_entries:
            # Leaf system — has its own run directories.
            result.append((d, d.name))
        else:
            children = sorted(c for c in d.iterdir() if c.is_dir())
            is_collection = any(
                (c / 'prep').exists()
                or list(c.glob('run*'))
                or list(c.glob('n_run*'))
                for c in children
            )
            if is_collection:
                for child in children:
                    result.append((child, f"{d.name}__{child.name}"))
            else:
                # Not a recognisable collection; pass through so it fails clearly.
                result.append((d, d.name))
    return result


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
    
    # Detect minimized.pdb for all phases.  Energy-minimized coordinates are used:
    #   protein  → extract protein context atoms within AEV cutoff
    #   solvent  → extract solvent (water) context atoms within AEV cutoff
    #   vacuum   → detected but not used for additional context
    minimized_pdb = detect_minimized_pdb(prep_dir)  # type: Optional[Path]
    if minimized_pdb:
        print(f"  Minimized PDB: {minimized_pdb.name}")

    # For protein systems, also keep a fallback standalone protein PDB path.
    protein_pdb = None          # Path to a standalone protein PDB (fallback only)
    if solvent_state == 'protein':
        if minimized_pdb:
            print(f"  → Will extract protein context from minimized coordinates")
        else:
            protein_pdb = detect_protein_pdb(prep_dir)
            if protein_pdb:
                print(f"  Protein PDB: {protein_pdb.name}")
                print(f"  → Using protein context for accurate AEVs")
            else:
                warnings.warn(f"[{system_name}] Protein state but no protein PDB or minimized.pdb found")
    
    # Collect all substituent PDB files
    # Collect substituent PDB files in sequential bias order.
    # If active_subs_ordered is provided (master-prep systems), resolve each
    # master sub name to its _frag.pdb path.  Otherwise fall back to a glob
    # which works for conventional preps that already contain only active subs.
    active_subs_ordered = metadata.get('active_subs_ordered')
    if active_subs_ordered:
        sub_pdbs = []
        for site_label in sorted(active_subs_ordered.keys(),
                                  key=lambda s: int(s.replace('site', ''))):
            for master_sub in active_subs_ordered[site_label]:
                frag = prep_dir / f"{master_sub}_frag.pdb"
                if frag.exists():
                    sub_pdbs.append(frag)
                else:
                    warnings.warn(
                        f"[{system_name}] Active sub frag PDB not found: {frag.name}"
                    )
    else:
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
            if solvent_state in ('protein', 'prot') and core_pdb and (minimized_pdb or protein_pdb):
                # Build protein context for this specific substituent.
                # Preferred path: extract protein atoms from minimized.pdb using
                # coordinate-based exclusion of core/sub atoms, then apply the AEV
                # spatial cutoff.  This gives the most accurate environment because
                # the coordinates reflect the actual minimized geometry.
                protein_context = None  # tuple (coords, elements) or None

                if minimized_pdb:
                    try:
                        protein_context = extract_environment_atoms_from_minimized(
                            minimized_pdb=minimized_pdb,
                            sub_pdb=pdb_path,
                            core_pdb=core_pdb,
                            aev_cutoff=aev_cutoff,
                            prep_dir=prep_dir,
                        )
                        if protein_context is None:
                            warnings.warn(
                                f"[{system_name}] No protein atoms within {aev_cutoff} Å "
                                f"of {pdb_path.name} in minimized.pdb"
                            )
                    except Exception as e:
                        warnings.warn(
                            f"[{system_name}] Could not extract protein context from "
                            f"minimized.pdb for {pdb_path.name}: {e}. Falling back to "
                            f"protein PDB if available."
                        )

                # Fallback to standalone protein PDB when minimized.pdb fails / absent
                effective_protein_pdb = protein_context if protein_context is not None else (
                    str(protein_pdb) if protein_pdb else None
                )

                features_dict = get_atom_features_with_context(
                    substituent_pdb=str(pdb_path),
                    core_pdb=str(core_pdb),
                    protein_pdb=effective_protein_pdb,
                    rtf_entry=rtf_data,
                    include_charges=True,
                    include_atom_ids=False,
                    prep_dir=str(prep_dir),
                    aev_cutoff=aev_cutoff,
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
                    if protein_context is not None:
                        print(f"    Protein source: minimized.pdb ({len(protein_context[0])} atoms within cutoff)")
                    elif protein_pdb:
                        print(f"    Protein source: {protein_pdb.name} (fallback)")
                    if context_info.get('reference_subs_from_other_sites'):
                        print(f"    Reference subs from other sites:")
                        for ref_info in context_info['reference_subs_from_other_sites']:
                            print(f"      - {ref_info['pdb']}: {ref_info['filtered_atoms']} atoms "
                                  f"(removed {ref_info['removed_duplicates']} duplicates)")
                    first_sub_processed = True
            elif solvent_state in ('solvent', 'water', 'solv') and core_pdb:
                # Solvent path: core + other-site subs (via prep_dir) + water from
                # minimized.pdb within AEV cutoff.
                solvent_context = None  # type: Optional[Tuple[List, List]]

                if minimized_pdb:
                    try:
                        solvent_context = extract_environment_atoms_from_minimized(
                            minimized_pdb=minimized_pdb,
                            sub_pdb=pdb_path,
                            core_pdb=core_pdb,
                            aev_cutoff=aev_cutoff,
                            prep_dir=prep_dir,
                        )
                        if solvent_context is None:
                            warnings.warn(
                                f"[{system_name}] No solvent atoms within {aev_cutoff} Å "
                                f"of {pdb_path.name} in minimized.pdb"
                            )
                    except Exception as e:
                        warnings.warn(
                            f"[{system_name}] Could not extract solvent context from "
                            f"minimized.pdb for {pdb_path.name}: {e}"
                        )

                features_dict = get_atom_features_with_context(
                    substituent_pdb=str(pdb_path),
                    core_pdb=str(core_pdb),
                    solvent_context=solvent_context,
                    rtf_entry=rtf_data,
                    include_charges=True,
                    include_atom_ids=False,
                    prep_dir=str(prep_dir),
                    aev_cutoff=aev_cutoff,
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
                    print(f"    Solvent included: {context_info.get('solvent_included', False)}")
                    if solvent_context is not None:
                        print(f"    Solvent source: minimized.pdb ({len(solvent_context[0])} atoms within cutoff)")
                    else:
                        print(f"    Solvent source: none (minimized.pdb unavailable or empty within cutoff)")
                    if context_info.get('reference_subs_from_other_sites'):
                        print(f"    Reference subs from other sites:")
                        for ref_info in context_info['reference_subs_from_other_sites']:
                            print(f"      - {ref_info['pdb']}: {ref_info['filtered_atoms']} atoms "
                                  f"(removed {ref_info['removed_duplicates']} duplicates)")
                    first_sub_processed = True

            elif core_pdb:
                # Vacuum/gas path: core + other-site subs (via prep_dir) — no additional
                # environment atoms (no protein, no solvent).
                features_dict = get_atom_features_with_context(
                    substituent_pdb=str(pdb_path),
                    core_pdb=str(core_pdb),
                    rtf_entry=rtf_data,
                    include_charges=True,
                    include_atom_ids=False,
                    prep_dir=str(prep_dir),
                    aev_cutoff=aev_cutoff,
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
                    if context_info.get('reference_subs_from_other_sites'):
                        print(f"    Reference subs from other sites:")
                        for ref_info in context_info['reference_subs_from_other_sites']:
                            print(f"      - {ref_info['pdb']}: {ref_info['filtered_atoms']} atoms "
                                  f"(removed {ref_info['removed_duplicates']} duplicates)")
                    first_sub_processed = True

            else:
                # Fallback: no core available — substituent-only AEVs
                aevs = get_substituent_aevs(str(pdb_path))
                charges = extract_charges_from_rtf(rtf_path, pdb_path.name) if rtf_data else None

                # Print verbose info for substituent-only fallback
                if verbose and not first_sub_processed:
                    print(f"\n  [VERBOSE] AEV Context for {pdb_path.name}:")
                    print(f"    Mode: Substituent-only (no core PDB found)")
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
    aev_cutoff: float = 5.1,
    verbose: bool = False
) -> List[Dict]:
    """Generate training datasets for all pretraining systems.
    
    Args:
        pretraining_root: Root directory containing all pretraining systems
        output_root: Root directory for output files
        skip_systems: List of system names to skip (default: ['14benz_pair_combos'])
        aev_cutoff: AEV spatial cutoff in Angstroms (default: 5.1 Å, matches ANI-2x radial cutoff)
        verbose: If True, print detailed context information for each system
        
    Returns:
        List of statistics dicts for each system
    """
    if skip_systems is None:
        skip_systems = ['14benz_pair_combos', '1_analysis_scripts']

    # Find all top-level system directories, then expand any collection dirs.
    top_dirs = sorted(
        d for d in pretraining_root.iterdir()
        if d.is_dir() and d.name not in skip_systems
    )
    leaf_systems = _collect_leaf_system_dirs(top_dirs)

    print(f"Found {len(leaf_systems)} pretraining systems to process "
          f"(from {len(top_dirs)} top-level directories)")

    all_stats = []
    errors = []

    for system_dir, unique_name in leaf_systems:
        output_path = output_root / f"{unique_name}_training_data.pt"

        if output_path.exists():
            print(f"  Skipping {unique_name} (dataset already exists)")
            # Load minimal stats from existing file for the summary
            try:
                import torch as _torch
                existing = _torch.load(output_path, weights_only=False)
                all_stats.append({
                    'system_name': unique_name,
                    'total_atoms': existing['features'].shape[0] if 'features' in existing else 0,
                    'num_substituents': existing.get('num_substituents', 0),
                })
            except Exception:
                pass
            continue

        try:
            stats = generate_training_data_for_system(
                system_dir, output_path,
                aev_cutoff=aev_cutoff,
                verbose=verbose,
            )
            all_stats.append(stats)
        except Exception as e:
            error_msg = f"Failed to process {unique_name}: {e}"
            warnings.warn(error_msg)
            errors.append(error_msg)
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Successfully processed {len(all_stats)}/{len(leaf_systems)} systems")
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    print(f"\nTotal atoms across all systems: {sum(s['total_atoms'] for s in all_stats):,}")
    
    return all_stats


# ---------------------------------------------------------------------------
# Bond-topology-aware dataset generation (for AtomBondGNN pretraining)
# ---------------------------------------------------------------------------

def generate_bond_training_data_for_system(
    system_dir: Path,
    output_path: Path,
    aev_cutoff: float = 5.1,
    verbose: bool = False,
) -> Dict:
    """Generate per-substituent bond-topology training data for AtomBondGNN pretraining.

    Each substituent is stored as a dict containing its AEV tensor, partial
    charges, integer atom-type IDs, and a bidirectional bond edge index.  The
    on-disk format is a list of such dicts (keyed ``'substituents'``) rather
    than one flat tensor, because substituents have variable atom counts and
    individual bond graphs.

    Args:
        system_dir: Path to pretraining system directory.
        output_path: Where to save the dataset (.pt file).
        aev_cutoff: AEV spatial cutoff distance in Angstroms (default: 5.1 Å).
        verbose: Print per-substituent context information for the first sub.

    Returns:
        Statistics dict with keys: system_name, num_substituents, total_atoms,
        aev_length, output_path.
    """
    system_name = system_dir.name
    print(f"\nProcessing (bond) {system_name}...")

    metadata = load_system_metadata(system_dir)
    prep_dir = metadata['prep_dir']
    solvent_state = metadata['solvent_state']
    print(f"  Prep directory: {prep_dir}")
    print(f"  Solvent state: {solvent_state}")

    core_pdb = detect_core_pdb(prep_dir)
    minimized_pdb = detect_minimized_pdb(prep_dir)
    if minimized_pdb:
        print(f"  Minimized PDB: {minimized_pdb.name}")

    protein_pdb = None
    if solvent_state == 'protein' and not minimized_pdb:
        protein_pdb = detect_protein_pdb(prep_dir)

    # Collect substituent PDB files
    active_subs_ordered = metadata.get('active_subs_ordered')
    if active_subs_ordered:
        sub_pdbs = []
        for site_label in sorted(active_subs_ordered.keys(),
                                  key=lambda s: int(s.replace('site', ''))):
            for master_sub in active_subs_ordered[site_label]:
                frag = prep_dir / f"{master_sub}_frag.pdb"
                if frag.exists():
                    sub_pdbs.append(frag)
                else:
                    warnings.warn(f"[{system_name}] Active sub frag PDB not found: {frag.name}")
    else:
        sub_pdbs = sorted(prep_dir.glob('site*_sub*_frag.pdb'))

    if not sub_pdbs:
        raise ValueError(f"No substituent PDB files found in {prep_dir}")

    print(f"  Found {len(sub_pdbs)} substituent PDB files")

    all_substituents: List[Dict] = []
    total_atoms = 0
    first_sub_processed = False

    for pdb_path in sub_pdbs:
        try:
            # Parse RTF for charges + bond fallback
            rtf_path = pdb_path.parent / pdb_path.name.replace('_frag.pdb', '_pres.rtf')
            rtf_data = None
            if rtf_path.exists():
                try:
                    rtf_data = parse_rtf_file(str(rtf_path))
                except Exception as e:
                    warnings.warn(f"[{system_name}] Could not parse RTF {rtf_path.name}: {e}")

            # ── Context-aware AEV computation (identical branch logic to
            #    generate_training_data_for_system, but with include_atom_ids=True) ──

            if solvent_state in ('protein', 'prot') and core_pdb and (minimized_pdb or protein_pdb):
                protein_context = None
                if minimized_pdb:
                    try:
                        protein_context = extract_environment_atoms_from_minimized(
                            minimized_pdb=minimized_pdb,
                            sub_pdb=pdb_path,
                            core_pdb=core_pdb,
                            aev_cutoff=aev_cutoff,
                            prep_dir=prep_dir,
                        )
                    except Exception as e:
                        warnings.warn(
                            f"[{system_name}] Could not extract protein context for "
                            f"{pdb_path.name}: {e}"
                        )
                effective_protein_pdb = protein_context if protein_context is not None else (
                    str(protein_pdb) if protein_pdb else None
                )
                features_dict = get_atom_features_with_context(
                    substituent_pdb=str(pdb_path),
                    core_pdb=str(core_pdb),
                    protein_pdb=effective_protein_pdb,
                    rtf_entry=rtf_data,
                    include_charges=True,
                    include_atom_ids=True,
                    prep_dir=str(prep_dir),
                    aev_cutoff=aev_cutoff,
                )

            elif solvent_state in ('solvent', 'water', 'solv') and core_pdb:
                solvent_context = None
                if minimized_pdb:
                    try:
                        solvent_context = extract_environment_atoms_from_minimized(
                            minimized_pdb=minimized_pdb,
                            sub_pdb=pdb_path,
                            core_pdb=core_pdb,
                            aev_cutoff=aev_cutoff,
                            prep_dir=prep_dir,
                        )
                    except Exception as e:
                        warnings.warn(
                            f"[{system_name}] Could not extract solvent context for "
                            f"{pdb_path.name}: {e}"
                        )
                features_dict = get_atom_features_with_context(
                    substituent_pdb=str(pdb_path),
                    core_pdb=str(core_pdb),
                    solvent_context=solvent_context,
                    rtf_entry=rtf_data,
                    include_charges=True,
                    include_atom_ids=True,
                    prep_dir=str(prep_dir),
                    aev_cutoff=aev_cutoff,
                )

            elif core_pdb:
                features_dict = get_atom_features_with_context(
                    substituent_pdb=str(pdb_path),
                    core_pdb=str(core_pdb),
                    rtf_entry=rtf_data,
                    include_charges=True,
                    include_atom_ids=True,
                    prep_dir=str(prep_dir),
                    aev_cutoff=aev_cutoff,
                )

            else:
                # Substituent-only fallback
                aevs_only = get_substituent_aevs(str(pdb_path))
                charges_only = (
                    extract_charges_from_rtf_metadata(rtf_data)
                    if rtf_data else None
                )
                features_dict = {
                    'aevs': aevs_only,
                    'charges': charges_only,
                    'atom_ids': None,
                }

            if verbose and not first_sub_processed:
                ctx = features_dict.get('context_info', {})
                print(f"\n  [VERBOSE] Bond dataset AEV context for {pdb_path.name}:")
                print(f"    Context sources: {', '.join(ctx.get('context_sources', []))}")
                print(f"    Total context atoms: {ctx.get('total_context_atoms', 0)}")
                first_sub_processed = True

            aevs = features_dict['aevs']           # [N, 2288]
            charges = features_dict.get('charges')  # [N] or None
            atom_ids = features_dict.get('atom_ids')  # [N] int or None

            num_atoms = aevs.shape[0]

            if charges is None:
                warnings.warn(
                    f"[{system_name}] No charges for {pdb_path.name}, using zeros"
                )
                charges = torch.zeros(num_atoms, dtype=torch.float32)
            if len(charges) != num_atoms:
                warnings.warn(
                    f"[{system_name}] Charge count mismatch for {pdb_path.name}: "
                    f"{len(charges)} vs {num_atoms} atoms. Using zeros."
                )
                charges = torch.zeros(num_atoms, dtype=torch.float32)

            if atom_ids is None:
                warnings.warn(
                    f"[{system_name}] No atom_ids for {pdb_path.name}, using zeros"
                )
                atom_ids = torch.zeros(num_atoms, dtype=torch.long)

            # Bond topology (RDKit primary; RTF BOND section fallback)
            rtf_bonds = rtf_data.get('bonds', []) if rtf_data else []
            bond_edge_index, bond_edge_attr = get_bond_edge_index_from_pdb(
                str(pdb_path), rtf_bonds=rtf_bonds
            )

            all_substituents.append({
                'aev': aevs,                         # [N, 2288]
                'charges': charges,                  # [N]
                'atom_ids': atom_ids,                # [N] int64
                'bond_edge_index': bond_edge_index,  # [2, 2E]
                'bond_edge_attr': bond_edge_attr,    # [2E, 1]
                'pdb_name': pdb_path.name,
            })
            total_atoms += num_atoms

        except Exception as e:
            warnings.warn(f"[{system_name}] Error processing {pdb_path.name}: {e}")
            continue

    if not all_substituents:
        raise ValueError(f"No valid substituents generated for {system_dir.name}")

    aev_length = all_substituents[0]['aev'].shape[1]
    print(f"  Generated {len(all_substituents)} substituents, {total_atoms} total atoms")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'dataset_type': 'bond_topology',
        'substituents': all_substituents,
        'system_name': metadata['system_name'],
        'solvent_state': metadata['solvent_state'],
        'num_substituents': len(all_substituents),
        'total_atoms': total_atoms,
        'aev_length': aev_length,
    }, output_path)

    print(f"  Saved to {output_path}")
    return {
        'system_name': metadata['system_name'],
        'num_substituents': len(all_substituents),
        'total_atoms': total_atoms,
        'aev_length': aev_length,
        'output_path': str(output_path),
    }


def generate_all_bond_pretraining_datasets(
    pretraining_root: Path,
    output_root: Path,
    skip_systems: Optional[List[str]] = None,
    aev_cutoff: float = 5.1,
    verbose: bool = False,
) -> List[Dict]:
    """Generate bond-topology training datasets for all pretraining systems.

    Mirrors :func:`generate_all_pretraining_datasets` but calls
    :func:`generate_bond_training_data_for_system` so that each saved file
    contains per-substituent bond graphs required by :class:`AtomBondGNN`.

    Args:
        pretraining_root: Root directory containing all pretraining systems.
        output_root: Root directory for output files (datasets stored here).
        skip_systems: System directory names to skip.
        aev_cutoff: AEV spatial cutoff in Angstroms (default: 5.1 Å).
        verbose: Print verbose AEV context info for first substituent per system.

    Returns:
        List of statistics dicts for each processed system.
    """
    if skip_systems is None:
        skip_systems = ['1_analysis_scripts']

    # Find all top-level system directories, then expand any collection dirs.
    top_dirs = sorted(
        d for d in pretraining_root.iterdir()
        if d.is_dir() and d.name not in skip_systems
    )
    leaf_systems = _collect_leaf_system_dirs(top_dirs)

    print(f"Found {len(leaf_systems)} pretraining systems to process "
          f"(from {len(top_dirs)} top-level directories, bond topology mode)")

    all_stats: List[Dict] = []
    errors: List[str] = []

    for system_dir, unique_name in leaf_systems:
        output_path = output_root / f"{unique_name}_training_data.pt"

        if output_path.exists():
            print(f"  Skipping {unique_name} (dataset already exists)")
            try:
                import torch as _torch
                existing = _torch.load(output_path, weights_only=False)
                all_stats.append({
                    'system_name': unique_name,
                    'num_substituents': existing.get('num_substituents', 0),
                    'total_atoms': existing.get('total_atoms', 0),
                })
            except Exception:
                pass
            continue

        try:
            stats = generate_bond_training_data_for_system(
                system_dir, output_path,
                aev_cutoff=aev_cutoff,
                verbose=verbose,
            )
            all_stats.append(stats)
        except Exception as e:
            error_msg = f"Failed to process {unique_name}: {e}"
            warnings.warn(error_msg)
            errors.append(error_msg)
            continue

    print(f"\n{'='*70}")
    print(f"SUMMARY: Successfully processed {len(all_stats)}/{len(leaf_systems)} systems")
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    total_subs = sum(s.get('num_substituents', 0) for s in all_stats)
    total_atoms = sum(s.get('total_atoms', 0) for s in all_stats)
    print(f"\nTotal substituents: {total_subs:,}")
    print(f"Total atoms across all systems: {total_atoms:,}")

    return all_stats
