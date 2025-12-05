"""Workflow utility functions for training with SLURM job management.

This module contains reusable functions for:
- Fixing system-specific simulation scripts
- Managing simulation success checking and metric parsing
- SLURM job submission and monitoring utilities

Note: For manifest loading, use mllf.cli.workflow.load_manifest()
"""
from pathlib import Path
from typing import List, Dict, Optional
import json
import re

# Re-export load_manifest from cli.workflow for convenience
from mllf.cli.workflow import load_manifest


def fix_msld_flat_for_single_site(combo_path: Path, 
                                   site_atoms: Dict[int, str] = None) -> bool:
    """Modify msld_flat.py to delete only atoms that overlap with present sites.
    
    For multi-site λ-dynamics simulations, certain atoms in the base structure
    may overlap with specific substituents. When only a subset of sites are
    present in a combination, we should only delete atoms that would overlap
    with the present sites.
    
    This function reads mapping.json to determine which original sites are
    present, then modifies the atom deletion command in msld_flat.py accordingly.
    
    Example usage:
        # 14benz system
        site_atoms = {1: 'C4 H4', 2: 'C5 H5'}
        fix_msld_flat_for_single_site(combo_path, site_atoms)
        
        # Indole system
        site_atoms = {1: 'C2 H2', 2: 'C6 H6'}
        fix_msld_flat_for_single_site(combo_path, site_atoms)
    
    Args:
        combo_path: Path to combination directory containing msld_flat.py
            and mapping.json.
        site_atoms: Dictionary mapping site numbers to atom selection strings.
            Example: {1: 'C4 H4', 2: 'C5 H5'} for 14benz system.
            If None, defaults to 14benz atoms for backward compatibility.
    
    Returns:
        True if file was modified, False if skipped or unchanged.
    """
    # Default to 14benz atoms for backward compatibility
    if site_atoms is None:
        site_atoms = {1: 'C4 H4', 2: 'C5 H5'}
    msld_flat = combo_path / 'msld_flat.py'
    if not msld_flat.exists():
        return False
    
    # Read mapping.json to determine ORIGINAL sites present
    mapping_file = combo_path / 'mapping.json'
    if not mapping_file.exists():
        return False
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Extract unique original site numbers from entries that have site info
    original_sites = set()
    for entry in mapping:
        site = entry.get('original_site')
        if site is not None:
            original_sites.add(site)
    
    # Determine what to delete based on original sites
    # Only modify if exactly one site is present
    if len(original_sites) != 1:
        return False  # Either no sites identified or multiple sites - leave unchanged
    
    original_site = list(original_sites)[0]
    
    # Determine which atoms to delete based on ORIGINAL site number
    atoms_to_delete = site_atoms.get(original_site)
    if atoms_to_delete is None:
        # Site not in mapping - leave as is
        return False
    
    # Read and modify the msld_flat.py content
    content = msld_flat.read_text()
    
    # Replace the delete line - handle both original and already-modified versions
    # Pattern matches any combination of atoms in the selection
    old_pattern = r"select\.store_selection\('todelete',pycharmm\.SelectAtoms\(\)\.by_res_and_type\(ligseg,resnum,'[^']+'\)\)"
    new_line = f"select.store_selection('todelete',pycharmm.SelectAtoms().by_res_and_type(ligseg,resnum,'{atoms_to_delete}'))"
    
    new_content = re.sub(old_pattern, new_line, content)
    
    # Only write if something changed
    if new_content != content:
        msld_flat.write_text(new_content)
        return True
    
    return False


def check_simulation_success(output_file: Path) -> bool:
    """Check if a simulation completed successfully.
    
    Args:
        output_file: Path to simulation output file.
    
    Returns:
        True if simulation terminated normally, False otherwise.
    """
    from mllf.file_handling.read_output import terminated_normally
    
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
        return terminated_normally(output_text)
    except Exception:
        return False


def parse_simulation_metrics(output_file: Path) -> Dict[str, List]:
    """Parse raw populations and transitions from simulation output.
    
    This is a convenience wrapper that extracts aggregate metrics
    (total populations per block, total transitions per site) as lists
    for use in reward computation. For full parsed data, use
    mllf.cli.sim.parse_simulation_results() instead.
    
    Args:
        output_file: Path to simulation output file.
    
    Returns:
        Dict with 'populations' and 'transitions' lists. Returns empty
        lists if parsing fails.
    """
    from mllf.file_handling.read_output import (
        parse_single_population,
        parse_transitions_and_rates
    )
    
    raw_metrics = {'populations': [], 'transitions': []}
    
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
        
        population_data = parse_single_population(output_text)
        transitions_data, _ = parse_transitions_and_rates(output_text)
        
        # Extract total populations per block
        for block_id, block_info in population_data.items():
            counts_dict = block_info.get('counts', {})
            total_count = sum(counts_dict.values())
            raw_metrics['populations'].append(total_count)
        
        # Extract total transitions per site
        for site_id, trans_dict in transitions_data.items():
            total_trans = sum(trans_dict.values())
            raw_metrics['transitions'].append(total_trans)
    except Exception:
        pass
    
    return raw_metrics
