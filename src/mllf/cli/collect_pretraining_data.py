"""Collect pretraining data from MSLD run directories.

This script extracts data from run#/ directories containing:
- prep/ directory with site/sub RTF files
- output file with population and transition data
- variables*.py or variables*.inp file with bias coefficients

The extracted data is organized into a format suitable for offline/pretraining.
Only essential graph information (atom types, charges, solvent state) is saved,
not the full prep directory.
"""
from __future__ import annotations

import argparse
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json

from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.file_handling.read_output import (
    parse_single_population,
    parse_transitions_and_rates,
    terminated_normally
)
from mllf.file_handling.read_bias_coeff import read_bias_coeff


def find_run_directories(base_path: Path) -> List[Path]:
    """Find all run directories in the base path.
    
    Handles two directory structures:
    1. MSLD runs: base_path/run#/ (each contains prep/, output, variables)
    2. Generated combos: base_path/comb_###/run_###/ (shared prep/ at comb level)
    
    Args:
        base_path: Base directory to search for run# directories
        
    Returns:
        List of Path objects for run directories, sorted numerically
    """
    run_dirs = []
    
    # Check if this is a generated combo structure (has comb_* directories)
    combo_dirs = list(base_path.glob("comb_*"))
    combo_dirs = [d for d in combo_dirs if d.is_dir()]
    
    def _has_run_data(p: Path) -> bool:
        """Return True if the directory contains the minimum files for a completed run."""
        has_vars = bool(list(p.glob("variables*.py")) or list(p.glob("variables*.inp")))
        has_output = (p / "output").exists() or (p / "output.out").exists()
        return has_vars or has_output

    if combo_dirs:
        # Generated combo structure - look for run_### inside each comb_###
        # Exclude directories with _failed in name or missing required files
        for combo_dir in combo_dirs:
            run_dirs.extend([
                d for d in combo_dir.glob("run_*")
                if d.is_dir() and '_failed' not in d.name.lower() and _has_run_data(d)
            ])
    else:
        # MSLD run structure - look for run# at base level
        # Exclude directories with _failed in name or missing required files
        run_dirs = list(base_path.glob("run*"))
        run_dirs = [
            d for d in run_dirs
            if d.is_dir() and '_failed' not in d.name.lower() and _has_run_data(d)
        ]
    
    # Sort numerically by extracting run number
    def get_run_num(p: Path) -> int:
        match = re.search(r'run[_]?(\d+)', p.name)
        return int(match.group(1)) if match else 0
    
    return sorted(run_dirs, key=get_run_num)


def parse_bias_from_py(
    variables_file: Path,
    prep_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Parse bias coefficients from variables*.py file.

    Delegates to :func:`~mllf.file_handling.read_bias_coeff.read_bias_coeff`
    which handles both formats:

    * **Flat format** (older, auto-generated pretraining files): ``b`` is a
      1-D list, ``c``/``x``/``s`` are NxN matrices.
    * **Matrix + scalar format** (``variablesflat.py`` from ALF): ``b`` is a
      2-D ``[[b1, b2, …]]`` row (normalised to 1-D), plus ``c``/``x``/``s``
      matrices and redundant individual ``cs*``/``xs*``/``ss*`` scalars.

    Only the clean matrix keys (``b``, ``c``, ``x``, ``s``) and
    ``_nsubs_per_site`` are returned — the raw scalar subdicts are discarded
    so stored ``variables.py`` files stay compact.

    Args:
        variables_file: Path to variables*.py file.
        prep_dir:       Optional prep directory used to locate ``alf_info.py``
                        when ``alf_info_string`` is absent from the file.

    Returns:
        Dict with ``b``, ``c``, ``x``, ``s`` matrix keys and
        ``_nsubs_per_site`` (when determinable).
    """
    raw = read_bias_coeff(str(variables_file), prep_dir=prep_dir)

    # Keep only the matrix keys and topology metadata; discard lams/cs/xs/ss
    # scalar subdicts that bloat stored files.
    result: Dict[str, Any] = {}
    for key in ('b', 'c', 'x', 's', '_nsubs_per_site'):
        if key in raw:
            result[key] = raw[key]

    if not result.get('b'):
        raise ValueError(f"Could not parse bias vector 'b' from {variables_file}")

    return result


def parse_bias_from_inp(variables_file: Path) -> Dict[str, Any]:
    """Parse bias coefficients from variables*.inp file (old format).
    
    Old format has individual coefficient entries like:
      set lams1s1 = 0.00        # baseline bias for site1_sub1
      set cs1s1s1s2 = -0.00     # quadratic coefficient
      set xs1s1s2s3 = 1.50      # skew coefficient
    
    We need to convert these to matrix format for compatibility with the
    graph building code that expects c, x, s matrices.
    
    Args:
        variables_file: Path to variables*.inp file
        
    Returns:
        Dict with bias coefficients in matrix format (c, x, s, b keys)
    """
    # Read old format (returns {"lams": {...}, "cs": {...}, "xs": {...}, "ss": {...}})
    old_data = read_bias_coeff(str(variables_file))
    
    # Determine dimensions from lams entries (baseline biases)
    # lams format: lams1s1, lams1s2, ..., lams2s1, lams2s2, ...
    site_sub_counts = {}
    for lam_key in old_data.get("lams", {}).keys():
        match = re.match(r'lams(\d+)s(\d+)', lam_key)
        if match:
            site_num = int(match.group(1))
            sub_num = int(match.group(2))
            if site_num not in site_sub_counts:
                site_sub_counts[site_num] = set()
            site_sub_counts[site_num].add(sub_num)
    
    if not site_sub_counts:
        # Try to infer from cs entries: cs1s1s1s2 means site1_sub1 to site1_sub2
        for cs_key in old_data.get("cs", {}).keys():
            match = re.match(r'cs(\d+)s(\d+)s(\d+)s(\d+)', cs_key)
            if match:
                site1, sub1, site2, sub2 = map(int, match.groups())
                for site, sub in [(site1, sub1), (site2, sub2)]:
                    if site not in site_sub_counts:
                        site_sub_counts[site] = set()
                    site_sub_counts[site].add(sub)
    
    # Calculate total number of blocks (sum of substituents across all sites)
    num_blocks = sum(len(subs) for subs in site_sub_counts.values())
    
    if num_blocks == 0:
        raise ValueError(f"Could not determine number of blocks from {variables_file}")
    
    # Create mapping from site_sub to block index
    block_idx = 0
    site_sub_to_idx = {}
    for site_num in sorted(site_sub_counts.keys()):
        for sub_num in sorted(site_sub_counts[site_num]):
            site_sub_to_idx[(site_num, sub_num)] = block_idx
            block_idx += 1
    
    # Initialize matrices
    c_matrix = [[0.0] * num_blocks for _ in range(num_blocks)]
    x_matrix = [[0.0] * num_blocks for _ in range(num_blocks)]
    s_matrix = [[0.0] * num_blocks for _ in range(num_blocks)]
    b_vector = [0.0] * num_blocks
    
    # Fill b vector from lams entries (baseline biases)
    for key, value in old_data.get("lams", {}).items():
        match = re.match(r'lams(\d+)s(\d+)', key)
        if match:
            site_num = int(match.group(1))
            sub_num = int(match.group(2))
            idx = site_sub_to_idx.get((site_num, sub_num))
            if idx is not None:
                b_vector[idx] = float(value)
    
    # Fill c matrix from cs entries (quadratic coefficients)
    # Note: Original format stores full matrix, not just upper triangle
    for key, value in old_data.get("cs", {}).items():
        match = re.match(r'cs(\d+)s(\d+)s(\d+)s(\d+)', key)
        if match:
            site1, sub1, site2, sub2 = map(int, match.groups())
            i = site_sub_to_idx.get((site1, sub1))
            j = site_sub_to_idx.get((site2, sub2))
            if i is not None and j is not None:
                c_matrix[i][j] = float(value)
    
    # Fill x matrix from xs entries (skew coefficients)
    for key, value in old_data.get("xs", {}).items():
        match = re.match(r'xs(\d+)s(\d+)s(\d+)s(\d+)', key)
        if match:
            site1, sub1, site2, sub2 = map(int, match.groups())
            i = site_sub_to_idx.get((site1, sub1))
            j = site_sub_to_idx.get((site2, sub2))
            if i is not None and j is not None:
                x_matrix[i][j] = float(value)
                # Note: x is typically asymmetric
    
    # Fill s matrix from ss entries (end-state coefficients)
    # Note: Original format stores full matrix, not just upper triangle
    for key, value in old_data.get("ss", {}).items():
        match = re.match(r'ss(\d+)s(\d+)s(\d+)s(\d+)', key)
        if match:
            site1, sub1, site2, sub2 = map(int, match.groups())
            i = site_sub_to_idx.get((site1, sub1))
            j = site_sub_to_idx.get((site2, sub2))
            if i is not None and j is not None:
                s_matrix[i][j] = float(value)
    
    return {
        "b": b_vector,
        "c": c_matrix,
        "x": x_matrix,
        "s": s_matrix,
        # Topology metadata: not a bias coefficient, stripped before writing variables.py
        "_nsubs_per_site": [len(site_sub_counts[s]) for s in sorted(site_sub_counts.keys())],
    }


def convert_to_json_serializable(data: Any) -> Any:
    """Convert data structures to JSON-serializable format.
    
    Handles:
    - Tuple keys in dicts (convert to string)
    - Float keys in dicts (convert to string)
    - Nested structures
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Convert tuple/float keys to strings
            if isinstance(key, tuple):
                str_key = f"{key[0]}_{key[1]}"
            elif isinstance(key, float):
                str_key = str(key)
            else:
                str_key = str(key)
            result[str_key] = convert_to_json_serializable(value)
        return result
    elif isinstance(data, (list, tuple)):
        return [convert_to_json_serializable(item) for item in data]
    else:
        return data


def parse_output_file(output_file: Path, run_dir: Path) -> Dict[str, Any]:
    """Parse population and transition data from output file.
    
    Uses existing file_handling utilities to parse CHARMM output.
    Assumes normal termination unless '_failed' appears in run directory name.
    
    Args:
        output_file: Path to output file
        run_dir: Path to run directory (used to check for _failed)
        
    Returns:
        Dict with 'populations', 'transitions', 'rates', and 'terminated_normally'
    """
    content = output_file.read_text()
    
    # Check termination status based on directory name
    # Assume normal termination unless _failed in path
    terminated = '_failed' not in str(run_dir).lower()
    
    # Parse populations using existing utility
    populations = parse_single_population(content)
    
    # Parse transitions and rates using existing utility  
    transitions, rates = parse_transitions_and_rates(content)
    
    # Convert to JSON-serializable format (handles tuple/float keys)
    return {
        'populations': convert_to_json_serializable(populations),
        'transitions': convert_to_json_serializable(transitions),
        'rates': convert_to_json_serializable(rates),
        'terminated_normally': terminated
    }


def extract_graph_info(source_prep: Path, solvent_state: str = "unknown") -> Dict[str, Any]:
    """Extract essential graph information from prep directory.
    
    Instead of copying the entire prep directory, extract only the information
    needed for graph construction: atom types, total charge, and solvent state.
    Only includes site#_sub# entries, filtering out full_ligand and top_all36_msld.
    
    Args:
        source_prep: Source prep directory (may be symlink)
        solvent_state: Solvent state ("solvent", "vacuum", "protein", or "unknown")
        
    Returns:
        Dict containing site/sub information with atom_types and total_charge
    """
    # Resolve symlink if present
    if source_prep.is_symlink():
        source_prep = source_prep.resolve()
    
    if not source_prep.exists() or not source_prep.is_dir():
        raise ValueError(f"prep directory not found: {source_prep}")
    
    # Parse all RTF files in prep directory
    rtf_data = parse_rtf_dir(str(source_prep))
    
    # Filter to only include site#_sub# entries
    filtered_sites = {
        key: value for key, value in rtf_data.items()
        if key.startswith("site") and "_sub" in key
    }
    
    # Add solvent state to the data
    graph_info = {
        "solvent_state": solvent_state,
        "sites": filtered_sites
    }
    
    return graph_info


def parse_active_subs_from_inp(
    prep_dir: Path,
    nsubs_per_site: List[int],
) -> Dict[str, List[str]]:
    """Parse the ordered list of active substituents from the CHARMM .inp file.

    The order in which RTF files are ``read rtf append``-ed in the main .inp file
    defines the sequential bias-variable numbering (lams1s1=position 1, etc.).
    For systems whose .inp lists the FULL master prep (more subs than
    nsubs_per_site indicates), falls back to sequential numbering (sub1, sub2, …).

    Args:
        prep_dir:       Prep directory to search for a CHARMM .inp file.
        nsubs_per_site: Number of active subs per site, from the bias variables.

    Returns:
        Dict mapping 'site1', 'site2', … to an ordered list of master sub names,
        e.g. {'site1': ['site1_sub3', 'site1_sub1', …], 'site2': ['site2_sub15', …]}
    """
    # Candidate .inp files: top-level only, exclude known non-MSLD files
    exclude = {'lpsites.inp', 'variablesflat.inp'}
    inp_candidates = [
        p for p in prep_dir.glob('*.inp')
        if p.name not in exclude and 'traj' not in p.name
    ]

    def _sequential(npsite: List[int]) -> Dict[str, List[str]]:
        return {
            f'site{s}': [f'site{s}_sub{i}' for i in range(1, n + 1)]
            for s, n in enumerate(npsite, start=1)
        }

    if not inp_candidates or not nsubs_per_site:
        return _sequential(nsubs_per_site)

    # Parse the RTF read order from the first valid .inp file.
    # Only consider uncommented lines (CHARMM comment prefix: ! or !!).
    pattern = re.compile(
        r'read\s+rtf\s+append.*?/?(site(\d+)_sub(\d+))_pres\.rtf',
        re.IGNORECASE,
    )
    comment_re = re.compile(r'^\s*!')
    by_site: Dict[int, List[str]] = {}
    all_active_ordered: List[str] = []
    for inp_file in inp_candidates:
        try:
            for line in inp_file.read_text().splitlines():
                if comment_re.match(line):
                    continue  # skip CHARMM comment lines
                m = pattern.search(line)
                if m:
                    master_sub = m.group(1)   # e.g. 'site1_sub3'
                    site_num = int(m.group(2))
                    by_site.setdefault(site_num, []).append(master_sub)
                    all_active_ordered.append(master_sub)
            if by_site:
                break   # found RTF reads in this file; stop
        except Exception:
            continue

    if not by_site:
        return _sequential(nsubs_per_site)

    # First attempt: per-site grouping.  Works when the CHARMM file-naming
    # site prefix matches the MSLD site index (e.g. BACE, MCL1, PTP1B, P38).
    result: Dict[str, List[str]] = {}
    per_site_ok = True
    for s_idx, n_active in enumerate(nsubs_per_site):
        s_num = s_idx + 1
        inp_subs = by_site.get(s_num, [])
        if len(inp_subs) == n_active:
            # Exact match: .inp lists exactly the right subs in the right order.
            result[f'site{s_num}'] = inp_subs
        else:
            per_site_ok = False
            break
    if per_site_ok:
        return result

    # Second attempt: global sequential.  Handles systems where the active subs
    # for one MSLD site span multiple CHARMM file-naming site prefixes (e.g. JNK1
    # uses site2_sub* files for its single MSLD site's sequential positions 2-6).
    # Take the first sum(nsubs_per_site) active reads in file order and
    # distribute them to MSLD sites by cumulative count.
    total_expected = sum(nsubs_per_site)
    if len(all_active_ordered) >= total_expected:
        result = {}
        offset = 0
        for s_idx, n_active in enumerate(nsubs_per_site):
            s_num = s_idx + 1
            result[f'site{s_num}'] = all_active_ordered[offset:offset + n_active]
            offset += n_active
        return result

    # Final fallback: sequential sub names (sub1, sub2, …) per site.
    # Used when the .inp over-lists subs with no usable ordering signal.
    return _sequential(nsubs_per_site)


def detect_solvent_state(run_dir: Path) -> str:
    """Detect solvent state from directory path.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Solvent state: 'solv', 'gas', 'protein', or 'unknown'
    """
    path_str = str(run_dir).lower()
    
    if "protein" in path_str or "prot" in path_str:
        return "protein"
    elif "vac" in path_str or "vacuum" in path_str or "gas" in path_str:
        return "gas"
    elif "solv" in path_str or "water" in path_str or "aq" in path_str:
        return "solv"
    else:
        return "unknown"


def collect_run_data(
    run_dir: Path,
    solvent_state: Optional[str] = None,
    nsubs_per_site_override: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Collect all data from a single run directory.
    
    Handles two directory structures:
    1. MSLD runs: run_dir contains prep/, output, variables*.py
    2. Generated combos: run_dir/output.out, run_dir/variables.py, prep/ at parent level
    
    Args:
        run_dir: Path to run directory
        solvent_state: Optional solvent state override (if None, auto-detect)
        nsubs_per_site_override: If provided, use this nsubs_per_site instead of
            the value read from alf_info.py.  Needed when prep/ is a full-system
            prep but only a subset of substituents are active (e.g. combo runs).
        
    Returns:
        Tuple of (bias_data, output_results, graph_info)
    """
    # Find variables file (try .py first, then .inp)
    variables_files = list(run_dir.glob("variables*.py"))
    if not variables_files:
        variables_files = list(run_dir.glob("variables*.inp"))
    
    if not variables_files:
        raise ValueError(f"No variables file found in {run_dir}")
    
    variables_file = variables_files[0]

    # Find prep directory early so parse_bias_from_py can read alf_info.py
    # for matrix-format variables files (variablesflat.py) that lack _nsubs_per_site.
    prep_path = run_dir / 'prep'
    if not prep_path.exists():
        prep_path = run_dir.parent / 'prep'

    # Parse bias coefficients
    if variables_file.suffix == '.py':
        bias_data = parse_bias_from_py(
            variables_file,
            prep_dir=prep_path if prep_path.exists() else None,
        )
    else:
        bias_data = parse_bias_from_inp(variables_file)

    # Find output file - check both 'output' and 'output.out'
    output_file = run_dir / 'output'
    if not output_file.exists():
        output_file = run_dir / 'output.out'
    if not output_file.exists():
        raise ValueError(f"output file not found in {run_dir}")
    
    output_results = parse_output_file(output_file, run_dir)
    
    # Find prep directory - check local first, then parent (for generated combos)
    if not prep_path.exists():
        raise ValueError(f"prep directory not found in {run_dir} or {run_dir.parent}")

    # Detect or use provided solvent state
    if solvent_state is None:
        solvent_state = detect_solvent_state(run_dir)
    
    graph_info = extract_graph_info(prep_path, solvent_state)

    # Build the active-substituent mapping from the CHARMM .inp file.
    # The read-RTF order in the .inp defines the sequential bias numbering
    # (lams1s1 = position 1, lams1s2 = position 2, …).
    # For master preps that list ALL subs, parse_active_subs_from_inp falls back
    # to sequential numbering (site1_sub1, site1_sub2, …).
    # nsubs_per_site_override takes priority (e.g. combo runs where prep/ is full-system).
    if nsubs_per_site_override is not None:
        nsubs_per_site = nsubs_per_site_override
        bias_data['_nsubs_per_site'] = nsubs_per_site_override
    else:
        nsubs_per_site = bias_data.get('_nsubs_per_site') or []
    active_subs_ordered = parse_active_subs_from_inp(prep_path, nsubs_per_site)

    # Re-key graph_info['sites']:
    #   sequential key 'site1_sub1' ← master sub at position 0 (e.g. site1_sub3)
    all_rtf_sites = graph_info['sites']
    graph_info['sites'] = {}
    for site_label, ordered_masters in active_subs_ordered.items():
        for seq_idx, master_sub in enumerate(ordered_masters):
            seq_key = f"{site_label}_sub{seq_idx + 1}"
            entry = dict(all_rtf_sites.get(master_sub, {}))
            entry['sub'] = seq_idx + 1  # Sequential sub number (for correct graph node ordering)
            graph_info['sites'][seq_key] = entry

    graph_info['active_subs_ordered'] = active_subs_ordered

    return bias_data, output_results, graph_info


def create_pretraining_entry(
    run_dir: Path,
    output_dir: Path,
    combo_name: Optional[str] = None,
    solvent_state: Optional[str] = None,
    nsubs_per_site_override: Optional[List[int]] = None,
) -> Path:
    """Create a pretraining directory entry from a run directory.
    
    Args:
        run_dir: Source run# directory
        output_dir: Output directory for pretraining data
        combo_name: Optional name for the combination (default: run directory name)
        solvent_state: Optional solvent state override (if None, auto-detect)
        
    Returns:
        Path to created pretraining entry directory
    """
    if combo_name is None:
        combo_name = run_dir.name
    
    # Collect data
    bias_data, output_results, graph_info = collect_run_data(
        run_dir, solvent_state, nsubs_per_site_override=nsubs_per_site_override
    )
    
    # Extract topology metadata before writing YAML (not stored in variables.py)
    nsubs_per_site = bias_data.pop('_nsubs_per_site', None)
    
    # Create output directory
    entry_dir = output_dir / combo_name
    entry_dir.mkdir(parents=True, exist_ok=True)
    
    # Save graph info as JSON instead of copying prep directory
    graph_info_file = entry_dir / 'graph_info.json'
    with open(graph_info_file, 'w') as f:
        json.dump(graph_info, f, indent=2)
    
    # Write bias data to variables.py
    bias_yaml = yaml.dump(bias_data, default_flow_style=False, sort_keys=False)
    variables_content = f'''"""Auto-generated variables file for pretraining data."""
import yaml

bias_string = """
{bias_yaml.strip()}
"""

bias = yaml.safe_load(bias_string)
'''
    (entry_dir / 'variables.py').write_text(variables_content)
    
    # Write output results to JSON
    with open(entry_dir / 'simulation_results.json', 'w') as f:
        json.dump(output_results, f, indent=2)
    
    # Write metadata
    # Calculate total transitions (sum across sites, using only HIGHEST lambda value per site)
    total_trans = 0
    for trans_dict in output_results['transitions'].values():
        if isinstance(trans_dict, dict):
            # Use only the highest lambda value for this site
            if trans_dict:
                max_lambda = max(trans_dict.keys(), key=lambda x: float(x))
                total_trans += trans_dict[max_lambda]
        else:
            total_trans += trans_dict
    
    # Count populated blocks (blocks with non-zero counts at HIGHEST lambda value)
    num_populated = 0
    for block_data in output_results['populations'].values():
        if isinstance(block_data, dict) and 'counts' in block_data:
            counts = block_data['counts']
            if counts:
                # Use only the highest lambda value
                max_lambda = max(counts.keys(), key=lambda x: float(x))
                if counts[max_lambda] > 0:
                    num_populated += 1
    
    # Derive site topology from bias data (authoritative: bias encodes only the
    # active substituents for this group; the prep dir may be a superset).
    if nsubs_per_site is not None:
        num_substituents = sum(nsubs_per_site)
        num_sites = len(nsubs_per_site)
    else:
        # Fallback for .py format files that don't carry _nsubs_per_site
        unique_sites = set(k.split('_')[0] for k in graph_info['sites'])
        num_sites = len(unique_sites)
        num_substituents = len(graph_info['sites'])
        nsubs_per_site = []
    
    metadata = {
        'source_run_dir': str(run_dir),
        'combo_name': combo_name,
        'terminated_normally': output_results['terminated_normally'],
        'total_transitions': total_trans,
        'num_populated_blocks': num_populated,
        'num_sites': num_sites,
        'num_substituents': num_substituents,
        'nsubs_per_site': nsubs_per_site,
        'solvent_state': graph_info['solvent_state']
    }
    
    with open(entry_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return entry_dir


def main():
    """Main entry point for collecting pretraining data."""
    parser = argparse.ArgumentParser(
        description="Collect pretraining data from MSLD run directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all run directories from a base path
  python -m mllf.cli.collect_pretraining_data \\
      /path/to/wLambda \\
      --output-dir pretraining/14benz_solv
  
  # Collect specific runs with custom names
  python -m mllf.cli.collect_pretraining_data \\
      /path/to/wLambda \\
      --output-dir pretraining/14benz_solv \\
      --runs run1 run5 run10
  
  # Process with custom combo name pattern
  python -m mllf.cli.collect_pretraining_data \\
      /path/to/wLambda \\
      --output-dir pretraining/14benz_solv \\
      --name-pattern "14benz_solv_{:03d}"
        """
    )
    
    parser.add_argument(
        'base_path',
        type=Path,
        help='Base directory containing run# subdirectories'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Output directory for pretraining data'
    )
    
    parser.add_argument(
        '--runs',
        nargs='+',
        help='Specific run directories to process (default: all run* directories)'
    )
    
    parser.add_argument(
        '--name-pattern',
        help='Name pattern for combinations (e.g., "combo_{:03d}"). Use {} or {:03d} for run number.'
    )
    
    parser.add_argument(
        '--solvent-state',
        choices=['solvent', 'vacuum', 'protein', 'unknown'],
        help='Solvent state for all runs (default: auto-detect from path)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing files'
    )
    
    args = parser.parse_args()
    
    base_path = args.base_path.resolve()
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return 1
    
    # Find run directories
    if args.runs:
        run_dirs = [base_path / run_name for run_name in args.runs]
        run_dirs = [d for d in run_dirs if d.exists() and d.is_dir()]
    else:
        run_dirs = find_run_directories(base_path)
    
    if not run_dirs:
        print(f"No run directories found in {base_path}")
        return 1
    
    print(f"Found {len(run_dirs)} run directories")
    
    # Process each run directory
    success_count = 0
    error_count = 0
    
    for i, run_dir in enumerate(run_dirs, start=1):
        # Determine combo name
        if args.name_pattern:
            # Extract run number
            match = re.search(r'run(\d+)', run_dir.name)
            run_num = int(match.group(1)) if match else i
            combo_name = args.name_pattern.format(run_num)
        else:
            # For generated combo structure, include parent combo name
            if run_dir.parent != base_path and run_dir.parent.name.startswith('comb_'):
                combo_name = f"{run_dir.parent.name}_{run_dir.name}"
            else:
                combo_name = run_dir.name
        
        print(f"\n[{i}/{len(run_dirs)}] Processing {run_dir.name} -> {combo_name}")
        
        if args.dry_run:
            print(f"  Would create: {args.output_dir / combo_name}")
            try:
                bias_data, output_results, graph_info = collect_run_data(run_dir, args.solvent_state)
                print(f"  Bias keys: {list(bias_data.keys())}")
                print(f"  Populations: {len(output_results['populations'])} sites")
                if output_results['transitions']:
                    print(f"  Transition pairs: {len(output_results['transitions'])}")
                print(f"  Terminated normally: {output_results['terminated_normally']}")
                print(f"  Solvent state: {graph_info['solvent_state']}")
                print(f"  Sites found: {len(graph_info['sites'])}")
                success_count += 1
            except Exception as e:
                print(f"  Error: {e}")
                error_count += 1
        else:
            try:
                entry_dir = create_pretraining_entry(run_dir, args.output_dir, combo_name, args.solvent_state)
                print(f"  ✓ Created: {entry_dir}")
                success_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count} successful, {error_count} errors")
    
    if not args.dry_run and success_count > 0:
        print(f"Pretraining data written to: {args.output_dir.resolve()}")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
