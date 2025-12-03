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
    
    if combo_dirs:
        # Generated combo structure - look for run_### inside each comb_###
        for combo_dir in combo_dirs:
            run_dirs.extend([d for d in combo_dir.glob("run_*") if d.is_dir()])
    else:
        # MSLD run structure - look for run# at base level
        run_dirs = list(base_path.glob("run*"))
        run_dirs = [d for d in run_dirs if d.is_dir()]
    
    # Sort numerically by extracting run number
    def get_run_num(p: Path) -> int:
        match = re.search(r'run[_]?(\d+)', p.name)
        return int(match.group(1)) if match else 0
    
    return sorted(run_dirs, key=get_run_num)


def parse_bias_from_py(variables_file: Path) -> Dict[str, Any]:
    """Parse bias coefficients from variables*.py file.
    
    Args:
        variables_file: Path to variables*.py file
        
    Returns:
        Dict with bias coefficients (b, c, x, s keys and potentially others)
    """
    content = variables_file.read_text()
    
    # Extract the bias_string YAML content - handle both """ and '''
    match = re.search(r'bias_string\s*=\s*["\']' + r'{3}' + r'(.*?)["\']' + r'{3}', content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find bias_string in {variables_file}")
    
    bias_yaml = match.group(1)
    bias_data = yaml.safe_load(bias_yaml)
    
    if bias_data is None:
        raise ValueError(f"Failed to parse bias_string YAML in {variables_file}")
    
    if not isinstance(bias_data, dict):
        raise ValueError(f"Expected dict from YAML, got {type(bias_data)} in {variables_file}")
    
    return bias_data


def parse_bias_from_inp(variables_file: Path) -> Dict[str, Any]:
    """Parse bias coefficients from variables*.inp file (old format).
    
    Args:
        variables_file: Path to variables*.inp file
        
    Returns:
        Dict with bias coefficients (lams, cs, xs, ss keys)
    """
    return read_bias_coeff(str(variables_file))


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


def parse_output_file(output_file: Path) -> Dict[str, Any]:
    """Parse population and transition data from output file.
    
    Uses existing file_handling utilities to parse CHARMM output.
    
    Args:
        output_file: Path to output file
        
    Returns:
        Dict with 'populations', 'transitions', 'rates', and 'terminated_normally'
    """
    content = output_file.read_text()
    
    # Check termination status
    terminated = terminated_normally(content)
    
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


def detect_solvent_state(run_dir: Path) -> str:
    """Detect solvent state from directory path.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Solvent state: "solvent", "vacuum", "protein", or "unknown"
    """
    path_str = str(run_dir).lower()
    
    if "solv" in path_str or "water" in path_str:
        return "solvent"
    elif "vac" in path_str:
        return "vacuum"
    elif "protein" in path_str or "prot" in path_str:
        return "protein"
    else:
        return "unknown"


def collect_run_data(run_dir: Path, solvent_state: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Collect all data from a single run directory.
    
    Handles two directory structures:
    1. MSLD runs: run_dir contains prep/, output, variables*.py
    2. Generated combos: run_dir/output.out, run_dir/variables.py, prep/ at parent level
    
    Args:
        run_dir: Path to run directory
        solvent_state: Optional solvent state override (if None, auto-detect)
        
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
    
    # Parse bias coefficients
    if variables_file.suffix == '.py':
        bias_data = parse_bias_from_py(variables_file)
    else:
        bias_data = parse_bias_from_inp(variables_file)
    
    # Find output file - check both 'output' and 'output.out'
    output_file = run_dir / 'output'
    if not output_file.exists():
        output_file = run_dir / 'output.out'
    if not output_file.exists():
        raise ValueError(f"output file not found in {run_dir}")
    
    output_results = parse_output_file(output_file)
    
    # Find prep directory - check local first, then parent (for generated combos)
    prep_path = run_dir / 'prep'
    if not prep_path.exists():
        # Generated combo structure - prep is at parent level
        prep_path = run_dir.parent / 'prep'
    if not prep_path.exists():
        raise ValueError(f"prep directory not found in {run_dir} or {run_dir.parent}")
    
    # Detect or use provided solvent state
    if solvent_state is None:
        solvent_state = detect_solvent_state(run_dir)
    
    graph_info = extract_graph_info(prep_path, solvent_state)
    
    return bias_data, output_results, graph_info


def create_pretraining_entry(
    run_dir: Path,
    output_dir: Path,
    combo_name: Optional[str] = None,
    solvent_state: Optional[str] = None
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
    bias_data, output_results, graph_info = collect_run_data(run_dir, solvent_state)
    
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
    # Calculate total transitions (sum of all transition counts)
    total_trans = 0
    for trans_dict in output_results['transitions'].values():
        if isinstance(trans_dict, dict):
            total_trans += sum(trans_dict.values())
        else:
            total_trans += trans_dict
    
    # Count populated blocks (blocks with non-zero counts)
    num_populated = 0
    for block_data in output_results['populations'].values():
        if isinstance(block_data, dict) and 'counts' in block_data:
            if sum(block_data['counts'].values()) > 0:
                num_populated += 1
    
    metadata = {
        'source_run_dir': str(run_dir),
        'combo_name': combo_name,
        'terminated_normally': output_results['terminated_normally'],
        'total_transitions': total_trans,
        'num_populated_blocks': num_populated,
        'num_sites': len(graph_info['sites']),
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
