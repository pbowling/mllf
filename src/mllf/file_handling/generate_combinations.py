"""Generate all combinations of site/sub files into separate directories.

This utility scans an input directory for files matching the pattern
`site{site}_sub{sub}_{label}.{ext}` (e.g., `site1_sub2_pres.rtf`, `site1_sub2_frag.pdb`)
and creates output subdirectories, one per combination. For each combination, it copies
the relevant files, renaming them so sub-indices start at 1 within the new directory.

Each generated combination directory contains:
- `prep/`: Copy of input prep directory with renamed RTF/PDB files
- `msld_flat.py`: Simulation script (if included via --include pattern)
- `mapping.json`: Records original file paths and new names
- `info.py`: Configuration dict with nsubs, nblocks, temp, etc.
- `run.sh`: Executable SLURM submission script for running simulations

Example:
  input_dir/
    site1_sub1_pres.rtf
    site1_sub1_frag.pdb
    site1_sub2_pres.rtf
    site1_sub2_frag.pdb
    site1_sub3_pres.rtf
    site1_sub3_frag.pdb

Running:
  python -m mllf.file_handling.generate_combinations input_dir --out combos_out

Will produce directories like:
  combos_out/comb_0001_site1_1__site1_2/
    ├── prep/
    │   ├── site1_sub1_pres.rtf    (renamed if necessary, see mapping.json)
    │   ├── site1_sub1_frag.pdb
    │   ├── site1_sub2_pres.rtf    (renamed if necessary, see mapping.json)
    │   ├── site1_sub2_frag.pdb
    │   ├── top_all36_msld.rtf     (unchanged from input prep/)
    │   ├── par_all36_msld.prm     (unchanged from input prep/)
    │   └── ... (other prep files)
    ├── msld_flat.py               (if included via --include)
    ├── mapping.json
    ├── info.py
    └── run.sh

Combination Generation Logic:

- Generates both within-site and cross-site combinations
- Within-site: Each substituent can be the "anchor" with others as tail

  - Anchor is always first, tail is sorted
  - Example: anchor=1 generates [1,2], [1,3], [1,2,3], etc.
  - Example: anchor=2 generates [2,1], [2,3], [2,1,3], etc.
  - Minimum 2 substituents per combination

- Cross-site: Cartesian product of within-site selections across sites

  - Each site contributes >= 2 substituents
  - Example: site1 has 75 selections, site2 has 186 selections
  - Generates 75 × 186 = 13,950 cross-site combinations

- Total combinations grow significantly with multiple sites

Additional Features:
- RTF PRES tokens are automatically renumbered to match new indices
- Include patterns allow copying extra files (e.g., prep/, msld_flat.py)
- Archive mode creates .tar.gz files for storage
"""
from __future__ import annotations

import argparse
import json
import re
import itertools
import warnings
from pathlib import Path
from shutil import copy2
from shutil import make_archive
from typing import Dict, List, Tuple


SITE_SUB_RE = re.compile(r"site(\d+)_sub(\d+)_([A-Za-z0-9_-]+)\.([A-Za-z0-9]+)$", re.IGNORECASE)


def find_site_sub_files(input_dir: Path) -> Dict[int, Dict[int, Dict[Tuple[str, str], Path]]]:
    """Scan input_dir and prep subdirectory for site/sub files.

    Only files matching the pattern `site{site}_sub{sub}_{label}.{ext}` are considered.
    For example: site1_sub2_pres.rtf, site1_sub2_frag.pdb
    Searches both input_dir and input_dir/prep if it exists.

    Args:
        input_dir: Directory containing site/sub files or prep subdirectory.

    Returns:
        Nested dict mapping site ID -> sub ID -> (label, ext) -> file path.
    """
    found: Dict[int, Dict[int, Dict[str, Path]]] = {}
    
    # Search in both the input_dir and input_dir/prep
    search_dirs = [input_dir]
    prep_dir = input_dir / 'prep'
    if prep_dir.exists() and prep_dir.is_dir():
        search_dirs.append(prep_dir)
    
    for search_path in search_dirs:
        for p in search_path.iterdir():
            if not p.is_file():
                continue
            m = SITE_SUB_RE.match(p.name)
            if not m:
                continue
            site = int(m.group(1))
            sub = int(m.group(2))
            label = m.group(3)
            ext = m.group(4)
            # store keyed by (label, ext) so we preserve per-file suffixes like
            # `_pres.rtf` and `_frag.pdb` and allow arbitrary additional files.
            # Don't overwrite if already found (input_dir takes precedence over prep/)
            key = (label, ext)
            if key not in found.setdefault(site, {}).setdefault(sub, {}):
                found[site][sub][key] = p.resolve()
    return found


def all_site_sub_combinations(found: Dict[int, Dict[int, Dict[str, Path]]], max_subs_per_site: int = 10) -> List[Tuple[List[int], List[int]]]:
    """Generate all within-site and cross-site ordered combinations.

    For each site independently, enumerate all subsets of substituents of size >= 2
    using rotating anchor strategy. Then, generate cross-site combinations by
    selecting at least 2 substituents from each involved site.

    Within-site strategy:
    - For each sub as "anchor", generate combinations from remaining subs
    - Example with subs [1,2,3,4,5]:
      - Anchor 1: [1,2], [1,3], [1,4], [1,5], [1,2,3], [1,2,4], ...
      - Anchor 2: [2,1], [2,3], [2,4], [2,5], [2,1,3], [2,1,4], ...
    - Combination size is limited to max_subs_per_site (no combinations larger than this)

    Cross-site strategy:
    - For each combination of sites (pairs, triplets, etc.), generate all valid
      combinations where each site contributes >= 2 subs
    - Uses rotating anchor within each site component
    - Example: site1[1,2] × site2[3,4] generates multiple ordered combinations
    - Each site's contribution is limited to max_subs_per_site

    Args:
        found: Nested dict mapping site -> sub -> {(label, ext): Path}.
        max_subs_per_site: Maximum number of substituents per site in any combination (default: 10).
                          Combinations will include at most this many subs per site, but all
                          available subs can participate in different combinations.

    Returns:
        List of (sites_list, subs_list, subs_per_site_counts) tuples where:
        - sites_list: List of site IDs (e.g., [1] for within-site or [1, 2] for cross-site)
        - subs_list: List of selected sub indices (e.g., [1, 2, 3, 4])
          For cross-site, subs are concatenated from each site in order
        - subs_per_site_counts: Tuple of counts for cross-site (e.g., (2, 2) means
          first 2 subs from site1, next 2 from site2), or None for within-site
    """
    import warnings
    
    combos = []
    sites_with_enough_subs = {}
    
    for site, subs in found.items():
        if len(subs) < 2:
            continue
        
        sorted_subs = sorted(subs.keys())
        sites_with_enough_subs[site] = sorted_subs
        
        # Warn if site has more subs than the limit
        if len(sorted_subs) > max_subs_per_site:
            warnings.warn(
                f"Site {site} has {len(sorted_subs)} substituents. Individual combinations will be "
                f"limited to at most {max_subs_per_site} substituents per site. "
                f"To include larger combinations, increase max_subs_per_site parameter.",
                UserWarning
            )
    
    if not sites_with_enough_subs:
        return combos
    
    # Part 1: Within-site combinations
    for site, subs in sites_with_enough_subs.items():
        # Try each sub as anchor
        for anchor in subs:
            remaining = [s for s in subs if s != anchor]
            
            # Generate all combinations of size >= 1 from remaining subs
            # Combined with anchor, this gives combinations of size >= 2
            # Limit to max_subs_per_site - 1 (since anchor takes 1 slot)
            max_tail_size = min(len(remaining), max_subs_per_site - 1)
            for r in range(1, max_tail_size + 1):
                for tail_combo in itertools.combinations(remaining, r):
                    # Anchor always first, followed by sorted tail
                    combo_list = [anchor] + list(tail_combo)
                    # For within-site, return 3-tuple with None as third element
                    combos.append(([site], combo_list, None))
    
    # Part 2: Cross-site combinations
    # Generate combinations of sites (pairs, triplets, etc.)
    all_sites = sorted(sites_with_enough_subs.keys())
    if len(all_sites) >= 2:
        # For each combination of 2 or more sites
        for num_sites in range(2, len(all_sites) + 1):
            for site_combo in itertools.combinations(all_sites, num_sites):
                # For each site in this combination, generate within-site sub selections
                # Each site must contribute at least 2 subs, but no more than max_subs_per_site
                site_sub_options = []
                for site in site_combo:
                    subs = sites_with_enough_subs[site]
                    # Generate all possible within-site selections (size >= 2, <= max_subs_per_site) with rotating anchor
                    site_selections = []
                    for anchor in subs:
                        remaining = [s for s in subs if s != anchor]
                        # Limit tail size to ensure total doesn't exceed max_subs_per_site
                        max_tail_size = min(len(remaining), max_subs_per_site - 1)
                        for r in range(1, max_tail_size + 1):
                            for tail_combo in itertools.combinations(remaining, r):
                                combo_list = [anchor] + list(tail_combo)
                                site_selections.append(combo_list)
                    site_sub_options.append(site_selections)
                
                # Generate cartesian product of all site selections
                for cross_site_selection in itertools.product(*site_sub_options):
                    # Build sites list and concatenated subs list
                    # Also track the distribution of subs per site for later processing
                    sites_list = list(site_combo)
                    subs_list = []
                    subs_per_site_counts = []
                    
                    for sub_selection in cross_site_selection:
                        subs_list.extend(sub_selection)
                        subs_per_site_counts.append(len(sub_selection))
                    
                    # Store as tuple: (sites, subs, subs_per_site_counts)
                    # The counts tell us how to split the subs list back into per-site selections
                    combos.append((sites_list, subs_list, tuple(subs_per_site_counts)))
    
    return combos


def make_combo_dir_name(counter: int, sites: List[int], subs: List[int]) -> str:
    """Generate a directory name for a combination.

    Args:
        counter: Sequential combination number (for comb_NNNN prefix).
        sites: List of site IDs in this combination.
        subs: List of substituent IDs in this combination.
          For cross-site combos, subs are ordered by site.

    Returns:
        Directory name like:
        - Within-site: 'comb_0001_site1_2__site1_3__site1_4'
        - Cross-site: 'comb_0001_site1_1__site1_2__site2_3__site2_4'
    """
    parts = []
    
    if len(sites) == 1:
        # Within-site combination: all subs belong to same site
        s = sites[0]
        for sub in subs:
            parts.append(f"site{s}_{sub}")
    else:
        # Cross-site combination: distribute subs across sites
        # Need to figure out which subs belong to which site
        # For now, assume subs are evenly distributed or use simple split
        # This is a heuristic - the actual mapping is stored in mapping.json
        
        # Simple strategy: divide subs among sites as evenly as possible
        subs_per_site = len(subs) // len(sites)
        remainder = len(subs) % len(sites)
        
        idx = 0
        for i, site in enumerate(sites):
            # Give extra sub to first 'remainder' sites
            count = subs_per_site + (1 if i < remainder else 0)
            for _ in range(count):
                if idx < len(subs):
                    parts.append(f"site{site}_{subs[idx]}")
                    idx += 1
    
    joined = "__".join(parts)
    return f"comb_{counter:04d}_{joined}"


def renumber_pres_tokens(content: str, old_site: int, old_sub: int, new_site: int, new_sub: int) -> str:
    """Renumber PRES tokens in RTF file content.
    
    Args:
        content: RTF file content as string.
        old_site: Original site number.
        old_sub: Original substituent number.
        new_site: New site number.
        new_sub: New substituent number.
    
    Returns:
        Updated content with renumbered PRES tokens.
    """
    old_token = f"p{old_site}_{old_sub}"
    new_token = f"p{new_site}_{new_sub}"
    
    def _replace_pres_line(m: re.Match) -> str:
        line = m.group(0)
        if old_token in line:
            return line.replace(old_token, new_token)
        return line
    
    new_content, nsub = re.subn(r"(?m)^PRES.*$", _replace_pres_line, content, count=1)
    return new_content if nsub else content


def list_possible_combinations(input_dir: Path, out_dir: Path, max_subs_per_site: int = 10) -> List[Dict]:
    """List all possible combinations without creating directories.
    
    Args:
        input_dir: Directory containing site{n}_sub{m}_{label}.{ext} files.
        out_dir: Output directory where combination subdirs would be created.
        max_subs_per_site: Maximum number of substituents to consider per site (default: 10).
    
    Returns:
        List of dicts with keys: 'name', 'path', 'sites', 'subs', 'subs_per_site_counts'
    """
    found = find_site_sub_files(input_dir)
    if not found:
        raise RuntimeError(f"No site_sub files found in {input_dir}")
    
    # Check for sites with only 1 substituent and raise error
    single_sub_sites = [site for site, subs in found.items() if len(subs) == 1]
    if single_sub_sites:
        site_list = ", ".join(f"site{s}" for s in sorted(single_sub_sites))
        raise RuntimeError(
            f"Sites with only 1 substituent detected: {site_list}. "
            f"MSLD simulations require at least 2 substituents per site. "
            f"Please add more substituents or add these sites from the core structure files "
            f"(e.g., core.pdb and core.rtf if using msld-py-prep)."
        )
    
    eligible = {s: subs for s, subs in found.items() if len(subs) >= 2}
    if not eligible:
        raise RuntimeError(f"No eligible sites with >=2 substituents found in {input_dir}")
    
    combos = all_site_sub_combinations(eligible, max_subs_per_site=max_subs_per_site)
    combo_list = []
    
    for cnt, combo_data in enumerate(combos, start=1):
        if len(combo_data) == 3 and combo_data[2] is not None:
            sites, subs, subs_per_site_counts = combo_data
        else:
            sites, subs = combo_data[0], combo_data[1]
            subs_per_site_counts = None
        
        name = make_combo_dir_name(cnt, sites, subs)
        combo_path = out_dir / name
        
        combo_list.append({
            'name': name,
            'path': str(combo_path),
            'sites': sites,
            'subs': subs,
            'subs_per_site_counts': subs_per_site_counts,
            'counter': cnt
        })
    
    return combo_list


def create_single_combination_dir(input_dir: Path, out_dir: Path, combo_info: Dict, include_patterns: List[str] | None = None) -> Path:
    """Create a single combination directory with renamed files and support files.
    
    Args:
        input_dir: Directory containing site{n}_sub{m}_{label}.{ext} files.
        out_dir: Output directory where combination subdir will be created.
        combo_info: Dict with 'name', 'sites', 'subs', 'subs_per_site_counts', 'counter'.
        include_patterns: Glob patterns for extra files to copy (e.g., ['prep/*', '*.py']).
    
    Returns:
        Path to created directory.
    """
    found = find_site_sub_files(input_dir)
    
    sites = combo_info['sites']
    subs = combo_info['subs']
    subs_per_site_counts = combo_info.get('subs_per_site_counts')
    name = combo_info['name']
    
    combo_path = out_dir / name
    combo_path.mkdir(parents=True, exist_ok=True)
    
    mapping = []
    
    # Determine which subs belong to which site
    per_site_selected = {}
    if subs_per_site_counts is not None:
        idx = 0
        for i, site in enumerate(sites):
            count = subs_per_site_counts[i]
            per_site_selected[site] = subs[idx:idx+count]
            idx += count
    elif len(sites) == 1:
        per_site_selected[sites[0]] = list(subs)
    else:
        raise ValueError(f"Cross-site combo without counts: {sites}, {subs}")
    
    # Create prep directory and process files
    prep_in = input_dir / 'prep'
    prep_out = combo_path / 'prep'
    prep_out.mkdir(exist_ok=True)
    
    # Build old->new index mapping for renumbering
    # Map (original_site, original_sub) -> (new_site, new_sub_idx)
    old_to_new = {}
    
    # Create site renumber map to handle site renumbering
    site_renumber_map = {site: idx + 1 for idx, site in enumerate(sorted(sites))}
    
    # For each site, renumber substituents sequentially within that site
    for site in sites:
        selected_subs = per_site_selected[site]
        new_site = site_renumber_map[site]
        for new_sub_idx, sub in enumerate(selected_subs, start=1):
            old_to_new[(site, sub)] = (new_site, new_sub_idx)
    
    # Copy and rename site/sub files
    for site, selected_subs in per_site_selected.items():
        for sub in selected_subs:
            if site not in found or sub not in found[site]:
                continue
            
            new_site, new_sub_idx = old_to_new[(site, sub)]
            
            for (label, ext), src_path in found[site][sub].items():
                new_name = f"site{new_site}_sub{new_sub_idx}_{label}.{ext}"
                dst_path = prep_out / new_name
                
                if ext.lower() == 'rtf':
                    content = src_path.read_text()
                    content = renumber_pres_tokens(content, site, sub, new_site, new_sub_idx)
                    dst_path.write_text(content)
                else:
                    copy2(src_path, dst_path)
                
                mapping.append({
                    'original': str(src_path),
                    'new_name': new_name,
                    'original_site': site,
                    'original_sub': sub,
                    'new_site': new_site,
                    'new_sub': new_sub_idx,
                })
    
    # Copy support files from prep directory
    if prep_in.exists():
        for item in prep_in.iterdir():
            if item.is_file() and not SITE_SUB_RE.match(item.name):
                dst = prep_out / item.name
                if not dst.exists():
                    copy2(item, dst)
    
    # Write mapping.json
    (combo_path / 'mapping.json').write_text(json.dumps(mapping, indent=2))
    
    # Write info.py
    # Calculate nsubs as list of counts per site
    if subs_per_site_counts is not None:
        nsubs_list = subs_per_site_counts
    else:
        # Single site case
        nsubs_list = [len(subs)]
    
    info_content = f"""import numpy as np
import os

info = {{}}
info['name'] = '{name}'
info['nsubs'] = {nsubs_list}
info['nblocks'] = np.sum(info['nsubs'])
info['ncentral'] = 0
info['nreps'] = 1
info['nnodes'] = 1
info['enginepath'] = os.environ.get('CHARMMEXEC', '')
info['temp'] = 298.15
"""
    (combo_path / 'info.py').write_text(info_content)
    
    # Generate run.sh
    run_sh = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={name}.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu2080 --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --time=01:00:00

module load charmm/charmm/c51a1

python3 msld_flat.py > output.out 2>&1
"""
    run_script = combo_path / 'run.sh'
    run_script.write_text(run_sh)
    run_script.chmod(0o755)
    
    # Copy additional files matching include patterns
    if include_patterns:
        for pattern in include_patterns:
            if '/' in pattern:
                continue
            for src in input_dir.glob(pattern):
                if src.is_file():
                    copy2(src, combo_path / src.name)
    
    return combo_path


def create_combination_dirs(input_dir: Path, out_dir: Path, dry_run: bool = False, include_patterns: List[str] | None = None, max_subs_per_site: int = 10) -> List[Path]:
    """Create combination directories with renamed files and support files.

    For each valid combination:
    1. Create directory with name like 'comb_0001_site1_2__site1_3'
    2. Copy and rename RTF/PDB files (sub indices start at 1)
    3. Update PRES tokens in RTF files to match new indices
    4. Copy prep/ directory with renamed files and unchanged support files
    5. Generate mapping.json with file tracking info
    6. Generate info.py with configuration dictionary
    7. Generate run.sh executable script for job submission
    8. Copy any additional files matching include_patterns (e.g., msld_flat.py)

    Args:
        input_dir: Directory containing site{n}_sub{m}_{label}.{ext} files.
        out_dir: Output directory where combination subdirs will be created.
        dry_run: If True, print actions without creating files.
        include_patterns: Glob patterns for extra files to copy (e.g., ['prep/*', '*.py']).
        max_subs_per_site: Maximum number of substituents per site in any single combination (default: 10).
                          All substituents can still participate, but each combination is limited to this size per site.

    Returns:
        List of created directory paths.
    """
    found = find_site_sub_files(input_dir)
    if not found:
        raise RuntimeError(f"No site_sub files found in {input_dir}")

    # Check for sites with only 1 substituent and raise error
    single_sub_sites = [site for site, subs in found.items() if len(subs) == 1]
    if single_sub_sites:
        site_list = ", ".join(f"site{s}" for s in sorted(single_sub_sites))
        raise RuntimeError(
            f"Sites with only 1 substituent detected: {site_list}. "
            f"MSLD simulations require at least 2 substituents per site. "
            f"Please add more substituents or remove these sites from the core structure files "
            f"(e.g., core.pdb and core.rtf if using msld-py-prep)."
        )

    # Only consider sites that have at least two substituents. Sites with a
    # single available substituent are not informative for generating
    # combinations and are therefore excluded.
    eligible = {s: subs for s, subs in found.items() if len(subs) >= 2}
    if not eligible:
        raise RuntimeError(f"No eligible sites with >=2 substituents found in {input_dir}")
    found = eligible

    combos = all_site_sub_combinations(found, max_subs_per_site=max_subs_per_site)
    created_dirs: List[Path] = []
    cnt = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for combo_data in combos:
        # Handle both 2-tuple (backward compat) and 3-tuple (with counts)
        if len(combo_data) == 3 and combo_data[2] is not None:
            sites, subs, subs_per_site_counts = combo_data
        else:
            sites, subs = combo_data[0], combo_data[1]
            subs_per_site_counts = None
            
        name = make_combo_dir_name(cnt, sites, subs)
        combo_path = out_dir / name
        cnt += 1
        mapping = []
        if dry_run:
            print(f"DRY: would create {combo_path}")
        else:
            combo_path.mkdir(exist_ok=True)

        # For each site in this combo, determine which subs belong to it.
        per_site_selected = {}
        
        if subs_per_site_counts is not None:
            # Cross-site combination: use counts to split subs
            idx = 0
            for i, site in enumerate(sites):
                count = subs_per_site_counts[i]
                per_site_selected[site] = subs[idx:idx+count]
                idx += count
        elif len(sites) == 1:
            # Within-site combination: all subs belong to the single site
            per_site_selected[sites[0]] = list(subs)
        else:
            # Shouldn't reach here with new logic, but fallback
            raise ValueError(f"Cross-site combo without counts: {sites}, {subs}")

        # Store the file info for later copying to prep directory
        # Don't copy to combo root - only to prep subdirectory
        
        # Renumber sites to start from 1 if needed
        # If we only have site2, rename it to site1 in the output
        all_sites = sorted(per_site_selected.keys())
        site_renumber_map = {old_site: new_site for new_site, old_site in enumerate(all_sites, start=1)}

        # Copy prep directory if it exists in input_dir
        prep_src = input_dir / 'prep'
        if prep_src.exists() and prep_src.is_dir():
            prep_dest = combo_path / 'prep'
            if dry_run:
                print(f"DRY: would create prep directory {prep_dest}")
            else:
                prep_dest.mkdir(exist_ok=True)
            
            # Build mapping from (site, original_sub) -> new_sub_index
            sub_renaming = {}
            for site, selected_subs in per_site_selected.items():
                for new_index, chosen_sub in enumerate(selected_subs, start=1):
                    sub_renaming[(site, chosen_sub)] = new_index
            
            # Copy all non-site-specific files from prep directory
            # Site-specific files will be handled separately with renaming
            for prep_file in prep_src.iterdir():
                if not prep_file.is_file():
                    continue
                
                # Check if this is a site/sub specific file (RTF or PDB)
                # Pattern: site{N}_sub{M}_{label}.{ext}
                match = re.match(r'site(\d+)_sub(\d+)_(.+)', prep_file.name)
                if match:
                    # Skip all site-specific files - they'll be handled with renaming below
                    continue
                
                # Copy non-site-specific files (full_ligand.*, par_all36_msld.prm, etc.)
                dest_file = prep_dest / prep_file.name
                if dry_run:
                    print(f"DRY: would copy prep file {prep_file} -> {dest_file}")
                else:
                    copy2(prep_file, dest_file)
                    mapping.append({
                        'site': None,
                        'original_sub': None,
                        'original_path': str(prep_file.resolve()),
                        'new_name': f"prep/{prep_file.name}",
                        'dest_path': str(dest_file),
                        'note': 'prep_directory',
                    })
            
            # Now rename and copy the selected RTF/PDB files into prep directory
            for site, selected_subs in per_site_selected.items():
                # Use renumbered site ID
                renumbered_site = site_renumber_map[site]
                for new_index, chosen_sub in enumerate(selected_subs, start=1):
                    files_for_sub = found[site].get(chosen_sub, {})
                    for (label, ext), src_path in files_for_sub.items():
                        new_name = f"site{renumbered_site}_sub{new_index}_{label}.{ext}"
                        dest_in_prep = prep_dest / new_name
                        
                        if dry_run:
                            print(f"DRY: would copy and rename {src_path} -> {dest_in_prep}")
                        else:
                            # Copy directly from source to prep with new name
                            copy2(src_path, dest_in_prep)
                            
                            # If this is an RTF file, update PRES token inside
                            if ext.lower() == "rtf":
                                try:
                                    txt = dest_in_prep.read_text()
                                    old_token = f"p{site}_{chosen_sub}"
                                    new_token = f"p{renumbered_site}_{new_index}"
                                    
                                    def _replace_pres_line(m: re.Match) -> str:
                                        line = m.group(0)
                                        if old_token in line:
                                            return line.replace(old_token, new_token)
                                        return line
                                    
                                    new_txt, nsub = re.subn(r"(?m)^PRES.*$", _replace_pres_line, txt, count=1)
                                    if nsub:
                                        dest_in_prep.write_text(new_txt)
                                except Exception:
                                    pass
                            
                            mapping.append({
                                'site': site,
                                'original_sub': chosen_sub,
                                'original_path': str(src_path),
                                'new_name': f"prep/{new_name}",
                                'dest_path': str(dest_in_prep),
                                'note': 'prep_renamed',
                            })

        # optionally copy additional files matching include_patterns into each combo dir
        if include_patterns:
            for pat in include_patterns:
                for extra in input_dir.glob(pat):
                    if not extra.is_file():
                        continue
                    dest = combo_path / extra.name
                    mapping.append({
                        'site': None,
                        'original_sub': None,
                        'original_path': str(extra.resolve()),
                        'new_name': extra.name,
                        'dest_path': str(dest),
                        'note': 'included_extra',
                    })
                    if dry_run:
                        print(f"DRY: would copy extra {extra} -> {dest}")
                    else:
                        copy2(extra, dest)

        # write mapping file
        if dry_run:
            print(f"DRY: would write mapping.json in {combo_path}")
        else:
            mapping_path = combo_path / 'mapping.json'
            with mapping_path.open('w') as fh:
                json.dump({'combo': name, 'entries': mapping}, fh, indent=2)
            
            # write info.py file
            info_path = combo_path / 'info.py'
            # Count number of substituents per site for this combination
            # Use renumbered site order (sorted by renumbered site ID)
            nsubs_per_site = []
            sorted_sites = sorted(per_site_selected.keys(), key=lambda s: site_renumber_map[s])
            for site in sorted_sites:
                nsubs_per_site.append(len(per_site_selected[site]))
            
            # Generate info.py configuration file
            info_content = f"""import numpy as np
import os

info = {{}}
info['name'] = '{name}'
info['nsubs'] = {nsubs_per_site}
info['nblocks'] = np.sum(info['nsubs'])
info['ncentral'] = 0
info['nreps'] = 1
info['nnodes'] = 1
info['enginepath'] = os.environ.get('CHARMMEXEC', '')
info['temp'] = 298.15
"""
            info_path.write_text(info_content)

            # Generate run.sh submission script
            run_sh_path = combo_path / 'run.sh'
            run_sh_content = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={name}.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu2080 --gres=gpu:1 
#SBATCH --export=ALL
#SBATCH --time=01:00:00

module load charmm/charmm/c51a1

# Run msld_flat.py with variables.py from this directory
python3 msld_flat.py --vars-file variables.py --out-dir . > output.out
"""
            run_sh_path.write_text(run_sh_content)
            # Make run.sh executable
            run_sh_path.chmod(0o755)

            created_dirs.append(combo_path)

    return created_dirs


def archive_combo_dirs(out_dir: Path, pattern: str = 'comb_*', remove: bool = False):
    """Archive combination directories as .tar.gz files.

    For each matching directory, create a gzipped tar archive `{dir}.tar.gz`
    in the same parent directory. Optionally removes original directories
    after successful archiving to save disk space.

    Args:
        out_dir: Directory containing combination subdirectories.
        pattern: Glob pattern for matching directories (default: 'comb_*').
        remove: If True, remove original directories after archiving.

    Returns:
        List of created archive file paths.
    """
    from glob import glob
    matches = sorted(glob(str(out_dir / pattern)))
    archived = []
    for path in matches:
        p = Path(path)
        if not p.is_dir():
            continue
        base_name = str(p)
        # make_archive will append .tar.gz when format='gztar'
        archive_name = base_name + '.tar.gz'
        # Use make_archive (creates base_name.tar.gz)
        try:
            # make_archive takes base_name without extension
            make_archive(base_name, 'gztar', root_dir=base_name)
            archived.append(Path(archive_name))
            if remove:
                # remove the directory tree
                import shutil

                shutil.rmtree(base_name)
        except Exception as e:
            print(f"Failed to archive {base_name}: {e}")
    return archived


def main():
    p = argparse.ArgumentParser(description="Generate combination directories from site_sub files")
    p.add_argument('input_dir', type=Path, help='Directory containing site{n}_sub{m}.* files')
    p.add_argument('--out', '-o', dest='out_dir', type=Path, default=Path('combos'), help='Output base directory')
    p.add_argument('--dry-run', action='store_true', help='Show actions without copying files')
    p.add_argument('--include', '-i', dest='include', action='append', default=[], help='Glob pattern(s) of additional files to copy into each combo dir (relative to input_dir). Can be provided multiple times.')
    p.add_argument('--max-subs', type=int, default=10, help='Maximum number of substituents per site in any single combination (default: 10). All substituents can still participate, but each combination is limited to this size per site.')
    p.add_argument('--archive', dest='archive', action='store_true', help='Create .tar.gz archives of generated combo directories matching pattern comb_*')
    p.add_argument('--archive-remove', dest='archive_remove', action='store_true', help='Remove combo directories after successful archiving')
    args = p.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    include_patterns = args.include if args.include else None
    created = create_combination_dirs(input_dir, args.out_dir, dry_run=args.dry_run, include_patterns=include_patterns, max_subs_per_site=args.max_subs)
    print(f"Created {len(created)} combination dirs under {args.out_dir}")

    if args.archive:
        archived = archive_combo_dirs(args.out_dir, pattern='comb_*', remove=args.archive_remove)
        print(f"Archived {len(archived)} combo directories")


if __name__ == '__main__':
    main()
