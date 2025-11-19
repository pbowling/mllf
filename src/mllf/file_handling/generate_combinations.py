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
- Generates within-site combinations of size >= 2 (pairs, triplets, etc.)
- First substituent is fixed as anchor; remaining form unordered sets
- Example: [1,2,3] and [1,3,2] are considered identical (same tail)
- But [2,1,3] is different (different anchor)
- This reduces combinatorial explosion while maintaining diversity

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


def all_site_sub_combinations(found: Dict[int, Dict[int, Dict[str, Path]]]) -> List[Tuple[List[int], List[int]]]:
    """Generate all within-site ordered combinations with rotating anchor.

    For each site independently, enumerate all subsets of substituents of size >= 2.
    For EACH substituent as anchor, generate all unordered combinations with the
    remaining substituents. This creates ordered tuples where position matters.

    Strategy:
    - For each sub as "anchor", generate combinations from remaining subs
    - Example with subs [1,2,3,4,5]:
      - Anchor 1: [1,2], [1,3], [1,4], [1,5], [1,2,3], [1,2,4], ...
      - Anchor 2: [2,1], [2,3], [2,4], [2,5], [2,1,3], [2,1,4], ...
      - etc.

    Args:
        found: Nested dict mapping site -> sub -> {(label, ext): Path}.

    Returns:
        List of (sites_list, subs_list) tuples where:
        - sites_list: List containing single site ID (e.g., [1])
        - subs_list: List of selected sub indices (e.g., [1, 2, 3])
    """
    combos = []
    # For each site, iterate over each substituent as anchor
    for site in sorted(found.keys()):
        subs = sorted(found[site].keys())
        if len(subs) < 2:
            continue  # Need at least 2 subs to form a combination
        
        # Try each sub as anchor
        for anchor in subs:
            remaining = [s for s in subs if s != anchor]
            
            # Generate all combinations of size >= 1 from remaining subs
            # Combined with anchor, this gives combinations of size >= 2
            for r in range(1, len(remaining) + 1):
                for tail_combo in itertools.combinations(remaining, r):
                    # Anchor always first, followed by sorted tail
                    combo_list = [anchor] + list(tail_combo)
                    combos.append(([site], combo_list))
    return combos


def make_combo_dir_name(counter: int, sites: List[int], subs: List[int]) -> str:
    """Generate a directory name for a combination.

    Args:
        counter: Sequential combination number (for comb_NNNN prefix).
        sites: List of site IDs in this combination.
        subs: List of substituent IDs in this combination.

    Returns:
        Directory name like 'comb_0001_site1_2__site1_3__site1_4'.
    """
    parts = []
    # If there is a single site with multiple selected subs, list each sub
    # for that site. If there are multiple sites and equal-length subs list,
    # pair them elementwise. Otherwise fall back to pairing as much as
    # possible.
    if len(sites) == 1 and len(subs) >= 1:
        s = sites[0]
        # use `_to_` to indicate ordering/direction between selected subs
        for sub in subs:
            parts.append(f"site{s}_{sub}")
    elif len(sites) == len(subs):
        for s, sub in zip(sites, subs):
            parts.append(f"site{s}_{sub}")
    else:
        # fallback: pair up to the shorter length, then append remaining
        for i in range(min(len(sites), len(subs))):
            parts.append(f"site{sites[i]}_{subs[i]}")
        # if any subs left (unlikely), append them using last site
        if len(subs) > len(sites):
            last_site = sites[-1]
            for sub in subs[len(sites):]:
                parts.append(f"site{last_site}_{sub}")
    joined = "__".join(parts)
    return f"comb_{counter:04d}_{joined}"


def create_combination_dirs(input_dir: Path, out_dir: Path, dry_run: bool = False, include_patterns: List[str] | None = None) -> List[Path]:
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

    Returns:
        List of created directory paths.
    """
    found = find_site_sub_files(input_dir)
    if not found:
        raise RuntimeError(f"No site_sub files found in {input_dir}")

    # Only consider sites that have at least two substituents. Sites with a
    # single available substituent are not informative for generating
    # combinations and are therefore excluded.
    eligible = {s: subs for s, subs in found.items() if len(subs) >= 2}
    if not eligible:
        raise RuntimeError(f"No eligible sites with >=2 substituents found in {input_dir}")
    found = eligible

    combos = all_site_sub_combinations(found)
    created_dirs: List[Path] = []
    cnt = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for sites, subs in combos:
        name = make_combo_dir_name(cnt, sites, subs)
        combo_path = out_dir / name
        cnt += 1
        mapping = []
        if dry_run:
            print(f"DRY: would create {combo_path}")
        else:
            combo_path.mkdir(exist_ok=True)

        # For each site in this combo, we will rename selected subs starting at 1
        # within the combo dir. Since we pick exactly one sub per site here, the
        # new sub index is 1 for each site. If later we support multiple subs per
        # site in a combo, we'd enumerate and assign 1..k.
        # Handle per-site selection. `sites` is a list of sites; `subs` may
        # represent multiple chosen subs for a single site (common case). We
        # support two patterns:
        #  - len(sites) == len(subs): one chosen sub per site (legacy behavior)
        #  - len(sites) == 1 and len(subs) >= 1: multiple chosen subs within one site
        if len(sites) == len(subs):
            # one-to-one mapping: process each (site, chosen_sub)
            per_site_selected = {site: [sub] for site, sub in zip(sites, subs)}
        else:
            # assume the common case: single site with multiple chosen subs
            per_site_selected = {}
            if len(sites) == 1:
                per_site_selected[sites[0]] = list(subs)
            else:
                # fallback: try to pair as much as possible
                for i, site in enumerate(sites):
                    per_site_selected[site] = [subs[i]] if i < len(subs) else []

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
    p.add_argument('--archive', dest='archive', action='store_true', help='Create .tar.gz archives of generated combo directories matching pattern comb_*')
    p.add_argument('--archive-remove', dest='archive_remove', action='store_true', help='Remove combo directories after successful archiving')
    args = p.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    include_patterns = args.include if args.include else None
    created = create_combination_dirs(input_dir, args.out_dir, dry_run=args.dry_run, include_patterns=include_patterns)
    print(f"Created {len(created)} combination dirs under {args.out_dir}")

    if args.archive:
        archived = archive_combo_dirs(args.out_dir, pattern='comb_*', remove=args.archive_remove)
        print(f"Archived {len(archived)} combo directories")


if __name__ == '__main__':
    main()
