"""Generate all combinations of site/sub files into separate directories.

This utility scans an input directory for files named like
`site{site}_sub{sub}.{ext}` (for example `site1_sub2.rtf`, `site1_sub2.pdb`)
and then creates a set of output subdirectories, one per combination of
site selections. For each combination it copies the relevant files into the
directory and renames the files so that sub-indices start at 1 for each site
within the new directory.

Example:
  input_dir/
    site1_sub1.rtf
    site1_sub2.rtf
    site2_sub1.rtf

Running:
  python -m mllf.file_handling.generate_combinations input_dir --out combos_out

Will produce directories like:
  combos_out/comb_0001_site1_1/  (site1_sub1)
  combos_out/comb_0002_site1_2/  (site1_sub2)
  combos_out/comb_0003_site2_1/  (site2_sub1)
  combos_out/comb_0004_site1_1__site2_1/  (site1_sub1 + site2_sub1)

Each combination dir contains the renamed files and a `mapping.json` which
records original file paths and their new names inside the combo dir.

Notes / assumptions:
- This implementation generates all non-empty subsets of sites, and for each
  chosen subset enumerates the Cartesian product of subs at those sites.
  This produces single-site and multi-site combos.
- If you need multi-substituent-per-site combinations (e.g. pairs within the
  same site) we can extend the script; tell me and I'll add that mode.
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
    """Scan input_dir and return mapping: site -> sub -> {ext: Path}.

    Only files matching the pattern `site{site}_sub{sub}.{ext}` are considered.
    """
    found: Dict[int, Dict[int, Dict[str, Path]]] = {}
    for p in input_dir.iterdir():
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
        # `_pres.rft` and `_frag.pdb` and allow arbitrary additional files.
        found.setdefault(site, {}).setdefault(sub, {})[(label, ext)] = p.resolve()
    return found


def all_site_sub_combinations(found: Dict[int, Dict[int, Dict[str, Path]]]) -> List[Tuple[List[int], List[int]]]:
    """Return list of within-site combinations.

    For each site independently, enumerate all non-empty subsets of its
    substituents of size >= 2 (pairs, triplets, ...). Each returned entry is
    a tuple (sites_list, subs_list) where sites_list contains a single site
    id and subs_list contains the selected sub indices for that site.
    """
    combos = []
    # Use ordered permutations so that directional possibilities (A->B vs B->A)
    # are treated as distinct combinations. For each site, enumerate all
    # permutations of length >=2.
    for site in sorted(found.keys()):
        subs = sorted(found[site].keys())
        # only consider ordered tuples of size >= 2
        for r in range(2, len(subs) + 1):
            for combo in itertools.permutations(subs, r):
                combos.append(([site], list(combo)))
    return combos


def make_combo_dir_name(counter: int, sites: List[int], subs: List[int]) -> str:
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

        for site, selected_subs in per_site_selected.items():
            for new_index, chosen_sub in enumerate(selected_subs, start=1):
                files_for_sub = found[site].get(chosen_sub, {})
                for (label, ext), src_path in files_for_sub.items():
                    new_name = f"site{site}_sub{new_index}_{label}.{ext}"
                    dest = combo_path / new_name
                    mapping.append({
                        'site': site,
                        'original_sub': chosen_sub,
                        'original_label': label,
                        'original_ext': ext,
                        'original_path': str(src_path),
                        'new_name': new_name,
                        'dest_path': str(dest),
                    })
                    if dry_run:
                        print(f"DRY: would copy {src_path} -> {dest}")
                    else:
                        copy2(src_path, dest)

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
            created_dirs.append(combo_path)

    return created_dirs


def archive_combo_dirs(out_dir: Path, pattern: str = 'comb_*', remove: bool = False):
    """Archive all combo directories under out_dir matching pattern.

    For each matching directory, create a gzipped tar archive
    `{dir}.tar.gz` in the same parent directory. If `remove` is True,
    the original directory will be removed after successful archiving.
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
