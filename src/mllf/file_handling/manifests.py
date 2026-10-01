"""Manifest and combo-directory bookkeeping for the online-training workflow.

Pure I/O / bookkeeping helpers used to build per-system manifests (lists of
combo directory paths), lazily create combo directories on demand, and
efficiently look up combo metadata for very large combination spaces without
loading a whole system's ``combo_metadata.json`` into memory at once.

Moved here from ``training/workflow.py`` as part of making the online CB
training driver importable as a library (see ``mllf.cli.train_workflow``).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json

import numpy as np

from mllf.cli.workflow import load_manifest
from mllf.file_handling.generate_combinations import (
    create_single_combination_dir,
    find_site_sub_files,
    list_possible_combinations,
)


def count_combinations_fast(sys_entry: Dict, max_subs_per_site: int = 10) -> int:
    """Count combinations for a system using arithmetic only — O(1) memory.

    Calls find_site_sub_files (a directory scan) then counts via combinatorics
    without ever building the combination list.  Used to classify systems as
    'small' (safe to run in parallel) vs 'large' (run alone).
    """
    from math import comb as _comb
    from itertools import combinations as _icombinations

    input_dir = Path(sys_entry['input_dir'])
    # Per-system override takes precedence over the global config value.
    max_subs = sys_entry.get('max_subs_per_site', max_subs_per_site)

    try:
        found = find_site_sub_files(input_dir)
    except Exception:
        return 0
    eligible = {s: subs for s, subs in found.items() if len(subs) >= 2}
    if not eligible:
        return 0

    # Number of within-site selections per site (rotating-anchor strategy).
    per_site = {}
    for site, subs in eligible.items():
        n = len(subs)
        max_tail = min(n - 1, max_subs - 1)
        # n anchors × sum_r C(n-1, r)  for r in 1..max_tail
        per_site[site] = n * sum(_comb(n - 1, r) for r in range(1, max_tail + 1))

    total = sum(per_site.values())  # within-site totals

    # Cross-site: cartesian product of per-site selection counts.
    all_sites = sorted(eligible.keys())
    for k in range(2, len(all_sites) + 1):
        for site_combo in _icombinations(all_sites, k):
            cross = 1
            for site in site_combo:
                cross *= per_site[site]
            total += cross

    return total


def build_system_manifest(sys_entry: Dict, max_combos: Optional[int] = None,
                           seed: int = 42) -> Dict:
    """Create or load the manifest for one system.

    Creating a manifest calls list_possible_combinations which can build a very
    large list of dicts in memory.  To avoid OOM when many systems are processed
    in parallel, callers should only use this function (via a thread pool) for
    systems whose manifest *already exists*.  For systems that need creation,
    call create_and_load_system_manifest directly (sequentially).

    Returns a dict with keys 'sys_entry', 'combos' (list[str]), and 'log' (list[str]).
    """
    sys_name = sys_entry.get('name', sys_entry.get('input_dir', 'unknown'))
    input_dir = Path(sys_entry['input_dir'])
    out_dir = Path(sys_entry['out_dir'])

    log = [
        f"\n=== System: {sys_name} ===",
        f"  input_dir : {input_dir}",
        f"  out_dir   : {out_dir}",
        f"  solvent   : {sys_entry.get('solvent_state', 'solv')}",
    ]

    # Manifest and metadata live in input_dir so they survive simulation cleanup.
    manifest_path = input_dir / 'manifest.txt'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Call create_and_load_system_manifest for systems needing creation."
        )
    log.append(f"  Manifest already exists at {manifest_path}")
    sys_combos = load_manifest(str(manifest_path))
    total = len(sys_combos)
    if max_combos is not None and total > max_combos:
        rng = np.random.RandomState(seed ^ (hash(sys_name) & 0x7fffffff))
        idxs = rng.choice(total, size=max_combos, replace=False)
        sys_combos = [sys_combos[j] for j in sorted(idxs)]
        log.append(f"  Sampled {max_combos:,} / {total:,} combos (max_combos_per_system cap)")
    else:
        log.append(f"  Loaded {total:,} combos")
    return {'sys_entry': sys_entry, 'combos': sys_combos, 'log': log}


def create_and_load_system_manifest(sys_entry: Dict, max_subs_per_site: int = 10,
                                      max_combos: Optional[int] = None,
                                      seed: int = 42) -> Dict:
    """Generate combinations, write manifest + metadata to input_dir, then load.

    This is the memory-intensive path (list_possible_combinations builds all
    combos in memory).  Large systems (estimated combos >= large_combo_threshold)
    should be called one at a time to avoid OOM.
    """
    sys_name = sys_entry.get('name', sys_entry.get('input_dir', 'unknown'))
    input_dir = Path(sys_entry['input_dir'])
    out_dir = Path(sys_entry['out_dir'])

    log = [
        f"\n=== System: {sys_name} ===",
        f"  input_dir : {input_dir}",
        f"  out_dir   : {out_dir}",
        f"  solvent   : {sys_entry.get('solvent_state', 'solv')}",
        f"  Creating combinations in {out_dir} ...",
    ]

    # Per-system override takes precedence over the global config value.
    effective_max_subs = sys_entry.get('max_subs_per_site', max_subs_per_site)
    log.append(f"  max_subs_per_site : {effective_max_subs}")
    all_combo_info = list_possible_combinations(
        input_dir, out_dir,
        max_subs_per_site=effective_max_subs
    )
    log.append(f"  Found {len(all_combo_info)} possible combinations")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (input_dir / 'manifest.txt').open('w') as f:
        for info in all_combo_info:
            f.write(info['path'] + '\n')
    combo_metadata_path = input_dir / 'combo_metadata.json'
    with open(combo_metadata_path, 'w') as f:
        json.dump(all_combo_info, f, indent=2)
    log.append(f"  Saved metadata to {combo_metadata_path}")

    sys_combos = load_manifest(str(input_dir / 'manifest.txt'))
    total = len(sys_combos)
    if max_combos is not None and total > max_combos:
        rng = np.random.RandomState(seed ^ (hash(sys_name) & 0x7fffffff))
        idxs = rng.choice(total, size=max_combos, replace=False)
        sys_combos = [sys_combos[j] for j in sorted(idxs)]
        log.append(f"  Sampled {max_combos:,} / {total:,} combos (max_combos_per_system cap)")
    else:
        log.append(f"  Loaded {total:,} combos")
    return {'sys_entry': sys_entry, 'combos': sys_combos, 'log': log}


def iter_json_array_streaming(path, chunk_size: int = 1 << 20):
    """Yield each top-level element of a JSON array file one at a time,
    without ever holding the whole array (or the whole file) in memory.

    combo_metadata.json is written as ``json.dump(list_of_dicts, f)``. For
    large systems (e.g. 12fuzed at ~1.3 GB / 4M+ entries on disk) a plain
    ``json.load()`` inflates to several GB of Python objects for a single
    call — enough to OOM a constrained SLURM allocation on its own. This
    reads the file in fixed-size chunks and incrementally decodes one
    top-level array element at a time via ``json.JSONDecoder.raw_decode``
    (the real JSON parser, so it's robust to braces/commas inside string
    values) — peak memory is O(chunk_size + one element), not O(file size).
    """
    decoder = json.JSONDecoder()
    buf = ''
    started = False
    with open(path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk and not buf:
                return
            buf += chunk
            while True:
                i, n = 0, len(buf)
                while i < n and buf[i] in ' \t\r\n,':
                    i += 1
                if not started:
                    if i < n and buf[i] == '[':
                        i += 1
                        started = True
                    else:
                        break  # need more data (or malformed input)
                    while i < n and buf[i] in ' \t\r\n,':
                        i += 1
                if i < n and buf[i] == ']':
                    return
                if i >= n:
                    buf = buf[i:]
                    break  # need more data
                try:
                    obj, end = decoder.raw_decode(buf, i)
                except json.JSONDecodeError:
                    buf = buf[i:]
                    break  # incomplete element — read more data
                yield obj
                buf = buf[end:]
            if not chunk:
                return


def build_combo_dir_lookup(
    combos: List[str],
    combo_system_map: Dict[str, Dict],
) -> Dict[str, Any]:
    """Pre-scan combo_metadata.json for combos whose directories don't exist.

    Streams each affected system's metadata file via
    iter_json_array_streaming (never loading the full array into memory),
    extracting only the entries needed.

    Memory profile:
      - Peak during scan of one system: O(chunk_size)  →  effectively constant
      - Retained after all scans: O(n_missing_combos × entry_size)  →  tiny

    This avoids the catastrophic memory pattern of loading all metadata for
    a system at once (12fuzed has 4.18 M combos ≈ 1.3 GB on disk, several GB
    once parsed into Python dicts — enough on its own to OOM constrained
    SLURM nodes).
    """
    # Group missing combos by their system's input_dir.
    missing_by_system: Dict[str, Set[str]] = {}
    for c in combos:
        if not Path(c).exists():
            sys_cfg = combo_system_map.get(c, {})
            idir = str(sys_cfg.get('input_dir', ''))
            if idir:
                missing_by_system.setdefault(idir, set()).add(c)

    if not missing_by_system:
        return {}

    n_systems = len(missing_by_system)
    n_missing = sum(len(v) for v in missing_by_system.values())
    print(f"  [dir lookup] {n_missing} combos need creation across "
          f"{n_systems} system(s) — scanning metadata ...")

    lookup: Dict[str, Any] = {}
    for idir, needed_paths in missing_by_system.items():
        meta_path = Path(idir) / 'combo_metadata.json'
        if not meta_path.exists():
            continue
        remaining = set(needed_paths)
        for info in iter_json_array_streaming(meta_path):
            if info['path'] in remaining:
                lookup[info['path']] = info
                remaining.discard(info['path'])
                if not remaining:
                    break

    print(f"  [dir lookup] found {len(lookup)}/{n_missing} entries")
    return lookup


def ensure_combo_dir_exists(combo_path: Path, system_config: Dict,
                             global_config: Dict, verbose: bool = False,
                             _lookup: Optional[Dict] = None) -> Path:
    """Ensure a combination directory exists, creating it lazily if needed.

    *_lookup* is an optional {combo_path_str: combo_info} dict pre-built by
    build_combo_dir_lookup for the current epoch.  When provided it avoids
    re-reading combo_metadata.json; when absent the function falls back to a
    full JSON scan (correct but slow for large systems).
    """
    if combo_path.exists():
        return combo_path

    combo_info = (_lookup or {}).get(str(combo_path))

    if combo_info is None:
        # Fallback: full JSON scan.  Should only be reached if the pre-scan
        # was skipped or returned an incomplete lookup (e.g. sample_combo
        # initialisation before train_epoch).
        combo_metadata_path = Path(system_config['input_dir']) / 'combo_metadata.json'
        if not combo_metadata_path.exists():
            raise FileNotFoundError(f"Combo metadata not found at {combo_metadata_path}")
        for info in iter_json_array_streaming(combo_metadata_path):
            if info['path'] == str(combo_path):
                combo_info = info
                break

    if combo_info is None:
        raise ValueError(f"Combo info not found for {combo_path}")

    include_patterns = system_config.get('include_patterns', ['msld_flat.py'])
    input_dir = Path(system_config['input_dir'])
    output_dir = Path(system_config['out_dir'])

    if verbose:
        print(f"  Creating combo directory on-demand: {combo_path.name}")
    created_path = create_single_combination_dir(
        input_dir, output_dir, combo_info, include_patterns=include_patterns,
        verbose=verbose,
    )
    return created_path

