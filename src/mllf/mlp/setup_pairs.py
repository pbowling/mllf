"""Utilities to assemble training pairs for the MLP.

This module discovers training run directories named like
`system_solvent_fnex` (for example `14benz_solv_5.5`), parses RTF fragment
files to extract atom types and charges, finds the matching variables file
(variables*.py or variables*.inp) with bias coefficients, and assembles a
structure mapping each site_sub to a pair dict containing:

- system, solvent, fnex
- site, sub
- atom_types (list)
- total_charge (float)
- bias coefficients (lams, cs, xs, ss when available)

The main entrypoint is `assemble_pairs(root_training_dir)` which returns a
dictionary keyed by run_dir -> mapping of site_sub -> pair dict.
"""

from __future__ import annotations

import os
import re
import glob
from typing import Dict, Any, Optional

from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.file_handling.read_bias_coeff import read_bias_coeff


VARS_RE = re.compile(r'variables(\d+)\.(py|inp)$')


def parse_run_dirname(dirname: str) -> Dict[str, Optional[str]]:
    """Parse a run directory name into metadata: system, solvent, fnex.

    Expected formats:
    - system_solvent_fnex
    - system_solvent
    """
    parts = dirname.split('_')
    system = parts[0] if parts else None
    solvent = parts[1] if len(parts) > 1 else None
    fnex = parts[2] if len(parts) > 2 else None
    return {'system': system, 'solvent': solvent, 'fnex': fnex}


def find_variables_file(directory: str) -> Optional[str]:
    """Find the variables file in a directory. Prefer the highest-numbered file.

    Returns absolute path or None if not found.
    """
    candidates = []
    for ext in ('py', 'inp'):
        for path in glob.glob(os.path.join(directory, f'variables*.{ext}')):
            m = VARS_RE.search(os.path.basename(path))
            if m:
                num = int(m.group(1))
            else:
                num = 0
            candidates.append((num, path))
    if not candidates:
        return None
    # pick highest num
    candidates.sort(key=lambda x: x[0], reverse=True)
    return os.path.abspath(candidates[0][1])


def assemble_pairs_for_run(run_path: str) -> Dict[str, Dict[str, Any]]:
    """Assemble pairs for a single run directory.

    Returns a mapping: site{site}_sub{sub} -> pair dict
    """
    meta = parse_run_dirname(os.path.basename(run_path))

    # parse RTF fragments
    rtf_dir = os.path.join(run_path)
    fragments = parse_rtf_dir(rtf_dir)

    # find variables file and parse biases
    vars_file = find_variables_file(run_path)
    biases = None
    if vars_file:
        biases = read_bias_coeff(vars_file)

    pairs: Dict[str, Dict[str, Any]] = {}

    for key, frag in fragments.items():
        site = frag.get('site')
        sub = frag.get('sub')
        entry: Dict[str, Any] = {
            'system': meta.get('system'),
            'solvent': meta.get('solvent'),
            'fnex': meta.get('fnex'),
            'site': site,
            'sub': sub,
            'atom_types': frag.get('atom_types', []),
            'total_charge': frag.get('total_charge', 0.0),
            'filename': frag.get('filename'),
            'biases': {},
        }

        # attach biases if available
        if biases:
            # Build a lams vector for this site: collect lams{site}s{j} for all j
            lams_vector = []
            if site is not None and 'lams' in biases:
                # find all keys matching lams{site}s{j}
                pattern = re.compile(rf'^lams{site}s(\d+)$')
                matches = []
                for k, v in biases['lams'].items():
                    m = pattern.match(k)
                    if m:
                        j = int(m.group(1))
                        matches.append((j, v))
                if matches:
                    matches.sort(key=lambda x: x[0])
                    lams_vector = [float(v) for _, v in matches]
            if lams_vector:
                entry['biases']['lams_vector'] = lams_vector

            # build ordered pairwise linear biases for this site: map (a,b) -> lams[b]-lams[a]
            # stored under 'pairwise_lams' as keys 'pair_{a}_{b}' and numeric values.
            if lams_vector:
                pairwise = {}
                nsubs = len(lams_vector)
                for a in range(1, nsubs + 1):
                    for b in range(1, nsubs + 1):
                        if a == b:
                            continue
                        # indices in lams_vector are zero-based
                        val = float(lams_vector[b - 1]) - float(lams_vector[a - 1])
                        pair_key = f'pair_{a}_{b}'
                        pairwise[pair_key] = val
                if pairwise:
                    entry['biases']['pairwise_lams'] = pairwise

            # Build pairwise biases across groups (lams, cs, xs, ss) and store under 'pairwise_biases'
            # For lams we can directly compute per-sub differences; for cs/xs/ss we derive a per-sub
            # representative scalar by averaging any entries that reference that sub index (heuristic).
            pairwise_all = {}
            # helper to compute pairwise mapping from a per-sub list
            def compute_pairwise_from_list(vals):
                out = {}
                m = len(vals)
                for a in range(1, m + 1):
                    for b in range(1, m + 1):
                        if a == b:
                            continue
                        out[f'pair_{a}_{b}'] = float(vals[b - 1]) - float(vals[a - 1])
                return out

            if lams_vector:
                pairwise_all['lams'] = compute_pairwise_from_list(lams_vector)

            # Build deterministic pairwise mappings for cs/xs/ss.
            # For each ordered pair (a,b) we first look for an explicit scalar
            # entry in the biases[group] keys that references both subs
            # (e.g., 'cs1s1s1s2'). If found, use that value for pair_{a}_{b} and
            # -value for pair_{b}_{a}. If no explicit keys are present for the
            # group, fall back to deriving a per-sub scalar list and computing
            # differences.
            for group in ('cs', 'xs', 'ss'):
                if group not in biases:
                    continue
                group_map = biases[group]
                # determine number of subs from lams_vector if available, else
                # infer from keys in group_map
                nsubs = None
                if lams_vector:
                    nsubs = len(lams_vector)
                else:
                    # infer max sub index referenced in group_map
                    max_sub = 0
                    for kk in group_map.keys():
                        nums = re.findall(r"(\d+)", kk)
                        if nums:
                            max_sub = max(max_sub, max(int(x) for x in nums))
                    nsubs = max_sub

                if not nsubs or nsubs < 1:
                    continue

                grp_pairs = {}
                # try explicit per-pair keys first
                for a in range(1, nsubs + 1):
                    for b in range(1, nsubs + 1):
                        if a == b:
                            continue
                        found = False
                        for k, v in group_map.items():
                            nums = re.findall(r"(\d+)", k)
                            if len(nums) >= 4:
                                s1 = int(nums[0])
                                aa = int(nums[1])
                                s2 = int(nums[2])
                                bb = int(nums[3])
                                if site is not None and s1 == site and s2 == site and aa == a and bb == b:
                                    grp_pairs[f'pair_{a}_{b}'] = float(v)
                                    # ensure reverse order present with opposite sign
                                    grp_pairs[f'pair_{b}_{a}'] = -float(v)
                                    found = True
                                    break
                        if not found:
                            # leave to fallback derivation
                            continue

                # if any explicit pairs were found, fill missing entries from
                # derived per-sub scalars (so explicit mappings take precedence)
                if grp_pairs:
                    # derive per-sub scalars by grouping on primary sub index
                    nums_map = {}
                    for k, v in group_map.items():
                        nums = re.findall(r"(\d+)", k)
                        if not nums:
                            continue
                        primary = int(nums[0])
                        nums_map.setdefault(primary, []).append(float(v))
                    scalars = []
                    max_sub = nsubs
                    for i in range(1, max_sub + 1):
                        vals = nums_map.get(i, [])
                        if vals:
                            scalars.append(sum(vals) / len(vals))
                        else:
                            scalars.append(0.0)
                    if scalars:
                        computed = compute_pairwise_from_list(scalars)
                        for k2, v2 in computed.items():
                            grp_pairs.setdefault(k2, v2)
                    pairwise_all[group] = grp_pairs
                else:
                    # no explicit pairs, derive entirely from per-sub scalars
                    nums_map = {}
                    for k, v in group_map.items():
                        nums = re.findall(r"(\d+)", k)
                        if not nums:
                            continue
                        primary = int(nums[0])
                        nums_map.setdefault(primary, []).append(float(v))
                    if nums_map:
                        max_sub = nsubs
                        scalars = []
                        for i in range(1, max_sub + 1):
                            vals = nums_map.get(i, [])
                            if vals:
                                scalars.append(sum(vals) / len(vals))
                            else:
                                scalars.append(0.0)
                        pairwise_all[group] = compute_pairwise_from_list(scalars)

            if pairwise_all:
                entry['biases']['pairwise_biases'] = pairwise_all

                # Build a unified per-pair mapping so each ordered pair has
                # exactly one scalar for each group (lams, cs, xs, ss).
                # Precedence: use computed/merged pairwise_all values for
                # cs/xs/ss, and pairwise_lams for lams (which is the
                # difference lams[b]-lams[a]). If a value is missing for a
                # group, the entry is omitted (or left as None).
                pairs_map = {}
                # collect all pair keys across groups
                pair_keys = set()
                # lams pairwise map (from earlier computation) preferred
                lams_map = entry['biases'].get('pairwise_lams', {})
                if isinstance(lams_map, dict):
                    pair_keys.update(lams_map.keys())
                for grp, grp_map in pairwise_all.items():
                    if isinstance(grp_map, dict):
                        pair_keys.update(grp_map.keys())

                for pk in sorted(pair_keys):
                    pd = {}
                    # extract numeric a,b from pair key
                    m = re.match(r'pair_(\d+)_(\d+)', pk)
                    if not m:
                        continue
                    a = int(m.group(1))
                    b = int(m.group(2))

                    # Only include pairs that involve this fragment's sub index.
                    # For a fragment with `sub`, keep only pairs where a==sub or b==sub.
                    if sub is not None and not (a == sub or b == sub):
                        continue

                    # lams: prefer explicit pairwise_lams, else use pairwise_all['lams']
                    if pk in lams_map:
                        pd['lams'] = float(lams_map[pk])
                    else:
                        lm = pairwise_all.get('lams', {})
                        if pk in lm:
                            pd['lams'] = float(lm[pk])

                    # For cs/xs/ss, prefer explicit scalar keys in the original
                    # biases dict (e.g., 'cs1s1s1s2') if present; otherwise fall
                    # back to pairwise_all computed values.
                    for grp in ('cs', 'xs', 'ss'):
                        # construct explicit key name for this run/site/a/b
                        explicit_key = f"{grp}{site}s{a}s{site}s{b}"
                        val = None
                        if grp in biases and explicit_key in biases[grp]:
                            val = biases[grp][explicit_key]
                        else:
                            gm = pairwise_all.get(grp, {})
                            if pk in gm:
                                val = gm[pk]
                        if val is not None:
                            pd[grp] = float(val)

                    pairs_map[pk] = pd

                if pairs_map:
                    entry['biases']['pairs'] = pairs_map

            # For cs/xs/ss, parse integer indices from keys and include entries that reference this site
            def key_numbers(k: str):
                nums = re.findall(r"(\d+)", k)
                return [int(x) for x in nums]

            for group in ('cs', 'xs', 'ss'):
                if group in biases:
                    for k, v in biases[group].items():
                        nums = key_numbers(k)
                        if site is not None and site in nums:
                            entry['biases'].setdefault(group, {})[k] = v

        pairs[key] = entry

    return pairs


def assemble_pairs(root_training_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Walk run subdirectories under root_training_dir and assemble pairs for each run.

    Returns mapping: run_dir_name -> (site_sub -> pair dict)
    """
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not os.path.isdir(root_training_dir):
        raise ValueError(f"Not a directory: {root_training_dir}")

    for entry in sorted(os.listdir(root_training_dir)):
        run_path = os.path.join(root_training_dir, entry)
        if not os.path.isdir(run_path):
            continue
        try:
            pairs = assemble_pairs_for_run(run_path)
        except Exception:
            pairs = {}
        results[entry] = pairs

    return results


if __name__ == '__main__':
    # quick smoke test to run against the example training files directory
    # If a run directory name is passed as the first argument, print only that
    # run's pairs (e.g., `14benz_vac_5.5`). Otherwise default to that name.
    import json
    import sys

    here = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'training_files')
    here = os.path.abspath(here)
    out = assemble_pairs(here)

    run_name = sys.argv[1] if len(sys.argv) > 1 else '14benz_vac_5.5'
    if run_name not in out:
        available = sorted(out.keys())
        print(f"Run '{run_name}' not found. Available runs:\n{available}")
        sys.exit(1)

    run = out[run_name]
    # Print a readable per-fragment listing: site_sub -> pairs -> bias coefficients
    for site_sub in sorted(run.keys()):
        entry = run[site_sub]
        print(f"{site_sub}:")
        pairs = entry.get('biases', {}).get('pairs', {})
        if not pairs:
            print("  (no pairs)")
            continue
        # list pair keys
        keys = sorted(pairs.keys())
        print(f"  pairs ({len(keys)}): {keys}")
        # print each pair with its bias coefficients
        for pk in keys:
            coeffs = pairs.get(pk, {})
            if not coeffs:
                print(f"    {pk}: {{}}")
                continue
            pretty = json.dumps(coeffs, indent=4)
            # indent multi-line JSON for readability
            pretty_indented = '\n'.join('    ' + line for line in pretty.splitlines())
            print(pretty_indented)
        print()
