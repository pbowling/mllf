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

            # derive per-sub scalars for cs/xs/ss by averaging entries that include the sub index
            def derive_per_sub_scalar(group_dict):
                # group_dict: mapping key->value where keys contain integers referring to sub indices
                if not group_dict:
                    return []
                nums_map = {}
                for k, v in group_dict.items():
                    nums = re.findall(r"(\d+)", k)
                    if not nums:
                        continue
                    # assume the first number found is the primary sub index for this entry
                    primary = int(nums[0])
                    nums_map.setdefault(primary, []).append(float(v))
                # build list from 1..max_sub, fill missing with 0.0
                if not nums_map:
                    return []
                max_sub = max(nums_map.keys())
                scalars = []
                for i in range(1, max_sub + 1):
                    vals = nums_map.get(i, [])
                    if vals:
                        scalars.append(sum(vals) / len(vals))
                    else:
                        scalars.append(0.0)
                return scalars

            for group in ('cs', 'xs', 'ss'):
                if group in biases:
                    scalars = derive_per_sub_scalar(biases[group])
                    if scalars:
                        pairwise_all[group] = compute_pairwise_from_list(scalars)

            if pairwise_all:
                entry['biases']['pairwise_biases'] = pairwise_all

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
