"""Helper script to scaffold a training example for 14benz_solv_5.5.

This script provides utilities to:
- load the combos list produced by the generator
- split combos into train/val/test manifests
- scaffold a small set of per-combo directories that contain the
  necessary files to run a single simulation (copied RTF fragments,
  prep directory, `msld_flat.py`, `run.sh` and a `variables.py` built
  from a toy predicted graph).

The goal is to create the architecture for training without creating
all combination directories by default. Use the functions below from
your environment or run this file as a script.
"""
from __future__ import annotations

import os
import shutil
import random
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple

from mllf.cb.graph import Graph, EdgeCoeffs
from mllf.file_handling.write_bias_coeff import write_bias_inp_from_graph, write_variables_py_from_inp


def load_combos(combos_file: str) -> List[str]:
    """Return a list of combo names (skips header lines)."""
    combos = []
    with open(combos_file, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            # skip header line like 'Found X eligible sites...'
            if ln.startswith('Found '):
                continue
            combos.append(ln)
    return combos


def split_combos(combos: List[str], train_frac=0.8, val_frac=0.1, seed: int = 12345) -> Dict[str, List[str]]:
    """Split the combo list into train/val/test and return a dict.

    Does a randomized split (deterministic with seed).
    """
    random.seed(seed)
    combos_shuf = combos[:]
    random.shuffle(combos_shuf)
    n = len(combos_shuf)
    ntrain = int(n * train_frac)
    nval = int(n * val_frac)
    train = combos_shuf[:ntrain]
    val = combos_shuf[ntrain:ntrain + nval]
    test = combos_shuf[ntrain + nval:]
    return {"train": train, "val": val, "test": test}


def write_manifests(manifests: Dict[str, List[str]], out_dir: str):
    """Write train/val/test manifest files into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    for name, items in manifests.items():
        p = Path(out_dir) / f"{name}.txt"
        with open(p, 'w', encoding='utf-8') as fh:
            for it in items:
                fh.write(it + '\n')


def parse_combo_tokens(combo_name: str) -> List[Tuple[int, int]]:
    """Parse combo token names like 'comb_0001_site1_1__site2_3' into a
    list of (site, sub) tuples.
    """
    parts = combo_name.split('__')
    toks = []
    for p in parts:
        # skip the comb_xxx prefix
        if p.startswith('comb_'):
            continue
        # token like site1_2 or site2_3
        if p.startswith('site'):
            rem = p[len('site'):]
            try:
                site_str, sub_str = rem.split('_')
                site = int(site_str)
                sub = int(sub_str)
                toks.append((site, sub))
            except Exception:
                # ignore unknown token format
                continue
    return toks


def scaffold_combo(combo_name: str, example_dir: str, out_root: str, copy_templates: bool = True, predict_seed: int = None) -> str:
    """Create a scaffolded directory for a single combo.

    Copies necessary template files from example_dir and writes a
    variables.py generated from a toy predicted graph. Returns the path
    to the created combo directory.
    """
    toks = parse_combo_tokens(combo_name)
    if not toks:
        raise ValueError(f"Cannot parse combo tokens from {combo_name}")

    combo_dir = Path(out_root) / combo_name
    if combo_dir.exists():
        # be idempotent: do nothing if already exists
        return str(combo_dir)
    combo_dir.mkdir(parents=True, exist_ok=True)

    ex = Path(example_dir)

    # copy msld_flat.py and run.sh if present
    for fname in ('msld_flat.py', 'run.sh'):
        src = ex / fname
        if src.exists() and copy_templates:
            shutil.copy2(src, combo_dir / fname)

    # copy prep directory (shallow copy)
    prep_src = ex / 'prep'
    if prep_src.exists() and copy_templates:
        dst_prep = combo_dir / 'prep'
        shutil.copytree(prep_src, dst_prep)

    # copy RTF fragment files that match site/sub tokens
    for site, sub in toks:
        # pattern: site{site}_sub{sub}_pres.rtf
        candidate = ex / f"site{site}_sub{sub}_pres.rtf"
        if candidate.exists() and copy_templates:
            shutil.copy2(candidate, combo_dir / candidate.name)

    # Create a toy predicted graph and write variables.py
    # Graph nodes = number of substituents in this combo
    nsubs = len(toks)
    g = Graph(nsubs if nsubs > 0 else 1)

    # fill edges with a simple deterministic toy predictor
    rng = random.Random(predict_seed)
    for i in range(g.num_nodes):
        for j in range(i + 1, g.num_nodes):
            coeffs = EdgeCoeffs(
                linear=float(rng.uniform(-0.5, 0.5)),
                quadratic=float(rng.uniform(-0.1, 0.1)),
                skew=float(rng.uniform(-0.05, 0.05)),
                end=float(rng.uniform(-0.01, 0.01)),
            )
            g.set_edge(i, j, coeffs)

    # write a temporary .inp bias file then convert to variables.py using existing helpers
    tmpdir = tempfile.mkdtemp(prefix='mllf_tmp_')
    try:
        inp_path = Path(tmpdir) / 'bias.inp'
        vars_py = combo_dir / 'variables.py'
        # default per-site counts: assume each site has 1 substituent (since we don't parse RTF here)
        sub_counts = [1] * g.num_nodes
        write_bias_inp_from_graph(g, str(inp_path), sub_counts=sub_counts)
        write_variables_py_from_inp(str(inp_path), str(vars_py))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Write a metadata.json describing what was scaffolded
    meta = {
        'combo': combo_name,
        'tokens': toks,
        'n_subs': nsubs,
        'files': [p.name for p in combo_dir.iterdir() if p.is_file()],
    }
    import json

    with open(combo_dir / 'scaffold_metadata.json', 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)

    return str(combo_dir)


def scaffold_sample(example_dir: str, out_root: str, n: int = 5, seed: int = 12345) -> List[str]:
    """Scaffold up to `n` combos from the example combos list into out_root.

    Returns list of created directories.
    """
    combos_file = Path(example_dir) / 'combos_14benz_solv_5.5.txt'
    combos = load_combos(str(combos_file))
    manifests = split_combos(combos, seed=seed)
    # take first n from train set
    winners = manifests['train'][:n]
    created = []
    for i, c in enumerate(winners):
        created.append(scaffold_combo(c, example_dir, out_root, predict_seed=seed + i))
    return created


if __name__ == '__main__':
    # simple CLI: scaffold a small sample under examples/cb/14benz_solv_5.5/scaffolded
    here = Path(__file__).resolve().parent
    outdir = here / 'scaffolded'
    print('Loading combos...')
    created = scaffold_sample(str(here), str(outdir), n=5)
    print('Created sample combo dirs:')
    for c in created:
        print(' -', c)
