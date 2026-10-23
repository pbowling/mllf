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

from mllf.cb.cost_ledger import simulated_ns
from mllf.file_handling.generate_combinations import is_combo_dir_name
from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.file_handling.read_output import (
    parse_single_population,
    parse_transitions_and_rates,
    terminated_normally,
    parse_single_ddg,
    parse_pair_profile,
    resolve_output_file,
)
from mllf.file_handling.read_bias_coeff import (
    FLAT_TO_MATRIX_NONLINEAR,
    NONLINEAR_SIGN_KEY,
    read_bias_coeff,
)


def find_run_directories(base_path: Path) -> List[Path]:
    """Find completed runs under a direct or generated-combo dataset root.

    Supports ``run*/`` and ``comb_*/run_*/`` layouts; excludes failed or
    incomplete directories and sorts results by run number.
    """
    run_dirs = []
    
    # Check if this is a generated combo structure (has comb_* directories)
    combo_dirs = [d for d in base_path.iterdir() if d.is_dir() and is_combo_dir_name(d.name)]
    
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

    Sign convention: the flat ``cs*``/``xs*``/``ss*`` values are what CHARMM
    receives (``ldbv ... @coeff``), which is the NEGATIVE of the c/x/s matrix
    convention used by ALF's variables*.py / variablesflat.py and by every
    other pretraining ``variables.py`` (ALF writes both forms side by side,
    e.g. matrix ``c[0][1] = +19.53`` with ``cs1s1s1s2 = -19.53``). The
    nonlinear entries are therefore negated here; ``lams*`` (b) is stored
    unchanged in both conventions.
    
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
                c_matrix[i][j] = FLAT_TO_MATRIX_NONLINEAR * float(value)
    
    # Fill x matrix from xs entries (skew coefficients)
    for key, value in old_data.get("xs", {}).items():
        match = re.match(r'xs(\d+)s(\d+)s(\d+)s(\d+)', key)
        if match:
            site1, sub1, site2, sub2 = map(int, match.groups())
            i = site_sub_to_idx.get((site1, sub1))
            j = site_sub_to_idx.get((site2, sub2))
            if i is not None and j is not None:
                x_matrix[i][j] = FLAT_TO_MATRIX_NONLINEAR * float(value)
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
                s_matrix[i][j] = FLAT_TO_MATRIX_NONLINEAR * float(value)
    
    return {
        "b": b_vector,
        "c": c_matrix,
        "x": x_matrix,
        "s": s_matrix,
        # Topology metadata: not a bias coefficient, stripped before writing variables.py
        "_nsubs_per_site": [len(site_sub_counts[s]) for s in sorted(site_sub_counts.keys())],
        # Marks c/x/s as already converted to the matrix convention, so
        # fix_inp_nonlinear_signs() never flips this run again.
        NONLINEAR_SIGN_KEY: "matrix",
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
        Dict with 'populations', 'transitions', 'rates', 'terminated_normally',
        the per-pair fields and 'simulated_ns' (run length, None if unknown)
    """
    content = output_file.read_text()
    
    # Check termination status based on directory name
    # Assume normal termination unless _failed in path
    terminated = '_failed' not in str(run_dir).lower()
    
    # Parse populations using existing utility
    populations = parse_single_population(content)
    
    # Parse transitions and rates using existing utility  
    transitions, rates = parse_transitions_and_rates(content)
    
    # Parse per-pair DDG data (NaN = no transitions between that pair)
    ddg_raw = parse_single_ddg(content)
    # Per-pair barrier / skew / end signals from the lambda trajectory
    # (quadratic, skew and end pair rewards)
    profile = parse_pair_profile(content, pairs=list(ddg_raw.keys()))

    # Convert to JSON-serializable format (handles tuple/float keys)
    return {
        'populations': convert_to_json_serializable(populations),
        'transitions': convert_to_json_serializable(transitions),
        'rates': convert_to_json_serializable(rates),
        'terminated_normally': terminated,
        'ddg_pairs': {f"{lo}_{hi}": val for (lo, hi), val in ddg_raw.items()},
        **_profile_fields(profile),
        'simulated_ns': simulated_ns(output_file),
    }


PROFILE_FIELDS = (('barrier_pairs', 'barrier'), ('skew_pairs', 'skew'), ('end_pairs', 'end'),
                  ('transition_pairs', 'transitions'))


def _profile_fields(profile: Dict[Tuple[int, int], Dict[str, Optional[float]]]) -> Dict[str, Dict]:
    """simulation_results.json fields: 'barrier_pairs', 'skew_pairs', 'end_pairs' and
    'transition_pairs' (per-pair transitions at the λ ≥ 0.99 cutoff)."""
    return {field: {f"{lo}_{hi}": vals[key] for (lo, hi), vals in profile.items()}
            for field, key in PROFILE_FIELDS}


def _backfill_profile_one(sim_path: Path, dry_run: bool, overwrite: bool) -> Tuple[str, str]:
    """Backfill one run's λ-profile fields. Returns (status, message)."""
    run_dir = sim_path.parent
    try:
        with open(sim_path) as f:
            sim_results = json.load(f)
    except (OSError, ValueError) as exc:
        return 'error', f"unreadable {sim_path}: {exc}"
    has_profile = all(field in sim_results for field, _ in PROFILE_FIELDS)
    if has_profile and 'simulated_ns' in sim_results and not overwrite:
        return 'present', ''
    output_file = resolve_output_file(run_dir)
    if output_file is None:
        return 'no_output', str(run_dir)
    if has_profile and not overwrite:
        # Only the run length is missing (collected before it was recorded).
        ns = simulated_ns(output_file)
        if not dry_run:
            sim_results['simulated_ns'] = ns
            with open(sim_path, 'w') as f:
                json.dump(sim_results, f, indent=2)
        return 'updated', f"{run_dir}: simulated_ns = {ns}"
    try:
        content = output_file.read_text()
        pairs = [tuple(int(v) for v in k.split('_')) for k in sim_results.get('ddg_pairs', {})] or None
        fields = _profile_fields(parse_pair_profile(content, pairs=pairs))
    except Exception as exc:
        return 'error', f"could not parse lambda profile for {run_dir}: {exc}"
    fields['simulated_ns'] = simulated_ns(output_file)
    n_ok = sum(1 for v in fields['barrier_pairs'].values() if v is not None)
    if not dry_run:
        sim_results.update(fields)
        with open(sim_path, 'w') as f:
            json.dump(sim_results, f, indent=2)
    return 'updated', f"{run_dir}: {n_ok}/{len(fields['barrier_pairs'])} pairs with a barrier"


def _backfill_profile_task(args) -> Tuple[str, str]:
    return _backfill_profile_one(*args)


def backfill_barrier_pairs(pretraining_root: Path, dry_run: bool = False,
                           overwrite: bool = False, workers: int = 1) -> Tuple[int, int]:
    """Add per-pair lambda-profile signals to existing simulation_results.json files.

    Mirrors ``backfill_ddg_pairs``. Writes 'barrier_pairs', 'skew_pairs' and
    'end_pairs' (``parse_pair_profile``) -- the quadratic, skew and end pair
    reward signals; runs without them fall back to the legacy signals -- and
    'transition_pairs' (per-pair transitions at the λ ≥ 0.99 cutoff). Reads
    the run directory's own ``output.out`` when present, else the source run
    named by metadata.json's ``source_run_dir`` (nothing is copied). Also writes
    'simulated_ns' (the run's length; pretraining judges short runs by the
    online short-run success test), alone for runs that already have the
    profile fields. Stores the
    raw signals, not rewards, so reward targets/widths can change without a
    new backfill.

    Args:
        pretraining_root: Root directory containing pretraining subdirectories.
        dry_run: If True, report what would be done without modifying files.
        overwrite: Recompute runs that already have all three fields.
        workers: Parallel processes (each run writes only its own file).

    Returns:
        (n_updated, n_skipped) counts.
    """
    sim_results_files = sorted(pretraining_root.rglob('simulation_results.json'))
    total = len(sim_results_files)
    print(f"Found {total} simulation_results.json files under {pretraining_root}"
          + (f" ({workers} workers)" if workers > 1 else ""), flush=True)

    tasks = [(p, dry_run, overwrite) for p in sim_results_files]
    counts = {'updated': 0, 'present': 0, 'no_output': 0, 'error': 0}
    shown = {'updated': 0, 'no_output': 0, 'error': 0}
    limits = {'updated': 10 if dry_run else 0, 'no_output': 10, 'error': 20}

    def _consume(results):
        for n, (status, msg) in enumerate(results, 1):
            counts[status] += 1
            if status in shown and shown[status] < limits[status]:
                shown[status] += 1
                prefix = {'updated': 'Would update', 'no_output': 'No output file for',
                          'error': 'Warning:'}[status]
                print(f"  {prefix} {msg}", flush=True)
            if n % 1000 == 0:
                print(f"  ... {n}/{total} runs checked", flush=True)

    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            _consume(pool.imap_unordered(_backfill_profile_task, tasks, chunksize=16))
    else:
        _consume(_backfill_profile_one(*t) for t in tasks)

    n_skipped = counts['present'] + counts['no_output'] + counts['error']
    verb = "would update" if dry_run else "updated"
    print(f"\nLambda-profile backfill summary: {counts['updated']} {verb}, {counts['present']} already had "
          f"the fields, {counts['no_output']} without an output file, {counts['error']} errors", flush=True)
    return counts['updated'], n_skipped


def _source_variables_format(run_dir: Path) -> Optional[str]:
    """'py' or 'inp' for the variables file the collector would have read from
    this run's source (``metadata.json`` → ``source_run_dir``), else None."""
    metadata_path = run_dir / 'metadata.json'
    if not metadata_path.exists():
        return None
    try:
        source = json.loads(metadata_path.read_text()).get('source_run_dir')
    except (OSError, ValueError):
        return None
    if not source or not Path(source).is_dir():
        return None
    source = Path(source)
    # Same precedence as collect_run_data(): .py first, then .inp
    if any(source.glob('variables*.py')):
        return 'py'
    if any(source.glob('variables*.inp')):
        return 'inp'
    return None


def fix_inp_nonlinear_signs(pretraining_root: Path, dry_run: bool = False) -> Tuple[int, int, int]:
    """Correct c/x/s signs in runs collected from variables*.inp before the fix.

    ``parse_bias_from_inp`` used to copy the flat cs/xs/ss values straight into
    the matrices, but those values are the negative of the matrix convention
    that every ``.py``-derived run uses (see its docstring). Affected runs are
    identified by their source: if ``source_run_dir`` has no variables*.py but
    does have variables*.inp, the stored c/x/s are negated back and the file is
    marked with ``_nonlinear_sign_convention: matrix``. Marked runs and
    ``.py``-sourced runs are left untouched, so rerunning is safe. b is never
    changed.

    Returns:
        (n_fixed, n_already_ok, n_unresolved) -- unresolved means the source
        directory or its variables file could not be found.
    """
    n_fixed = n_ok = n_unresolved = 0
    unresolved_examples: List[str] = []
    var_files = sorted(pretraining_root.rglob('variables.py'))
    print(f"Found {len(var_files)} variables.py files under {pretraining_root}")
    for var_path in var_files:
        run_dir = var_path.parent
        text = var_path.read_text()
        m = re.search(r'bias_string\s*=\s*"""(.*?)"""', text, re.DOTALL)
        if not m:
            continue
        bias = yaml.safe_load(m.group(1))
        if not isinstance(bias, dict) or bias.get(NONLINEAR_SIGN_KEY) == "matrix":
            n_ok += 1
            continue
        fmt = _source_variables_format(run_dir)
        if fmt == 'py':
            n_ok += 1
            continue
        if fmt is None:
            n_unresolved += 1
            if len(unresolved_examples) < 5:
                unresolved_examples.append(str(run_dir))
            continue
        for key in ('c', 'x', 's'):
            if key in bias:
                bias[key] = [[(-float(v) if float(v) != 0.0 else 0.0) for v in row] for row in bias[key]]
        bias[NONLINEAR_SIGN_KEY] = "matrix"
        if dry_run:
            if n_fixed < 5:
                print(f"  Would fix {run_dir}")
        else:
            bias_yaml = yaml.dump(bias, default_flow_style=False, sort_keys=False)
            var_path.write_text(text[:m.start(1)] + "\n" + bias_yaml.strip() + "\n" + text[m.end(1):])
        n_fixed += 1
    verb = "would fix" if dry_run else "fixed"
    print(f"\n.inp sign fix: {n_fixed} {verb}, {n_ok} already correct (.py source or marked), "
          f"{n_unresolved} unresolved (no source dir/variables file)")
    for ex in unresolved_examples:
        print(f"  unresolved e.g. {ex}")
    return n_fixed, n_ok, n_unresolved


def backfill_ddg_pairs(pretraining_root: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Add per-pair DDG data to existing simulation_results.json files.

    For each pretraining run under pretraining_root that has a
    simulation_results.json but no 'ddg_pairs' key, reads the source
    CHARMM output file (resolved from metadata.json's source_run_dir)
    and adds the per-pair DDG presence information.

    A pair whose DDG is NaN (None in the result) had no lambda-space
    transitions and therefore no reliable bias estimate; a finite value
    means transitions occurred.

    Args:
        pretraining_root: Root directory containing pretraining subdirectories.
        dry_run: If True, report what would be done without modifying files.

    Returns:
        (n_updated, n_skipped) counts.
    """
    n_updated = 0
    n_skipped = 0

    sim_results_files = sorted(pretraining_root.rglob('simulation_results.json'))
    print(f"Found {len(sim_results_files)} simulation_results.json files under {pretraining_root}")

    for sim_path in sim_results_files:
        run_dir = sim_path.parent

        with open(sim_path) as f:
            sim_results = json.load(f)

        if 'ddg_pairs' in sim_results:
            n_skipped += 1
            continue

        metadata_path = run_dir / 'metadata.json'
        if not metadata_path.exists():
            print(f"  Warning: No metadata.json in {run_dir.name}, skipping")
            n_skipped += 1
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        source_run_dir = metadata.get('source_run_dir')
        if not source_run_dir:
            print(f"  Warning: No source_run_dir in {run_dir.name} metadata, skipping")
            n_skipped += 1
            continue

        source_path = Path(source_run_dir)
        output_file: Optional[Path] = None
        for candidate in (source_path / 'output.out', source_path / 'output'):
            if candidate.exists():
                output_file = candidate
                break

        if output_file is None:
            print(f"  Warning: No output file in source {source_run_dir}, skipping")
            n_skipped += 1
            continue

        try:
            ddg_raw = parse_single_ddg(output_file.read_text())
        except Exception as exc:
            print(f"  Warning: Could not parse DDG for {run_dir.name}: {exc}")
            n_skipped += 1
            continue

        ddg_serialized: Dict[str, Optional[float]] = {
            f"{lo}_{hi}": val for (lo, hi), val in ddg_raw.items()
        }

        n_transitions = sum(1 for v in ddg_serialized.values() if v is not None)
        if dry_run:
            print(
                f"  Would update {run_dir.name}: "
                f"{n_transitions}/{len(ddg_serialized)} pairs with transitions"
            )
        else:
            sim_results['ddg_pairs'] = ddg_serialized
            with open(sim_path, 'w') as f:
                json.dump(sim_results, f, indent=2)

        n_updated += 1

    print(f"\nBackfill summary: {n_updated} updated, {n_skipped} skipped")
    return n_updated, n_skipped


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


_ECHO_PREFIX_RE = re.compile(r'^\s*[A-Z]*>\s*')
_DEFINE_RE = re.compile(r'^define\s+(site\d+_sub\d+)\b', re.IGNORECASE)
_SELE_ATOM_RE = re.compile(r'\batom\s+\S+\s+\S+\s+(\S+)', re.IGNORECASE)
_CALL_RE = re.compile(r'^call\s+(\d+)\s+sele\s+(site\d+_sub\d+)\b', re.IGNORECASE)
_RTF_ATOM_RE = re.compile(r'^ATOM\s+(\S+)', re.IGNORECASE)
_READ_RTF_RE = re.compile(r'read\s+rtf\s+append.*?/?(site(\d+)_sub(\d+))_pres\.rtf', re.IGNORECASE)
_PRES_FILE_RE = re.compile(r'^(site\d+_sub\d+)_pres\.rtf$')


def _sequential_subs(nsubs_per_site: List[int]) -> Dict[str, List[str]]:
    return {
        f'site{s}': [f'site{s}_sub{i}' for i in range(1, n + 1)]
        for s, n in enumerate(nsubs_per_site, start=1)
    }


def _split_by_site(ordered: List[str], nsubs_per_site: List[int]) -> Dict[str, List[str]]:
    """Distribute an ordered list of master subs to MSLD sites by cumulative count."""
    result, offset = {}, 0
    for s, n in enumerate(nsubs_per_site, start=1):
        result[f'site{s}'] = ordered[offset:offset + n]
        offset += n
    return result


def _master_sub_atoms(prep_dir: Path) -> Dict[str, set]:
    """Atom names of each master substituent: {'site1_sub3': {'C040', ...}}."""
    atoms = {}
    for p in prep_dir.glob('site*_sub*_pres.rtf'):
        m = _PRES_FILE_RE.match(p.name)
        if not m:
            continue
        names = set()
        for line in p.read_text(errors='ignore').splitlines():
            am = _RTF_ATOM_RE.match(line.strip())
            if am:
                names.add(am.group(1).upper())
        atoms[m.group(1)] = names
    return atoms


def _block_selections(lines) -> List[set]:
    """Atom-name sets of the MSLD substituent blocks, in block order.

    Reads ``define siteS_subJ`` selections and the ``call N sele siteS_subJ``
    lines of the BLOCK setup from a CHARMM script or from its echo in a
    CHARMM output (``CHARMM>`` / ``BLOCK>`` prefixes stripped). Block N carries
    bias position N - 1, so the calls, not the ``read rtf`` or ``patch``
    order, define the bias numbering; the selection labels are local names.
    Returns [] if either part is missing.
    """
    defines: Dict[str, set] = {}
    calls: Dict[int, str] = {}
    current = None
    for raw in lines:
        line = _ECHO_PREFIX_RE.sub('', raw).strip()
        if not line or line.startswith('!'):
            continue
        dm = _DEFINE_RE.match(line)
        if dm:
            current = dm.group(1).lower()
            defines.setdefault(current, set())
            continue
        if current is not None:
            am = _SELE_ATOM_RE.search(line)
            if am:
                defines[current].add(am.group(1).upper())
            if re.search(r'\bend\b', line, re.IGNORECASE) and not line.endswith('-'):
                current = None
            continue
        cm = _CALL_RE.match(line)
        if cm:
            calls.setdefault(int(cm.group(1)), cm.group(2).lower())
    if not calls or not defines:
        return []
    return [defines.get(label, set()) for _, label in sorted(calls.items())]


def _map_selections_to_masters(selections: List[set],
                               master_atoms: Dict[str, set]) -> Optional[List[str]]:
    """Master sub for each block selection: the one defining most of its atoms.

    Atoms not in any substituent RTF (lone pairs added by lpsites) are ignored.
    None unless every selection maps to a single master covering at least half
    of its atoms, each master used once.
    """
    known = set().union(*master_atoms.values()) if master_atoms else set()
    out = []
    for sel in selections:
        sel = sel & known
        if not sel:
            return None
        scored = sorted(((len(sel & names), master) for master, names in master_atoms.items()),
                        reverse=True)
        best, master = scored[0]
        if best * 2 < len(sel) or (len(scored) > 1 and scored[1][0] == best):
            return None
        out.append(master)
    return out if len(set(out)) == len(out) else None


def parse_active_subs_from_output(
    output_file: Path,
    prep_dir: Path,
    nsubs_per_site: List[int],
) -> Optional[Dict[str, List[str]]]:
    """Active substituents in bias order, from what the run actually executed.

    The CHARMM output echoes the substituent ``define`` selections and the
    BLOCK ``call`` lines; each selection is matched to the master
    ``siteS_subJ_pres.rtf`` in ``prep_dir`` that defines its atoms. This does
    not depend on which (or how many) .inp files sit in prep/. Returns None
    when the output lacks the echo or the blocks don't match nsubs_per_site.
    """
    if not nsubs_per_site or not output_file or not Path(output_file).exists():
        return None
    try:
        with open(output_file, errors='ignore') as fh:
            selections = _block_selections(fh)
    except OSError:
        return None
    if len(selections) != sum(nsubs_per_site):
        return None
    ordered = _map_selections_to_masters(selections, _master_sub_atoms(prep_dir))
    return _split_by_site(ordered, nsubs_per_site) if ordered else None


def parse_active_subs_from_inp(
    prep_dir: Path,
    nsubs_per_site: List[int],
) -> Dict[str, List[str]]:
    """Parse the ordered list of active substituents from the CHARMM .inp files.

    Fallback for runs whose output does not echo the BLOCK setup (see
    ``parse_active_subs_from_output``, which is preferred). Every candidate
    .inp in prep/ is considered, in name order, so the result does not depend
    on directory listing order (a prep can hold several, e.g. p38's benz.inp
    listing all subs next to the p38.inp that ran):

    1. ``define``/``call`` selections mapped to master RTFs by atom names
       (the bias numbering itself), when a file's blocks match nsubs_per_site;
    2. ``read rtf append`` order, from a file whose per-site counts match
       nsubs_per_site exactly;
    3. the first sum(nsubs_per_site) reads distributed by cumulative count,
       only from a file with exactly that many reads (systems whose one MSLD
       site spans several file-naming site prefixes, e.g. JNK1);
    4. sequential numbering (sub1, sub2, ...).

    Within a tier, candidates giving different answers are reported, and the
    first in name order is used.

    Args:
        prep_dir:       Prep directory to search for CHARMM .inp files.
        nsubs_per_site: Number of active subs per site, from the bias variables.

    Returns:
        Dict mapping 'site1', 'site2', … to an ordered list of master sub names,
        e.g. {'site1': ['site1_sub3', 'site1_sub1', …], 'site2': ['site2_sub15', …]}
    """
    exclude = {'lpsites.inp', 'variablesflat.inp'}
    inp_candidates = sorted(
        p for p in prep_dir.glob('*.inp')
        if p.name not in exclude and 'traj' not in p.name
    )
    if not inp_candidates or not nsubs_per_site:
        return _sequential_subs(nsubs_per_site)

    master_atoms = _master_sub_atoms(prep_dir)
    total = sum(nsubs_per_site)
    tiers: List[List[Tuple[str, Dict[str, List[str]]]]] = [[], [], []]
    for inp_file in inp_candidates:
        try:
            lines = inp_file.read_text(errors='ignore').splitlines()
        except Exception:
            continue
        selections = _block_selections(lines)
        if len(selections) == total:
            ordered = _map_selections_to_masters(selections, master_atoms)
            if ordered:
                tiers[0].append((inp_file.name, _split_by_site(ordered, nsubs_per_site)))
        reads: List[Tuple[int, str]] = []
        for line in lines:
            if line.lstrip().startswith('!'):
                continue
            m = _READ_RTF_RE.search(line)
            if m:
                reads.append((int(m.group(2)), m.group(1)))
        by_site: Dict[int, List[str]] = {}
        for site_num, master in reads:
            by_site.setdefault(site_num, []).append(master)
        if all(len(by_site.get(s, [])) == n for s, n in enumerate(nsubs_per_site, start=1)) \
                and len(reads) == total:
            tiers[1].append((inp_file.name, {f'site{s}': by_site[s]
                                             for s in range(1, len(nsubs_per_site) + 1)}))
        elif len(reads) == total:
            tiers[2].append((inp_file.name, _split_by_site([m for _, m in reads], nsubs_per_site)))

    for tier in tiers:
        if not tier:
            continue
        distinct = {json.dumps(r, sort_keys=True) for _, r in tier}
        if len(distinct) > 1:
            print(f"  Warning: {prep_dir}: .inp files disagree on the active substituents "
                  f"({', '.join(n for n, _ in tier)}); using {tier[0][0]}")
        return tier[0][1]
    return _sequential_subs(nsubs_per_site)


def _parse_nsubs_from_mapping(mapping_path: Path) -> Optional[List[int]]:
    """Derive nsubs_per_site from a generated-combo mapping.json file.

    mapping.json records how master subs were renumbered for the combo's prep.
    The maximum new_sub per new_site gives the number of active subs for that
    site (assuming sequential 1-based renumbering).

    Args:
        mapping_path: Path to the combo-level mapping.json.

    Returns:
        nsubs_per_site list, or None if the file cannot be parsed.
    """
    try:
        with open(mapping_path) as fh:
            entries = json.load(fh)
        subs_per_site: Dict[int, int] = {}
        for entry in entries:
            site = entry.get('new_site')
            sub  = entry.get('new_sub')
            if site is None or sub is None:
                continue
            if site not in subs_per_site or sub > subs_per_site[site]:
                subs_per_site[site] = sub
        if not subs_per_site:
            return None
        return [subs_per_site[s] for s in sorted(subs_per_site)]
    except Exception:
        return None


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

    # Build the active-substituent mapping: which master sub carries each
    # sequential bias position (lams1s1 = position 1, lams1s2 = position 2, …).
    # The BLOCK calls define it (parse_active_subs_from_output); the prep .inp
    # files are the fallback, and sequential numbering (site1_sub1, …) the last.
    # nsubs_per_site_override takes priority (e.g. combo runs where prep/ is full-system).
    #
    # Auto-detect from mapping.json for generated-combo runs: alf_info.py in the
    # combo prep stores the FULL system nsubs, but mapping.json records the actual
    # active subs for this specific combo.
    if nsubs_per_site_override is None:
        mapping_path = run_dir.parent / 'mapping.json'
        if mapping_path.exists():
            nsubs_per_site_override = _parse_nsubs_from_mapping(mapping_path)

    if nsubs_per_site_override is not None:
        nsubs_per_site = nsubs_per_site_override
        bias_data['_nsubs_per_site'] = nsubs_per_site_override
    else:
        nsubs_per_site = bias_data.get('_nsubs_per_site') or []
    # What the run executed (its output's BLOCK setup) first; the prep .inp
    # files only when the output doesn't echo it.
    active_subs_ordered = (parse_active_subs_from_output(output_file, prep_path, nsubs_per_site)
                           or parse_active_subs_from_inp(prep_path, nsubs_per_site))

    # Re-key graph_info['sites']:
    #   sequential key 'site1_sub1' ← master sub at position 0 (e.g. site1_sub3)
    all_rtf_sites = graph_info['sites']

    # Safety net: validate that every master_sub referenced by active_subs_ordered
    # actually has RTF data available (i.e. exists as a key in all_rtf_sites, the
    # subs whose RTF files are physically present in prep/).  This guards against
    # parse_active_subs_from_inp() mis-parsing a *stale* or master-template .inp
    # file that was copied into a generated-combo prep directory unchanged (its
    # "read rtf append" order no longer matches the combo's renamed/renumbered
    # RTF files).  When that happens, the previous behaviour was to silently emit
    # empty stub entries (missing 'site'/'atom_types'/etc.), corrupting graph
    # construction downstream.  Generated-combo preps always have their RTF files
    # renamed/renumbered sequentially 1:1 with the bias-variable ordering, so
    # falling back to plain sequential naming is always correct for them.
    def _is_fully_resolved(mapping: Dict[str, List[str]]) -> bool:
        return all(
            master_sub in all_rtf_sites
            for ordered_masters in mapping.values()
            for master_sub in ordered_masters
        )

    if not _is_fully_resolved(active_subs_ordered):
        active_subs_ordered = {
            f'site{s}': [f'site{s}_sub{i}' for i in range(1, n + 1)]
            for s, n in enumerate(nsubs_per_site, start=1)
        }

    graph_info = rekey_graph_info(graph_info, active_subs_ordered, nsubs_per_site)

    return bias_data, output_results, graph_info


def rekey_graph_info(graph_info: Dict[str, Any], active_subs_ordered: Dict[str, List[str]],
                     nsubs_per_site: List[int]) -> Dict[str, Any]:
    """Key graph_info['sites'] by bias position (in place; returned).

    ``graph_info['sites']`` holds every master sub in prep/ (from
    ``extract_graph_info``). Each active one becomes 'site{S}_sub{k}' for MSLD
    site S and bias position k, with 'site' = S and 'sub' = k (graph node
    order) and 'master' = the master sub ('site2_sub21'), whose RTF supplied
    the atom data and whose fragment PDB is the substituent's structure. A
    master's own site prefix can differ from its MSLD site (JNK1).
    """
    all_rtf_sites = graph_info['sites']
    graph_info['sites'] = {}
    for site_label, ordered_masters in active_subs_ordered.items():
        site_num = int(site_label.replace('site', ''))
        for seq_idx, master_sub in enumerate(ordered_masters):
            entry = dict(all_rtf_sites.get(master_sub, {}))
            entry['site'] = site_num
            entry['sub'] = seq_idx + 1
            entry['master'] = master_sub
            graph_info['sites'][f"{site_label}_sub{seq_idx + 1}"] = entry
    graph_info['active_subs_ordered'] = active_subs_ordered
    graph_info['nsubs_per_site'] = nsubs_per_site
    return graph_info


def create_pretraining_entry(
    run_dir: Path,
    output_dir: Path,
    combo_name: Optional[str] = None,
    solvent_state: Optional[str] = None,
    nsubs_per_site_override: Optional[List[int]] = None,
    min_transitions: int = 0,
) -> Optional[Path]:
    """Create a pretraining directory entry from a run directory.
    
    Args:
        run_dir: Source run# directory
        output_dir: Output directory for pretraining data
        combo_name: Optional name for the combination (default: run directory name)
        solvent_state: Optional solvent state override (if None, auto-detect)
        min_transitions: Skip runs with fewer total transitions than this threshold.
        
    Returns:
        Path to created pretraining entry directory, or None if skipped.
    """
    if combo_name is None:
        combo_name = run_dir.name
    
    # Collect data
    bias_data, output_results, graph_info = collect_run_data(
        run_dir, solvent_state, nsubs_per_site_override=nsubs_per_site_override
    )
    
    # Extract topology metadata before writing YAML (not stored in variables.py)
    nsubs_per_site = bias_data.pop('_nsubs_per_site', None)

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

    # Apply minimum-transitions filter before touching the filesystem
    if total_trans < min_transitions:
        print(f"  Skipping {combo_name}: {total_trans} transitions < {min_transitions}")
        return None

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


def _relabel_one(entry_dir: Path, dry_run: bool) -> Tuple[str, str]:
    """Re-derive one collected run's active substituents; rewrite graph_info.json
    if they (or the node site/master fields) differ. Returns (status, message)."""
    try:
        meta = json.loads((entry_dir / 'metadata.json').read_text())
        gi_path = entry_dir / 'graph_info.json'
        old = json.loads(gi_path.read_text())
        src = Path(meta.get('source_run_dir', ''))
        prep = src / 'prep' if (src / 'prep').is_dir() else src.parent / 'prep'
        if not prep.is_dir():
            return 'no_source', str(entry_dir)
        out = src / 'output' if (src / 'output').exists() else src / 'output.out'
        nsubs = old.get('nsubs_per_site') or meta.get('nsubs_per_site') or []
        new_subs = (parse_active_subs_from_output(out, prep, nsubs)
                    or parse_active_subs_from_inp(prep, nsubs))
        new = rekey_graph_info(extract_graph_info(prep, old.get('solvent_state', 'unknown')),
                               new_subs, nsubs)
        if not all(e.get('atom_types') for e in new['sites'].values()):
            return 'error', f"{entry_dir}: unresolved substituents {new_subs}"
        if {k: (v.get('site'), v.get('sub'), v.get('master')) for k, v in new['sites'].items()} == \
                {k: (v.get('site'), v.get('sub'), v.get('master')) for k, v in old.get('sites', {}).items()}:
            return 'same', str(entry_dir)
        changed = old.get('active_subs_ordered') != new_subs
        if not dry_run:
            backup = entry_dir / 'graph_info.orig.json'
            if not backup.exists():
                backup.write_text(gi_path.read_text())
            gi_path.write_text(json.dumps(new, indent=2))
        return ('relabeled' if changed else 'annotated'), \
            f"{entry_dir}: {old.get('active_subs_ordered')} -> {new_subs}"
    except Exception as e:
        return 'error', f"{entry_dir}: {e}"


def _relabel_task(args):
    return _relabel_one(*args)


def relabel_active_subs(root: Path, datasets: Optional[List[str]] = None,
                        dry_run: bool = False, workers: int = 1) -> Dict[str, int]:
    """Rewrite graph_info.json of collected runs with the substituents each run
    actually executed (``parse_active_subs_from_output``, .inp fallback).

    For data collected before that parser: the first .inp found in prep/ set
    the labels, so a prep holding several (p38's benz.inp next to p38.inp) or
    defining its blocks out of read order (BACE, thrombin, JNK1) mislabeled
    the substituents. 'relabeled' = substituents changed; 'annotated' = only
    the node site/master fields added. The original is kept once as
    graph_info.orig.json. ``datasets``: dataset directory names to limit to.
    """
    entries = sorted(p.parent for p in root.glob('*/*/graph_info.json')
                     if not datasets or p.parent.parent.name in datasets)
    print(f"Relabeling active substituents in {len(entries)} runs under {root}"
          + (" (dry run)" if dry_run else ""), flush=True)
    counts = {k: 0 for k in ('relabeled', 'annotated', 'same', 'no_source', 'error')}
    by_dataset: Dict[str, str] = {}
    tasks = [(e, dry_run) for e in entries]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            results = list(pool.imap(_relabel_task, tasks, chunksize=8))
    else:
        results = [_relabel_one(*t) for t in tasks]
    for (entry_dir, _), (status, msg) in zip(tasks, results):
        counts[status] += 1
        if status == 'relabeled':
            by_dataset.setdefault(entry_dir.parent.name, msg.split(': ', 1)[1])
        elif status == 'error':
            print(f"  Warning: {msg}", flush=True)
    for ds, change in sorted(by_dataset.items()):
        print(f"  {ds}: {change}", flush=True)
    verb = 'would be' if dry_run else 'were'
    print(f"\nRelabel summary ({verb}): " + ', '.join(f"{v} {k}" for k, v in counts.items()), flush=True)
    return counts


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
        nargs='?',
        help='Base directory containing run# subdirectories (not required with --backfill-ddg)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=False,
        default=None,
        help='Output directory for pretraining data (not required with --backfill-ddg)'
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
        '--min-transitions',
        type=int,
        default=0,
        metavar='N',
        help='Skip runs with fewer than N total transitions (default: 0 = keep all)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing files'
    )

    parser.add_argument(
        '--backfill-ddg',
        metavar='PRETRAINING_ROOT',
        type=Path,
        default=None,
        help=(
            'Backfill per-pair DDG data into existing simulation_results.json files '
            'under PRETRAINING_ROOT.  Reads source output files via metadata.json '
            'source_run_dir.  Pass --dry-run to preview without writing.'
        ),
    )

    parser.add_argument(
        '--backfill-barrier',
        metavar='PRETRAINING_ROOT',
        type=Path,
        default=None,
        help=(
            'Backfill per-pair lambda-profile signals (barrier / skew / end: the '
            'quadratic, skew and end pair rewards) and the run length (simulated_ns) '
            'into existing simulation_results.json files under PRETRAINING_ROOT. '
            "Reads each run's output.out (or its metadata.json source_run_dir). "
            'Pass --dry-run to preview without writing.'
        ),
    )

    parser.add_argument(
        '--fix-inp-signs',
        metavar='PRETRAINING_ROOT',
        type=Path,
        default=None,
        help=(
            'Correct c/x/s signs in variables.py files collected from variables*.inp '
            'sources before parse_bias_from_inp negated them (identified via '
            "metadata.json source_run_dir). Safe to rerun. Pass --dry-run to preview."
        ),
    )

    parser.add_argument(
        '--relabel-subs',
        metavar='PRETRAINING_ROOT',
        type=Path,
        default=None,
        help=(
            'Rewrite graph_info.json of collected runs under PRETRAINING_ROOT with the '
            'substituents each run actually executed (its output\'s BLOCK setup, read via '
            'metadata.json source_run_dir); keeps graph_info.orig.json. Limit with '
            '--datasets. Pass --dry-run to preview.'
        ),
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='With --relabel-subs: dataset directory names to process (default: all)',
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='With --backfill-barrier: recompute runs that already have the λ-profile fields',
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        metavar='N',
        help='With --backfill-barrier: parallel worker processes (default 1)',
    )

    args = parser.parse_args()

    # --- Maintenance modes: mutually exclusive with normal collection mode ---
    if args.relabel_subs is not None:
        root = args.relabel_subs.resolve()
        if not root.exists():
            print(f"Error: root does not exist: {root}")
            return 1
        counts = relabel_active_subs(root, datasets=args.datasets, dry_run=args.dry_run,
                                     workers=max(1, args.workers))
        return 1 if counts['error'] else 0

    if args.fix_inp_signs is not None:
        root = args.fix_inp_signs.resolve()
        if not root.exists():
            print(f"Error: root does not exist: {root}")
            return 1
        fix_inp_nonlinear_signs(root, dry_run=args.dry_run)
        return 0

    if args.backfill_barrier is not None:
        root = args.backfill_barrier.resolve()
        if not root.exists():
            print(f"Error: Backfill root does not exist: {root}")
            return 1
        backfill_barrier_pairs(root, dry_run=args.dry_run, overwrite=args.overwrite,
                               workers=max(1, args.workers))
        return 0

    if args.backfill_ddg is not None:
        root = args.backfill_ddg.resolve()
        if not root.exists():
            print(f"Error: Backfill root does not exist: {root}")
            return 1
        backfill_ddg_pairs(root, dry_run=args.dry_run)
        return 0

    # Normal collection mode — base_path and --output-dir are required
    if args.base_path is None:
        parser.error("base_path is required unless a --backfill-*/--fix-inp-signs/--relabel-subs mode is specified")
    if args.output_dir is None:
        parser.error("--output-dir is required unless a --backfill-*/--fix-inp-signs/--relabel-subs mode is specified")

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
    skipped_count = 0
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
            if run_dir.parent != base_path and is_combo_dir_name(run_dir.parent.name):
                combo_name = f"{run_dir.parent.name}_{run_dir.name}"
            else:
                combo_name = run_dir.name
        
        print(f"\n[{i}/{len(run_dirs)}] Processing {run_dir.name} -> {combo_name}")
        
        if args.dry_run:
            print(f"  Would create: {args.output_dir / combo_name}")
            try:
                bias_data, output_results, graph_info = collect_run_data(run_dir, args.solvent_state)
                # Compute dry-run transition count
                dry_trans = 0
                for trans_dict in output_results['transitions'].values():
                    if isinstance(trans_dict, dict):
                        if trans_dict:
                            max_lam = max(trans_dict.keys(), key=lambda x: float(x))
                            dry_trans += trans_dict[max_lam]
                    else:
                        dry_trans += trans_dict
                if dry_trans < args.min_transitions:
                    print(f"  Would skip: {dry_trans} transitions < {args.min_transitions}")
                else:
                    print(f"  Bias keys: {list(bias_data.keys())}")
                    print(f"  Populations: {len(output_results['populations'])} sites")
                    if output_results['transitions']:
                        print(f"  Transition pairs: {len(output_results['transitions'])}")
                    print(f"  Total transitions (highest λ): {dry_trans}")
                    print(f"  Terminated normally: {output_results['terminated_normally']}")
                    print(f"  Solvent state: {graph_info['solvent_state']}")
                    print(f"  Sites found: {len(graph_info['sites'])}")
                success_count += 1
            except Exception as e:
                print(f"  Error: {e}")
                error_count += 1
        else:
            try:
                entry_dir = create_pretraining_entry(
                    run_dir, args.output_dir, combo_name, args.solvent_state,
                    min_transitions=args.min_transitions,
                )
                if entry_dir is None:
                    skipped_count += 1
                else:
                    print(f"  ✓ Created: {entry_dir}")
                    success_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count} successful, {skipped_count} skipped (< {args.min_transitions} transitions), {error_count} errors")
    
    if not args.dry_run and success_count > 0:
        print(f"Pretraining data written to: {args.output_dir.resolve()}")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    exit(main())
