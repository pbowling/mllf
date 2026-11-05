"""Per-run model inputs and targets: Uni-Mol node embeddings and
directed edges for a run's substituents, the bias targets read from its
``variables.py``, and the record of which structure each run used.
"""
import collections
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from mllf.cb.environment_consensus import build_site_consensus
from mllf.cb.pair_weights import (
    compute_pairwise_confidence_weights,
    compute_pairwise_measured_mask,
)
from mllf.cb.pairs import build_directed_pairs
from mllf.cb.policy import fill_site_size
from mllf.cb.unimol_representation import (
    get_substituent_dual_embeddings,
    get_substituent_unimol_with_environment,
)
from mllf.file_handling.generate_combinations import is_combo_dir_name
from mllf.pretrain.config import UnimolSettings


# ── Representation tracking ────────────────────────────────────────────────────

def initialize_representation_tracker(output_dir: Path) -> Path:
    """Initialize CSV file for tracking which representations are used.

    Args:
        output_dir: Output directory for pretraining

    Returns:
        Path to tracking CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_file = output_dir / 'pretraining_representations_used.csv'

    # Create header if file doesn't exist
    if not tracking_file.exists():
        with open(tracking_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'system', 'run_idx', 'structure_type', 'source',
                'prep_dir_used', 'explicit_mapping', 'timestamp'
            ])
            writer.writeheader()

    return tracking_file


def track_representation_choice(
    tracking_file: Path,
    system_name: str,
    run_idx: int,
    structure_type: str,
    source: str,
    prep_dir_used: Path,
    explicit_mapping: bool = False
) -> None:
    """Log which representation (minimized/unrelaxed) was used for a run.

    Args:
        tracking_file: Path to tracking CSV file
        system_name: System name (e.g., 'indolizine_prot')
        run_idx: Run index
        structure_type: 'minimized' or 'unrelaxed'
        source: Where structures came from (e.g., 'pretraining/prep', 'training/systems')
        prep_dir_used: Full path to prep directory used
        explicit_mapping: Whether an explicit mapping was used
    """
    tracking_file = Path(tracking_file)

    with open(tracking_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'system', 'run_idx', 'structure_type', 'source',
            'prep_dir_used', 'explicit_mapping', 'timestamp'
        ])
        writer.writerow({
            'system': system_name,
            'run_idx': run_idx,
            'structure_type': structure_type,
            'source': source,
            'prep_dir_used': str(prep_dir_used.resolve()),
            'explicit_mapping': str(explicit_mapping),
            'timestamp': datetime.now().isoformat(),
        })


# ── Structure type selection and unrelaxed path detection ───────────────────

# ── Structure type selection and path detection ────────────────────────────


def _select_structure_type_for_run(
    run_idx: int,
    strategy: str,
    system_name: Optional[str] = None,
    skip_patterns: Optional[List[str]] = None,
    seed: int = 42
) -> str:
    """Determine which structure type to use for a given run.

    **Strategies:**
    - 'minimized': Always use minimized (pretraining) structures
    - 'unrelaxed': Always use unrelaxed (crystallographic) structures
    - 'random': Randomly alternate between minimized and unrelaxed per run

    **System-Specific Skipping:**
    If skip_patterns is provided and the system_name matches any pattern,
    unrelaxed structures are skipped regardless of strategy (falls back to minimized).

    Args:
        run_idx: Index of the run (used for random seeding if needed)
        strategy: Strategy to use ('minimized', 'unrelaxed', 'random')
        system_name: Name of the system (for pattern matching against skip list)
        skip_patterns: List of patterns to match against system_name (case-insensitive)
                      Uses 'if pattern.lower() in system_name.lower()'
        seed: Random seed for reproducibility in 'random' mode

    Returns:
        str: 'minimized' or 'unrelaxed'
    """
    strategy = strategy.lower()

    # Check if this system should skip unrelaxed
    should_skip_unrelaxed = False
    if system_name is not None and skip_patterns:
        system_name_lower = system_name.lower()
        for pattern in skip_patterns:
            if pattern.lower() in system_name_lower:
                should_skip_unrelaxed = True
                break

    if strategy == 'minimized' or should_skip_unrelaxed:
        return 'minimized'
    elif strategy == 'unrelaxed':
        return 'unrelaxed'
    elif strategy == 'random':
        rng = np.random.RandomState(seed + run_idx)  # Deterministic per run
        return 'minimized' if rng.rand() < 0.5 else 'unrelaxed'
    else:
        raise ValueError(
            f"Unknown structure selection strategy: {strategy}. Use 'minimized', 'unrelaxed', or "
            "'random'.")


# ── Uni-Mol embedding computation for pretraining ────────────────────────────

def _node_master_sub(site_info: Dict, site_id: int, sub_id: int) -> str:
    """Master substituent ('site1_sub12') whose fragment PDB a graph node uses.

    graph_info keys nodes by bias position ('sub' = position in its site),
    which differs from the master numbering of the prep files whenever a
    group runs a subset of a system's substituents. The master comes from the
    node's 'master' field, else from the RTF file it was read from; only
    graph_info without either falls back to the position.
    """
    master = site_info.get('master')
    if master:
        return master
    m = re.match(r'(site\d+_sub\d+)_pres\.rtf$', str(site_info.get('filename') or ''))
    return m.group(1) if m else f"site{site_id}_sub{sub_id}"


def compute_unimol_embeddings_and_edges(
    run_dir: Path,
    graph_info_path: Path,
    structure_type: str = 'minimized',
    unrelaxed_system_mappings: Optional[Dict[str, Path]] = None,
    custom_search_paths: Optional[Dict[str, List[str]]] = None,
    cache_dir: Optional[Path] = None,
    run_idx: int = 0,
    env_cutoff: float = 8.0,
    use_environment_difference: bool = True,
    consensus_dict: Optional[Dict[str, Dict[str, Optional[set]]]] = None,
    env_input: str = 'dual',
    ligand_other_sites: bool = False,
    ligand_minimized: bool = False,
    site_size_input: bool = False,
) -> tuple:
    """Compute Uni-Mol embeddings for all substituents in a run.

    env_input='phase' or 'phase_ligmean' keeps the [ligand, full] layout but puts
    the phase flags (unimol_representation.phase_embedding) in the full half,
    computing no environment embedding (see UnimolPolicy env_input).
    ligand_other_sites: the ligand half also holds every other site's
    reference substituent (unimol_representation.other_site_reference_subs).
    site_size_input: the full half also holds the run's site size
    (policy.fill_site_size; unimol.site_size_input).
    ligand_minimized: for a 'minimized' run, the ligand half takes its
    coordinates from the prep's minimized.pdb (unimol.ligand_from_minimized).

    Supports both minimized (pretraining) and unrelaxed (crystallographic) structures.
    Both representations are constructed from the same prep directory - the choice only
    affects semantic interpretation (e.g., which PDB files represent the initial vs relaxed state).

    **Embedding Modes:**
    - **Standard mode** (use_environment_difference=False): Single 512D embedding per node
      - Input to policy: [ligand+environment diff, ligand+environment mean]
      - Backward compatible with original UnimolPolicy design
    - **Dual embedding mode** (use_environment_difference=True): Two separate 512D
      embeddings per node
      - Ligand-only: core + sub (captures substituent-specific information)
      - Full: core + sub + environment + ref_subs (captures ligand+environment context)
      - Edge input to policy: [diff_ligand (antisymmetric), mean_full (symmetric)]
      - This allows MLPs to learn substituent-dependent and environment-dependent info in parallel

    **Structure Types:**
    - 'minimized': Use minimized (relaxed, MD-equilibrated) representation
    - 'unrelaxed': Use unrelaxed (crystallographic/experimental) representation

    Both use the same prep directory files; the selection is logged for tracking.

    **Consensus Filtering:**
    If consensus_dict is provided, environment atoms are filtered to only atoms present in
    ALL substituents at a site. Each system has its own consensus built from its substituents.

    Args:
        run_dir: Directory containing variables.py and graph_info.json
        graph_info_path: Path to graph_info.json
        structure_type: 'minimized' or 'unrelaxed' (default: 'minimized')
        unrelaxed_system_mappings: Dict mapping system names to alternate prep paths
            (optional override)
        custom_search_paths: Dict mapping system solvent_state to custom PDB search paths
                           e.g., {'protein': ['pdb/protein.pdb'], 'solv': ['solvent.pdb']}
        cache_dir: Optional directory to cache/load embeddings
        run_idx: Run index (for reference in tracking)
        env_cutoff: Environment distance cutoff in Ångströms (default: 8.0)
        use_environment_difference: If True, compute dual embeddings:
            [ligand-only(512), full(512)] = 1024D per node.
            If False, use standard embedding: ligand+environment(512) per node.
        consensus_dict: Optional nested dict mapping systems to site consensus atoms.
            Structure: {prep_dir_str: {site_key: {(resnum, chain, atomname), ...}, ...}, ...}
            If provided, environment atoms are filtered to consensus atoms for each site.
            Use None for sites with no consensus (environment not filtered for that site).

    Returns:
        (unimol_embeddings [N, emb_dim], edge_index [2, E], nsubs_per_site, representation_source)
        where:
        - N = total substituents
        - E = directed pairs within sites
        - emb_dim = 1024 when use_environment_difference=True (dual embeddings)
        - emb_dim = 512 when use_environment_difference=False (standard embeddings)
        - representation_source = dict with 'structure_type', 'prep_dir_used', 'source',
          'explicit_mapping'
    """

    # Load graph_info to get site structure
    with open(graph_info_path) as f:
        graph_info = json.load(f)

    sites = graph_info.get("sites", {})

    # Get nsubs_per_site from graph_info if available (preferred)
    nsubs_per_site = graph_info.get('nsubs_per_site', None)

    if nsubs_per_site is None:
        # Reconstruct from sites structure: count subs per site
        # Sites can be keyed as "site1_sub1" or as "1" with "subs" list
        site_sub_counts = defaultdict(int)
        for site_key, site_info in sites.items():
            site_id = site_info.get("site")
            if site_id is not None:
                site_sub_counts[site_id] += 1
        if site_sub_counts:
            nsubs_per_site = [site_sub_counts[s] for s in sorted(site_sub_counts.keys())]
        else:
            raise ValueError(f"Could not determine nsubs_per_site from {graph_info_path}")

    n_nodes = sum(nsubs_per_site)

    # Build mapping from (site_id, sub_id) to global node index
    # Iterate through sites and map each (site, sub) to its position
    sub_to_global_idx = {}
    global_idx = 0
    for site_id in sorted(set(site_info["site"] for site_info in sites.values())):
        subs_for_site = sorted([
            site_info["sub"]
            for site_info in sites.values()
            if site_info["site"] == site_id
        ])
        for local_idx, sub_id in enumerate(subs_for_site):
            sub_to_global_idx[(site_id, sub_id)] = global_idx
            global_idx += 1

    # Compute Uni-Mol embeddings for each substituent
    prep_dir = run_dir.parent / "prep"
    if not prep_dir.is_dir():
        raise FileNotFoundError(f"Prep dir not found: {prep_dir}")

    # Determine which prep directory to use
    # Both minimized and unrelaxed use the same prep_dir; only the choice is tracked
    # Handle nested combos: need to look TWO levels up if in comb_* directory
    sys_name = prep_dir.parent.name
    if is_combo_dir_name(sys_name):
        sys_name = prep_dir.parent.parent.name
    active_prep_dir = prep_dir
    explicit_mapping = False
    source = 'pretraining/prep'

    # Check if explicit mapping provided for unrelaxed
    if (structure_type == 'unrelaxed' and unrelaxed_system_mappings
            and sys_name in unrelaxed_system_mappings):
        active_prep_dir = Path(unrelaxed_system_mappings[sys_name])
        explicit_mapping = True
        source = 'training/systems'
        if not active_prep_dir.exists():
            raise FileNotFoundError(f"Mapped prep dir not found: {active_prep_dir}")

    # Initialize representation source tracking
    representation_source = {
        'structure_type': structure_type,
        'prep_dir_used': str(active_prep_dir.resolve()),
        'source': source,
        'explicit_mapping': explicit_mapping,
    }

    # Determine embedding dimension based on representation type
    # Dual embeddings (ligand-only + full) = 1024D, standard = 512D
    emb_dim = 1024 if use_environment_difference else 512
    unimol_embeddings = torch.zeros((n_nodes, emb_dim), dtype=torch.float32)

    # Read core PDB once (needed for environment loading)
    core_pdb = active_prep_dir / "core.pdb"
    if not core_pdb.exists():
        raise FileNotFoundError(f"Core PDB not found: {core_pdb}")

    # Get solvent state from graph_info if available (for environment path selection)
    solvent_state = graph_info.get('solvent_state', 'solv')

    # Determine custom search paths for this system
    sub_custom_paths = None
    if custom_search_paths is not None:
        sub_custom_paths = custom_search_paths.get(solvent_state)

    # Compute embedding for each substituent with environment context
    prep_dir_str = str(prep_dir)  # Key for consensus lookup

    for site_key, site_info in sites.items():
        site_id = site_info.get("site")
        sub_id = site_info.get("sub")

        if site_id is None or sub_id is None:
            continue

        try:
            sub_pdb = active_prep_dir / f"{_node_master_sub(site_info, site_id, sub_id)}_frag.pdb"
            if not sub_pdb.exists():
                print(f"  Warning: Sub PDB not found: {sub_pdb}, using zeros")
                continue

            # Get consensus atoms for this site if available.
            # consensus_dict is keyed by [prep_dir_str][site_name] where
            # site_name is per-SITE (e.g. "site1"), NOT per-node like
            # `site_key` (e.g. "site1_sub1") — derive the site-level lookup
            # key from site_id instead of using site_key directly.
            consensus_atoms = None
            if consensus_dict is not None:
                system_consensus = consensus_dict.get(prep_dir_str)
                if system_consensus is not None:
                    consensus_atoms = system_consensus.get(f"site{site_id}")

            # Compute Uni-Mol embeddings
            if use_environment_difference:
                # Compute dual embeddings: ligand-only and full
                # Stack as [512 ligand + 512 full] = 1024D for edge computation
                emb_ligand, emb_full = get_substituent_dual_embeddings(
                    sub_pdb=str(sub_pdb),
                    core_pdb=str(core_pdb),
                    prep_dir=active_prep_dir,
                    env_cutoff=env_cutoff,
                    atom_limit=256,
                    custom_search_paths=sub_custom_paths,
                    skip_minimized=(structure_type == 'unrelaxed'),
                    consensus_atoms=consensus_atoms,  # Filter by consensus if provided
                    cache_dir=cache_dir,
                    save_cache=(cache_dir is not None),
                    env_input=env_input,
                    ligand_other_sites=ligand_other_sites,
                    ligand_minimized=ligand_minimized and structure_type == 'minimized',
                )
                # Concatenate for storage: UnimolPolicy._forward_edges() will split [ligand, full]
                emb = np.concatenate([emb_ligand, emb_full])  # [1024]
            else:
                # Compute standard embedding with environment context (512D)
                emb = get_substituent_unimol_with_environment(
                    sub_pdb=str(sub_pdb),
                    core_pdb=str(core_pdb),
                    prep_dir=active_prep_dir,
                    env_cutoff=env_cutoff,
                    atom_limit=256,
                    custom_search_paths=sub_custom_paths,
                    cache_dir=cache_dir,
                    save_cache=(cache_dir is not None),
                    skip_minimized=(structure_type == 'unrelaxed'),
                    consensus_atoms=consensus_atoms,  # Filter by consensus if provided
                )
            global_idx = sub_to_global_idx[(site_id, sub_id)]
            unimol_embeddings[global_idx] = torch.from_numpy(emb).float()

        except Exception as e:
            print(f"  Warning: Could not compute embedding for site{site_id}_sub{sub_id}: {e}")

    # Build edge_index for all directed pairs within each site
    pairs = build_directed_pairs(nsubs_per_site)
    src_list = [p[0] for p in pairs]
    dst_list = [p[1] for p in pairs]
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    if site_size_input:
        fill_site_size(unimol_embeddings, nsubs_per_site)

    return unimol_embeddings, edge_index, nsubs_per_site, representation_source


# ── per-system graph cache helpers ──────────────────────────────────────────


def extract_targets_from_variables(run_dir, nsubs_per_site: list, pairs: list):
    """Extract bias-coefficient targets from a single run's variables.py.

    This is cheap (pure Python, no AEV computation).  Call once per run after
    the shared graph structure has been built via ``_build_graph_structure``.

    For UnimolPolicy, each edge gets ONE target row with 4 values: [linear, quadratic, skew, end].
    This matches the policy architecture where each edge produces 4 bias predictions.

    Returns a list of [linear, quadratic, skew, end] targets (one per directed
    edge), or None if the variables file cannot be parsed, is missing, or the
    bias matrix size does not match nsubs_per_site.
    """
    try:
        variables_path = Path(run_dir) / 'variables.py'
        if not variables_path.exists():
            return None
        content = variables_path.read_text()

        # Find the YAML block — try all quoting styles; track the matching closer
        yaml_start = -1
        close_delim = None
        for open_d, close_d in (('bias_string = """', '"""'),
                                ('bias_string="""', '"""'),
                                ("bias_string = '''", "'''"),
                                ("bias_string='''", "'''")):
            pos = content.find(open_d)
            if pos != -1:
                yaml_start = pos + len(open_d)
                close_delim = close_d
                break
        if yaml_start == -1:
            return None

        yaml_end = content.find(close_delim, yaml_start)
        if yaml_end == -1:
            return None

        bias_data = yaml.safe_load(content[yaml_start:yaml_end])
        b_list = bias_data['b']
        b_vector = np.array(b_list[0] if isinstance(b_list[0], list) else b_list, dtype=float)
        n_nodes = sum(nsubs_per_site)
        if len(b_vector) != n_nodes:
            return None  # size mismatch — incompatible run
        c_matrix = np.array(bias_data['c'], dtype=float)
        x_matrix = np.array(bias_data['x'], dtype=float)
        s_matrix = np.array(bias_data['s'], dtype=float)

        targets = []
        for (i, j) in pairs:
            linear = float(b_vector[j] - b_vector[i])
            quadratic = float(c_matrix[i, j]) if i < j else float(c_matrix[j, i])
            skew = float(x_matrix[i, j])
            end = float(s_matrix[i, j])
            # One row per edge with all 4 bias types (matches UnimolPolicy output)
            targets.append([linear, quadratic, skew, end])
        return targets
    except Exception:
        return None


# ── Inputs for every run ───────────────────────────────────────────────────


@dataclass
class RunInputs:
    """What behavior cloning trains on, per run (indexed like ``runs``)."""
    graph_cache: List            # (embeddings [N, D], edge_index [2, E], targets [E][4]) or None
    n_ok: int                    # runs with inputs
    nsubs_by_system: Dict[str, List[int]]   # prep dir -> substituents per site
    consensus: Dict              # prep dir -> site name -> consensus atoms (or None)


def _system_label(prep_dir: Path) -> str:
    """System name of a prep dir (``system/prep`` or ``system/comb_ID/prep``)."""
    name = prep_dir.parent.name
    return prep_dir.parent.parent.name if is_combo_dir_name(name) else name


def _site_fragments(prep_dir: Path) -> Dict[str, List[Path]]:
    """site name -> fragment PDBs, in the prep dir and its comb_* subdirectories."""
    site_to_subs: Dict[str, List[Path]] = {}
    combo_dirs = sorted(d for d in prep_dir.iterdir() if d.is_dir() and is_combo_dir_name(d.name))
    for folder in [prep_dir] + combo_dirs:
        for sub_pdb in sorted(folder.glob('site*_sub*_frag.pdb')):
            site_num = sub_pdb.stem.split('_')[0]  # e.g., "site1"
            subs = site_to_subs.setdefault(site_num, [])
            if sub_pdb not in subs:
                subs.append(sub_pdb)
    return site_to_subs


def _core_pdb(prep_dir: Path) -> Optional[str]:
    if (prep_dir / 'core.pdb').exists():
        return str(prep_dir / 'core.pdb')
    for comb_dir in sorted(d for d in prep_dir.iterdir()
                           if d.is_dir() and is_combo_dir_name(d.name)):
        if (comb_dir / 'core.pdb').exists():
            return str(comb_dir / 'core.pdb')
    return None


def build_environment_consensus(systems: List[str], unimol: UnimolSettings,
                                output_dir: Path) -> Dict:
    """consensus[prep_dir][site name]: the environment atoms shared by every
    substituent of that site (None for vacuum systems or on failure); saved
    to environment_consensus.json."""
    print("\nBuilding environment consensus atoms per system/site...")
    consensus_dict: Dict = {}
    for prep_dir_str in sorted(systems):
        prep_dir = Path(prep_dir_str)
        if not prep_dir.is_dir():
            continue
        site_to_subs = _site_fragments(prep_dir)
        if not site_to_subs:
            continue
        consensus_dict[prep_dir_str] = {}
        core_pdb = _core_pdb(prep_dir)
        if not core_pdb:
            continue
        system_label = _system_label(prep_dir)
        # No environment to define in vacuum
        if 'vac' in system_label.lower() or 'vacuum' in system_label.lower():
            print(f"\n[CONSENSUS] Skipping consensus for vacuum system: {system_label} "
                  "(no environment)")
            for site_name in sorted(site_to_subs.keys()):
                consensus_dict[prep_dir_str][site_name] = None
            continue
        for site_name in sorted(site_to_subs.keys()):
            sub_pdbs = [str(p) for p in sorted(site_to_subs[site_name])]
            try:
                consensus_dict[prep_dir_str][site_name] = build_site_consensus(
                    site_name, sub_pdbs, core_pdb, prep_dir,
                    env_cutoff=unimol.environment_cutoff, system_name=system_label)
            except Exception as e:
                print("  Warning: Failed to build consensus for "
                      f"{prep_dir_str}/{site_name}: {e}")
                consensus_dict[prep_dir_str][site_name] = None

    if not consensus_dict:
        print()
        return consensus_dict
    # Saved for reference/debugging (sets of atom tuples as sorted lists)
    consensus_json = {
        prep: {site: (sorted(list(a) for a in atoms) if atoms is not None else None)
               for site, atoms in sites.items()}
        for prep, sites in consensus_dict.items() if sites}
    if consensus_json:
        consensus_file = output_dir / 'environment_consensus.json'
        with open(consensus_file, 'w') as f:
            json.dump(consensus_json, f, indent=2)
        print(f"  Saved consensus to {consensus_file.name}\n")
    return consensus_dict


def _nsubs_from_graph_info(gi_path: Path) -> List[int]:
    """Substituents per site (older runs only list their sites: count them,
    as compute_unimol_embeddings_and_edges does)."""
    with open(gi_path) as f:
        gi = json.load(f)
    nsubs_per_site = gi.get('nsubs_per_site') or []
    if not nsubs_per_site:
        per_site = collections.Counter(
            s.get('site') for s in (gi.get('sites') or {}).values()
            if s.get('site') is not None)
        nsubs_per_site = [per_site[s] for s in sorted(per_site)]
    return nsubs_per_site


def _prep_dir(run: Dict) -> Path:
    return Path(run["run_dir"]).parent / "prep"


def build_run_inputs(runs: List[Dict], unimol: UnimolSettings, output_dir: Path) -> RunInputs:
    """Embeddings, edges and targets for every run.

    All runs of a system (prep directory) share substituent coordinates, so
    with a per-system structure selection (minimized / unrelaxed) the
    embeddings are built once per system and only the targets (each run's
    variables.py) differ. With 'random' the structure is chosen, embedded
    and recorded per run.
    """
    structure_selection = unimol.structure_selection
    use_system_cache = unimol.per_system
    tracking_file = initialize_representation_tracker(output_dir)
    embed_kwargs = unimol.embedding_kwargs()

    # Persistent on-disk embedding cache: survives separate submissions, so a
    # rerun skips systems it already embedded (system_graph_cache below is
    # in-memory, for this process only).
    embed_cache_dir: Optional[Path] = None
    if unimol.cache_embeddings:
        embed_cache_dir = (Path(unimol.cache_dir) if unimol.cache_dir
                           else Path(output_dir) / 'embeddings_cache')
        embed_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Uni-Mol embedding cache: ENABLED ({embed_cache_dir})")
    else:
        print("  Uni-Mol embedding cache: disabled (set unimol.cache_embeddings: true to enable)")

    print(f"\n{'=' * 60}")
    if structure_selection != 'minimized':
        print(f"Structure selection: {structure_selection}")
        print("Computing representations per-run (system cache disabled)")
        print(f"Tracking: {tracking_file}")
        print(f"{'=' * 60}\n")
    else:
        print(f"Grouping {len(runs)} runs by system and building per-system graphs...")
        print(f"{'=' * 60}")

    system_to_runs: Dict[str, List[int]] = {}
    for run_idx, run in enumerate(runs):
        system_to_runs.setdefault(str(_prep_dir(run)), []).append(run_idx)
    print(f"Found {len(system_to_runs)} unique systems")

    consensus_dict: Dict = {}
    if unimol.use_environment_consensus:
        consensus_dict = build_environment_consensus(list(system_to_runs), unimol, output_dir)

    system_graph_cache = {}   # prep_dir -> (embeddings, edge_index, nsubs_per_site)
    nsubs_by_system = {}      # prep_dir -> nsubs_per_site (for the pairwise weights)
    if use_system_cache:
        n_systems_ok = n_systems_fail = 0
        print(f"\nBuilding per-system graphs ({len(system_to_runs)} systems)...")
        for system_idx, (prep_dir_str, run_indices) in enumerate(sorted(system_to_runs.items())):
            try:
                run_dir = Path(runs[run_indices[0]]["run_dir"])
                gi_path = run_dir / "graph_info.json"
                if not gi_path.exists():
                    n_systems_fail += 1
                    continue
                unimol_emb, edge_idx, nsubs_per_site, _ = compute_unimol_embeddings_and_edges(
                    run_dir, gi_path, structure_type=structure_selection,
                    unrelaxed_system_mappings={}, run_idx=0, consensus_dict=consensus_dict,
                    cache_dir=embed_cache_dir, **embed_kwargs)
                system_graph_cache[prep_dir_str] = (unimol_emb, edge_idx, nsubs_per_site)
                nsubs_by_system[prep_dir_str] = nsubs_per_site
                n_systems_ok += 1
                prep_dir = Path(prep_dir_str)
                display_name = prep_dir.parent.name
                if is_combo_dir_name(display_name):
                    display_name = prep_dir.parent.parent.name + '/' + display_name
                print(f"  [{system_idx + 1}/{len(system_to_runs)}] {display_name}: "
                      f"{len(run_indices)} runs, {unimol_emb.size(0)} subs, {edge_idx.size(1)} "
                      "edges")
            except Exception as e:
                print(f"  System {prep_dir_str}: {e}")
                n_systems_fail += 1
        print(
            f"\nPer-system graph cache ready: {n_systems_ok} systems built, {n_systems_fail} "
            "failed")
    else:
        print("Skipping system cache (using per-run structure selection)")
        print("Building per-system structure cache (for pairwise weights)...")
        for prep_dir_str, run_indices in sorted(system_to_runs.items()):
            try:
                gi_path = Path(runs[run_indices[0]]["run_dir"]) / "graph_info.json"
                if gi_path.exists():
                    nsubs_by_system[prep_dir_str] = _nsubs_from_graph_info(gi_path)
            except Exception:
                pass

    graph_cache = [None] * len(runs)
    n_ok = n_fail = 0
    print(f"\n{'=' * 60}")
    if use_system_cache:
        print("Pre-building per-run targets (using system cache)...")
    else:
        print("Pre-building per-run embeddings and targets...")
        print(f"(structure selection: {structure_selection})")
    print(f"{'=' * 60}")

    for run_idx, run in enumerate(runs):
        try:
            run_dir = Path(run["run_dir"])
            prep_dir = run_dir.parent / "prep"
            prep_dir_str = str(prep_dir)
            gi_path = run_dir / "graph_info.json"
            if not gi_path.exists():
                n_fail += 1
                continue

            if use_system_cache:
                if prep_dir_str not in system_graph_cache:
                    n_fail += 1
                    continue
                unimol_emb, edge_idx, nsubs_per_site = system_graph_cache[prep_dir_str]
            else:
                system_name = _system_label(prep_dir)
                struct_type_used = _select_structure_type_for_run(
                    run_idx, structure_selection, system_name=system_name,
                    skip_patterns=unimol.skip_unrelaxed_for_systems)
                (unimol_emb, edge_idx, nsubs_per_site,
                 rep_source) = compute_unimol_embeddings_and_edges(
                    run_dir, gi_path, structure_type=struct_type_used,
                    unrelaxed_system_mappings={}, run_idx=run_idx,
                    consensus_dict=consensus_dict, cache_dir=embed_cache_dir, **embed_kwargs)
                track_representation_choice(
                    tracking_file, system_name=system_name, run_idx=run_idx,
                    structure_type=struct_type_used, source=rep_source['source'],
                    prep_dir_used=Path(rep_source['prep_dir_used']),
                    explicit_mapping=rep_source['explicit_mapping'])
                # the structure the run's graph was built with sizes its
                # pairwise weights (the structure cache above may be empty)
                nsubs_by_system[prep_dir_str] = nsubs_per_site

            pairs = build_directed_pairs(nsubs_per_site)
            targets = extract_targets_from_variables(run_dir, nsubs_per_site, pairs)
            if targets and edge_idx.size(1) > 0:
                graph_cache[run_idx] = (unimol_emb, edge_idx, targets)
                n_ok += 1
            else:
                n_fail += 1

            if (run_idx + 1) % 100 == 0 or (run_idx + 1) == len(runs):
                pct = 100 * n_ok / (run_idx + 1) if (run_idx + 1) > 0 else 0
                print(f"  {run_idx + 1}/{len(runs)} runs ({n_ok} cached, {n_fail} failed, "
                      f"{pct:.0f}%)...")
        except Exception as e:
            print(f"  Run {run_idx}: {e}")
            n_fail += 1

    print(f"\nRun cache ready: {n_ok} runs cached, {n_fail} failed")
    if not use_system_cache:
        print(f"Representation tracking: {tracking_file}")
    return RunInputs(graph_cache, n_ok, nsubs_by_system, consensus_dict)


def build_pairwise_weights(runs: List[Dict], nsubs_by_system: Dict[str, List[int]],
                           data_hygiene: bool) -> List:
    """Per-run edge confidence weights (population balance, DDG reliability):
    [E] per run, or [E][4] under data hygiene with a bias type's weight zeroed
    where the run didn't measure what sets that target. None where unavailable."""
    print(f"\n{'=' * 60}")
    print("Building per-run pairwise confidence weights...")
    print(f"{'=' * 60}")
    weights_cache = [None] * len(runs)
    n_weights_ok = 0
    kept = np.zeros(4)    # data hygiene: per bias type, edges whose target counts
    edges_total = 0
    for run_idx, run in enumerate(runs):
        try:
            prep_dir_str = str(_prep_dir(run))
            if prep_dir_str not in nsubs_by_system:
                continue      # (and no progress line, as before)
            if "sim_results" in run:
                pairs = build_directed_pairs(nsubs_by_system[prep_dir_str])
                weights = compute_pairwise_confidence_weights(
                    run["sim_results"], nsubs_by_system[prep_dir_str], pairs)
                if weights is not None and data_hygiene:
                    mask = compute_pairwise_measured_mask(run["sim_results"], pairs, weights)
                    weights = [[w * m for m in row] for w, row in zip(weights, mask)]
                    kept += (np.array(weights) > 0).sum(axis=0)
                    edges_total += len(weights)
                if weights is not None:
                    weights_cache[run_idx] = weights
                    n_weights_ok += 1
        except Exception:
            pass   # this run gets no pairwise weighting
        if (run_idx + 1) % 100 == 0 or (run_idx + 1) == len(runs):
            print(f"  {run_idx + 1}/{len(runs)} runs ({n_weights_ok} with pairwise weights)...")

    print(f"\nPairwise confidence weights ready: {n_weights_ok} runs with weights")
    if n_weights_ok > 0:
        print("  Pairwise AWR weighting: ENABLED")
    if data_hygiene:
        frac = kept / max(edges_total, 1)
        print(f"  Data hygiene: targets kept only where measured (of {edges_total} edges): "
              f"linear {frac[0]:.1%}, quadratic {frac[1]:.1%}, skew {frac[2]:.1%}, "
              f"end {frac[3]:.1%}; {len(runs) - n_weights_ok} runs without usable "
              "populations contribute nothing")
    return weights_cache
