"""Graph + Uni-Mol embedding construction for the online-training loop.

Builds the per-combo PyG-like ``Data`` object (dual 1024D Uni-Mol node
embeddings, directed within-site edges) that ``mllf.cb.reinforce_trainer``'s
``train_epoch`` samples policy actions on.

Moved here from ``training/workflow.py`` as part of making the online CB
training driver importable as a library (see ``mllf.cli.train_workflow``).
"""
from pathlib import Path
from typing import Dict, Optional
import re
import warnings

import numpy as np
import torch

from mllf.cb.graph_utils import build_directed_pairs
from mllf.cb.unimol_representation import (
    build_environment_consensus,
    get_substituent_dual_embeddings,
)
from mllf.file_handling.read_rtf import parse_rtf_dir


def build_graph_and_data(combo_dir: str, system_config: Dict,
                          global_config: Dict, device: str = 'cpu'):
    """Build graph and compute Uni-Mol embeddings for the ligand system.

    Constructs the full ligand (core + substituent + ref subs + environment)
    and computes 512D Uni-Mol representation using pretrained model.

    Uses system_config['solvent_state'] to determine which PDB files to load
    (protein, solvent, or neither for gas-phase).
    """
    combo_path = Path(combo_dir)

    rtf_results = parse_rtf_dir(combo_dir)
    rtf_dir = combo_dir
    if not rtf_results:
        prep_dir = combo_path / 'prep'
        if prep_dir.exists():
            rtf_results = parse_rtf_dir(str(prep_dir))
            rtf_dir = str(prep_dir)
    if not rtf_results:
        raise FileNotFoundError(
            f"No RTF files found in {combo_dir} or {combo_dir}/prep"
        )

    solvent_state = system_config.get('solvent_state', None)

    # Persistent Uni-Mol embedding cache (unimol.cache_embeddings / unimol.cache_dir
    # in the config, same convention as pretrain_policy.py). Keyed by each
    # combo's own (site/sub PDB path, core.pdb path, prep dir) — NOT
    # canonicalized back to training/systems/<name>/prep/, because core.pdb
    # can be mutated per-combo for single-site combinations
    # (augment_core_with_excluded_sub1 merges in excluded sites' sub1 atoms),
    # so a system-wide canonical core.pdb would silently compute the wrong
    # embedding for those combos. Still gives a real win: any combo
    # re-selected across epochs (validation combos every epoch; resampled
    # training combos) hits the cache instead of recomputing, and the cache
    # survives process restarts.
    unimol_cfg = global_config.get('unimol', {})
    embed_cache_dir: Optional[Path] = None
    if unimol_cfg.get('cache_embeddings', False):
        _cd = unimol_cfg.get('cache_dir')
        embed_cache_dir = Path(_cd) if _cd else Path(global_config.get('output', {}).get('base_dir', 'checkpoints')) / 'embeddings_cache'

    # For Uni-Mol, we need to construct the full ligand geometry
    # Find the substituent and core PDB files
    pdb_dir = combo_path / 'prep'
    if not pdb_dir.exists():
        pdb_dir = combo_path

    # Parse core PDB
    core_pdb_candidates = list(pdb_dir.glob('core*.pdb'))
    if not core_pdb_candidates:
        raise FileNotFoundError(f"No core PDB found in {pdb_dir}")
    core_pdb = str(core_pdb_candidates[0])

    # Parse first substituent PDB (for now, use site1_sub1)
    sub_pdb_candidates = list(pdb_dir.glob('site*_sub*.pdb'))
    if not sub_pdb_candidates:
        raise FileNotFoundError(f"No substituent PDB found in {pdb_dir}")
    sub_pdb = str(sub_pdb_candidates[0])

    # Note: Environment loading now handled by get_substituent_unimol_with_environment()
    # which uses intelligent fallback chain (minimized.pdb preferred, with distance filtering)
    # and will print debug info about what environment files are loaded

    # Compute Uni-Mol embeddings for each substituent
    # Combo directories contain ALL substituents at each site (even if only one is "active")
    # We compute embeddings for ALL of them to match the simulator's expectations
    sub_pdb_files = sorted(list(pdb_dir.glob('site*_sub*.pdb')))
    if not sub_pdb_files:
        raise FileNotFoundError(f"No substituent PDB files found in {pdb_dir}")
    
    # Parse all substituents at each site
    site_to_sub_map = {}  # site_idx -> dict of {sub_idx: pdb_path}
    for pdb_file in sub_pdb_files:
        match = re.search(r'site(\d+)_sub(\d+)', pdb_file.name)
        if match:
            site_idx = int(match.group(1)) - 1  # Convert to 0-indexed
            sub_idx = int(match.group(2))  # Keep 1-indexed for internal consistency
            site_to_sub_map.setdefault(site_idx, {})[sub_idx] = str(pdb_file)
    
    if not site_to_sub_map:
        raise ValueError(f"Could not extract site/substituent structure from {combo_dir}")
    
    # Build nsubs_per_site list: count of substituents at each site
    max_site = max(site_to_sub_map.keys())
    nsubs_per_site = [len(site_to_sub_map.get(i, {})) for i in range(max_site + 1)]
    
    # Extract yaml-configured custom PDB paths (if any)
    # Support protein_pdbs (list) and solvent_pdb (str) from yaml config
    custom_search_paths = []
    if 'protein_pdbs' in system_config:
        protein_pdbs = system_config['protein_pdbs']
        if isinstance(protein_pdbs, list):
            custom_search_paths.extend(protein_pdbs)
        else:
            custom_search_paths.append(protein_pdbs)
    if 'solvent_pdb' in system_config:
        solvent_pdb = system_config['solvent_pdb']
        if isinstance(solvent_pdb, list):
            custom_search_paths.extend(solvent_pdb)
        else:
            custom_search_paths.append(solvent_pdb)
    
    # Compute Uni-Mol embeddings for each substituent at each site
    # Use dual embeddings [ligand_only (512D) + full (512D)] = 1024D
    # This matches pretraining configuration: edge input = [diff_ligand, mean_full]
    #
    # Build a per-site consensus atom set from the CORE scaffold (identical
    # for every substituent at a site, since the core doesn't change when
    # swapping substituents). This matches pretrain_policy.py's approach and
    # ensures the environment half of the embedding is consistent across
    # subs at a site, exactly as during pretraining.
    combo_name = combo_path.name
    print(f"  Computing dual Uni-Mol embeddings for {combo_name} (system: {system_config.get('name', 'unknown')}, solvent_state: {solvent_state})")
    if custom_search_paths:
        print(f"    Custom environment PDB paths: {custom_search_paths}")

    site_consensus = {}  # site_idx -> consensus_atoms set
    if solvent_state and solvent_state.lower() != 'gas':
        for site_idx in sorted(site_to_sub_map.keys()):
            try:
                consensus_atoms = build_environment_consensus(
                    core_pdb=core_pdb,
                    prep_dir=pdb_dir,
                    env_cutoff=8.0,
                    custom_search_paths=custom_search_paths if custom_search_paths else None,
                )
                if consensus_atoms:
                    site_consensus[site_idx] = consensus_atoms
                    print(f"    Site {site_idx}: consensus contains {len(consensus_atoms)} atoms")
                else:
                    print(f"    Site {site_idx}: no consensus found, using distance-based fallback")
            except Exception as e:
                print(f"    Site {site_idx}: consensus building failed ({e}), using distance-based fallback")
    else:
        print(f"    Skipping consensus building for gas-phase system (no environment)")

    unimol_embeddings = []
    for site_idx in sorted(site_to_sub_map.keys()):
        # Get all substituents at this site, sorted by sub_idx
        sub_dict = site_to_sub_map[site_idx]
        site_consensus_atoms = site_consensus.get(site_idx)

        for sub_idx in sorted(sub_dict.keys()):
            sub_pdb_path = sub_dict[sub_idx]
            try:
                # Use dual embedding mode: [ligand_only (512D) + full (512D)]
                # Skip minimized.pdb since online training uses unrelaxed structures only
                emb_ligand, emb_full = get_substituent_dual_embeddings(
                    sub_pdb=sub_pdb_path,
                    core_pdb=core_pdb,
                    prep_dir=pdb_dir,
                    include_other_sites=False,
                    env_cutoff=8.0,
                    atom_limit=256,
                    use_cuda=device == 'cuda',
                    custom_search_paths=custom_search_paths if custom_search_paths else None,
                    skip_minimized=True,  # Online training uses unrelaxed structures only
                    consensus_atoms=site_consensus_atoms,  # Filter by site consensus
                    cache_dir=embed_cache_dir,
                    save_cache=(embed_cache_dir is not None),
                )
                # Concatenate dual embeddings: [512 ligand + 512 full] = 1024D
                sub_emb = np.concatenate([emb_ligand, emb_full])  # [1024]
                if sub_emb.ndim == 1:
                    sub_emb = sub_emb[np.newaxis, :]  # [1024] -> [1, 1024]
                unimol_embeddings.append(sub_emb)
            except Exception as e:
                warnings.warn(f"Failed to compute dual Uni-Mol embedding for {Path(sub_pdb_path).name}: {e}")
                # Create a zero embedding as fallback (1024D)
                unimol_embeddings.append(np.zeros((1, 1024), dtype=np.float32))
    
    if unimol_embeddings:
        unimol_emb_array = np.vstack(unimol_embeddings)  # [N, 1024] where N = total number of subs
    else:
        raise ValueError(f"No Uni-Mol embeddings computed for {combo_dir}")
    
    # Build directed edges within each site (using the build_directed_pairs pattern)
    pairs = build_directed_pairs(nsubs_per_site)
    
    # Create one edge per directed pair.
    # Policy._forward_edges outputs [E, 4] where all 4 bias types are predicted per edge.
    # This matches the pretraining analysis script which processes per-edge predictions directly.
    # Edge type is set to None (no routing) so all 4 MLPs process each edge, producing [E, 2*4] output.
    src_list = []
    dst_list = []
    
    for (src, dst) in pairs:
        src_list.append(src)
        dst_list.append(dst)
    
    # Convert to edge_index tensor
    if src_list:
        edge_index = torch.LongTensor([src_list, dst_list])  # [2, E]
        edge_type = None  # No routing; all MLPs process all edges (legacy path)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = None
    
    # Create site_index: which site each substituent belongs to
    site_index = []
    for site_idx, nsubs in enumerate(nsubs_per_site):
        site_index.extend([site_idx] * nsubs)
    site_index = torch.LongTensor(site_index)
    
    # Prepare embedding as node features (now 1024D for dual embeddings)
    unimol_emb_t = torch.from_numpy(unimol_emb_array).float().to(device)  # [N, 1024]
    
    # Create a simple data-like object with required attributes
    class UnimolData:
        def __init__(self, x, edge_index, edge_type, edge_attr, site_index):
            self.x = x
            self.edge_index = edge_index.to(device)
            self.edge_type = edge_type if edge_type is None else edge_type.to(device)
            self.edge_attr = edge_attr
            self.site_index = site_index.to(device)
    
    edge_attr = torch.zeros((edge_index.shape[1], 1), dtype=torch.float32) if edge_index.shape[1] > 0 else torch.zeros((0, 1), dtype=torch.float32)
    data = UnimolData(unimol_emb_t, edge_index, edge_type, edge_attr, site_index)
    
    extras = {
        'relation_names': ['linear', 'quadratic', 'skew', 'end'],
        'base_relation_map': {
            'linear': 'linear',
            'quadratic': 'quadratic',
            'skew': 'skew',
            'end': 'end',
        },
        'nsubs_per_site': nsubs_per_site,
    }

    return data, extras
