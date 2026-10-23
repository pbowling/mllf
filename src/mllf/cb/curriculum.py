"""Curriculum-stage filtering and stratified combo sampling.

Pure filtering/sampling helpers that select which combo directories a given
curriculum stage or training epoch trains on, plus a small system-name ->
integer-index map used by ``ContextBaseline``'s per-system embedding.

Moved here from ``training/workflow.py`` as part of making the online CB
training driver importable as a library (see ``mllf.cli.train_workflow``).
"""
from pathlib import Path
from typing import Dict, List, Optional
import logging
import re

import numpy as np
import torch


def build_system_index(systems_cfg: List[Dict]) -> Dict[str, int]:
    """Stable system-name -> integer index map for ``ContextBaseline``'s
    system embedding, built once from the config's ``systems:`` list order.

    Index 0 is reserved for an 'unknown' fallback (a combo whose system name
    can't be resolved via ``combo_system_map``), so real systems start at 1
    and a resume from an older checkpoint with a different systems: list
    still loads (unknown/new names just fall back to the same shared slot
    rather than erroring).
    """
    idx = {'__unknown__': 0}
    for i, s in enumerate(systems_cfg):
        idx[s.get('name', f'system_{i}')] = i + 1
    return idx


def filter_combos_by_curriculum(combos: List[Path],
                                 min_sites: int, max_sites: int,
                                 min_subs_per_site: int = 1,
                                 max_subs_per_site: Optional[int] = None,
                                 whitelist: Optional[List[str]] = None) -> List[Path]:
    """Keep combos whose encoded site and substituent counts meet the limits.

    If provided, ``whitelist`` further restricts results to matching names.
    """
    filtered = []
    for combo_path in combos:
        combo_name = Path(combo_path).name
        parts = combo_name.split('__')
        subs_per_site: Dict[int, int] = {}
        for part in parts:
            if 'site' not in part:
                continue
            site_match = re.search(r'site(\d+)_subs_([\d_]+)', part)
            if site_match:
                site_id = int(site_match.group(1))
                sub_ids = site_match.group(2).split('_')
                subs_per_site[site_id] = len(sub_ids)
        if not subs_per_site:
            continue
        num_sites = len(subs_per_site)
        site_counts = list(subs_per_site.values())
        if not (min_sites <= num_sites <= max_sites):
            continue
        if min(site_counts) < min_subs_per_site:
            continue
        if max_subs_per_site is not None and max(site_counts) > max_subs_per_site:
            continue
        if whitelist is not None and combo_name not in whitelist:
            continue
        filtered.append(combo_path)
    return filtered


def filter_combos_by_transitions(combos: List, min_transitions: int,
                                  lookback: int = 3) -> List:
    """Keep combos whose recent results meet the per-site transition floor."""
    qualifying = []
    for combo_dir in combos:
        combo_path = Path(combo_dir)
        run_dirs = sorted(
            combo_path.glob('run_*'),
            key=lambda p: int(p.name.split('_')[1]) if p.name.split('_')[1].isdigit() else -1
        )
        recent = [rd for rd in reversed(run_dirs) if (rd / 'epoch_results.pt').exists()][:lookback]
        qualified = False
        for rd in recent:
            try:
                sd = torch.load(rd / 'epoch_results.pt', weights_only=False)
                trans = sd.get('transitions', {})
                tv = list(trans.values()) if isinstance(trans, dict) else list(trans)
                if tv and min(tv) >= min_transitions:
                    qualified = True
                    break
            except Exception:
                pass
        if qualified:
            qualifying.append(combo_dir)
    return qualifying


def apply_stage_combo_filter(train_combos, stage, curriculum_config, stage_idx, seed,
                               apply_transition_filter: bool = True,
                               combo_system_map: Optional[Dict[str, Dict]] = None):
    """Apply curriculum + optional transition filter for a given stage config.

    A stage may list ``systems: [name, ...]`` to train on only those systems
    (e.g. a solvent leg first, then add the protein leg in a later stage);
    needs ``combo_system_map`` (combo path -> system config).
    """
    if stage.get('systems'):
        if combo_system_map is None:
            raise ValueError("stage 'systems' needs combo_system_map")
        keep = set(stage['systems'])
        train_combos = [c for c in train_combos
                        if combo_system_map.get(c, {}).get('name') in keep]
        print(f"  Stage systems: {sorted(keep)} ({len(train_combos)} combos)")
    combos = filter_combos_by_curriculum(
        train_combos,
        min_sites=stage['min_sites'],
        max_sites=stage['max_sites'],
        min_subs_per_site=stage.get('min_subs_per_site', 1),
        max_subs_per_site=stage.get('max_subs_per_site', None),
        whitelist=stage.get('combo_whitelist'),
    )
    print(f"  Filtered to {len(combos)} combos for this stage")

    trans_filter = stage.get('min_transitions_for_stage') if apply_transition_filter else None
    if trans_filter is not None:
        lookback = stage.get('transition_filter_lookback', 3)
        before = len(combos)
        combos = filter_combos_by_transitions(combos, trans_filter, lookback)
        print(f"  Transition filter (≥{trans_filter} trans, lookback={lookback}): "
              f"{len(combos)}/{before} combos qualify")
        if combos:
            names = [Path(c).name for c in combos]
            print(f"  Qualifying combos: {', '.join(names)}")

    return combos


def sample_combos_stratified(system_combo_pools: Dict[str, List[str]],
                              n_total: int = 500,
                              min_per_system: int = 5,
                              seed: Optional[int] = None) -> List[str]:
    """Sample combos with uniform minimum allocation per system, then proportional.
    
    Guarantees each system gets at least min_per_system combos, then allocates
    remaining combos proportionally to system pool size.
    
    Args:
        system_combo_pools: Dict mapping system_name -> list of combo paths
        n_total: Total number of combos to sample
        min_per_system: Minimum combos per system
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled combo paths
    """
    rng = np.random.RandomState(seed)
    n_systems = len(system_combo_pools)
    
    # Check feasibility
    min_total = min_per_system * n_systems
    if min_total > n_total:
        logging.warning(f"n_total={n_total} < min required {min_total}; "
                       f"reducing min_per_system to {n_total // n_systems}")
        min_per_system = max(1, n_total // n_systems)
    
    # Allocate minimum to each system first
    allocated_per_system = {name: min_per_system for name in system_combo_pools}
    remaining = n_total - (min_per_system * n_systems)
    
    # Get system sizes
    system_sizes = {name: len(combos) for name, combos in system_combo_pools.items()}
    total_size = sum(system_sizes.values())
    
    # Allocate remaining proportionally
    if total_size > 0 and remaining > 0:
        for system_name in allocated_per_system:
            if total_size > 0:
                extra = int(remaining * system_sizes[system_name] / total_size)
                allocated_per_system[system_name] += extra
    
    # Collect sampled combos
    sampled_combos = []
    for system_name, n_samples in allocated_per_system.items():
        combo_list = system_combo_pools[system_name]
        # Cap at actual pool size
        n_samples = min(n_samples, len(combo_list))
        if n_samples > 0:
            sampled = rng.choice(combo_list, size=n_samples, replace=False).tolist()
            sampled_combos.extend(sampled)
    
    # Final cap to ensure exactly n_total (may be slightly less due to rounding)
    if len(sampled_combos) > n_total:
        sampled_combos = sampled_combos[:n_total]
    
    return sampled_combos
