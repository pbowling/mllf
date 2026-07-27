"""Consensus environment atom filtering for pretraining.

Builds consensus atom sets for each site to prevent odd environment noise
in the mean embeddings used by the policy. By using only atoms that appear
across ALL substituents at a site, we ensure more stable environmental context.

Key Functions:
- build_site_consensus(): Build consensus atoms for all subs at a site
- save_consensus_atoms(): Save consensus to disk
- load_consensus_atoms(): Load consensus from disk
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import warnings

from mllf.cb.unimol_representation import (
    build_environment_consensus,
    get_substituent_unimol_with_environment,
    get_substituent_dual_embeddings,
)


def build_site_consensus(
    site_name: str,
    sub_pdbs: List[str],
    core_pdb: str,
    prep_dir: Path,
    env_cutoff: float = 8.0,
    custom_search_paths: Optional[List[str]] = None,
    system_name: Optional[str] = None,
) -> Optional[set]:
    """Build consensus environment atoms for a site using the core scaffold.
    
    Uses the core scaffold to determine which environment atoms should be included
    for all substituents at a site. This ensures consistent environment representation
    across pretraining, analysis, and online training.
    
    **Strategy:**
    1. Use core scaffold coordinates to identify environment atoms within env_cutoff
    2. Return this consistent set (capped at 256 atoms, keeping closest)
    3. Use this same set when computing embeddings for all subs at the site
    
    This approach is superior to sub-based consensus because:
    - The core is the same for all substituents (minimized or not)
    - Environment doesn't change when swapping substituents
    - Consistent across pretraining, analysis, and online training
    - Simpler and faster (no need to iterate over all subs)
    
    **Example:**
    ```python
    # Build consensus for site1 using its core
    sub_pdbs = [
        'pretraining/SYSTEM/comb_ID/prep/site1_sub1_frag.pdb',
        'pretraining/SYSTEM/comb_ID/prep/site1_sub2_frag.pdb',
        ...
    ]
    core_pdb = 'pretraining/SYSTEM/comb_ID/prep/core.pdb'
    prep_dir = Path('pretraining/SYSTEM/comb_ID/prep')
    
    consensus = build_site_consensus(
        'site1', sub_pdbs, core_pdb, prep_dir,
        env_cutoff=8.0,
        system_name='12benz_solvent_group1',  # For logging
    )
    
    # Then use this when computing embeddings:
    emb = get_substituent_unimol_with_environment(
        sub_pdbs[0], core_pdb, prep_dir,
        consensus_atoms=consensus,  # Filter by consensus
    )
    ```
    
    Args:
        site_name: Name of site for logging (e.g., 'site1', 'site2')
        sub_pdbs: List of paths to substituent PDB files for all subs at site
            (used only for logging/context; core is used for consensus building)
        core_pdb: Path to core PDB (used to define environment)
        prep_dir: Prep directory containing environment files
        env_cutoff: Distance cutoff (Å) for identifying relevant environment (default: 8.0)
        custom_search_paths: Optional custom PDB search paths (from yaml config)
        system_name: Optional system name for logging (e.g., '12benz_solvent_group1')
        
    Returns:
        Set of (file_index, resnum, chain, atomname) tuples (max 256), or None if
        no consensus found
    """
    system_label = f" [{system_name}]" if system_name else ""
    print(f"\n[CONSENSUS] Building {site_name} consensus from core{system_label}")
    
    consensus = build_environment_consensus(
        core_pdb=core_pdb,
        prep_dir=prep_dir,
        env_cutoff=env_cutoff,
        custom_search_paths=custom_search_paths,
    )
    
    if consensus:
        print(f"[CONSENSUS] {site_name}: Found {len(consensus)} consensus atoms{system_label}")
        return consensus
    else:
        print(f"[CONSENSUS] {site_name}: No consensus atoms found (environment may not exist){system_label}")
        return None


def save_consensus_atoms(
    consensus_dict: Dict[str, set],
    run_dir: Path,
    filename: str = "environment_consensus.json",
) -> Path:
    """Save consensus atoms to disk for reference and debugging.
    
    Stores consensus atom identifiers as JSON for inspection and reuse.
    Each consensus atom is stored as [file_index, resnum, chain, atomname].
    
    Args:
        consensus_dict: Dict mapping site_name (e.g., 'site1') to consensus set
        run_dir: Directory to save consensus file
        filename: Filename for consensus file (default: "environment_consensus.json")
        
    Returns:
        Path where consensus was saved
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert sets to serializable format
    consensus_serializable = {}
    for site_name, atom_set in consensus_dict.items():
        if atom_set:
            # Convert (resnum, chain, atomname) tuples to lists
            consensus_serializable[site_name] = sorted([list(atom_id) for atom_id in atom_set])
        else:
            consensus_serializable[site_name] = None
    
    save_path = run_dir / filename
    with open(save_path, 'w') as f:
        json.dump(consensus_serializable, f, indent=2)
    
    print(f"[CONSENSUS] Saved consensus atoms to {save_path}")
    return save_path


def load_consensus_atoms(
    run_dir: Path,
    filename: str = "environment_consensus.json",
) -> Dict[str, Optional[set]]:
    """Load consensus atoms from disk.
    
    Args:
        run_dir: Directory containing consensus file
        filename: Filename of consensus file
        
    Returns:
        Dict mapping site_name to consensus set (or None if no consensus)
    """
    load_path = run_dir / filename
    
    if not load_path.exists():
        print(f"[CONSENSUS] No consensus file found at {load_path}")
        return {}
    
    with open(load_path, 'r') as f:
        consensus_data = json.load(f)
    
    # Convert lists back to sets of tuples
    consensus_dict = {}
    for site_name, atom_list in consensus_data.items():
        if atom_list is not None:
            consensus_dict[site_name] = set((tuple(atom_id) for atom_id in atom_list))
        else:
            consensus_dict[site_name] = None
    
    print(f"[CONSENSUS] Loaded consensus atoms from {load_path}")
    return consensus_dict


def compute_run_embeddings_with_consensus(
    run_dir: Path,
    consensus_dict: Dict[str, Optional[set]],
    use_dual_embeddings: bool = False,
    env_cutoff: float = 8.0,
    use_cuda: bool = False,
) -> Dict[str, Dict]:
    """Compute all embeddings for a run using consensus atom filtering.
    
    This is a helper function showing how to integrate consensus filtering
    into the embedding computation workflow.
    
    **Expected run_dir structure:**
    ```
    run_dir/
        site1_sub1_frag.pdb
        site1_sub2_frag.pdb
        ...
        core.pdb
        minimized.pdb (or other environment files)
    ```
    
    Args:
        run_dir: Directory containing substituent and environment PDB files
        consensus_dict: Dict from load_consensus_atoms() or build_site_consensus()
        use_dual_embeddings: If True, compute both ligand-only and ligand+env (default: False)
        env_cutoff: Distance cutoff for environment filtering (default: 8.0)
        use_cuda: Whether to use GPU (default: False)
        
    Returns:
        Dict mapping sub_pdb_name to embedding(s)
    """
    run_dir = Path(run_dir)
    core_pdb = run_dir / "core.pdb"
    
    if not core_pdb.exists():
        raise FileNotFoundError(f"Core PDB not found in {run_dir}")
    
    results = {}
    sub_pdbs = sorted(run_dir.glob("site*_sub*_frag.pdb"))
    
    for sub_pdb in sub_pdbs:
        # Extract site number to get consensus for this sub
        site_str = sub_pdb.stem.split('_')[0]  # e.g., "site1"
        consensus = consensus_dict.get(site_str)
        
        print(f"\nComputing embedding for {sub_pdb.name}")
        print(f"  Using consensus: {site_str} ({len(consensus)} atoms if consensus provided)" if consensus else "  No consensus")
        
        try:
            if use_dual_embeddings:
                emb_lig, emb_env = get_substituent_dual_embeddings(
                    str(sub_pdb),
                    str(core_pdb),
                    run_dir,
                    env_cutoff=env_cutoff,
                    use_cuda=use_cuda,
                    consensus_atoms=consensus,
                )
                results[sub_pdb.name] = {'ligand_only': emb_lig, 'with_environment': emb_env}
            else:
                emb = get_substituent_unimol_with_environment(
                    str(sub_pdb),
                    str(core_pdb),
                    run_dir,
                    env_cutoff=env_cutoff,
                    use_cuda=use_cuda,
                    consensus_atoms=consensus,
                )
                results[sub_pdb.name] = {'embedding': emb}
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results[sub_pdb.name] = {'error': str(e)}
    
    return results
