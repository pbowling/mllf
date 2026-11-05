"""Environment atoms shared by every substituent at a site (env_input "dual")."""

import logging
from pathlib import Path
from typing import List, Optional

from mllf.cb.unimol_representation import build_environment_consensus


log = logging.getLogger(__name__)


def build_site_consensus(
    site_name: str,
    sub_pdbs: List[str],
    core_pdb: str,
    prep_dir: Path,
    env_cutoff: float = 8.0,
    custom_search_paths: Optional[List[str]] = None,
    system_name: Optional[str] = None,
) -> Optional[set]:
    """Select nearby environment atoms using the shared core structure.

    The selection is based on ``core_pdb`` so every substituent at the site
    uses the same environment context.

    Args:
        site_name: Site label for status messages.
        sub_pdbs: Currently unused; retained for call-site compatibility.
        core_pdb: Core structure used to select environment atoms.
        prep_dir: Directory containing environment structures.
        env_cutoff: Maximum core-to-environment distance in angstroms.
        custom_search_paths: Additional paths to search for structures.
        system_name: Optional system label for status messages.

    Returns:
        Selected atom identifiers, or None when none are found.
    """
    system_label = f" [{system_name}]" if system_name else ""
    log.debug(f"\nBuilding {site_name} consensus from core{system_label}")

    consensus = build_environment_consensus(
        core_pdb=core_pdb,
        prep_dir=prep_dir,
        env_cutoff=env_cutoff,
        custom_search_paths=custom_search_paths,
    )

    if consensus:
        log.debug(f"{site_name}: Found {len(consensus)} consensus atoms{system_label}")
        return consensus
    else:
        log.debug(
            f"{site_name}: No consensus atoms found (environment may not "
            f"exist){system_label}")
        return None
