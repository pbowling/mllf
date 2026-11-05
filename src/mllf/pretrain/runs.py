"""Loading collected pretraining runs (``collect_pretraining_data`` output)."""
import json
import logging
from pathlib import Path
from typing import Dict, List

from mllf.file_handling.generate_combinations import is_combo_dir_name


log = logging.getLogger(__name__)


def load_pretraining_runs(pretraining_dir: Path) -> List[Dict]:
    """Load all collected pretraining runs.

    Supports both flat and nested directory structures:
    - Flat: pretraining_dir/run_001/, pretraining_dir/run_002/, ...
    - Nested: pretraining_dir/combo_XXX/run_001/, pretraining_dir/combo_XXX/run_002/, ...

    Args:
        pretraining_dir: Directory with collected run data

    Returns:
        List of dicts with run_dir, metadata, sim_results for each run
    """
    runs = []

    def _load_run(run_dir: Path):
        """Helper to load a single run directory."""
        # Load metadata
        metadata_file = run_dir / "metadata.json"
        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Skip if simulation didn't terminate normally
        if not metadata.get("terminated_normally", False):
            log.info(f"  Skipping {run_dir.name}: did not terminate normally")
            return None

        # Load simulation results
        results_file = run_dir / "simulation_results.json"
        if not results_file.exists():
            log.info(f"  Skipping {run_dir.name}: no simulation_results.json")
            return None

        with open(results_file, 'r') as f:
            sim_results = json.load(f)

        return {
            "run_dir": run_dir,
            "source_dir": metadata.get("source_run_dir"),
            "metadata": metadata,
            "sim_results": sim_results,
        }

    # Iterate through top-level directories
    for entry in sorted(pretraining_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Check if this is a run directory (has metadata.json)
        if (entry / "metadata.json").exists():
            # Flat structure: entry is a run directory
            run_data = _load_run(entry)
            if run_data:
                runs.append(run_data)
        else:
            # Nested structure: entry might contain run subdirectories
            # Look one level deeper for run directories
            for subentry in sorted(entry.iterdir()):
                if subentry.is_dir() and (subentry / "metadata.json").exists():
                    run_data = _load_run(subentry)
                    if run_data:
                        runs.append(run_data)

    return runs


def read_manifest(path) -> List[str]:
    """Pretraining directories listed in a manifest: a JSON list, a JSON
    object with a 'directories' list, or one path per line."""
    content = Path(path).read_text().strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return [line.strip() for line in content.split('\n') if line.strip()]
    if isinstance(data, dict) and 'directories' in data:
        return data['directories']
    if isinstance(data, list):
        return data
    raise ValueError("Manifest file must contain a 'directories' key or be a JSON array")


def system_label(pretraining_dir: Path) -> str:
    """The system a pretraining directory belongs to: its name, or its
    parent's for a comb_* directory."""
    label = pretraining_dir.name
    return pretraining_dir.parent.name if is_combo_dir_name(label) else label


def load_runs_from_dirs(pretraining_dirs: List) -> List[Dict]:
    """Every run of every directory, each tagged with ``system_label``
    (matched by ``PretrainOptions.system_emphasis``)."""
    all_runs = []
    for d in pretraining_dirs:
        pretrain_dir = Path(d)
        log.info(f"\nLoading from {pretrain_dir}...")
        runs = load_pretraining_runs(pretrain_dir)
        log.info(f"  Loaded {len(runs)} runs")
        label = system_label(pretrain_dir)
        for run in runs:
            run['system_label'] = label
        all_runs.extend(runs)
    return all_runs
