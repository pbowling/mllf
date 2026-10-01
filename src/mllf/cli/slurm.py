"""SLURM job submission for one online-training epoch's simulation.

Distinct from ``mllf.cli.sim.run_simulation_batch`` (which runs a manifest of
combos concurrently via an arbitrary shell command, for the lighter
``mllf.cli.workflow.run_from_config`` pipeline): this module writes and
submits a single per-combo, per-epoch ``run_epoch.sh`` SLURM script for the
online CB-policy training driver (``mllf.cli.train_workflow`` /
``mllf.cb.reinforce_trainer``), which polls job completion itself rather than
blocking on submission.

Moved here from ``training/workflow.py`` as part of making the online CB
training driver importable as a library.

All SLURM specifics (partition, GRES, wall time, environment module) come
from the config's ``slurm:`` block and are optional except ``time``, which
has a generic 1-hour default. ``partition``/``gres``/``module`` are omitted
from the generated script entirely when not set, so this works unmodified on
a cluster with no GPUs, no environment-modules system, or a account-level
default partition -- nothing here assumes any particular cluster's resource
names.
"""
from pathlib import Path
from typing import Dict
import subprocess

# The only default with no cluster-specific meaning; everything else in the
# `slurm:` config block is opt-in (see submit_epoch_simulation).
_SLURM_TIME = '01:00:00'


def submit_epoch_simulation(
    combo_path: str,
    epoch_dir: str,
    config: Dict,
    solvent_state: str = 'solv',
) -> str:
    """Write run_epoch.sh to *epoch_dir* and submit it via sbatch.

    Reads the config's ``slurm:`` block:
      - ``partition``: passed as ``#SBATCH -p``; omitted (cluster/account
        default partition applies) if not set.
      - ``gres``: passed as ``#SBATCH --gres``; omitted (no GPU requested)
        if not set.
      - ``time``: passed as ``#SBATCH --time``; defaults to '01:00:00'.
      - ``module``: passed to ``module load``; the line is omitted entirely
        (e.g. for clusters without an environment-modules system, or where
        the simulator is already on PATH) if not set.

    Returns the SLURM job ID string (e.g. '12345').
    """
    combo_path = Path(combo_path)
    epoch_dir = Path(epoch_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    slurm_cfg = config.get('slurm', {})
    partition = slurm_cfg.get('partition')
    gres = slurm_cfg.get('gres')
    time_limit = slurm_cfg.get('time', _SLURM_TIME)
    module = slurm_cfg.get('module')

    epoch_label = epoch_dir.name  # e.g. 'run_007'
    job_name = f"{combo_path.name}_{epoch_label}"
    slurm_log = epoch_dir / f"{job_name}.%j.out"
    vars_relative = (epoch_dir / 'variables.py').relative_to(combo_path)
    out_relative = epoch_dir.relative_to(combo_path)

    sbatch_directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={slurm_log}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
    ]
    if partition:
        sbatch_directives.append(f"#SBATCH -p {partition}")
    if gres:
        sbatch_directives.append(f"#SBATCH --gres={gres}")
    sbatch_directives.append("#SBATCH --export=ALL")
    sbatch_directives.append(f"#SBATCH --time={time_limit}")

    module_load_block = f"module load {module}\n\n" if module else ""

    script_content = f"""#!/bin/bash
{chr(10).join(sbatch_directives)}

{module_load_block}echo $(hostname)

# Change to combo directory and run with epoch-specific variables
cd {combo_path}
python3 msld_flat.py --vars-file {vars_relative} --out-dir {out_relative} > {out_relative}/output.out 2>&1
"""
    script_path = epoch_dir / 'run_epoch.sh'
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    result = subprocess.run(
        ['sbatch', str(script_path)],
        cwd=str(epoch_dir),
        capture_output=True, text=True, check=True,
    )
    job_id = result.stdout.strip().split()[-1]
    return job_id
