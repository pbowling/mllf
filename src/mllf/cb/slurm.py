"""SLURM job submission for one online-training epoch's simulation.

Writes and submits a per-combo, per-epoch ``run_epoch.sh`` for the online
training driver, which polls job completion itself rather than blocking on
submission. The script's options come from the config's ``slurm:`` block
(any sbatch option, node exclusion, modules and setup commands); see
``mllf.slurm``.
"""
import subprocess
from pathlib import Path
from typing import Dict

from mllf.slurm import job_script, quote


def submit_epoch_simulation(
    combo_path: str,
    epoch_dir: str,
    config: Dict,
    solvent_state: str = 'solv',
) -> str:
    """Write run_epoch.sh to *epoch_dir* and submit it via sbatch.

    The ``#SBATCH`` options, ``module load`` and setup lines come from
    ``config['slurm']`` (``mllf.slurm``). Returns the SLURM job ID string
    (e.g. '12345').
    """
    combo_path = Path(combo_path)
    epoch_dir = Path(epoch_dir)
    epoch_dir.mkdir(parents=True, exist_ok=True)

    epoch_label = epoch_dir.name  # e.g. 'run_007'
    job_name = f"{combo_path.name}_{epoch_label}"
    slurm_log = epoch_dir / f"{job_name}.%j.out"
    vars_relative = (epoch_dir / 'variables.py').relative_to(combo_path)
    out_relative = epoch_dir.relative_to(combo_path)
    script_content = job_script((config or {}).get('slurm'), job_name, slurm_log, [
        'echo $(hostname)',
        '',
        '# Change to combo directory and run with epoch-specific variables',
        f'cd {quote(combo_path)}',
        f'python3 msld_flat.py --vars-file {quote(vars_relative)} '
        f'--out-dir {quote(out_relative)} > {quote(out_relative)}/output.out 2>&1',
    ])
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
