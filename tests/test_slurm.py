"""Tests for mllf.cli.slurm (migrated from training/workflow.py).

Covers submit_epoch_simulation: writing run_epoch.sh with the right SLURM
directives (defaults and config overrides) and parsing the sbatch job id,
without ever invoking a real `sbatch`.
"""
import os
import stat
from types import SimpleNamespace

import pytest

from mllf.cli import slurm as slurm_mod
from mllf.cli.slurm import submit_epoch_simulation


class _FakeCompletedProcess(SimpleNamespace):
    pass


def _fake_sbatch_run(job_id='12345'):
    """Return a stand-in for subprocess.run that mimics sbatch's stdout."""
    calls = []

    def _run(cmd, cwd=None, capture_output=None, text=None, check=None):
        calls.append({'cmd': cmd, 'cwd': cwd})
        return _FakeCompletedProcess(
            returncode=0, stdout=f'Submitted batch job {job_id}\n', stderr='',
        )
    _run.calls = calls
    return _run


def test_submit_epoch_simulation_minimal_config_omits_site_specific_directives(tmp_path, monkeypatch):
    """No `slurm:` block at all: no partition, no GRES, no module load -- must
    still produce a runnable script (portable to a GPU-less, module-less,
    default-partition-only cluster)."""
    combo_path = tmp_path / 'combo'
    combo_path.mkdir()
    epoch_dir = combo_path / 'run_001'

    fake_run = _fake_sbatch_run(job_id='98765')
    monkeypatch.setattr(slurm_mod.subprocess, 'run', fake_run)

    job_id = submit_epoch_simulation(str(combo_path), str(epoch_dir), config={})

    assert job_id == '98765'
    script_path = epoch_dir / 'run_epoch.sh'
    assert script_path.exists()

    # Executable bit set (0o755).
    mode = stat.S_IMODE(os.stat(script_path).st_mode)
    assert mode == 0o755

    content = script_path.read_text()
    # Generic default: kept even with no `slurm:` block.
    assert '#SBATCH --time=01:00:00' in content
    assert 'python3 msld_flat.py --vars-file' in content
    # Cluster/account-specific directives: omitted entirely when unset.
    assert '#SBATCH -p ' not in content
    assert '#SBATCH --gres=' not in content
    assert 'module load' not in content

    # sbatch invoked with the written script, run from epoch_dir.
    assert len(fake_run.calls) == 1
    call = fake_run.calls[0]
    assert call['cmd'] == ['sbatch', str(script_path)]
    assert call['cwd'] == str(epoch_dir)


def test_submit_epoch_simulation_honors_config_slurm_overrides(tmp_path, monkeypatch):
    combo_path = tmp_path / 'combo'
    combo_path.mkdir()
    epoch_dir = combo_path / 'run_002'

    monkeypatch.setattr(slurm_mod.subprocess, 'run', _fake_sbatch_run())

    config = {
        'slurm': {
            'partition': 'gpu-test',
            'gres': 'gpu:2',
            'time': '02:30:00',
            'module': 'charmm/other',
        }
    }
    submit_epoch_simulation(str(combo_path), str(epoch_dir), config=config)

    content = (epoch_dir / 'run_epoch.sh').read_text()
    assert '#SBATCH -p gpu-test' in content
    assert '#SBATCH --gres=gpu:2' in content
    assert '#SBATCH --time=02:30:00' in content
    assert 'module load charmm/other' in content


def test_submit_epoch_simulation_partition_without_gres(tmp_path, monkeypatch):
    """A CPU-only cluster: partition set, no GRES requested at all."""
    combo_path = tmp_path / 'combo'
    combo_path.mkdir()
    epoch_dir = combo_path / 'run_004'

    monkeypatch.setattr(slurm_mod.subprocess, 'run', _fake_sbatch_run())

    config = {'slurm': {'partition': 'cpu'}}
    submit_epoch_simulation(str(combo_path), str(epoch_dir), config=config)

    content = (epoch_dir / 'run_epoch.sh').read_text()
    assert '#SBATCH -p cpu' in content
    assert '#SBATCH --gres=' not in content
    assert 'module load' not in content


def test_submit_epoch_simulation_raises_on_sbatch_failure(tmp_path, monkeypatch):
    import subprocess as real_subprocess

    combo_path = tmp_path / 'combo'
    combo_path.mkdir()
    epoch_dir = combo_path / 'run_003'

    def _failing_run(cmd, cwd=None, capture_output=None, text=None, check=None):
        raise real_subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(slurm_mod.subprocess, 'run', _failing_run)

    with pytest.raises(real_subprocess.CalledProcessError):
        submit_epoch_simulation(str(combo_path), str(epoch_dir), config={})
