"""Tests for mllf.slurm (job scripts from the slurm: block) and mllf.cb.slurm.

Covers submit_epoch_simulation: writing run_epoch.sh with the right SLURM
directives (defaults and config overrides) and parsing the sbatch job id,
without ever invoking a real `sbatch`.
"""
import os
import stat
from types import SimpleNamespace

import pytest

from mllf import slurm as job
from mllf.cb import slurm as slurm_mod
from mllf.cb.slurm import submit_epoch_simulation


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


def test_submit_epoch_simulation_minimal_config_omits_site_specific_directives(
        tmp_path, monkeypatch):
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
    assert '--partition' not in content
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
    assert '#SBATCH --partition=gpu-test' in content
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
    assert '#SBATCH --partition=cpu' in content
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


# ---------------------------------------------------------------- mllf.slurm
def _directives(cfg):
    return job.sbatch_directives(cfg, 'j', 'j.%j.out')


def test_any_sbatch_option_from_the_block():
    lines = _directives({'partition': 'gpu2080', 'gres': 'gpu:1', 'qos': 'normal',
                         'mem_per_cpu': '4G', 'mail_type': ['FAIL', 'END'],
                         'requeue': True, 'nodelist': None, 'overcommit': False})
    assert lines[:2] == ['#SBATCH --job-name=j', '#SBATCH --output=j.%j.out']
    for want in ('#SBATCH --partition=gpu2080', '#SBATCH --gres=gpu:1', '#SBATCH --qos=normal',
                 '#SBATCH --mem-per-cpu=4G', '#SBATCH --mail-type=FAIL,END',
                 '#SBATCH --requeue'):
        assert want in lines
    assert not any('nodelist' in ln or 'overcommit' in ln for ln in lines)


def test_defaults_can_be_overridden_or_dropped():
    lines = _directives({'cpus_per_task': 4, 'export': None, 'time': '2-00:00:00'})
    assert '#SBATCH --cpus-per-task=4' in lines and '#SBATCH --ntasks=1' in lines
    assert '#SBATCH --time=2-00:00:00' in lines
    assert not any('--export' in ln for ln in lines)


def test_exclude_list_string_and_file(tmp_path):
    bad = tmp_path / 'bad_nodes.txt'
    bad.write_text('gollum007  # flaky GPU\ngollum009, gollum003\n')
    lines = _directives({'exclude': ['gollum003', 'gollum005'], 'exclude_file': str(bad)})
    assert '#SBATCH --exclude=gollum003,gollum005,gollum007,gollum009' in lines
    assert '#SBATCH --exclude=gollum[001-004]' in _directives({'exclude': 'gollum[001-004]'})
    # a missing file excludes nothing
    assert not any('exclude' in ln for ln in _directives({'exclude_file': tmp_path / 'none'}))


def test_extra_module_and_setup_lines():
    text = job.job_script({'extra': ['--signal=B:USR1@60', '#SBATCH --hint=nomultithread'],
                           'module': ['charmm/c51a1', 'cuda'],
                           'setup': 'conda activate charmm'}, 'j', 'o', ['run'])
    assert '#SBATCH --signal=B:USR1@60\n#SBATCH --hint=nomultithread\n' in text
    assert text.endswith('module load charmm/c51a1\nmodule load cuda\n'
                         'conda activate charmm\n\nrun\n')


@pytest.mark.parametrize('key', ['job_name', 'output', 'job-name'])
def test_job_name_and_output_are_reserved(key):
    with pytest.raises(ValueError, match='set by mllf'):
        _directives({key: 'x'})


def test_retry_picks_up_newly_excluded_nodes(tmp_path):
    bad = tmp_path / 'bad_nodes.txt'
    bad.write_text('gollum001\n')
    cfg = {'exclude_file': str(bad)}
    script = tmp_path / 'run_epoch.sh'
    script.write_text(job.job_script(cfg, 'j', 'o', ['run']))
    assert '#SBATCH --exclude=gollum001' in script.read_text()
    bad.write_text('gollum001\ngollum002\n')       # a node goes bad mid-run
    job.refresh_exclude(script, cfg)
    text = script.read_text()
    assert text.count('--exclude') == 1 and '#SBATCH --exclude=gollum001,gollum002' in text
    bad.write_text('')
    job.refresh_exclude(script, cfg)
    assert '--exclude' not in script.read_text()


def test_combo_run_sh_uses_the_slurm_block(tmp_path):
    from mllf.file_handling.generate_combinations import create_combination_dirs
    src = tmp_path / 'in'
    src.mkdir()
    for site, sub in ((1, 1), (1, 2)):
        (src / f'site{site}_sub{sub}_pres.rtf').write_text(f'PRES s{site}s{sub}\n')
        (src / f'site{site}_sub{sub}_frag.pdb').write_text('END\n')
    dirs = create_combination_dirs(src, tmp_path / 'out',
                                   slurm={'partition': 'gpu', 'exclude': 'n1',
                                          'module': 'charmm'})
    text = (dirs[0] / 'run.sh').read_text()
    assert '#SBATCH --partition=gpu' in text and '#SBATCH --exclude=n1' in text
    assert 'module load charmm' in text and 'gpu2080' not in text
    plain = create_combination_dirs(src, tmp_path / 'out2')
    assert 'module load' not in (plain[0] / 'run.sh').read_text()
