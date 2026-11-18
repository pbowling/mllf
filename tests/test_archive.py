"""Archiving a run's combo directories (mllf.file_handling.archive, mllf archive)."""
import tarfile
from pathlib import Path

import pytest
import yaml

from mllf.cli import main as cli
from mllf.file_handling.archive import archive_config, archive_directory, archive_systems


def _combo_tree(root: Path, name: str) -> Path:
    out = root / 'out' / name
    for combo in ('pair01_site1_subs_1_2', 'pair02_site1_subs_1_3'):
        run = out / combo / 'run_000'
        run.mkdir(parents=True)
        (run / 'output.out').write_text('OK\n')
        (out / combo / 'variables.py').write_text('x = 1\n')
    return out


def test_archive_directory_holds_everything_and_removes_only_on_request(tmp_path):
    src = _combo_tree(tmp_path, 'sysA')
    kept = archive_directory(src, tmp_path / 'a' / 'sysA.tar.gz')
    with tarfile.open(kept) as tar:
        names = tar.getnames()
    assert 'sysA/pair01_site1_subs_1_2/run_000/output.out' in names and src.is_dir()
    archive_directory(src, tmp_path / 'b' / 'sysA.tar.gz', remove=True)
    assert not src.exists() and (tmp_path / 'b' / 'sysA.tar.gz').exists()
    assert not list((tmp_path / 'b').glob('.*tmp'))        # no temporary file left


@pytest.mark.parametrize('block, expected', [
    (None, None),
    ({'enabled': False}, None),
    (True, {'when': 'end', 'remove': False}),
    ({'when': 'system', 'remove': True, 'dir': 'arch'}, {'when': 'system', 'remove': True}),
])
def test_archive_config(block, expected, tmp_path):
    cfg = archive_config({'archive': block, 'output': {'base_dir': str(tmp_path)}})
    if expected is None:
        assert cfg is None
        return
    assert {k: cfg[k] for k in expected} == expected
    assert cfg['dir'] == tmp_path / (block.get('dir', 'archive') if isinstance(block, dict)
                                     else 'archive')


def test_archive_config_rejects_unknown_schedule():
    with pytest.raises(ValueError, match='archive.when'):
        archive_config({'archive': {'when': 'weekly'}})


def test_archive_systems_skips_systems_without_directories(tmp_path):
    _combo_tree(tmp_path, 'sysA')
    systems = [{'name': 'sysA', 'out_dir': str(tmp_path / 'out' / 'sysA')},
               {'name': 'sysB', 'out_dir': str(tmp_path / 'out' / 'sysB')}]
    written = archive_systems(systems, tmp_path / 'arch')
    assert [p.name for p in written] == ['sysA.tar.gz']


def test_mllf_archive_command(tmp_path):
    _combo_tree(tmp_path, 'sysA')
    cfg = tmp_path / 'c.yaml'
    cfg.write_text(yaml.safe_dump({
        'output': {'base_dir': str(tmp_path / 'ck')},
        'systems': [{'name': 'sysA', 'out_dir': str(tmp_path / 'out' / 'sysA')}]}))
    assert cli.main(['archive', str(cfg), '--remove']) == 0
    assert (tmp_path / 'ck' / 'archive' / 'sysA.tar.gz').exists()
    assert not (tmp_path / 'out' / 'sysA').exists()
    assert cli.main(['archive', str(cfg), '--systems', 'nope']) == 1
