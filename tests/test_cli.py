"""The ``mllf`` command: dispatch, help, verbosity and errors."""
import argparse
import contextlib
import io
import logging

import pytest

from mllf.cli import main as cli
from mllf.log import setup_logging


@pytest.fixture(autouse=True)
def default_logging():
    yield
    setup_logging(0)


def _exit_code(argv):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    return e.value.code


def test_help_lists_every_command(capsys):
    assert _exit_code(['-h']) == 0
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert name in out


@pytest.mark.parametrize('command', list(cli.COMMANDS))
def test_each_command_has_help_and_verbosity_options(command, capsys):
    assert _exit_code([command, '-h']) == 0
    out = capsys.readouterr().out
    assert f'usage: mllf {command}' in out and '--verbose' in out and '--quiet' in out


def test_version(capsys):
    from mllf import __version__
    assert _exit_code(['--version']) == 0
    assert capsys.readouterr().out.strip() == f'mllf {__version__}'


def test_no_command_prints_help_and_fails(capsys):
    assert cli.main([]) == 2
    assert 'usage: mllf' in capsys.readouterr().out


def test_quiet_and_verbose_are_exclusive(capsys):
    assert _exit_code(['-q', '-v', 'cost', 'x']) == 2
    assert 'not allowed' in capsys.readouterr().err


def test_command_is_found_after_global_options():
    assert cli._command_in(['-v', '-q', 'train', 'cfg.yaml']) == 'train'
    assert cli._command_in(['--version']) is None


@pytest.mark.parametrize('argv, verbose', [
    (['train', 'c.yaml'], None),
    (['-v', 'train', 'c.yaml'], True),
    (['train', '-v', 'c.yaml'], True),
    (['-q', 'train', 'c.yaml'], None),
])
def test_train_passes_config_and_verbose(monkeypatch, argv, verbose):
    import mllf.cb.training_loop as tl
    seen = {}
    monkeypatch.setattr(tl, 'run_training', lambda path, verbose=None: seen.update(
        path=path, verbose=verbose))
    assert cli.main(argv) == 0
    assert seen == {'path': 'c.yaml', 'verbose': verbose}


def test_usage_error_from_run_exits_2(tmp_path, capsys):
    cfg = tmp_path / 'c.yaml'
    cfg.write_text('systems: []\n')
    assert _exit_code(['export', str(cfg)]) == 2
    assert '--model is required' in capsys.readouterr().err


def test_missing_input_file_is_one_line_error(capsys):
    assert _exit_code(['replay', 'missing.yaml']) == 1
    err = capsys.readouterr().err
    assert 'mllf replay: error: no such file: missing.yaml' in err
    assert 'Traceback' not in err


def test_replay_config_overrides(tmp_path):
    from mllf.cb.bandit_config import with_hindsight_defaults
    from mllf.cli.replay import replay_config
    full = with_hindsight_defaults({'bandit': {'algorithm': 'hindsight'},
                                    'unimol': {'env_input': 'phase', 'site_size_input': True}})
    args = argparse.Namespace(cache='r.pt', manifest='m.json', consensus=None,
                              keep_online_runs=True)
    cfg = replay_config(full, args)
    assert cfg['replay_cache'] == 'r.pt' and cfg['replay_manifest'] == 'm.json'
    assert cfg['replay_pretraining_only'] is False
    assert cfg['env_input'] == 'phase' and cfg['site_size_input'] is True


def test_old_module_entry_points_forward(capsys):
    from mllf.cb import cost_ledger, export_biases
    from mllf.file_handling import generate_combinations
    for mod in (cost_ledger, export_biases, generate_combinations):
        with pytest.raises(SystemExit) as e:
            mod.main(['-h'])
        assert e.value.code == 0
    assert 'usage:' in capsys.readouterr().out


# ---------------------------------------------------------------- logging
def test_levels_follow_verbosity(capsys):
    log = logging.getLogger('mllf.test')
    setup_logging(-1)
    log.info('progress')
    log.warning('Warning: careful')
    setup_logging(0)
    log.info('progress again')
    log.debug('detail')
    setup_logging(2)
    log.debug('detail shown')
    out = capsys.readouterr().out.splitlines()
    assert out == ['Warning: careful', 'progress again', 'DEBUG mllf.test: detail shown']


def test_messages_follow_redirected_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        logging.getLogger('mllf.test').info('to the redirected stream')
    assert buf.getvalue() == 'to the redirected stream\n'
