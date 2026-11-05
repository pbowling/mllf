"""The ``mllf`` command.

    mllf [-h] [--version] [-v | -q] COMMAND [options]

Each command lives in ``mllf.cli.<module>`` with ``DESCRIPTION`` (and
optionally ``EPILOG``), ``add_arguments(parser)`` and ``run(args)``. Only the
chosen command's module is imported, so ``mllf -h`` stays fast; the heavy
imports (torch, Uni-Mol) happen inside ``run``.
"""
import argparse
import importlib
import sys

from mllf import __version__
from mllf.cli.common import UsageError
from mllf.log import setup_logging


# command -> (module, one-line help); in pipeline order
COMMANDS = {
    'combos': ('mllf.cli.combos', 'generate combination directories from site/sub files'),
    'collect': ('mllf.cli.collect_pretraining_data',
                'collect finished MSLD runs into pretraining records'),
    'pretrain': ('mllf.cli.pretrain', 'behavior-clone the policy on collected runs'),
    'replay': ('mllf.cli.replay', 'prebuild the regression replay set of a workflow config'),
    'train': ('mllf.cli.train', 'online training from a workflow config'),
    'export': ('mllf.cli.export', 'export production biases from a trained policy'),
    'cost': ('mllf.cli.cost', 'simulated ns per system: online run vs ALF'),
}

DESCRIPTION = 'Machine-learned bias coefficients for multisite lambda dynamics.'
EPILOG = ("Run 'mllf COMMAND -h' for a command's options. "
          "-v / -q may also follow the command.")


def _add_verbosity(parser: argparse.ArgumentParser, default) -> None:
    g = parser.add_mutually_exclusive_group()
    g.add_argument('-v', '--verbose', action='count', default=default,
                   help='more detail (-v: verbose output, -vv: debug messages)')
    g.add_argument('-q', '--quiet', action='store_const', const=-1, dest='verbose',
                   default=default, help='only warnings and errors')


def _command_in(argv) -> str:
    """The command named in argv: the first argument that is not an option
    (the global options take no values)."""
    return next((a for a in argv if not a.startswith('-')), None)


def build_parser(command: str = None) -> argparse.ArgumentParser:
    """The ``mllf`` parser, with full options for ``command`` only."""
    parser = argparse.ArgumentParser(prog='mllf', description=DESCRIPTION, epilog=EPILOG)
    parser.add_argument('--version', action='version', version=f'mllf {__version__}')
    _add_verbosity(parser, default=0)
    sub = parser.add_subparsers(dest='command', metavar='COMMAND', title='commands')
    for name, (module, help_text) in COMMANDS.items():
        if name != command:
            sub.add_parser(name, help=help_text)
            continue
        mod = importlib.import_module(module)
        p = sub.add_parser(name, help=help_text, description=mod.DESCRIPTION,
                           epilog=getattr(mod, 'EPILOG', None),
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        # also accepted after the command; SUPPRESS keeps the global value otherwise
        _add_verbosity(p, default=argparse.SUPPRESS)
        mod.add_arguments(p)
        p.set_defaults(_run=mod.run, _parser=p)
    return parser


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    command = _command_in(argv)
    parser = build_parser(command if command in COMMANDS else None)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    setup_logging(args.verbose)
    try:
        status = args._run(args)
    except UsageError as e:
        args._parser.error(str(e))      # usage line + message, exit status 2
    except FileNotFoundError as e:
        # a path given on the command line: one line, not a traceback (-vv keeps it)
        if args.verbose >= 2 or e.filename not in argv:
            raise
        args._parser.exit(1, f"mllf {args.command}: error: no such file: {e.filename}\n")
    return int(status or 0)


if __name__ == '__main__':
    sys.exit(main())
