"""Shared pieces of the command-line modules."""
import argparse


class UsageError(Exception):
    """A command-line usage error found after parsing; reported like argparse's
    own errors (usage line, message, exit status 2)."""


def run_command(parser: argparse.ArgumentParser, run, argv=None):
    """Parse ``argv`` with ``parser`` and call ``run(args)``; a UsageError
    becomes ``parser.error``. Returns run's exit status. Used by the ``python -m``
    entry points, which set up the console as the ``mllf`` command does."""
    from mllf.log import setup_logging
    args = parser.parse_args(argv)
    setup_logging(getattr(args, 'verbose', 0))
    try:
        return run(args)
    except UsageError as e:
        parser.error(str(e))
