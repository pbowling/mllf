"""Shared pieces of the command-line modules."""
import argparse


class UsageError(Exception):
    """A command-line usage error found after parsing; reported like argparse's
    own errors (usage line, message, exit status 2)."""


def run_command(parser: argparse.ArgumentParser, run, argv=None):
    """Parse ``argv`` with ``parser`` and call ``run(args)``; a UsageError
    becomes ``parser.error``. Returns run's exit status."""
    args = parser.parse_args(argv)
    try:
        return run(args)
    except UsageError as e:
        parser.error(str(e))
