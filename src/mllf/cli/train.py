"""``mllf train``: online training from a workflow config.

The training itself is ``mllf.cb.training_loop.run_training`` (setup in
``mllf.cb.training_setup``, one epoch in ``mllf.cb.reinforce_trainer``); see
:doc:`workflow` for the configuration reference.
"""
import argparse

DESCRIPTION = 'Run online CB-policy training from a YAML config'
EPILOG = ('Example: mllf train examples/workflow_14benz.yaml\n'
          'Rerunning the same command resumes from the latest checkpoint.')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('config', help='Path to workflow YAML config file')


def run(args: argparse.Namespace) -> None:
    from mllf.cb.training_loop import run_training
    # -v turns on the config's verbose detail (as 'verbose: true' does)
    run_training(args.config, verbose=True if getattr(args, 'verbose', 0) > 0 else None)


def main(argv=None):
    from mllf.cli.common import run_command
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return run_command(parser, run, argv)
