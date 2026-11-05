"""Command-line entry point for online CB-policy training.

Usage::

    python -m mllf.cli.train_workflow my_workflow_config.yaml

The training itself is ``mllf.cb.training_loop.run_training`` (setup in
``mllf.cb.training_setup``, one epoch in ``mllf.cb.reinforce_trainer``);
see :doc:`workflow` for the configuration reference.
"""
import argparse

from mllf.cb.training_loop import run_training

__all__ = ['run_training', 'main']


def main() -> None:
    p = argparse.ArgumentParser(
        description='Run online CB-policy training from a YAML config',
        epilog='Example: python -m mllf.cli.train_workflow examples/workflow_14benz.yaml',
    )
    p.add_argument('config', help='Path to workflow YAML config file')
    args = p.parse_args()
    run_training(args.config)


if __name__ == '__main__':
    main()
