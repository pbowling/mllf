"""Compatibility alias: ``python -m mllf.cli.train_workflow CONFIG`` is
``mllf train CONFIG`` (see ``mllf.cli.train``)."""
import sys

from mllf.cb.training_loop import run_training
from mllf.cli.train import main

__all__ = ['run_training', 'main']

if __name__ == '__main__':
    sys.exit(main())
