"""Example runner that reads a YAML config and runs the high-level workflow.

This script demonstrates how to:
  1. create combinations from a directory of site/sub files
  2. split the manifest into train/val
  3. pick one combo and run a quick training epoch
  4. run simulations concurrently
  5. compress completed runs

Create a YAML similar to `examples/workflow_sample.yaml` and pass it as the
only argument to this script.
"""
from pathlib import Path
import sys
from mllf.cli.workflow import run_from_config


def main():
    if len(sys.argv) < 2:
        print('No config provided, using examples/workflow_sample.yaml')
        cfg = str(Path(__file__).parent / 'workflow_sample.yaml')
    else:
        cfg = sys.argv[1]
    out = run_from_config(cfg)
    print('Workflow result:')
    print(out)


if __name__ == '__main__':
    main()
