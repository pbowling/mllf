"""``mllf export``: write production biases from a trained policy, or label a
finished run's exported groups."""
import argparse
import logging
from pathlib import Path

from mllf.cli.common import UsageError


log = logging.getLogger(__name__)

DESCRIPTION = 'Export production biases from a trained policy'
EPILOG = ('Examples:\n'
          '  mllf export config.yaml --model final_model.pt --bandit-state bandit_state.json\n'
          '  mllf export config.yaml --label <output.base_dir>')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('config', help='workflow YAML (systems, unimol, bandit, training)')
    parser.add_argument('--model', default=None,
                        help='checkpoint with policy_state_dict or policy_state')
    parser.add_argument('--bandit-state', default=None,
                        help='bandit_state.json with per-pair offsets')
    parser.add_argument('--systems', nargs='*', default=None,
                        help='system names (default: all in config)')
    parser.add_argument('--out', default='final_biases',
                        help='output root (one subdirectory per system)')
    parser.add_argument('--label', default=None, metavar='CHECKPOINT_DIR',
                        help='only label an existing run\'s exported groups (confidence files, '
                             '_unconverged names) and exit')


def run(args: argparse.Namespace) -> None:
    import torch
    import yaml

    from mllf.cb.bandit_config import with_hindsight_defaults
    from mllf.cb.export_biases import export_final_biases, label_final_biases
    from mllf.cb.training_setup import build_policy
    from mllf.cb.workflow_utils import load_bandit_state

    with open(args.config) as f:
        config = with_hindsight_defaults(yaml.safe_load(f))
    if args.label:
        n = label_final_biases(args.label, config, args.systems)
        log.info(f"labelled {n} exported group(s) under {Path(args.label) / 'final_biases'}")
        return
    if not args.model:
        raise UsageError('--model is required unless --label is given')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = build_policy(config, device, bayesian_default=True)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt.get('policy_state_dict') or ckpt['policy_state'])
    state = load_bandit_state(Path(args.bandit_state)) if args.bandit_state else None
    for sys_cfg in config.get('systems', []):
        if args.systems and sys_cfg.get('name') not in args.systems:
            continue
        path = export_final_biases(policy, sys_cfg, config, state,
                                   Path(args.out) / sys_cfg['name'], device=device)
        log.info(f"{sys_cfg['name']}: {path}")


def main(argv=None):
    from mllf.cli.common import run_command
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return run_command(parser, run, argv)
