"""``mllf replay``: prebuild the pretraining replay set that
``bandit.shared_update: regression`` replays (``hindsight_regression.build_replay_set``),
so the training driver does not stall building it. The trainer then reuses the
file at ``bandit.regression.replay_cache``."""
import argparse
import logging
from pathlib import Path


log = logging.getLogger(__name__)

DESCRIPTION = 'Prebuild the regression replay set of a workflow config'
EPILOG = ('The options override the config\'s bandit.regression keys of the same meaning.\n'
          'Example: mllf replay online_1.yaml --cache replay_online_1.pt')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('config', help='workflow YAML (bandit.regression block)')
    parser.add_argument('--cache', default=None, metavar='FILE',
                        help='output file (regression.replay_cache); must not exist yet')
    parser.add_argument('--manifest', default=None, metavar='FILE',
                        help="the pretrained model's pretraining_manifest.json "
                             '(regression.replay_manifest)')
    parser.add_argument('--consensus', default=None, metavar='FILE',
                        help="the pretrained model's environment_consensus.json "
                             '(regression.replay_consensus)')
    parser.add_argument('--keep-online-runs', action='store_true',
                        help='keep runs with a collection_tag (online-test runs collected into '
                             'the datasets), for a model pretrained on them '
                             '(replay_pretraining_only: false)')


def replay_config(full_config: dict, args: argparse.Namespace) -> dict:
    """The regression config the trainer would use, with the command-line overrides."""
    from mllf.cb.hindsight_regression import regression_config
    cfg = regression_config(full_config['bandit'])
    from mllf.cb.policy import unimol_options
    unimol = unimol_options(full_config)
    for key in ('env_input', 'ligand_other_sites', 'site_size_input'):
        cfg[key] = unimol[key]
    if args.cache:
        cfg['replay_cache'] = args.cache
    if args.manifest:
        cfg['replay_manifest'] = args.manifest
    if args.consensus:
        cfg['replay_consensus'] = args.consensus
    if args.keep_online_runs:
        cfg['replay_pretraining_only'] = False
    return cfg


def run(args: argparse.Namespace) -> int:
    import yaml

    from mllf.cb.bandit_config import with_hindsight_defaults
    from mllf.cb.hindsight_regression import build_replay_set

    with open(args.config) as f:
        full = with_hindsight_defaults(yaml.safe_load(f))   # the trainer's defaults
    cfg = replay_config(full, args)
    for k in ('replay_manifest', 'replay_consensus', 'replay_pretraining_only', 'replay_runs',
              'replay_cache', 'replay_selection', 'replay_systems_root',
              'replay_min_per_system', 'site_size_input'):
        log.info(f"  {k}: {cfg.get(k)}")
    if not cfg.get('replay_cache'):
        log.error("bandit.regression.replay_cache must be set (or pass --cache)")
        return 1
    if Path(cfg['replay_cache']).exists():
        log.error(f"{cfg['replay_cache']} exists; remove it or choose another --cache")
        return 1
    from mllf.log import log_to_file
    with log_to_file(Path(cfg['replay_cache']).with_suffix('.log')) as log_file:
        log.info(f"Log: {log_file}")
        data = build_replay_set(cfg, seed=0)
        log.info(f"replay runs: {len(data)}")
    return 0


def main(argv=None):
    from mllf.cli.common import run_command
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return run_command(parser, run, argv)
