"""``mllf scout``: one scouting run per group with the starting model's biases,
and the summary a stage's ``scouting:`` key applies (``mllf.cb.scouting``)."""
import argparse
import logging
from pathlib import Path


log = logging.getLogger(__name__)

DESCRIPTION = 'Scouting runs before a stage: submit them, then summarize their outcome'
EPILOG = ('Examples:\n'
          '  mllf scout workflow.yaml --out scouting --wait   # submit, wait, summarize\n'
          '  mllf scout workflow.yaml --out scouting           # submit only\n'
          '  mllf scout workflow.yaml --out scouting --summarize\n'
          'Point the stage at the result with  scouting: <out>.')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('config', help='workflow YAML (systems, combo sets, curriculum)')
    parser.add_argument('--out', required=True, metavar='DIR',
                        help='scouting directory (runs, scout_summary.json, pair_priority.json)')
    parser.add_argument('--model', default=None,
                        help='model (default: deploy.model_path, else pretrain.model_path)')
    parser.add_argument('--stage', default=None,
                        help='stage whose group_set is scouted (default: the first with one)')
    parser.add_argument('--systems', nargs='*', default=None, help='only these systems')
    parser.add_argument('--run-steps', nargs=2, type=int, default=None,
                        metavar=('EQUIL', 'PROD'),
                        help="MD steps of each run (default: the system's msld_flat.py)")
    parser.add_argument('--no-submit', action='store_true',
                        help='prepare the runs without submitting them')
    parser.add_argument('--wait', action='store_true',
                        help='wait for the submitted runs, then summarize')
    parser.add_argument('--summarize', action='store_true',
                        help='only summarize finished runs (no preparation)')
    parser.add_argument('--poll', type=float, default=60.0, metavar='SECONDS',
                        help='seconds between queue checks with --wait (default 60)')


def run(args: argparse.Namespace) -> int:
    import yaml

    from mllf.cb.bandit_config import with_hindsight_defaults
    from mllf.cb.scouting import prepare, summarize, wait_for_jobs
    from mllf.log import log_to_file

    with open(args.config) as f:
        config = with_hindsight_defaults(yaml.safe_load(f))
    out = Path(args.out)
    with log_to_file(out / 'scouting.log'):
        if not args.summarize:
            jobs = prepare(config, out, model=args.model, systems=args.systems,
                           stage_name=args.stage, run_steps=args.run_steps,
                           submit=not args.no_submit)
            if not args.wait:
                return 0
            log.info(f"Waiting for {len(jobs)} scouting job(s)...")
            wait_for_jobs(jobs.values(), poll_seconds=args.poll)
        summary = summarize(config, out, systems=args.systems, stage_name=args.stage)
        if summary['missing_runs']:
            log.warning(f"  Warning: no finished run for {', '.join(summary['missing_runs'])}; "
                        "training treats those groups as not scouted")
    return 0


def main(argv=None):
    from mllf.cli.common import run_command
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return run_command(parser, run, argv)
