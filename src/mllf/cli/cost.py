"""``mllf cost``: simulated nanoseconds per system in an online run, next to
ALF's cost for the same systems."""
import argparse
import json
import logging
from pathlib import Path


log = logging.getLogger(__name__)

DESCRIPTION = 'Simulated ns per system: online run vs ALF'
EPILOG = 'Example: mllf cost training/checkpoints_x --pretraining-root pretraining'


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('checkpoint_dir', help='online run output dir (holds cost_ledger.json)')
    parser.add_argument('--pretraining-root', default=None,
                        help='pretraining datasets dir, for ALF cost (omit to skip ALF)')
    parser.add_argument('--json', default=None, help='also write the table to this JSON file')


def run(args: argparse.Namespace) -> None:
    from mllf.cb.cost_ledger import (
        LEDGER_FILE,
        alf_cost,
        alf_dataset_dirs,
        load_ledger,
        summarize_ledger,
    )
    summary = summarize_ledger(load_ledger(Path(args.checkpoint_dir) / LEDGER_FILE))
    stages_path = Path(args.checkpoint_dir) / 'system_stages.json'
    stages = json.loads(stages_path.read_text()) if stages_path.exists() else {}
    rows = {}
    log.info(f"{'system':28s} {'train ns':>9s} {'runs':>6s} {'valid ns':>9s} {'ALF ns':>9s} "
             f"{'ALF runs':>9s}  stages")
    for system in sorted(summary):
        s = summary[system]
        alf = None
        if args.pretraining_root:
            per_group = alf_cost(alf_dataset_dirs(system, args.pretraining_root))
            if per_group:
                alf = {'runs': sum(g['runs'] for g in per_group.values()),
                       'ns': sum(g['ns'] for g in per_group.values()),
                       'missing': sum(g['missing'] for g in per_group.values()),
                       'groups': per_group}
        st = stages.get(system) or {}
        hist = ' > '.join(
            f"{h['stage']}({h['epochs']}, {h['reason']})" for h in st.get('history', []))
        rows[system] = {'online': s, 'alf': alf, 'stages': st}
        alf_cols = f"{alf['ns']:9.1f} {alf['runs']:9d}" if alf else f"{'-':>9s} {'-':>9s}"
        log.info(f"{system:28s} {s['train']['ns']:9.1f} {s['train']['runs']:6d} "
                 f"{s['validation']['ns']:9.1f} {alf_cols}  {hist}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


def main(argv=None):
    from mllf.cli.common import run_command
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG)
    add_arguments(parser)
    return run_command(parser, run, argv)
