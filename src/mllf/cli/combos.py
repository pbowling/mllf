"""``mllf combos``: generate combination directories from site/substituent files."""
import argparse
import logging
from pathlib import Path


log = logging.getLogger(__name__)

DESCRIPTION = "Generate combination directories from site_sub files"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('input_dir', type=Path,
                        help='Directory containing site{n}_sub{m}.* files')
    parser.add_argument('--out', '-o', dest='out_dir', type=Path, default=Path('combos'),
                        help='Output base directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show actions without copying files')
    parser.add_argument('--include', '-i', dest='include', action='append', default=[],
                        help='Glob pattern(s) of additional files to copy into each combo dir '
                             '(relative to input_dir). Can be provided multiple times.')
    parser.add_argument('--max-subs', type=int, default=10,
                        help='Maximum number of substituents per site in any single combination '
                             '(default: 10). All substituents can still participate, but each '
                             'combination is limited to this size per site.')
    parser.add_argument('--archive', dest='archive', action='store_true',
                        help='Create .tar.gz archives of generated combo directories matching '
                             'pattern comb_*')
    parser.add_argument('--slurm-config', type=Path, default=None, metavar='YAML',
                        help="YAML file whose 'slurm:' block sets each run.sh's sbatch options, "
                             'modules and setup (e.g. the workflow config); default: generic '
                             'options only')
    parser.add_argument('--archive-remove', dest='archive_remove', action='store_true',
                        help='Remove combo directories after successful archiving')


def run(args: argparse.Namespace) -> None:
    from mllf.file_handling.generate_combinations import (
        archive_combo_dirs,
        create_combination_dirs,
    )
    input_dir = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    slurm = None
    if args.slurm_config:
        import yaml
        slurm = (yaml.safe_load(args.slurm_config.read_text()) or {}).get('slurm')
    created = create_combination_dirs(
        input_dir, args.out_dir, dry_run=args.dry_run,
        include_patterns=args.include or None, max_subs_per_site=args.max_subs, slurm=slurm)
    log.info(f"Created {len(created)} combination dirs under {args.out_dir}")
    if args.archive:
        archived = archive_combo_dirs(args.out_dir, pattern='comb_*', remove=args.archive_remove)
        log.info(f"Archived {len(archived)} combo directories")


def main(argv=None):
    from mllf.cli.common import run_command
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    return run_command(parser, run, argv)
