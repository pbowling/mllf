"""``mllf archive``: pack a finished or stopped online run's combo directories
into one ``.tar.gz`` per system (``mllf.file_handling.archive``)."""
import argparse
import logging
from pathlib import Path


log = logging.getLogger(__name__)

DESCRIPTION = "Archive an online run's combo directories, one .tar.gz per system"
EPILOG = ('Settings default to the config\'s archive: block, or <output.base_dir>/archive.\n'
          'Example: mllf archive online_1.yaml --systems luis_p38_solvent --remove')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('config', help='workflow YAML (systems and output.base_dir)')
    parser.add_argument('--systems', nargs='*', default=None,
                        help='system names (default: every system in the config)')
    parser.add_argument('--dir', default=None, metavar='DIR',
                        help='archive directory (default: archive.dir, else '
                             '<output.base_dir>/archive)')
    parser.add_argument('--remove', action='store_true',
                        help='delete each combo directory once its archive is verified')


def run(args: argparse.Namespace) -> int:
    import yaml

    from mllf.file_handling.archive import archive_config, archive_systems

    with open(args.config) as f:
        config = yaml.safe_load(f)
    block = config.get('archive')
    block = dict(block) if isinstance(block, dict) else {}
    defaults = archive_config({**config, 'archive': {**block, 'enabled': True}})
    dest = Path(args.dir) if args.dir else defaults['dir']
    remove = args.remove or defaults['remove']
    systems = config.get('systems', [])
    unknown = set(args.systems or []) - {s.get('name') for s in systems}
    if unknown:
        log.error(f"Error: not in the config: {', '.join(sorted(unknown))}")
        return 1
    written = archive_systems(systems, dest, names=args.systems, remove=remove)
    log.info(f"{len(written)} system(s) archived in {dest}")
    return 0


def main(argv=None):
    from mllf.cli.common import run_command
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return run_command(parser, run, argv)
