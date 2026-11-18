"""``mllf production``: production replicas with fixed biases, and their
end-state free energies (``mllf.production``, ``mllf.analysis.production``)."""
import argparse
import logging
from pathlib import Path


log = logging.getLogger(__name__)

DESCRIPTION = ('Production replicas of one system with fixed biases, and the free energies '
               'of its end states (WHAM and Potts model)')
EPILOG = ('Examples:\n'
          '  mllf production run production.yaml\n'
          '  mllf production run production.yaml --combo SIMS/12benz_vac/group01_site1_subs_1_2_3 '
          '\\\n      --variables SIMS/12benz_vac/group01_site1_subs_1_2_3/run_004/variables.py '
          '--out prod/12benz_vac_g1\n'
          '  mllf production analyze prod/12benz_vac_g1 --discard-ns 1\n'
          'The config holds a production: block (prep, script, variables or combo; out_dir, '
          'replicas, ns), a slurm: block for the simulations and an optional analysis: block; '
          'see the documentation for every key.')

RUN_KEYS = ('combo', 'prep', 'script', 'variables', 'name', 'replicas', 'ns', 'equil_ns')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest='action', metavar='ACTION', required=True)
    p = sub.add_parser('run', help='set up and submit the replicas, then their analysis',
                       description='Set up one directory per replica and submit each; replicas '
                                   'already finished or queued are left alone, so running it '
                                   'again resubmits only the failed ones. The analysis job '
                                   'starts when the replicas end (analysis: run: false skips '
                                   'it).')
    p.add_argument('config', help='YAML with production:, slurm: and analysis: blocks')
    p.add_argument('--combo', help='combination directory with prep/, msld_flat.py and info.py')
    p.add_argument('--prep', help='prep directory')
    p.add_argument('--script', help='MSLD script (msld_flat.py)')
    p.add_argument('--variables', help='variables file with the biases')
    p.add_argument('--out', dest='out_dir', help='production directory')
    p.add_argument('--name', help='name of the jobs and lambda files (default: from --out)')
    p.add_argument('--replicas', type=int, help='independent replicas')
    p.add_argument('--ns', type=float, help='production length of each replica (ns)')
    p.add_argument('--equil-ns', type=float, dest='equil_ns',
                   help="equilibration before production (default: the script's esteps)")
    p.add_argument('--dry-run', action='store_true',
                   help='check the inputs and list what would be submitted')

    p = sub.add_parser('analyze', help='free energies of a production',
                       description='End-state free energies of a production by WHAM and by a '
                                   'Potts model, with bootstrap uncertainties over the replicas; '
                                   'unset options come from the production\'s analysis: block.')
    p.add_argument('out_dir', help='production directory (holds production.yaml)')
    p.add_argument('--out', default=None, metavar='DIR',
                   help='output directory (default: <production>/analysis)')
    p.add_argument('--discard-ns', type=float, default=None,
                   help='production discarded from the start of each replica (default: a '
                        'quarter of its length)')
    p.add_argument('--cutoff', type=float, default=None,
                   help='lambda at or above which a substituent is at its end state (0.99)')
    p.add_argument('--bootstrap', type=int, default=None, help='bootstrap samples (50)')
    p.add_argument('--stride', type=int, default=None, help='use every Nth frame (1)')
    p.add_argument('--no-potts', action='store_true', help='skip the Potts model')
    p.add_argument('--format', choices=('png', 'pdf'), default='png', dest='fmt')
    p.add_argument('--only', nargs='+', default=None, metavar='FIGURE',
                   help='make only these figures (populations, lambda_trajectories, profiles, '
                        'estimator_parity)')


def run(args: argparse.Namespace) -> int:
    import yaml

    from mllf.cli.common import UsageError
    from mllf.log import log_to_file
    if args.action == 'run':
        from mllf import production as prod
        with open(args.config) as f:
            raw = yaml.safe_load(f) or {}
        config = dict(raw.get('production') or {})
        config['slurm'] = raw.get('slurm') or {}
        config['analysis'] = raw.get('analysis') or {}
        for key in RUN_KEYS + ('out_dir',):
            v = getattr(args, key, None)
            if v is not None:
                config[key] = v
        try:
            # paths in the config are relative to it; command-line ones to here
            base = Path(args.config).resolve().parent
            for key in ('combo', 'prep', 'script', 'variables', 'out_dir'):
                if getattr(args, key, None) is not None:
                    config[key] = str(Path(getattr(args, key)).resolve())
            prod.submit(config, base=base, dry_run=args.dry_run)
        except ValueError as e:
            raise UsageError(str(e))
        return 0
    from mllf.analysis.common import require_plotting
    try:
        require_plotting()
    except ImportError as e:
        raise UsageError(str(e))
    from mllf.analysis.production import FIGURES, analyze_production
    unknown = [o for o in args.only or () if o not in FIGURES]
    if unknown:
        raise UsageError(f"unknown figure(s) {', '.join(unknown)}; choose from "
                         f"{', '.join(FIGURES)}")
    out = Path(args.out or Path(args.out_dir) / 'analysis')
    with log_to_file(out / 'analysis.log'):
        analyze_production(args.out_dir, out=out, discard_ns=args.discard_ns,
                           cutoff=args.cutoff, n_boot=args.bootstrap, stride=args.stride,
                           potts=False if args.no_potts else None, fmt=args.fmt,
                           only=args.only)
    return 0
