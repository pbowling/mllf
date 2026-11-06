"""``mllf analyze``: figures and a report for a pretrained model or an
online-training run (``mllf.analysis``)."""
import argparse
import logging
from pathlib import Path


log = logging.getLogger(__name__)

DESCRIPTION = 'Figures and a report for a pretrained model or an online-training run'
EPILOG = ('Examples:\n'
          '  mllf analyze online online_1.yaml --pretraining-root ../pretraining\n'
          '  mllf analyze online online_1.yaml --only success_over_epochs stage_timeline\n'
          '  mllf analyze pretrain models/pretraining_Nov02b\n'
          'Each writes figures/ (PNG, or PDF with --format pdf), tables/ (CSV) and '
          'report.txt.\n'
          'Needs matplotlib, pandas and pillow: pip install "mllf[analysis]".')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest='target', metavar='TARGET', required=True)

    p = sub.add_parser('online', help='an online-training run',
                       description='Figures and a report for an online-training run: success '
                                   'over epochs, run outcomes, stage timeline, convergence vs '
                                   'ALF and free-energy profiles.')
    p.add_argument('config', help='the run\'s workflow YAML')
    p.add_argument('--run-dir', default=None,
                   help="the run's output directory (default: output.base_dir)")
    p.add_argument('--out', default=None, metavar='DIR',
                   help='output directory (default: <run dir>/analysis)')
    p.add_argument('--pretraining-root', default=None, metavar='DIR',
                   help="pretraining datasets holding ALF's runs (default: the config's "
                        'alf_root); "none" skips the ALF comparison')
    p.add_argument('--no-validation', action='store_true',
                   help='leave held-out pairs out of success_over_epochs')
    p.add_argument('--free-energy', nargs='*', default=None, metavar='SYSTEM',
                   help="also draw free-energy profiles of the passed groups' runs (of these "
                        'systems; default all). Reads every run trajectory: minutes per system')
    _common(p)

    p = sub.add_parser('pretrain', help='a pretrained model',
                       description='Figures and a report for a pretrained model on the runs it '
                                   'was trained on: parity plots, predictions, embedding PCA, '
                                   'similarity between systems and dataset statistics.')
    p.add_argument('model_dir', help='pretraining output directory (best_policy*.pt, '
                                     'pretraining_manifest.json)')
    p.add_argument('--model', default=None,
                   help='checkpoint (default: best_policy_bayesian.pt, else best_policy.pt)')
    p.add_argument('--config', default=None,
                   help="pretraining config, for the unimol settings when the model's "
                        'pretrain_metadata.json lacks them')
    p.add_argument('--out', default=None, metavar='DIR',
                   help='output directory (default: <model dir>/analysis)')
    p.add_argument('--device', default=None, help='torch device (default: cuda if available)')
    p.add_argument('--similarity-full', action='store_true',
                   help='also draw similarity_full: per system and environment, with the full '
                        '515-D node input (ligand embedding, phase flags, site size)')
    p.add_argument('--dataset-stats', action='store_true',
                   help='also draw dataset_stats; lists every pretraining directory (minutes '
                        'for a full set; cached in <out>/cache)')
    p.add_argument('--max-runs-per-dataset', type=int, default=None, metavar='N',
                   help='analyze the N runs of each dataset with the most scored targets, of 50 '
                        'read at random (default 10; 0: every run, hours for a full set)')
    _common(p)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument('--format', choices=('png', 'pdf'), default='png', dest='fmt',
                   help='figure file format (default png; pdf is vector with editable text)')
    p.add_argument('--grayscale', action='store_true',
                   help='also write a grayscale PNG of each figure, to check print legibility')
    p.add_argument('--only', nargs='+', default=None, metavar='FIGURE',
                   help='make only these figures (the report and tables are always written)')
    p.add_argument('--workers', type=int, default=8, help='threads reading run files')
    p.add_argument('--rescan', action='store_true',
                   help='ignore the cached data in <out>/cache and read everything again')


def run(args: argparse.Namespace) -> int:
    from mllf.analysis.common import require_plotting
    from mllf.cli.common import UsageError
    from mllf.log import log_to_file
    try:
        require_plotting()
    except ImportError as e:
        raise UsageError(str(e))
    if args.target == 'online':
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        run_dir = Path(args.run_dir or (config.get('output') or {}).get('base_dir') or '.')
        out = Path(args.out or run_dir / 'analysis')
        root = args.pretraining_root or config.get('alf_root')
        if root and str(root).lower() == 'none':
            root = None
        from mllf.analysis.online import FIGURES, analyze_online
        _check_only(args.only, FIGURES, UsageError)
        with log_to_file(out / 'analysis.log'):
            analyze_online(config, out, run_dir=run_dir, pretraining_root=root, only=args.only,
                           validation=not args.no_validation, workers=args.workers,
                           rescan=args.rescan, free_energy=args.free_energy, fmt=args.fmt,
                           grayscale=args.grayscale)
        return 0
    from mllf.analysis.pretrain import FIGURES, analyze_pretrain
    _check_only(args.only, FIGURES, UsageError)
    out = Path(args.out or Path(args.model_dir) / 'analysis')
    with log_to_file(out / 'analysis.log'):
        analyze_pretrain(Path(args.model_dir), out, model=args.model, config_path=args.config,
                         device=args.device, max_runs=args.max_runs_per_dataset,
                         only=args.only, workers=args.workers, rescan=args.rescan,
                         fmt=args.fmt, grayscale=args.grayscale,
                         dataset_stats=args.dataset_stats,
                         similarity_full=args.similarity_full)
    return 0


def _check_only(only, figures, error) -> None:
    unknown = [o for o in only or () if o not in figures]
    if unknown:
        raise error(f"unknown figure(s) {', '.join(unknown)}; choose from {', '.join(figures)}")
