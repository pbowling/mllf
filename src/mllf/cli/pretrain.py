"""``mllf pretrain``: behavior-clone the policy on collected runs
(``mllf.pretrain.trainer.pretrain_with_runs``)."""
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List


log = logging.getLogger(__name__)

DESCRIPTION = "Pretrain MSLD policy on collected simulation data"
EPILOG = ("Example: mllf pretrain --pretraining-dir pretraining/14benz_solv "
          "--output-dir models/14benz --config examples/workflow_pretrain.yaml --bayesian-heads")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretraining-dir",
        type=str,
        action='append',
        default=None,
        help="Directory containing collected pretraining data (can specify multiple times). "
             "Mutually exclusive with --manifest-file.",
    )
    parser.add_argument(
        "--manifest-file",
        type=str,
        default=None,
        help="Path to a JSON manifest file containing a list of pretraining directories. One "
             "directory path per line, or a JSON object with 'directories' key. Use this instead "
             "of --pretraining-dir when you have too many directories (avoids 'Argument list too "
             "long' error).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save pretrained policy",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/workflow_pretrain.yaml",
        help="Config file (same format as workflow_pretrain.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50 for behavior cloning convergence)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--use-best-only",
        action="store_true",
        help="If set, filter to best run per system. Otherwise use all valid runs (default: use "
             "all)",
    )
    parser.add_argument(
        "--use-sparse-graphs",
        action="store_true",
        help=argparse.SUPPRESS,  # no effect (edges are always all within-site pairs)
    )
    parser.add_argument(
        "--no-filter-outliers",
        action="store_true",
        help="If set, disable statistical filtering of outlier runs. By default, runs with "
             "coefficients outside +/-N standard deviations are excluded.",
    )
    parser.add_argument(
        "--outlier-std-threshold",
        type=float,
        default=3.0,
        help="Number of standard deviations for outlier threshold (default: 3.0). Runs with any "
             "coefficient exceeding mean +/- N*std are excluded.",
    )
    parser.add_argument(
        "--min-transitions",
        type=int,
        default=None,
        metavar="N",
        help="Keep only runs where every site has >= N transitions at the highest lambda "
             "value (e.g. --min-transitions 3). Uses the raw transition counts from "
             "simulation_results.json; no reward formula involved. Takes precedence over "
             "--min-reward-threshold and --stratified-negative-fraction when set.",
    )
    parser.add_argument(
        "--min-reward-threshold",
        type=float,
        default=None,
        help="Minimum reward threshold for filtering runs (e.g., 5.0). If set, only runs with "
             "reward >= threshold are used for pretraining. Ignored when "
             "--stratified-negative-fraction is set.",
    )
    parser.add_argument(
        "--stratified-negative-fraction",
        type=float,
        default=None,
        metavar="F",
        help="If set, keep all positive-reward runs and sample negative-reward buckets with a "
             "quadratic ramp: worst bucket (-inf,-50] keeps 0%%, best bucket (-10,0) keeps this "
             "fraction. "
             "E.g. 0.55 ramps from 0%% to 55%% (quadratic schedule). "
             "When specified, --min-reward-threshold is ignored. Default: disabled.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10). Training stops if the MSE loss does "
             "not improve for this many consecutive epochs.",
    )
    parser.add_argument(
        "--reward-weighted",
        action="store_true",
        default=False,
        help="Weight each run's MSE loss by its reward (clamped >= 0, normalised so the mean "
             "weight = 1.0). High-reward runs contribute proportionally more gradient signal; "
             "zero/negative-reward runs are skipped. Useful when the training set contains "
             "many low-quality runs that compress the predicted coefficient range.",
    )
    parser.add_argument(
        "--freeze-encoder-after",
        type=int,
        default=None,
        metavar="N",
        help=argparse.SUPPRESS,  # no effect; accepted for old job scripts
    )
    parser.add_argument(
        "--awr-temperature",
        type=float,
        default=1.0,
        help="Temperature beta for AWR loss: each run is weighted by exp(r/beta). Higher beta: "
             "more uniform weighting; lower beta: sharper emphasis on high-reward runs. "
             "Default: 1.0.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Start from this checkpoint's policy_state. With --epochs 0 --bayesian-heads, "
             "only builds its Bayesian-head sibling (final_policy_bayesian.pt).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Continue an interrupted run from its checkpoint_epoch_NNN.pt (same data, "
             "settings and --output-dir): restores model, optimizer, LR schedule and "
             "early-stopping state, then runs the remaining epochs and the post-passes.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Use a random sample (seed 0) of at most this many runs, e.g. for a quick "
             "Bayesian post-pass with --init-checkpoint --epochs 0.",
    )
    parser.add_argument(
        "--no-data-hygiene",
        action="store_true",
        default=False,
        help="Disable data hygiene (enabled by default). With hygiene, each edge's "
             "target for each bias type counts only when the run measured what sets it "
             "(linear: both substituents sampled and a finite pair DDG; quadratic: "
             "midpoint barrier; skew/end: skew/end asymmetry of the lambda profile), "
             "and runs without usable populations contribute nothing. Makes run-level "
             "filters (reward, transitions, stratified sampling) unnecessary.",
    )
    parser.add_argument(
        "--no-per-pair-awr",
        action="store_true",
        default=False,
        help="Disable per-edge, per-dimension AWR (enabled by default). When enabled, "
             "each pair gets its own reward computed via compute_pair_reward() (same as "
             "online training), and pairs with better DDG/population/physical-fraction "
             "within a successful run drive more of the gradient than poorly-resolved "
             "pairs. When disabled, every pair in a run shares one scalar exp(run_reward/β) "
             "weight (legacy behaviour).",
    )
    parser.add_argument(
        "--bayesian-heads",
        action="store_true",
        default=False,
        help="After the normal BC loop finishes, also build and save a "
             "use_bayesian_heads=True sibling policy for online training with "
             "Thompson sampling (best_policy_bayesian.pt / "
             "final_policy_bayesian.pt). Trunks are copied from the trained "
             "deterministic policy and each BayesianLinearHead's posterior "
             "mean is warm-started from the deterministic readout; the "
             "deterministic BC loop and its own checkpoints are unaffected.",
    )
    parser.add_argument(
        "--bayesian-prior-precision",
        type=float,
        default=1.0,
        help="Prior precision (ridge strength) for each BayesianLinearHead "
             "when --bayesian-heads is set. Higher = stronger shrinkage "
             "toward zero and lower initial posterior variance. Default: 1.0.",
    )
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.3,
        help="Target Thompson-sampling exploration variance (physical bias-"
             "coefficient units) as a fraction of the real physical residual "
             "variance for each dimension, when --bayesian-heads is set. "
             "Raise (e.g. 0.5-1.0) for more exploration in online "
             "Bayesian-head training. Default: 0.3.",
    )
    parser.add_argument(
        "--emphasize-systems",
        type=str,
        action='append',
        default=None,
        metavar="NAME:WEIGHT",
        help="Up-weight a system's runs in the BC loss so it gets fit more "
             "accurately despite having few samples, e.g. "
             "--emphasize-systems luis_tyk2_solvent:5.0. Matched by "
             "case-insensitive substring against the --pretraining-dir "
             "basename. Repeat the flag to emphasize multiple systems.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="after training, write figures and a report to <output-dir>/analysis "
             "(mllf analyze pretrain; also the config key analysis: true)",
    )


def _emphasis(entries) -> Dict[str, float]:
    """--emphasize-systems NAME:WEIGHT entries as {name: weight}."""
    system_emphasis = {}
    for entry in entries or []:
        name, sep, weight_str = entry.rpartition(':')
        if not sep:
            log.warning(f"  Warning: --emphasize-systems entry '{entry}' missing ':WEIGHT'; "
                        "skipping")
            continue
        try:
            system_emphasis[name] = float(weight_str)
        except ValueError:
            log.warning(f"  Warning: --emphasize-systems entry '{entry}' has non-numeric weight; "
                        "skipping")
    if system_emphasis:
        log.info(f"System emphasis weights: {system_emphasis}")
    return system_emphasis


def _pretraining_dirs(args) -> List[str]:
    from mllf.pretrain.runs import read_manifest
    if args.manifest_file:
        if args.pretraining_dir:
            log.error("Error: Cannot specify both --manifest-file and --pretraining-dir")
            sys.exit(1)
        log.info(f"\nLoading pretraining directories from manifest: {args.manifest_file}")
        try:
            dirs = read_manifest(args.manifest_file)
        except FileNotFoundError:
            log.error(f"Error: Manifest file not found: {args.manifest_file}")
            sys.exit(1)
        except ValueError as e:
            log.error(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            log.error(f"Error reading manifest file: {e}")
            sys.exit(1)
        log.info(f"  Found {len(dirs)} directories in manifest")
        return dirs
    if args.pretraining_dir:
        return args.pretraining_dir
    log.error("Error: Must specify either --pretraining-dir or --manifest-file")
    sys.exit(1)


def run(args: argparse.Namespace) -> None:
    from mllf.log import log_to_file
    from mllf.pretrain.trainer import LOG_FILE

    dirs = _pretraining_dirs(args)     # argument errors before any output is written
    with log_to_file(Path(args.output_dir) / LOG_FILE):
        log.debug(f"Pretraining directories ({len(dirs)}): {dirs}")
        _write_manifest(Path(args.output_dir), dirs)
        _run(args, dirs)


def _write_manifest(output_dir: Path, dirs: List[str]) -> None:
    """pretraining_manifest.json: the directories pretraining read, as absolute
    paths (the regression replay's bandit.regression.replay_manifest)."""
    import json
    path = output_dir / 'pretraining_manifest.json'
    path.write_text(json.dumps({'directories': [str(Path(d).resolve()) for d in dirs]},
                               indent=1))


def _run(args: argparse.Namespace, dirs: List[str]) -> None:
    import yaml

    from mllf.pretrain.config import PretrainOptions
    from mllf.pretrain.runs import load_runs_from_dirs
    from mllf.pretrain.trainer import pretrain_with_runs

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    all_runs = load_runs_from_dirs(dirs)
    log.info(f"\nTotal runs from all directories: {len(all_runs)}")
    if args.max_runs is not None and len(all_runs) > args.max_runs:
        all_runs = random.Random(0).sample(all_runs, args.max_runs)
        log.info(f"Sampled {len(all_runs)} runs (--max-runs, seed 0)")
    system_emphasis = _emphasis(args.emphasize_systems)

    pretrain_with_runs(all_runs, Path(args.output_dir), config, PretrainOptions(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        use_best_only=args.use_best_only,
        filter_outliers=not args.no_filter_outliers,
        outlier_std_threshold=args.outlier_std_threshold,
        min_reward_threshold=args.min_reward_threshold,
        min_transitions=args.min_transitions,
        stratified_negative_fraction=args.stratified_negative_fraction,
        patience=args.patience,
        reward_weighted=args.reward_weighted,
        awr_temperature=args.awr_temperature,
        use_per_pair_awr=not args.no_per_pair_awr,
        bayesian_heads=args.bayesian_heads,
        bayesian_prior_precision=args.bayesian_prior_precision,
        exploration_fraction=args.exploration_fraction,
        system_emphasis=system_emphasis or None,
        data_hygiene=not args.no_data_hygiene,
        init_checkpoint=args.init_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
    ))
    if args.analyze or config.get('analysis'):
        _analyze(args)


def _analyze(args: argparse.Namespace) -> None:
    """``--analyze``: the model's analysis (mllf analyze pretrain); a failure
    is a warning, the trained model stands."""
    out = Path(args.output_dir) / 'analysis'
    try:
        from mllf.analysis.common import require_plotting
        from mllf.analysis.pretrain import analyze_pretrain
        require_plotting()
        log.info(f"\nAnalysis -> {out}")
        analyze_pretrain(Path(args.output_dir), out, config_path=args.config, device=args.device)
    except Exception as e:
        log.warning(f"  Warning: analysis not completed ({type(e).__name__}: {e}); run "
                    f"'mllf analyze pretrain {args.output_dir}' to retry")


def main(argv=None):
    from mllf.cli.common import run_command
    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG)
    add_arguments(parser)
    return run_command(parser, run, argv)


if __name__ == "__main__":
    main()
