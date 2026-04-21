#!/usr/bin/env python3
"""Build a warm-start bias coefficient mapping from a pretraining directory.

Scans a pretraining directory for run{N}/ subdirectories, ranks them by a
configurable scoring metric, and saves a JSON mapping of pretraining bias
coefficients indexed by (original_site, original_sub).  This mapping is
consumed by WarmStartMapper during RL training to initialise epoch 0
(or any chosen epoch) with physically meaningful bias coefficients rather
than random policy samples.

Usage (14benz example):
    python examples/build_warmstart_map.py \\
        --pretrain-dir /home/.../pretraining/14benz_solv \\
        --nsubs-per-site 6 5 \\
        --output /home/.../models/warmstart_14benz_solv.json

    # Average over top 3 runs:
    python examples/build_warmstart_map.py \\
        --pretrain-dir /home/.../pretraining/14benz_solv \\
        --nsubs-per-site 6 5 \\
        --output warmstart.json \\
        --top-k 3

The resulting JSON can be referenced in the workflow YAML:

    warmstart:
      mapping_path: /home/.../models/warmstart_14benz_solv.json
      epoch: 0   # which epoch uses the warm-start biases (default 0)
"""
import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mllf.cb.warmstart import build_warmstart_map, save_warmstart_map


def parse_args():
    p = argparse.ArgumentParser(
        description='Build a warm-start bias map from a pretraining directory.'
    )
    p.add_argument(
        '--pretrain-dir', required=True,
        help='Directory containing run{N}/ subdirs with variables.py and '
             'simulation_results.json (e.g. pretraining/14benz_solv).'
    )
    p.add_argument(
        '--nsubs-per-site', required=True, nargs='+', type=int,
        metavar='N',
        help='Number of substituents per site in the pretraining system, '
             'e.g. --nsubs-per-site 6 5 for a 2-site system with 6 and 5 subs.'
    )
    p.add_argument(
        '--output', required=True,
        help='Path for the output JSON file.'
    )
    p.add_argument(
        '--scoring', default='total_transitions',
        choices=['total_transitions', 'num_populated_blocks'],
        help='Metric used to rank pretraining runs (default: total_transitions).'
    )
    p.add_argument(
        '--top-k', type=int, default=1,
        help='Number of top-ranked runs to average (default: 1 = best run only).'
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Scanning pretraining directory: {args.pretrain_dir}")
    print(f"  nsubs_per_site: {args.nsubs_per_site}")
    print(f"  scoring:        {args.scoring}")
    print(f"  top_k:          {args.top_k}")

    try:
        map_dict = build_warmstart_map(
            pretrain_dir=args.pretrain_dir,
            nsubs_per_site=args.nsubs_per_site,
            scoring=args.scoring,
            top_k=args.top_k,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    save_warmstart_map(map_dict, args.output)
    print(f"\nBest run(s): {map_dict['source_runs']}")
    print(f"Score ({map_dict['score_type']}): {map_dict['score']:.1f}")


if __name__ == '__main__':
    main()
