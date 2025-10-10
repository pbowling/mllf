"""Training script for A2C using Stable Baselines3.

This script is intentionally minimal and safe to import. Running it will
execute a short training loop when invoked as a script.
"""
from typing import Optional

import os
import json
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

# SB3-based training was removed in favor of RLlib. The helpers below still
# build Graph objects from RTF directories for use by trainers (RLlib or
# custom PyTorch loops).

from .wrappers import make_env
from .graph import Graph
from ..file_handling.read_rtf import parse_rtf_dir


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError("PyYAML not installed; install it to load YAML configs")
        return yaml.safe_load(p.read_text())
    else:
        return json.loads(p.read_text())


def build_graphs_from_config(config_path: str):
    """Parse config and return a list of Graph objects (one per simulation).

    This helper preserves the RTF parsing and Graph construction logic that
    previously lived in the A2C trainer. Trainers (RLlib or custom loops) can
    call this to obtain Graphs with node metadata.
    """
    cfg = _load_config(config_path)
    sims = cfg if isinstance(cfg, list) else cfg.get('simulations')
    if not sims:
        raise RuntimeError("No simulations found in config")

    sim_graphs = []
    for sim in sims:
        prep_dir = sim.get('prep_dir')
        if prep_dir is None:
            continue
        rtf_results = parse_rtf_dir(prep_dir)
        # allow optional solvent override in sim config
        solvent = sim.get('solvent') if isinstance(sim, dict) else None
        graph_sim = Graph.from_rtf_results(rtf_results, solvent_override=solvent)
        sim_graphs.append(graph_sim)
    return sim_graphs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to JSON/YAML training config")
    p.add_argument("--timesteps", type=int, default=2000)
    p.add_argument("--resume", help="Path to an existing model to resume training from", default=None)
    p.add_argument("--save-name", help="Filename to save trained model as (inside out_dir)", default=None)
    args = p.parse_args()
    graphs = build_graphs_from_config(args.config)
    for i, g in enumerate(graphs):
        print(f"Graph {i}: nodes={g.num_nodes}, edges={len(g.edges)}")
