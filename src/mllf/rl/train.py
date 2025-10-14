"""Training script for A2C using Stable Baselines3.

This script is intentionally minimal and safe to import. Running it will
execute a short training loop when invoked as a script.
"""
from typing import Optional

import os
import sys
import json
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

# SB3-based training was removed in favor of RLlib. The helpers below still
# build Graph objects from RTF directories for use by trainers (RLlib or
# custom PyTorch loops).

from .graph import Graph
from ..file_handling.read_rtf import parse_rtf_dir
from ..mlp.setup_pairs import find_variables_file
import subprocess
import warnings
from ..file_handling.write_bias_coeff import create_variables_py_from_template


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


def build_graphs_from_config(config_path: str, initialize: bool = True, msld_script: str = None):
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
        # optionally perform initialization: ensure variables file exists and run the MD script
        if initialize:
            try:
                vars_file = find_variables_file(prep_dir)
                if vars_file is None:
                    warnings.warn(f"No variables file found in {prep_dir}; skipping initialization for this sim.")
                else:
                    # run msld_flat.py (or provided msld_script) in the prep_dir with vars-file argument
                    msld = msld_script or os.path.join(os.getcwd(), 'examples', 'rl', 'msld_flat.py')
                    # if the msld script is not present at msld, try prep_dir
                    if not os.path.exists(msld):
                        local_msld = os.path.join(prep_dir, 'msld_flat.py')
                        if os.path.exists(local_msld):
                            msld = local_msld
                    # Prefer using variablesflat.py as the initialization variables file if present
                    vars_template = os.path.join(prep_dir, 'variablesflat.py') if os.path.exists(os.path.join(prep_dir, 'variablesflat.py')) else vars_file
                    cmd = [sys.executable, msld, '--vars-file', vars_template]
                    # run the initialization script in the prep_dir so relative paths resolve
                    try:
                        subprocess.run(cmd, cwd=prep_dir, check=True)
                    except Exception as e:
                        warnings.warn(f"Initialization run failed for {prep_dir}: {e}")
                    # create a variables0.py file for step 0 with minimizeflag=False
                    try:
                        # choose template: prefer variablesflat.py in prep_dir when present
                        template = os.path.join(prep_dir, 'variablesflat.py') if os.path.exists(os.path.join(prep_dir, 'variablesflat.py')) else vars_file
                        # allow user to specify output directory for variables files
                        vars_out_dir = sim.get('vars_out_dir') if isinstance(sim, dict) else None
                        if vars_out_dir:
                            out_dir = os.path.abspath(vars_out_dir)
                        else:
                            out_dir = prep_dir
                        out0 = os.path.join(out_dir, 'variables0.py')
                        create_variables_py_from_template(template, out0, minimizeflag=False)
                        # if sim specifies n_steps, pre-create variables1..variables{n_steps-1}.py
                        nsteps_cfg = sim.get('n_steps') if isinstance(sim, dict) else None
                        if isinstance(nsteps_cfg, int) and nsteps_cfg > 1:
                            for step_idx in range(1, nsteps_cfg):
                                outp = os.path.join(out_dir, f'variables{step_idx}.py')
                                create_variables_py_from_template(template, outp, minimizeflag=False)
                    except Exception as e:
                        warnings.warn(f"Failed to create variables0.py in {prep_dir}: {e}")
                    # after initialization attempt, if variables file is a .py then set minimizeflag=False
                    if vars_file.endswith('.py'):
                        try:
                            with open(vars_file, 'r', encoding='utf-8') as fh:
                                txt = fh.read()
                            if 'minimizeflag' in txt:
                                txt_new = txt.replace('minimizeflag=True', 'minimizeflag=False')
                                if txt_new != txt:
                                    with open(vars_file, 'w', encoding='utf-8') as fh:
                                        fh.write(txt_new)
                        except Exception:
                            warnings.warn(f"Failed to update minimizeflag in {vars_file}")
            except Exception:
                warnings.warn(f"Initialization step encountered an error for sim at {prep_dir}")
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
