"""Orchestration for contextual-bandit training using the repo CB stack.

Features:
- Load a manifest of combo directories.
- Optionally run each combo's `./run.sh` (or custom command) and wait for completion.
- Parse simulation outputs via `mllf.file_handling.read_output` helpers.
- Build an mllf Graph from `variables.py` bias_string and convert to PyG Data.
- Instantiate `RGCNEncoder` + `EdgePolicy` and run policy updates using
  `mllf.cb.train.reinforce_train_step`.
- Progressive checkpointing with metadata and rotation.

This script is deliberately pragmatic: it provides a clear CLI and default
behaviour (dry-run/backfill reward from bias matrices) while allowing you to
plug-in a custom simulation command or reward function.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple, Optional

import yaml
import torch

from mllf.cb.graph import Graph, EdgeCoeffs
from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy
from mllf.cb import train as cb_train
from mllf.cb import graph_utils
from mllf.file_handling.read_output import parse_transitions_and_rates, parse_single_population, terminated_normally
from mllf.file_handling.generate_combinations import create_combination_dirs
from mllf.cli.sim import run_simulation_command, parse_simulation_results, run_simulation_batch

import math
import torch.nn as nn
import torch.nn.functional as F


def load_bias_from_variables(py_path: str) -> Dict[str, Any]:
    """Load the YAML bias mapping embedded inside a `variables.py` file.

    Args:
        py_path: Path to a `variables.py` file.

    Returns:
        Dict parsed from the triple-quoted `bias_string` YAML block, or {} on
        failure.
    """
    text = Path(py_path).read_text(encoding='utf-8')
    m = __import__('re').search(r"bias_string\s*=\s*(?:\"\"\"|''')([\s\S]*?)(?:\"\"\"|''')", text, __import__('re').S)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


def graph_from_bias(bias: Dict[str, Any]) -> Graph:
    """Build an mllf `Graph` from a bias dict (as produced by `load_bias_from_variables`).

    The returned Graph will have explicit EdgeCoeffs only for pairs where the
    input bias matrices contain non-zero values; otherwise the Graph will have
    connectivity (masks) but leave the coefficient values to be predicted by
    the policy/MLP.

    Args:
        bias: dict with keys 'b','c','x','s' (b may be a flat list, c/x/s NxN matrices).

    Returns:
        Graph instance sized by the length of flattened `b` (or 1 if missing).
    """
    b = bias.get('b', [])
    if isinstance(b, list) and b and isinstance(b[0], list):
        flat_b = [float(x) for row in b for x in row]
    elif isinstance(b, list):
        flat_b = [float(x) for x in b]
    else:
        flat_b = []

    N = len(flat_b) if flat_b else 1
    c = bias.get('c', [])
    x = bias.get('x', [])
    s = bias.get('s', [])

    g = Graph(int(N))
    for i in range(int(N)):
        for j in range(i + 1, int(N)):
            try:
                cval = float(c[i][j]) if c and len(c) > i and len(c[i]) > j else 0.0
            except Exception:
                cval = 0.0
            try:
                xval = float(x[i][j]) if x and len(x) > i and len(x[i]) > j else 0.0
            except Exception:
                xval = 0.0
            try:
                sval = float(s[i][j]) if s and len(s) > i and len(s[i]) > j else 0.0
            except Exception:
                sval = 0.0
            # do not set edges here — only mark existence by setting mask; leave
            # coefficient values to be estimated by the policy/MLP
            # but we create EdgeCoeffs if explicit values are present (non-zero)
            if any((cval, xval, sval)):
                g.set_edge(i, j, EdgeCoeffs(linear=0.0, quadratic=cval, skew=xval, end=sval))

    return g


# run_simulation_command/imported from mllf.cli.sim
# Using shared implementation from mllf.cli.sim: run_simulation_command


# parse_simulation_results: using shared implementation from mllf.cli.sim


def save_checkpoint(out_dir: str, policy: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int, step: int, keep: int = 5):
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    fname = od / f'cb_ckpt_epoch_{epoch:04d}_step_{step:06d}.pt'
    meta = {'epoch': epoch, 'step': step, 'ts': datetime.utcnow().isoformat()}
    torch.save({'policy': policy.state_dict(), 'optim': optim.state_dict(), 'meta': meta}, str(fname))

    # rotate
    ckpts = sorted(od.glob('cb_ckpt_epoch_*.pt'))
    if len(ckpts) > keep:
        # remove older
        for rm in ckpts[:-keep]:
            try:
                rm.unlink()
            except Exception:
                pass


def default_env_reward(actions: torch.Tensor, target_vals: List[float], sim_results: Dict[str, Any] = None) -> float:
    # If sim_results present, allow more complex mapping later. For now, use
    # negative MSE vs provided target_vals (per-edge targets).
    try:
        targ = torch.tensor(target_vals, dtype=actions.dtype, device=actions.device)
        mse = torch.mean((actions - targ) ** 2).item()
        return -mse
    except Exception:
        return float('-inf')


def build_data_and_targets_from_combo(combo_dir: str, base_bias: str = 'quadratic'):
    """Build PyG `data`, per-edge `targets`, and `extras` for a combo dir.

    Default behaviour: prefer RTF fragments (Graph.from_rtf_results) when
    available. If no RTF fragments are found, fall back to reading
    `variables.py` and constructing a Graph from the embedded YAML `bias_string`.

    Args:
        combo_dir: path to a combo directory containing either RTF fragments
                   or a `variables.py` file.
        base_bias: which base bias to use when constructing per-edge targets
                   ('quadratic', 'skew', 'end').

    Returns:
        Tuple (data, targets, extras) where `data` is a PyG Data object, and
        `targets` is a python list of per-directed-edge floats aligned to
        `data.edge_index`.

    Raises:
        FileNotFoundError if neither RTF fragments nor `variables.py` are
        present.
    """
    # Prefer building from RTF fragments (default behaviour per-user request)
    # bias dict may be absent when building from RTFs; initialize to empty
    bias: Dict[str, Any] = {}
    rtf_results = parse_rtf_dir(combo_dir)
    if rtf_results:
        g = Graph.from_rtf_results(rtf_results)
    else:
        vpy = Path(combo_dir) / 'variables.py'
        if vpy.exists():
            bias = load_bias_from_variables(str(vpy))
            g = graph_from_bias(bias)
        else:
            raise FileNotFoundError(f'No RTF fragments and no variables.py found in {combo_dir}')

    data, extras = graph_utils.build_pyg_graph_from_mllf_graph(g)

    # build per-edge multi-dimensional targets aligned to data.edge_index.
    # We produce a vector per directed edge containing values for each base
    # relation (e.g. [quadratic, skew, end]) in the order given by
    # extras['base_relation_map']. This supports edges that have different
    # relation types and allows the policy to predict all bases simultaneously.
    rel_names = extras.get('relation_names', [])
    base_map = extras.get('base_relation_map', {})
    # base_order determines output ordering for multi-dim targets
    # Ensure 'linear' (b) is included as a base if not present — some graphs
    # represent node-level linear biases as a base relation as well.
    base_order = list(base_map.keys()) if isinstance(base_map, dict) else ['quadratic', 'skew', 'end']
    if 'linear' not in base_order:
        base_order.append('linear')
    D = len(base_order)

    # map relation name -> base index
    relname_to_baseidx = {}
    for b_idx, (base, (fwd, bwd)) in enumerate(base_map.items()):
        relname_to_baseidx[fwd] = b_idx
        relname_to_baseidx[bwd] = b_idx

    # helper to access bias matrices by base name
    base_to_matrix = {
        'quadratic': bias.get('c', []),
        'skew': bias.get('x', []),
        'end': bias.get('s', []),
        'linear': bias.get('b', []),
    }

    targets = []
    ei = data.edge_index
    for k in range(ei.shape[1]):
        src = int(ei[0, k].item())
        dst = int(ei[1, k].item())
        rel_idx = int(data.edge_type[k].item()) if hasattr(data, 'edge_type') and data.edge_type.numel() > k else None
        rel_name = rel_names[rel_idx] if rel_idx is not None and rel_idx < len(rel_names) else None

        # initialize zero vector for all bases
        vec = [0.0 for _ in range(D)]
        if rel_name is not None:
            base_idx = relname_to_baseidx.get(rel_name)
            if base_idx is not None:
                base_name = base_order[base_idx]
                mat = base_to_matrix.get(base_name, [])
                try:
                    if base_name == 'linear':
                        # b may be a flat list of length N (per-node) or an NxN matrix.
                        bmat = mat
                        if isinstance(bmat, list) and bmat:
                            # matrix-like
                            if isinstance(bmat[0], list):
                                val = float(bmat[src][dst]) if len(bmat) > src and len(bmat[src]) > dst else 0.0
                            else:
                                # flat list: use average of node b's as edge-level target
                                try:
                                    lhs = float(bmat[src])
                                except Exception:
                                    lhs = 0.0
                                try:
                                    rhs = float(bmat[dst])
                                except Exception:
                                    rhs = 0.0
                                val = 0.5 * (lhs + rhs)
                        else:
                            val = 0.0
                    else:
                        val = float(mat[src][dst]) if mat and len(mat) > src and len(mat[src]) > dst else 0.0
                except Exception:
                    val = 0.0
                vec[base_idx] = val

        targets.append(vec)

    return data, targets, extras


def write_variables_from_actions(combo_dir: str, data, extras: dict, actions: torch.Tensor, out_name: str = 'variables.py') -> None:
    """Write a `variables.py` file in combo_dir from per-directed-edge actions.

    The function maps directed relation actions back to base biases (quadratic, skew, end)
    and averages forward/backward values for each undirected pair. It then writes
    a Python file containing a triple-quoted YAML `bias_string` with keys 'b','c','x','s'.
    """
    combo_dir = Path(combo_dir)
    N = int(data.x.shape[0])

    # build reverse mapping rel_idx -> rel_name
    rel_names = extras.get('relation_names', [])
    rel_to_base = {}
    base_map = extras.get('base_relation_map', {})
    for base, (fwd, bwd) in base_map.items():
        rel_to_base[fwd] = base
        rel_to_base[bwd] = base

    # base output ordering (used when edge actions are vector-valued)
    base_order = list(base_map.keys()) if isinstance(base_map, dict) else ['quadratic', 'skew', 'end']
    if 'linear' not in base_order:
        base_order.append('linear')

    # collect per-undirected-pair forward-only values. For each directed edge
    # we only accept the forward relation (e.g. 'quadratic_fwd') as the canonical
    # source of truth for the pair; the reverse value is the negative of the
    # forward one (AB = -BA). If forward is missing, we fall back to using the
    # backward value with inverted sign.
    per_base_forward = {name: {} for name in base_order}
    ei = data.edge_index
    et = data.edge_type
    for k in range(ei.shape[1]):
        src = int(ei[0, k].item())
        dst = int(ei[1, k].item())
        rel_idx = int(et[k].item()) if hasattr(data, 'edge_type') and data.edge_type.numel() > k else None
        rel_name = rel_names[rel_idx] if rel_idx is not None and rel_idx < len(rel_names) else None
        if rel_name is None:
            continue
        base = rel_to_base.get(rel_name)
        if base is None:
            continue
        # determine if this directed relation is forward or backward for the pair
        fwd_name, bwd_name = base_map.get(base, (f"{base}_fwd", f"{base}_bwd"))
        pair = (min(src, dst), max(src, dst))
        if rel_name == fwd_name and (pair not in per_base_forward[base]):
            # canonical forward value for this undirected pair
            try:
                a = actions[k]
                val = float(a.item()) if hasattr(a, 'dim') and a.dim() == 0 else float(a.detach().cpu().numpy().tolist())
            except Exception:
                try:
                    val = float(actions[k])
                except Exception:
                    val = 0.0
            per_base_forward[base][pair] = val
        elif rel_name == bwd_name and (pair not in per_base_forward[base]):
            # we only have backward; store inverse so forward= -backward
            try:
                a = actions[k]
                val = float(a.item()) if hasattr(a, 'dim') and a.dim() == 0 else float(a.detach().cpu().numpy().tolist())
            except Exception:
                try:
                    val = float(actions[k])
                except Exception:
                    val = 0.0
            per_base_forward[base][pair] = -val

    # assemble matrices
    def build_mat_for(base_name: str):
        mat = [[0.0 for _ in range(N)] for _ in range(N)]
        vals_map = per_base_forward.get(base_name, {})
        for (i, j), val in vals_map.items():
            try:
                v = float(val)
            except Exception:
                v = 0.0
            # forward/backward are inverses: set AB = v and BA = -v
            mat[i][j] = v
            mat[j][i] = -v
        return mat

    c_mat = build_mat_for('quadratic')
    x_mat = build_mat_for('skew')
    s_mat = build_mat_for('end')
    # derive per-node linear b from per-edge linear values (average incident edges)
    b_vec = [0.0 for _ in range(N)]
    linear_vals = per_base_forward.get('linear', {})
    if linear_vals:
        sums = [0.0 for _ in range(N)]
        counts = [0 for _ in range(N)]
        for (i, j), val in linear_vals.items():
            try:
                avg = float(val)
            except Exception:
                avg = 0.0
            sums[i] += avg
            sums[j] += avg
            counts[i] += 1
            counts[j] += 1
        for idx in range(N):
            if counts[idx] > 0:
                b_vec[idx] = sums[idx] / counts[idx]
            else:
                b_vec[idx] = 0.0

    # if caller provided a special attribute on actions container with node-level
    # b values (convention: actions may be a tuple (edge_actions, node_b_actions)
    # but older callers pass a plain tensor). Support either.
    if isinstance(actions, tuple) and len(actions) >= 2:
        edge_actions, node_b = actions[0], actions[1]
        try:
            node_b_list = [float(x) for x in (node_b.detach().cpu().numpy().tolist() if hasattr(node_b, 'detach') else list(node_b))]
            if len(node_b_list) >= N:
                b_vec = node_b_list[:N]
            else:
                # pad if shorter
                b_vec = node_b_list + [0.0] * (N - len(node_b_list))
        except Exception:
            pass
    else:
        # no node-level b provided; keep zeros
        pass

    bias_dict = {
        'b': b_vec,
        'c': c_mat,
        'x': x_mat,
        's': s_mat,
    }

    yaml_block = yaml.safe_dump(bias_dict, sort_keys=False)

    content = f"""# Auto-generated variables.py — bias_string contains YAML for bias matrices
bias_string = '''\
{yaml_block}
'''
"""
    (combo_dir / out_name).write_text(content, encoding='utf-8')


def run_training(manifest: str, out_dir: str, epochs: int = 10, lr: float = 1e-3, base_bias: str = 'quadratic', run_sims: bool = False, sim_cmd: Optional[str] = None, checkpoint_every: int = 100, keep: int = 5, dry_run: bool = False, w_trans: float = 1.0, w_pop: float = 1.0, pop_normalize: bool = True):
    """Train edge and node policies using a manifest of combo directories.

    Behaviour:
      - Loads combos from a manifest (one combo dir per line).
      - Builds a PyG graph for each combo (prefers RTF fragments by default).
      - Instantiates an RGCN encoder and an EdgePolicy; also creates an
        internal NodePolicy head to predict per-node linear biases `b`.
      - For each combo in each epoch: samples stochastic edge and node
        actions, writes `variables.py` for the combo (so the simulator can
        read them), optionally runs the simulator, parses outputs, computes
        a scalar reward, and uses REINFORCE to update both policies jointly.

    Args:
        manifest: path to manifest file listing combo directories, one per line.
        out_dir: directory to save checkpoints.
        epochs: number of epochs to run.
        lr: learning rate for optimizer.
        base_bias: which base bias ('quadratic','skew','end') to use for per-edge targets.
        run_sims: if True, run the simulator (via run_simulation_command) for each combo.
        sim_cmd: optional override shell command to execute in each combo dir.
        checkpoint_every: steps between checkpoint saves.
        keep: number of checkpoints to keep.
        dry_run: if True, do not execute simulator commands.
        w_trans: weight for transition-count component of the reward.
        w_pop: weight for population nonzero-blocks component of the reward.
        pop_normalize: whether to normalize population score by number of blocks.
    """
    with open(manifest, 'r', encoding='utf-8') as fh:
        combos = [ln.strip() for ln in fh if ln.strip()]

    if not combos:
        raise RuntimeError('No combos in manifest')

    dataset = []
    for c in combos:
        try:
            data, targets, extras = build_data_and_targets_from_combo(c, base_bias=base_bias)
            dataset.append((c, data, torch.tensor(targets, dtype=torch.float32), extras))
        except Exception as e:
            print(f'Warning: skipping {c}: {e}')

    if not dataset:
        raise RuntimeError('No valid combos in manifest')

    sample_data = dataset[0][1]
    sample_extras = dataset[0][3]
    in_dim = sample_data.x.shape[1]
    num_rels = int(sample_data.edge_attr.shape[1]) if hasattr(sample_data, 'edge_attr') else 1
    encoder = RGCNEncoder(in_dim=in_dim, hidden_dims=[64], out_dim=32, num_relations=num_rels)

    # determine per-edge output dimensionality from base_relation_map (e.g. quadratic, skew, end)
    base_map = sample_extras.get('base_relation_map', {}) if isinstance(sample_extras, dict) else {}
    base_order = list(base_map.keys()) if isinstance(base_map, dict) else []
    edge_out_dim = len(base_order) if base_order else 1

    policy = EdgePolicy.from_pyg_data(encoder, 32, sample_data, mlp_hidden=64, mlp_out_dim=edge_out_dim)
    policy.train()

    # We predict all biases from per-edge MLP outputs; do not use a separate
    # node-level stochastic head. The EdgePolicy will produce per-edge means
    # and log-stds for the requested coefficient types.
    optim = torch.optim.Adam(list(policy.parameters()), lr=lr)

    step = 0
    best_score = float('-inf')

    for epoch in range(1, epochs + 1):
        epoch_rewards = []
        for combo_dir, data, targets, extras in dataset:
            if run_sims and not dry_run:
                rc, so, se = run_simulation_command(combo_dir, cmd=sim_cmd)
                if rc != 0:
                    print(f'Warning: simulation for {combo_dir} returned {rc}, stderr={se[:200]}')

            # initial parse of existing simulation results (if any)
            sim_results = parse_simulation_results(combo_dir)

            def env_reward_fn(edge_actions, node_b_actions=None, _targets=targets, _w_trans=w_trans, _w_pop=w_pop, _pop_norm=pop_normalize, _run_sims=run_sims, _sim_cmd=sim_cmd, _dry_run=dry_run, _combo_dir=combo_dir, _data=data, _extras=extras):
                # Given sampled actions, write variables.py for this combo, run
                # the simulation (if enabled), parse outputs, and compute reward.
                try:
                    # write variables for these actions; support passing node-level b
                    if node_b_actions is not None:
                        write_variables_from_actions(_combo_dir, _data, _extras, (edge_actions, node_b_actions))
                    else:
                        write_variables_from_actions(_combo_dir, _data, _extras, edge_actions)

                    # execute simulation only if requested and not dry-run
                    if _run_sims and not _dry_run:
                        rc, so, se = run_simulation_command(_combo_dir, cmd=_sim_cmd)
                        if rc != 0:
                            # simulation failed; return a large negative reward
                            return float('-1e6')

                    sim = parse_simulation_results(_combo_dir)

                    trans = sim.get('transitions', {})
                    pops = sim.get('population', {})

                    # total transitions (sum over sites and lambdas)
                    total_trans = 0
                    for site, vals in (trans or {}).items():
                        for lv, cnt in vals.items():
                            try:
                                total_trans += int(cnt)
                            except Exception:
                                total_trans += 0

                    # count blocks with any non-zero population (across lambdas)
                    total_blocks = 0
                    nonzero_blocks = 0
                    for block, info in (pops or {}).items():
                        total_blocks += 1
                        counts = info.get('counts', {}) if isinstance(info, dict) else {}
                        block_sum = 0
                        for lv, cnt in counts.items():
                            try:
                                block_sum += int(cnt)
                            except Exception:
                                block_sum += 0
                        if block_sum > 0:
                            nonzero_blocks += 1

                    pop_score = (nonzero_blocks / total_blocks) if (_pop_norm and total_blocks > 0) else float(nonzero_blocks)
                    return float(_w_trans * float(total_trans) + _w_pop * float(pop_score))
                except Exception:
                    # if anything fails, fallback to negative MSE vs targets
                    return default_env_reward(edge_actions, _targets, None)

            # Custom REINFORCE step combining edge- and node-level stochastic policies
            policy.train()
            optim.zero_grad()

            # node embeddings (provided by encoder)
            node_emb = policy.forward_node_embeddings(data.x, data.edge_index, data.edge_type)

            # edge actions and log-probs (policy returns actions, logp, mean, log_std)
            edge_actions, edge_logp, edge_mean, edge_logstd = policy.get_actions(data.x, data.edge_index, data.edge_type, getattr(data, 'edge_attr', None), deterministic=False)

            # call environment to get scalar reward (run sims/writing variables inside env)
            with torch.no_grad():
                try:
                    reward = float(env_reward_fn(edge_actions.detach()))
                except Exception:
                    reward = float(default_env_reward(edge_actions.detach(), targets, None))

            # combine log-probs and apply REINFORCE update (edge-only)
            total_logp = edge_logp.sum()
            adv = reward - 0.0
            loss = -(total_logp * adv)
            loss.backward()
            optim.step()

            epoch_rewards.append(reward)
            step += 1

            if step % checkpoint_every == 0:
                save_checkpoint(out_dir, policy, optim, epoch, step, keep=keep)

        avg_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
        print(f'Epoch {epoch}: avg_reward={avg_reward:.6f}')
        if avg_reward > best_score:
            best_score = avg_reward
            save_checkpoint(out_dir, policy, optim, epoch, step, keep=keep)



def _cli():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', nargs='?', default=None, help='Path to a manifest file listing combo directories (one per line). If omitted, use --gen-combos to generate combos.')
    p.add_argument('--out', default='cb_training')
    p.add_argument('--gen-combos', dest='gen_combos', default=None, help='If provided, input directory containing site{n}_sub{m} files to generate combo dirs from')
    p.add_argument('--gen-out', dest='gen_out', default='combos', help='Output directory for generated combos (used with --gen-combos)')
    p.add_argument('--gen-dry-run', dest='gen_dry', action='store_true', help='Dry-run generation of combos (with --gen-combos)')
    p.add_argument('--run-batch', dest='run_batch', action='store_true', help='Run simulations concurrently for the manifest and summarize results (uses run_simulation_batch)')
    p.add_argument('--batch-workers', dest='batch_workers', type=int, default=4, help='Number of concurrent workers when running batch simulations')
    p.add_argument('--batch-timeout', dest='batch_timeout', type=int, default=None, help='Per-simulation timeout seconds for batch run')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bias', default='quadratic', choices=['quadratic', 'skew', 'end'])
    p.add_argument('--run-sims', action='store_true', help='Run ./run.sh in each combo directory when present')
    p.add_argument('--sim-cmd', default=None, help='Custom shell command to run for each combo (overrides --run-sims default)')
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--keep', type=int, default=5)
    p.add_argument('--dry-run', action='store_true', help='Do not execute simulation commands; useful for CI/dry testing')
    p.add_argument('--w-trans', type=float, default=1.0, help='Weight for transition counts in combined reward')
    p.add_argument('--w-pop', type=float, default=1.0, help='Weight for population nonzero-score in combined reward')
    p.add_argument('--pop-normalize', action='store_true', help='Normalize population score by number of blocks (fraction instead of count)')
    args = p.parse_args()
    manifest_to_use = args.manifest
    # If requested, generate combos and create a manifest automatically
    if args.gen_combos:
        input_dir = Path(args.gen_combos)
        out_dir = Path(args.gen_out)
        created = create_combination_dirs(input_dir, out_dir, dry_run=args.gen_dry)
        manifest_path = out_dir / 'manifest.txt'
        with manifest_path.open('w') as fh:
            for pth in created:
                fh.write(str(pth) + '\n')
        manifest_to_use = str(manifest_path)

    # If requested, run the batch concurrently and exit
    if args.run_batch:
        if manifest_to_use is None:
            raise SystemExit('No manifest specified for batch run (provide a manifest or --gen-combos)')
        summary = run_simulation_batch(manifest_to_use, sim_cmd=args.sim_cmd, max_workers=args.batch_workers, timeout=args.batch_timeout)
        # print summary JSON to stdout
        print(json.dumps(summary, indent=2))
        return

    run_training(
        manifest_to_use,
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        base_bias=args.bias,
        run_sims=args.run_sims,
        sim_cmd=args.sim_cmd,
        checkpoint_every=args.checkpoint_every,
        keep=args.keep,
        dry_run=args.dry_run,
        w_trans=args.w_trans,
        w_pop=args.w_pop,
        pop_normalize=args.pop_normalize,
    )


if __name__ == '__main__':
    _cli()
