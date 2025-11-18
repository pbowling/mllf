"""High-level workflow utilities for preparing combos, training and running sims.

This module centralizes common steps used by the example orchestration so
other scripts can call a simple YAML/JSON-driven CLI to create combos,
split manifests, run a quick training epoch, run simulations (concurrently)
and compress completed runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import random
import shutil
import tarfile
import os

import torch

from mllf.file_handling.generate_combinations import create_combination_dirs
from mllf.cb.graph import Graph, EdgeCoeffs
from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.cb import graph_utils
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy
from mllf.cli.sim import run_simulation_batch, parse_simulation_results


def load_bias_from_variables(py_path: str) -> Dict[str, Any]:
    text = Path(py_path).read_text(encoding='utf-8')
    m = __import__('re').search(r"bias_string\s*=\s*(?:\"\"\"|''')([\s\S]*?)(?:\"\"\"|''')", text, __import__('re').S)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


def graph_from_bias(bias: Dict[str, Any]) -> Graph:
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
            if any((cval, xval, sval)):
                g.set_edge(i, j, EdgeCoeffs(linear=0.0, quadratic=cval, skew=xval, end=sval))
    return g


def build_data_and_targets_from_combo(combo_dir: str, base_bias: str = 'quadratic', verify_graph: bool = False):
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

    # Optional verification: ensure PyG edges correspond to Graph.edge_mask
    if verify_graph:
        rel_names = extras.get('relation_names', [])
        rel_map = extras.get('base_relation_map', {})
        # build a set of directed edges present in data keyed by (src,dst,rel_name)
        present = set()
        ei = data.edge_index
        et = data.edge_type
        for k in range(ei.shape[1]):
            s = int(ei[0, k].item())
            d = int(ei[1, k].item())
            ridx = int(et[k].item()) if et.numel() > k else None
            rname = rel_names[ridx] if ridx is not None and ridx < len(rel_names) else None
            present.add((s, d, rname))

        for (i, j), mask in getattr(g, 'edge_mask', {}).items():
            for base in ('linear', 'quadratic', 'skew', 'end'):
                allowed = bool(mask.get(base, False))
                if not allowed:
                    # ensure no forward/backward edges for this base in present
                    fwd, bwd = rel_map.get(base, (f"{base}_fwd", f"{base}_bwd"))
                    if (i, j, fwd) in present or (j, i, bwd) in present:
                        raise RuntimeError(f"Graph verification failed: unexpected directed edge for base {base} on ({i},{j})")
                else:
                    fwd, bwd = rel_map.get(base, (f"{base}_fwd", f"{base}_bwd"))
                    if (i, j, fwd) not in present or (j, i, bwd) not in present:
                        raise RuntimeError(f"Graph verification failed: missing directed edges for base {base} on ({i},{j})")

    # build per-edge multi-dimensional targets aligned to data.edge_index.
    rel_names = extras.get('relation_names', [])
    base_map = extras.get('base_relation_map', {})
    # base_order determines output ordering for multi-dim targets
    base_order = list(base_map.keys()) if isinstance(base_map, dict) else ['quadratic', 'skew', 'end']
    if 'linear' not in base_order:
        base_order.append('linear')

    D = len(base_order)

    # map relation name -> base index
    relname_to_baseidx = {}
    for b_idx, (base, (fwd, bwd)) in enumerate(base_map.items()):
        relname_to_baseidx[fwd] = b_idx
        relname_to_baseidx[bwd] = b_idx

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

        vec = [0.0 for _ in range(D)]
        if rel_name is not None:
            base_idx = relname_to_baseidx.get(rel_name)
            if base_idx is not None:
                base_name = base_order[base_idx]
                mat = base_to_matrix.get(base_name, [])
                try:
                    if base_name == 'linear':
                        bmat = mat
                        if isinstance(bmat, list) and bmat:
                            if isinstance(bmat[0], list):
                                val = float(bmat[src][dst]) if len(bmat) > src and len(bmat[src]) > dst else 0.0
                            else:
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
    combo_dir = Path(combo_dir)
    N = int(data.x.shape[0])
    base_map = extras.get('base_relation_map', {})
    base_order = list(base_map.keys()) if isinstance(base_map, dict) else ['quadratic', 'skew', 'end']
    if 'linear' not in base_order:
        base_order.append('linear')

    # relation names produced by graph_utils (index -> name)
    rel_names = extras.get('relation_names', []) if isinstance(extras, dict) else []
    # reverse mapping from relation name to base (e.g. 'quadratic_fwd' -> 'quadratic')
    rel_to_base = {}
    if isinstance(base_map, dict):
        for base, pair in base_map.items():
            try:
                fwd, bwd = pair
            except Exception:
                continue
            rel_to_base[fwd] = base
            rel_to_base[bwd] = base

    # collect per-undirected-pair forward-only values. For each directed edge
    # the canonical forward relation (e.g. 'quadratic_fwd') is the source of
    # truth for the undirected pair. If only the backward relation is present
    # we invert its sign to produce the forward value (forward = -backward).
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
        fwd_name, bwd_name = base_map.get(base, (f"{base}_fwd", f"{base}_bwd"))
        pair = (min(src, dst), max(src, dst))
        # extract scalar value from action tensor/array
        try:
            a = actions[k]
            if hasattr(a, 'dim') and a.dim() == 0:
                val = float(a.item())
            else:
                vlist = a.detach().cpu().numpy().tolist() if hasattr(a, 'detach') else list(a)
                val = float(vlist) if not isinstance(vlist, list) else float(vlist[0])
        except Exception:
            try:
                val = float(actions[k])
            except Exception:
                val = 0.0

        if rel_name == fwd_name and (pair not in per_base_forward[base]):
            per_base_forward[base][pair] = val
        elif rel_name == bwd_name and (pair not in per_base_forward[base]):
            per_base_forward[base][pair] = -val

    # assemble antisymmetric matrices (AB = v, BA = -v)
    def build_mat_for(base_name: str):
        mat = [[0.0 for _ in range(N)] for _ in range(N)]
        vals_map = per_base_forward.get(base_name, {})
        for (i, j), val in vals_map.items():
            try:
                v = float(val)
            except Exception:
                v = 0.0
            mat[i][j] = v
            mat[j][i] = -v
        return mat

    c_mat = build_mat_for('quadratic')
    x_mat = build_mat_for('skew')
    s_mat = build_mat_for('end')

    # derive per-node linear b from per-edge linear forward values (average incident edges)
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


def default_env_reward(actions: torch.Tensor, target_vals: List[float]) -> float:
    try:
        targ = torch.tensor(target_vals, dtype=actions.dtype, device=actions.device)
        mse = torch.mean((actions - targ) ** 2).item()
        return -mse
    except Exception:
        return float('-inf')


def create_and_manifest(input_dir: str, out_dir: str, dry_run: bool = False) -> str:
    created = create_combination_dirs(Path(input_dir), Path(out_dir), dry_run=dry_run)
    manifest_path = Path(out_dir) / 'manifest.txt'
    with manifest_path.open('w') as fh:
        for p in created:
            fh.write(str(p) + '\n')
    return str(manifest_path)


def split_manifest(manifest: str, train_frac: float = 0.8, seed: int = 0) -> Tuple[str, str]:
    with open(manifest, 'r', encoding='utf-8') as fh:
        combos = [ln.strip() for ln in fh if ln.strip()]
    random.Random(seed).shuffle(combos)
    n = int(len(combos) * train_frac)
    train = combos[:n]
    val = combos[n:]
    mtrain = Path(manifest).parent / 'manifest.train.txt'
    mval = Path(manifest).parent / 'manifest.val.txt'
    mtrain.write_text('\n'.join(train) + ('\n' if train else ''), encoding='utf-8')
    mval.write_text('\n'.join(val) + ('\n' if val else ''), encoding='utf-8')
    return str(mtrain), str(mval)


def run_quick_epoch_for_combo(combo_dir: str, base_bias: str = 'quadratic') -> Dict[str, Any]:
    data, targets, extras = build_data_and_targets_from_combo(combo_dir, base_bias=base_bias)
    sample_data = data
    in_dim = sample_data.x.shape[1]
    num_rels = int(sample_data.edge_attr.shape[1]) if hasattr(sample_data, 'edge_attr') else 1
    encoder = RGCNEncoder(in_dim=in_dim, hidden_dims=[32], out_dim=16, num_relations=num_rels)
    base_map = extras.get('base_relation_map', {}) if isinstance(extras, dict) else {}
    edge_out_dim = len(list(base_map.keys())) if isinstance(base_map, dict) else 1
    policy = EdgePolicy.from_pyg_data(encoder, 16, sample_data, mlp_hidden=32, mlp_out_dim=edge_out_dim)
    policy.train()
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # one pass: sample actions, write variables, compute reward against targets, and update
    node_emb = policy.forward_node_embeddings(sample_data.x, sample_data.edge_index, getattr(sample_data, 'edge_type', None))
    edge_actions, edge_logp, edge_mean, edge_logstd = policy.get_actions(sample_data.x, sample_data.edge_index, getattr(sample_data, 'edge_type', None), getattr(sample_data, 'edge_attr', None), deterministic=False)
    # write variables and run sim could be done here; for quick epoch compute reward from targets
    reward = default_env_reward(edge_actions.detach(), targets)
    loss = -(edge_logp.sum() * float(reward))
    optim.zero_grad()
    loss.backward()
    optim.step()
    return {'reward': float(reward)}


def run_simulations_and_collect(manifest: str, sim_cmd: Optional[str] = None, max_workers: int = 4, timeout: Optional[int] = None) -> Dict[str, Any]:
    return run_simulation_batch(manifest, sim_cmd=sim_cmd, max_workers=max_workers, timeout=timeout)


def compress_runs(manifest: str, out_tar: str) -> str:
    base = Path(manifest).parent
    tar = Path(out_tar)
    # make a tar.gz of the directory containing combos
    shutil.make_archive(str(tar.with_suffix('')), 'gztar', root_dir=str(base))
    return str(tar.with_suffix('.tar.gz'))


def run_from_config(config_path: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    results: Dict[str, Any] = {}
    # Step 1: create combos
    if cfg.get('create_combos'):
        manifest = create_and_manifest(cfg['create_combos']['input_dir'], cfg['create_combos'].get('out_dir', 'combos'), dry_run=cfg['create_combos'].get('dry_run', False))
    else:
        manifest = cfg.get('manifest')

    results['manifest'] = manifest
    # Step 2: split
    if cfg.get('split') and manifest:
        train_m, val_m = split_manifest(manifest, cfg['split'].get('train_frac', 0.8), cfg['split'].get('seed', 0))
        results['train_manifest'] = train_m
        results['val_manifest'] = val_m

    # Step 3: pick example combo
    example_combo = None
    with open(manifest, 'r', encoding='utf-8') as fh:
        combos = [ln.strip() for ln in fh if ln.strip()]
    if combos:
        example_combo = combos[0]
        results['example_combo'] = example_combo

    # Step 4: build graph & run quick epoch
    if example_combo:
        results['quick_epoch'] = run_quick_epoch_for_combo(example_combo, base_bias=cfg.get('base_bias', 'quadratic'))

    # Step 5: optionally start simulations concurrently
    if cfg.get('run_sims'):
        sim_cmd = cfg.get('sim_cmd')
        max_workers = cfg.get('max_workers', 4)
        timeout = cfg.get('timeout')
        sim_results = run_simulations_and_collect(manifest, sim_cmd=sim_cmd, max_workers=max_workers, timeout=timeout)
        results['sim_results'] = sim_results

    # Step 6: compress if requested
    if cfg.get('compress_after'):
        out_tar = cfg['compress_after'].get('out_tar', str(Path(manifest).parent) + '.tar.gz')
        results['archive'] = compress_runs(manifest, out_tar)

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('config', help='YAML config describing workflow')
    args = p.parse_args()
    out = run_from_config(args.config)
    print(yaml.safe_dump(out, sort_keys=False))
