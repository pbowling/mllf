"""Real training harness for edge prediction using variables.py inputs.

This script provides a minimal supervised training loop using PyTorch to
predict pairwise edge values (e.g. c/x/s) from the `bias_string` in
`variables.py` files produced by the example scaffold. It is intended as a
starting point — replace model, loss, and dataset logic to match your task.

Behavior:
- Reads a manifest (one combo dir per line).
- For each combo, extracts bias YAML from `variables.py` and builds:
  - node features: b (flattened lams) -> shape (N,1)
  - edge_links: list of (i,j) for i<j
  - target per edge: scalar extracted from chosen matrix ('c','x','s')
- Uses a small GNN encoder + edge MLP to predict a scalar per edge.
- Trains with MSE loss and saves checkpoints (state_dict + metadata).

Notes:
- For simplicity this script uses batch size 1 (no padding). It is straightforward
  to extend with batching/padding or PyG batching for larger-scale runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import yaml
import torch
from torch.utils.data import Dataset

from __future__ import annotations

from mllf.cb.graph import Graph
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy
from mllf.cb import train as cb_train
from mllf.cb import graph_utils


def extract_bias_from_variables(py_path: str) -> Dict[str, Any]:
    text = Path(py_path).read_text(encoding='utf-8')
    m = re.search(r"bias_string\s*=\s*(?:\"\"\"|''')([\s\S]*?)(?:\"\"\"|''')", text, re.S)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


class VariablesGraphDataset(Dataset):
    """Dataset that yields (node_feats, edge_links, target_edges) for each combo dir.

    node_feats: Tensor (N, F) where F=1 (b scalar per node)
    edge_links: List[Tuple[int,int]] with 0-based node indices, i<j ordering
    target_edges: Tensor (E,) with scalar target per edge
    """

    def __init__(self, manifest_path: str, target: str = 'c'):
        with open(manifest_path, 'r', encoding='utf-8') as fh:
            lines = [l.strip() for l in fh if l.strip()]
        self.items = lines
        self.target = target

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        combo = self.items[idx]
        vpy = Path(combo) / 'variables.py'
        bias = extract_bias_from_variables(str(vpy))
        b = bias.get('b', [])
        # normalize shape: if b is nested list from YAML implementation, flatten
        if isinstance(b, list) and b and isinstance(b[0], list):
            """Contextual-bandit training harness using the project's CB stack.

            This script uses the project's RGCN encoder, EdgePolicy, and REINFORCE
            helper to perform policy-gradient training on scaffolded combos. It
            constructs a `Graph` from the `bias_string` in `variables.py` then converts
            that Graph into a PyG Data object (via `graph_utils`) so the CB policy can
            operate on it.

            Training proceeds as REINFORCE where the environment reward is a supervised
            negative MSE between sampled edge actions and the target matrix (selected
            from `c`, `x`, or `s`). This lets you train the policy to produce values
            close to the target biases using your contextual-bandit codepath.
            """



            def load_bias_from_variables(py_path: str) -> Dict[str, Any]:
                text = Path(py_path).read_text(encoding='utf-8')
                m = re.search(r"bias_string\s*=\s*(?:\"\"\"|''')([\s\S]*?)(?:\"\"\"|''')", text, re.S)
                if not m:
                    return {}
                try:
                    return yaml.safe_load(m.group(1)) or {}
                except Exception:
                    return {}


            def graph_from_bias(bias: Dict[str, Any]) -> Graph:
                """Build an mllf.cb.Graph from bias dict containing b/c/x/s matrices.

                Matrix shapes are expected to be N x N where N is total_subs.
                We set Graph.num_nodes = N and populate EdgeCoeffs with
                quadratic=c, skew=x, end=s, linear=0.
                """
                # extract flattened b and matrices
                b = bias.get('b', [])
                # b may be stored as a single-row yaml list-of-lists; flatten
                if isinstance(b, list) and b and isinstance(b[0], list):
                    flat_b = [float(x) for row in b for x in row]
                elif isinstance(b, list):
                    flat_b = [float(x) for x in b]
                else:
                    flat_b = []

                N = len(flat_b) if flat_b else 0
                if N == 0:
                    N = 1

                c = bias.get('c', [])
                x = bias.get('x', [])
                s = bias.get('s', [])

                g = Graph(N)

                # populate edges: for i<j, set edge coeffs from matrices if present
                from mllf.cb.graph import EdgeCoeffs

                for i in range(N):
                    for j in range(i + 1, N):
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
                        coeffs = EdgeCoeffs(linear=0.0, quadratic=cval, skew=xval, end=sval)
                        g.set_edge(i, j, coeffs)

                return g


            def build_data_and_targets(combo_dir: str, base_bias: str = 'quadratic') -> Tuple[Any, List[float]]:
                """Return (pyg_data, targets) where targets is per-directed-edge aligned with data.edge_index."""
                vpy = Path(combo_dir) / 'variables.py'
                bias = load_bias_from_variables(str(vpy))
                g = graph_from_bias(bias)
                data, extras = graph_utils.build_pyg_graph_from_mllf_graph(g)

                # build target per directed edge in data.edge_index
                rel_names = extras['relation_names']
                base_map = extras['base_relation_map']  # e.g. {'linear':('linear_fwd','linear_bwd'), ...}

                # reverse map relation_name -> base name
                rel_to_base = {}
                for base, (fwd, bwd) in base_map.items():
                    rel_to_base[fwd] = base
                    rel_to_base[bwd] = base

                # choose which base to use as supervised target (e.g., 'quadratic' -> from c matrix)
                target_matrix = None
                if base_bias == 'quadratic':
                    target_matrix = bias.get('c', [])
                elif base_bias == 'skew':
                    target_matrix = bias.get('x', [])
                elif base_bias == 'end':
                    target_matrix = bias.get('s', [])
                else:
                    target_matrix = bias.get('c', [])

                targets = []
                ei = data.edge_index
                for k in range(ei.shape[1]):
                    src = int(ei[0, k].item())
                    dst = int(ei[1, k].item())
                    rel_idx = int(data.edge_type[k].item()) if hasattr(data, 'edge_type') and data.edge_type.numel() > k else None
                    rel_name = rel_names[rel_idx] if rel_idx is not None else None
                    # default target 0.0
                    t = 0.0
                    if rel_name is not None:
                        base = rel_to_base.get(rel_name)
                        if base == base_bias:
                            try:
                                t = float(target_matrix[src][dst])
                            except Exception:
                                t = 0.0
                    targets.append(t)

                return data, targets


            def train_cb(manifest: str, out_dir: str, base_bias: str = 'quadratic', epochs: int = 10, lr: float = 1e-3, emb_dim: int = 32):
                # load combos
                with open(manifest, 'r', encoding='utf-8') as fh:
                    combos = [ln.strip() for ln in fh if ln.strip()]
                if not combos:
                    print('No combos found in manifest')
                    return

                # build data list
                dataset = []
                for c in combos:
                    data, targets = build_data_and_targets(c, base_bias=base_bias)
                    dataset.append((c, data, torch.tensor(targets, dtype=torch.float32)))

                # construct encoder from first data example
                sample_data = dataset[0][1]
                in_dim = sample_data.x.shape[1]
                num_rels = int(sample_data.edge_attr.shape[1]) if hasattr(sample_data, 'edge_attr') else 1
                encoder = RGCNEncoder(in_dim=in_dim, hidden_dims=[64], out_dim=emb_dim, num_relations=num_rels)

                # policy: create from first data
                policy = EdgePolicy.from_pyg_data(encoder, emb_dim, sample_data)
                policy.train()
                optim = torch.optim.Adam(policy.parameters(), lr=lr)

                # baseline (moving average)
                baseline = 0.0
                alpha = 0.05

                best_score = float('-inf')
                best_epoch = -1

                os.makedirs(out_dir, exist_ok=True)

                for epoch in range(1, epochs + 1):
                    epoch_rewards = []
                    for combo_name, data, targets in dataset:
                        # create env_reward_fn that compares actions -> targets
                        def env_reward_fn(actions, _targets=targets):
                            # actions: torch tensor [E]
                            # compute negative MSE as reward (higher is better)
                            try:
                                mse = torch.mean((actions - _targets) ** 2).item()
                            except Exception:
                                mse = float('inf')
                            return -mse

                        loss_val, reward = cb_train.reinforce_train_step(policy, optim, data, env_reward_fn, baseline=baseline)
                        # update baseline
                        baseline = (1 - alpha) * baseline + alpha * reward
                        epoch_rewards.append(reward)

                    avg_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
                    print(f'Epoch {epoch}: avg_reward={avg_reward:.6f}, baseline={baseline:.6f}')

                    # checkpoint
                    ckpt = {'policy': policy.state_dict(), 'optim': optim.state_dict(), 'epoch': epoch}
                    torch.save(ckpt, Path(out_dir) / f'cb_ckpt_epoch_{epoch}.pt')

                    # track best (maximize reward)
                    if avg_reward > best_score:
                        best_score = avg_reward
                        best_epoch = epoch

                print(f'Training complete. best_epoch={best_epoch}, best_reward={best_score}')


            def _cli():
                p = argparse.ArgumentParser()
                p.add_argument('manifest')
                p.add_argument('--out', default='cb_training')
                p.add_argument('--bias', default='quadratic', choices=['quadratic', 'skew', 'end'])
                p.add_argument('--epochs', type=int, default=10)
                p.add_argument('--lr', type=float, default=1e-3)
                args = p.parse_args()
                train_cb(args.manifest, args.out, base_bias=args.bias, epochs=args.epochs, lr=args.lr)


            if __name__ == '__main__':
                _cli()
