"""Simple hyperparameter grid search for PTMLP using k-fold CV.

This script uses the same dataset assembly as `train_mlp_pt.py`, then performs
3-fold cross-validation evaluating MSE on validation folds. It searches over
hidden layer tuples, learning rates, and weight decay values, and prints the
average validation MSE per config.

Run with:
    python examples/hp_search_pt.py
"""
from __future__ import annotations

import os
import sys
import re
import numpy as np
import torch
import torch.nn as nn
from itertools import product

# ensure src on path
repo_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mllf.mlp.setup_pairs import assemble_pairs
from mllf.mlp.pt_model import PTMLP, train_one_epoch, evaluate

ROOT = os.path.join(os.path.dirname(__file__), 'training_files')


def build_qual_dataset():
    runs = assemble_pairs(ROOT)
    vocab = {}
    idx = 0
    for run, pairs in runs.items():
        for key, p in pairs.items():
            for at in p.get('atom_types', []):
                if at not in vocab and len(vocab) < 20:
                    vocab[at] = idx
                    idx += 1

    X_list = []
    y_list = []
    for run, pairs in runs.items():
        sites = {}
        for key, p in pairs.items():
            site = p.get('site')
            sub = p.get('sub')
            if site is None or sub is None:
                continue
            sites.setdefault(site, {})[sub] = p
        for site, subs in sites.items():
            subs_ids = sorted(subs.keys())
            for a in subs_ids:
                for b in subs_ids:
                    if a == b:
                        continue
                    p_a = subs[a]
                    pw = p_a.get('biases', {}).get('pairwise_biases', {})
                    key = f'pair_{a}_{b}'
                    # require all groups present
                    ok = True
                    vals = []
                    for g in ('lams','cs','ss','xs'):
                        if g not in pw or key not in pw[g]:
                            ok = False
                            break
                        vals.append(float(pw[g][key]))
                    if not ok:
                        continue
                    p_b = subs[b]
                    fa = np.zeros((len(vocab) + 1,), dtype=float)
                    fb = np.zeros((len(vocab) + 1,), dtype=float)
                    for at in set(p_a.get('atom_types', [])):
                        if at in vocab:
                            fa[vocab[at]] = 1.0
                    for at in set(p_b.get('atom_types', [])):
                        if at in vocab:
                            fb[vocab[at]] = 1.0
                    fa[-1] = float(p_a.get('total_charge', 0.0))
                    fb[-1] = float(p_b.get('total_charge', 0.0))
                    fnex = p_a.get('fnex') or p_b.get('fnex')
                    try:
                        fnex_val = float(fnex) if fnex is not None else 0.0
                    except Exception:
                        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(fnex))
                        fnex_val = float(m.group(1)) if m else 0.0
                    feat = np.concatenate([fa, fb, [fnex_val, 0.0]])
                    X_list.append(feat)
                    y_list.append(vals)
    if not X_list:
        return None
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    return X, y


def k_fold_split(n, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.hstack([f for j, f in enumerate(folds) if j != i])
        yield train, val


def run_grid():
    out = build_qual_dataset()
    if out is None:
        print('No data')
        return
    X, y = out
    print('Dataset size:', X.shape)

    # normalize
    eps = 1e-8
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < eps] = 1.0
    Xn = (X - X_mean) / X_std
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std[y_std < eps] = 1.0
    yn = (y - y_mean) / y_std

    # grid
    hidden_options = [(64,16), (128,64), (64,32,16)]
    lrs = [1e-3, 1e-4]
    wds = [1e-4, 1e-5]
    batch_size = 16
    epochs = 60
    device = 'cpu'

    best = None
    results = []
    for hidden, lr, wd in product(hidden_options, lrs, wds):
        val_scores = []
        for train_idx, val_idx in k_fold_split(Xn.shape[0], k=3, seed=0):
            Xtr = torch.from_numpy(Xn[train_idx]).float()
            ytr = torch.from_numpy(yn[train_idx]).float()
            Xv = torch.from_numpy(Xn[val_idx]).float()
            yv = torch.from_numpy(yn[val_idx]).float()
            model = PTMLP(Xn.shape[1], hidden=hidden, out_dim=4).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            loss_fn = nn.MSELoss()
            for ep in range(epochs):
                _ = train_one_epoch(model, opt, loss_fn, Xtr, ytr, batch_size=batch_size, device=device)
            val_loss, _ = evaluate(model, loss_fn, Xv, yv, device=device)
            val_scores.append(val_loss)
        mean_val = float(np.mean(val_scores))
        results.append(((hidden, lr, wd), mean_val))
        print(f'config hidden={hidden} lr={lr} wd={wd} -> val MSE (norm) = {mean_val:.6f}')
        if best is None or mean_val < best[1]:
            best = ((hidden, lr, wd), mean_val)

    print('\nBest config:', best)

if __name__ == '__main__':
    run_grid()
