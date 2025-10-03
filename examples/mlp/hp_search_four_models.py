"""Grid-search and train four PTMLP models, one per bias group (lams, cs, ss, xs).

Features: atom-type difference counts (a - b), total-charge difference (a - b), and solvent one-hot.

For each bias group we perform a 3-fold CV grid search over small hyperparameter grid,
train a final model on the full dataset with the best config, and save the model and norms
to `models/ptmlp_{bias}.pth`.

Run with:
    python examples/mlp/hp_search_four_models.py
"""
from __future__ import annotations

import os
import sys
import re
import json
from itertools import product

import numpy as np
import torch
import torch.nn as nn

repo_root = os.path.dirname(os.path.dirname(__file__))

from mllf.mlp.setup_pairs import assemble_pairs
from mllf.mlp.pt_model import PTMLP, train_one_epoch, evaluate

ROOT = os.path.join(os.path.dirname(__file__), 'training_files')


def build_dataset_for_bias(bias_group: str, include_fnex: bool = False):
    runs = assemble_pairs(ROOT)

    # build vocab for atom types (top N)
    atom_vocab = {}
    av_idx = 0
    solvents = {}
    s_idx = 0

    # first pass: collect atom types and solvents
    for run, pairs in runs.items():
        for k, p in pairs.items():
            for at in p.get('atom_types', []):
                if at not in atom_vocab and len(atom_vocab) < 20:
                    atom_vocab[at] = av_idx
                    av_idx += 1
            sol = p.get('solvent') or 'none'
            if sol not in solvents:
                solvents[sol] = s_idx
                s_idx += 1

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
                    # we require the specific bias_group to be present
                    if bias_group not in pw or key not in pw[bias_group]:
                        continue
                    # build atom-type count difference vector
                    va = np.zeros((len(atom_vocab),), dtype=float)
                    vb = np.zeros((len(atom_vocab),), dtype=float)
                    for at in p_a.get('atom_types', []):
                        if at in atom_vocab:
                            va[atom_vocab[at]] += 1.0
                    for at in subs[b].get('atom_types', []):
                        if at in atom_vocab:
                            vb[atom_vocab[at]] += 1.0
                    diff = va - vb

                    # charge diff
                    charge_diff = float(p_a.get('total_charge', 0.0)) - float(subs[b].get('total_charge', 0.0))

                    # solvent one-hot
                    sol = p_a.get('solvent') or 'none'
                    sol_onehot = np.zeros((len(solvents),), dtype=float)
                    sol_onehot[solvents.get(sol, 0)] = 1.0

                    elems = [diff, sol_onehot, np.array([charge_diff])]
                    if include_fnex:
                        fnex = p_a.get('fnex') or subs[b].get('fnex')
                        try:
                            fnex_val = float(fnex) if fnex is not None else 0.0
                        except Exception:
                            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(fnex))
                            fnex_val = float(m.group(1)) if m else 0.0
                        elems.append(np.array([fnex_val]))

                    feat = np.concatenate(elems)
                    X_list.append(feat)
                    y_list.append(float(pw[bias_group][key]))

    if not X_list:
        return None
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float).reshape(-1, 1)
    meta = {
        'atom_vocab': atom_vocab,
        'solvents': solvents,
    }
    return X, y, meta


def k_fold_split(n, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.hstack([f for j, f in enumerate(folds) if j != i])
        yield train, val


def run_grid_and_save():
    bias_groups = ['lams', 'cs', 'ss', 'xs']
    results = {}
    models_dir = os.path.join(repo_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # shared grid
    hidden_options = [(64, 16), (128, 64), (64, 32, 16)]
    lrs = [1e-3, 1e-4]
    wds = [1e-4, 1e-5]
    batch_size = 16
    epochs = 60
    device = 'cpu'

    for bias in bias_groups:
        out = build_dataset_for_bias(bias, include_fnex=False)
        if out is None:
            print('No data for', bias)
            continue
        X, y, meta = out
        print(f'Bias {bias}: dataset size', X.shape)

        # normalize X and y
        eps = 1e-8
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < eps] = 1.0
        Xn = (X - X_mean) / X_std
        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0)
        y_std[y_std < eps] = 1.0
        yn = (y - y_mean) / y_std

        best = None
        best_cfg = None
        for hidden, lr, wd in product(hidden_options, lrs, wds):
            val_scores = []
            for train_idx, val_idx in k_fold_split(Xn.shape[0], k=3, seed=0):
                Xtr = torch.from_numpy(Xn[train_idx]).float()
                ytr = torch.from_numpy(yn[train_idx]).float()
                Xv = torch.from_numpy(Xn[val_idx]).float()
                yv = torch.from_numpy(yn[val_idx]).float()
                model = PTMLP(Xn.shape[1], hidden=hidden, out_dim=1).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                loss_fn = nn.MSELoss()
                for ep in range(epochs):
                    _ = train_one_epoch(model, opt, loss_fn, Xtr, ytr, batch_size=batch_size, device=device)
                val_loss, _ = evaluate(model, loss_fn, Xv, yv, device=device)
                val_scores.append(val_loss)
            mean_val = float(np.mean(val_scores))
            print(f'bias={bias} config hidden={hidden} lr={lr} wd={wd} -> val MSE (norm) = {mean_val:.6f}')
            if best is None or mean_val < best:
                best = mean_val
                best_cfg = (hidden, lr, wd)

        print('Best for', bias, best_cfg, best)
        results[bias] = {'best_cfg': best_cfg, 'best_val': best}

        # train final model on full data with best config and save
        hidden, lr, wd = best_cfg
        model = PTMLP(Xn.shape[1], hidden=hidden, out_dim=1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        for ep in range(epochs * 2 // 3):
            train_loss = train_one_epoch(model, opt, loss_fn, torch.from_numpy(Xn).float(), torch.from_numpy(yn).float(), batch_size=batch_size, device=device)
            if ep % 20 == 0:
                print(f'[final train] bias={bias} epoch {ep}: loss {train_loss:.6f}')

        model_path = os.path.join(models_dir, f'ptmlp_{bias}.pth')
        torch.save({'model_state_dict': model.state_dict(), 'X_mean': X_mean, 'X_std': X_std, 'y_mean': y_mean, 'y_std': y_std, 'meta': meta}, model_path)
        print('Saved', model_path)

    # write summary
    summary_path = os.path.join(models_dir, 'hp_search_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print('Wrote summary to', summary_path)


if __name__ == '__main__':
    run_grid_and_save()
