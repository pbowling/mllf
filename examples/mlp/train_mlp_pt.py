"""Train a PyTorch MLP on the assembled dataset from examples/training_files.

Featurization changes:
- atom-type difference counts (a - b) across a small vocabulary (top 20 seen types)
- total charge difference (a - b)
- optional fnex inclusion via include_fnex flag
"""
from __future__ import annotations

import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn

repo_root = os.path.dirname(os.path.dirname(__file__))

from mllf.mlp.setup_pairs import assemble_pairs
from mllf.mlp.pt_model import PTMLP, train_one_epoch, evaluate
from mllf.mlp.data_split import split_train_val_test

ROOT = os.path.join(os.path.dirname(__file__), 'training_files')


def build_dataset(include_fnex: bool = False):
    runs = assemble_pairs(ROOT)
    # build a small vocab of atom types seen
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
        # group fragments by site
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
                    p_b = subs[b]
                    pw = p_a.get('biases', {}).get('pairwise_biases', {})
                    key = f'pair_{a}_{b}'
                    lams = pw.get('lams', {}).get(key)
                    cs = pw.get('cs', {}).get(key)
                    ss = pw.get('ss', {}).get(key)
                    xs = pw.get('xs', {}).get(key)
                    if lams is None or cs is None or ss is None or xs is None:
                        continue

                    # compute atom-type count difference vector (a - b)
                    va = np.zeros((len(vocab),), dtype=float)
                    vb = np.zeros((len(vocab),), dtype=float)
                    for at in p_a.get('atom_types', []):
                        if at in vocab:
                            va[vocab[at]] += 1.0
                    for at in p_b.get('atom_types', []):
                        if at in vocab:
                            vb[vocab[at]] += 1.0
                    diff = va - vb

                    # total charge difference (a - b)
                    charge_diff = float(p_a.get('total_charge', 0.0)) - float(p_b.get('total_charge', 0.0))

                    elems = [diff]
                    if include_fnex:
                        fnex = p_a.get('fnex') or p_b.get('fnex')
                        try:
                            fnex_val = float(fnex) if fnex is not None else 0.0
                        except Exception:
                            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(fnex))
                            fnex_val = float(m.group(1)) if m else 0.0
                        elems.append(np.array([fnex_val]))

                    elems.append(np.array([charge_diff]))
                    feat = np.concatenate(elems)
                    X_list.append(feat)
                    y_list.append([float(lams), float(cs), float(ss), float(xs)])
                    # compute atom-type count difference (a - b)
                    fa_counts = [0] * (len(vocab))
                    fb_counts = [0] * (len(vocab))
                    for at in p_a.get('atom_types', []):
                        if at in vocab:
                            fa_counts[vocab[at]] += 1
                    for at in p_b.get('atom_types', []):
                        if at in vocab:
                            fb_counts[vocab[at]] += 1
                    diff = np.array([fa_counts[i] - fb_counts[i] for i in range(len(vocab))], dtype=float)

                    # total charge difference (a - b)
                    charge_diff = float(p_a.get('total_charge', 0.0)) - float(p_b.get('total_charge', 0.0))

                    # solvent currently unused; fnex optional (not included by default)
                    feat = np.concatenate([diff, [charge_diff, 0.0]])
                    X_list.append(feat)
                    y_list.append([float(lams), float(cs), float(ss), float(xs)])
    if not X_list:
        return None
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    return X, y


def main():
    out = build_dataset()
    if out is None:
        print('No data')
        return
    X, y = out
    n = X.shape[0]
    train_idx, val_idx, test_idx = split_train_val_test(n, seed=0)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # normalize
    eps = 1e-8
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < eps] = 1.0
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < eps] = 1.0
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    device = 'cpu'
    model = PTMLP(X_train.shape[1], hidden=(64, 16), out_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        loss = train_one_epoch(model, opt, loss_fn, torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), batch_size=16, device=device)
        if epoch % 10 == 0:
            val_loss, _ = evaluate(model, loss_fn, torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float(), device=device)
            print(f'epoch {epoch}: train {loss:.4f}, val {val_loss:.4f}')

    test_loss, preds = evaluate(model, loss_fn, torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float(), device=device)
    preds = preds.numpy() * y_std + y_mean
    y_test_orig = y_test * y_std + y_mean
    print('test loss (norm):', test_loss)
    overall_mse = float(((preds - y_test_orig) ** 2).mean())
    print('overall mse (orig units):', overall_mse)
    print('preds[:5]:', preds[:5])
    print('targets[:5]:', y_test_orig[:5])


if __name__ == '__main__':
    main()
