"""Train PTMLP on the full qualifying dataset with chosen hyperparameters and save weights.

Saves to `models/ptmlp_best.pth` in the repo root.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn as nn

repo_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mllf.mlp.pt_model import PTMLP, train_one_epoch, evaluate
from mllf.mlp.setup_pairs import assemble_pairs

ROOT = os.path.join(os.path.dirname(__file__), 'training_files')


def build_qual_dataset(include_fnex: bool = False):
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
                    va = np.zeros((len(vocab),), dtype=float)
                    vb = np.zeros((len(vocab),), dtype=float)
                    for at in p_a.get('atom_types', []):
                        if at in vocab:
                            va[vocab[at]] += 1.0
                    for at in p_b.get('atom_types', []):
                        if at in vocab:
                            vb[vocab[at]] += 1.0
                    diff = va - vb
                    charge_diff = float(p_a.get('total_charge', 0.0)) - float(p_b.get('total_charge', 0.0))
                    feat = np.concatenate([diff, [charge_diff]])
                    X_list.append(feat)
                    y_list.append(vals)
    if not X_list:
        return None
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    return X, y


def main():
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

    device = 'cpu'
    hidden = (64, 32, 16)
    lr = 1e-3
    wd = 1e-05
    batch_size = 16
    epochs = 80

    model = PTMLP(Xn.shape[1], hidden=hidden, out_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        train_loss = train_one_epoch(model, opt, loss_fn, torch.from_numpy(Xn).float(), torch.from_numpy(yn).float(), batch_size=batch_size, device=device)
        if ep % 20 == 0:
            print(f'epoch {ep}: train loss (norm) {train_loss:.6f}')

    out_dir = os.path.join(repo_root, 'models')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'ptmlp_best.pth')
    torch.save({'model_state_dict': model.state_dict(), 'X_mean': X_mean, 'X_std': X_std, 'y_mean': y_mean, 'y_std': y_std}, model_path)
    print('Saved model to', model_path)


if __name__ == '__main__':
    main()
