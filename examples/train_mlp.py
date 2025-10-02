"""Small script to assemble training data from examples and train SimpleMLP.

This script is intentionally lightweight for experimentation.
It builds a simple feature vector per pair:
- counts of atom types (small vocabulary discovered from data)
- total_charge of the fragment
- fnex (parsed from run directory name if present; otherwise 0.0)

Target: the pairwise linear bias for a canonical ordered pair (pair_1_2 if present;
fall back to first available pair in the mapping).

Run with:
    python examples/train_mlp.py

"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import numpy as np

repo_root = os.path.dirname(os.path.dirname(__file__))


from mllf.mlp.setup_pairs import assemble_pairs
from mllf.mlp.model import SimpleMLP
from mllf.mlp.data_split import split_train_val_test

ROOT = os.path.join(os.path.dirname(__file__), 'training_files')


def build_vocab(pairs_by_run):
    # build full counts then keep top-K most common atom types
    counts = Counter()
    for run, pairs in pairs_by_run.items():
        for key, p in pairs.items():
            counts.update(p.get('atom_types', []))
    K = 20
    most = [a for a, _ in counts.most_common(K)]
    return {a: i for i, a in enumerate(most)}


def featurize_sub(p, vocab):
    # atom type counts for a single sub
    # binary presence features for top-K vocab
    counts = [0.0] * len(vocab)
    for at in set(p.get('atom_types', [])):
        if at in vocab:
            counts[vocab[at]] = 1.0
    tq = float(p.get('total_charge', 0.0))
    return np.array(counts + [tq], dtype=float)


def choose_pair_target(p_a, a, p_b, b):
    # Get pairwise_biases mapping from the a entry (they are stored per-sub)
    pw = p_a.get('biases', {}).get('pairwise_biases', {})
    key = f'pair_{a}_{b}'
    try:
        lams = pw.get('lams', {}).get(key)
        cs = pw.get('cs', {}).get(key)
        ss = pw.get('ss', {}).get(key)
        xs = pw.get('xs', {}).get(key)
    except Exception:
        return None
    if lams is None or cs is None or ss is None or xs is None:
        return None
    return [float(lams), float(cs), float(ss), float(xs)]


def main():
    runs = assemble_pairs(ROOT)
    vocab = build_vocab(runs)

    # collect solvents
    solvents = sorted({p.get('solvent') for run in runs.values() for p in run.values() if p.get('solvent') is not None})
    solvent_to_onehot = {s: np.eye(len(solvents))[i] for i, s in enumerate(solvents)} if solvents else {}

    X_list = []
    y_list = []
    pair_keys = []

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
            # iterate ordered pairs
            subs_ids = sorted(subs.keys())
            for a in subs_ids:
                for b in subs_ids:
                    if a == b:
                        continue
                    p_a = subs[a]
                    p_b = subs[b]
                    target = choose_pair_target(p_a, a, p_b, b)
                    if target is None:
                        continue

                    # features: counts_a | counts_b | total_charge_a | total_charge_b | solvent_onehot | fnex
                    fa = featurize_sub(p_a, vocab)
                    fb = featurize_sub(p_b, vocab)
                    tq_a = float(p_a.get('total_charge', 0.0))
                    tq_b = float(p_b.get('total_charge', 0.0))
                    solvent = p_a.get('solvent') or p_b.get('solvent')
                    solvent_vec = solvent_to_onehot.get(solvent, np.zeros((len(solvents),), dtype=float))
                    fnex = p_a.get('fnex') or p_b.get('fnex')
                    try:
                        fnex_val = float(fnex) if fnex is not None else 0.0
                    except Exception:
                        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(fnex))
                        fnex_val = float(m.group(1)) if m else 0.0

                    feat = np.concatenate([fa, fb, [tq_a, tq_b], solvent_vec, [fnex_val]]).astype(float)
                    X_list.append(feat)
                    y_list.append(target)
                    pair_keys.append((run, site, a, b))

    if not X_list:
        print('No training data assembled, exiting.')
        return

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float).reshape(-1, 4)

    # split
    train_idx, val_idx, test_idx = split_train_val_test(X.shape[0], seed=0)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print('Data shapes:', X.shape, y.shape)
    print('Train/Val/Test sizes:', len(train_idx), len(val_idx), len(test_idx))

    # Normalize features (using training set statistics) to stabilize training
    # split first (we already computed train/val/test indices)
    X_train_raw, X_val_raw, X_test_raw = X_train.copy(), X_val.copy(), X_test.copy()
    y_train_raw, y_val_raw, y_test_raw = y_train.copy(), y_val.copy(), y_test.copy()

    eps = 1e-8
    X_mean = X_train_raw.mean(axis=0)
    X_std = X_train_raw.std(axis=0)
    X_std[X_std < eps] = 1.0
    X_train = (X_train_raw - X_mean) / X_std
    X_val = (X_val_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    # normalize targets per-column
    y_mean = y_train_raw.mean(axis=0)
    y_std = y_train_raw.std(axis=0)
    y_std[y_std < eps] = 1.0
    y_train = (y_train_raw - y_mean) / y_std
    y_val = (y_val_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std

    # build model to predict 4 coefficients (train on normalized targets)
    in_dim = X.shape[1]
    model = SimpleMLP([in_dim, 64, 16, 4], lr=1e-4, seed=0)

    before_norm = model.score_mse(X_test, y_test)
    print('Initial test MSE (normalized):', before_norm)

    model.fit(X_train, y_train, epochs=300, batch_size=16, weight_decay=1e-4)

    after_norm = model.score_mse(X_test, y_test)
    print('Final test MSE (normalized):', after_norm)

    # generate predictions and un-normalize to report MSE in original units
    preds_norm = model.predict(X_test)
    preds = preds_norm * y_std + y_mean

    # overall MSE in original units
    overall_mse = float(np.mean((preds - y_test_raw) ** 2))
    print('Final test MSE (original units, overall):', overall_mse)

    # per-group MSE in original units
    groups = ['lams', 'cs', 'ss', 'xs']
    for i, g in enumerate(groups):
        mse = float(np.mean((preds[:, i:i+1] - y_test_raw[:, i:i+1]) ** 2))
        print(f'MSE {g}:', mse)

    # show a few predictions
    print('preds (first 5):')
    print(preds[:5])
    print('targets (first 5):')
    print(y_test_raw[:5])


if __name__ == '__main__':
    main()
