"""Data splitting utilities for MLP training."""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def split_train_val_test(n_samples: int, *, train_frac: float = 0.8, val_frac: float = 0.1, test_frac: float = 0.1, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index arrays for train, val, test splits.

    Parameters
    - n_samples: total number of samples
    - train_frac/val_frac/test_frac: fractions summing to 1.0 (or will be normalized)
    - seed: random seed for reproducibility

    Returns (train_idx, val_idx, test_idx)
    """
    assert n_samples > 0
    fracs = np.array([train_frac, val_frac, test_frac], dtype=float)
    if not np.isclose(fracs.sum(), 1.0):
        fracs = fracs / fracs.sum()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)

    n_train = int(round(fracs[0] * n_samples))
    n_val = int(round(fracs[1] * n_samples))
    # ensure we don't overshoot due to rounding
    n_train = max(0, min(n_samples, n_train))
    n_val = max(0, min(n_samples - n_train, n_val))
    n_test = n_samples - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx
