import numpy as np

from mllf.mlp.model import SimpleMLP


def test_simple_mlp_train_on_synthetic_with_fnex():
    """Train SimpleMLP on a tiny synthetic dataset that includes an `fnex` input.

    Input shape: [n_samples, n_features] where features include some atomic-like
    values and a scalar `fnex` appended as the last column. Target: a linear
    combination of inputs so the MLP can learn to reduce MSE.
    """
    rng = np.random.default_rng(42)

    n_samples = 200
    # create 3 atomic-like features + 1 fnex scalar
    n_atomic = 3
    fnex_dim = 1
    in_dim = n_atomic + fnex_dim

    X_atomic = rng.normal(scale=1.0, size=(n_samples, n_atomic))
    fnex = rng.uniform(low=0.0, high=1.0, size=(n_samples, 1))
    X = np.hstack([X_atomic, fnex])

    # true weights: atomic weights + fnex weight
    true_w = np.array([1.5, -2.0, 0.75, 3.0])
    y = X @ true_w + 0.1 * rng.normal(size=(n_samples,))
    y = y.reshape(-1, 1)

    # split train/test
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # create a tiny MLP: input -> 16 -> 1
    model = SimpleMLP([in_dim, 16, 1], lr=1e-3, seed=1)

    # baseline mse before training
    before = model.score_mse(X_test, y_test)

    # train a bit
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    after = model.score_mse(X_test, y_test)

    # Expect training to reduce MSE
    assert after < before, f"MSE did not decrease (before={before}, after={after})"
