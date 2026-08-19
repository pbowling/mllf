"""Tests for BayesianLinearHead (NeuralLinear + Thompson Sampling last layer)."""
import numpy as np
import pytest
import torch

from mllf.cb.bayesian_head import BayesianLinearHead


def _make_regression_data(n=200, feat_dim=5, noise_std=0.1, seed=0):
    """Synthetic linear regression data: r = z @ w_true + bias_true + noise."""
    rng = np.random.RandomState(seed)
    z = torch.tensor(rng.randn(n, feat_dim), dtype=torch.float32)
    w_true = torch.tensor(rng.randn(feat_dim), dtype=torch.float32)
    bias_true = 0.7
    noise = torch.tensor(rng.randn(n) * noise_std, dtype=torch.float32)
    r = z @ w_true + bias_true + noise
    return z, r, w_true, bias_true


class TestBayesianLinearHeadPosterior:
    """Correctness of the closed-form posterior update against ridge regression."""

    def test_converges_to_ridge_solution(self):
        """With enough low-noise data, the posterior mean should closely match
        the closed-form ridge regression solution on the same (augmented) data."""
        feat_dim = 5
        z, r, w_true, bias_true = _make_regression_data(
            n=500, feat_dim=feat_dim, noise_std=0.01, seed=1)

        head = BayesianLinearHead(feat_dim=feat_dim, prior_precision=1e-6, noise_var=0.01 ** 2)
        head.update(z, r)

        # Reference ridge solution on the bias-augmented design matrix.
        z_aug = torch.cat([z, torch.ones(z.shape[0], 1)], dim=-1)
        lam = 1e-6
        ridge_w = torch.linalg.solve(
            z_aug.T @ z_aug + lam * torch.eye(feat_dim + 1), z_aug.T @ r)

        assert torch.allclose(head.mu, ridge_w, atol=1e-2)

    def test_batched_update_matches_sequential(self):
        """One batched update() call should produce the same posterior as
        folding in the same rows one at a time."""
        feat_dim = 4
        z, r, _, _ = _make_regression_data(n=30, feat_dim=feat_dim, seed=2)

        head_batched = BayesianLinearHead(feat_dim=feat_dim, prior_precision=1.0, noise_var=1.0)
        head_batched.update(z, r)

        head_seq = BayesianLinearHead(feat_dim=feat_dim, prior_precision=1.0, noise_var=1.0)
        for i in range(z.shape[0]):
            head_seq.update(z[i:i + 1], r[i:i + 1])

        assert torch.allclose(head_batched.mu, head_seq.mu, atol=1e-4)
        assert torch.allclose(head_batched.Lambda, head_seq.Lambda, atol=1e-4)
        assert head_batched.n_obs.item() == pytest.approx(head_seq.n_obs.item())

    def test_variance_shrinks_with_more_data(self):
        """Posterior predictive variance at a fixed query point should shrink
        (monotonically, on average) as more non-degenerate observations arrive."""
        feat_dim = 3
        head = BayesianLinearHead(feat_dim=feat_dim, prior_precision=1.0, noise_var=1.0)
        query = torch.ones(1, feat_dim)

        _, var0 = head.predict(query)
        rng = np.random.RandomState(3)
        prev_var = var0.item()
        for _ in range(5):
            z = torch.tensor(rng.randn(20, feat_dim), dtype=torch.float32)
            r = torch.tensor(rng.randn(20), dtype=torch.float32)
            head.update(z, r)
            _, var = head.predict(query)
            assert var.item() < prev_var
            prev_var = var.item()

    def test_zero_weight_rows_do_not_move_posterior(self):
        """Rows with weight 0 should not change mu/Lambda at all."""
        feat_dim = 4
        z, r, _, _ = _make_regression_data(n=10, feat_dim=feat_dim, seed=4)
        head = BayesianLinearHead(feat_dim=feat_dim)
        mu_before = head.mu.clone()
        lambda_before = head.Lambda.clone()

        head.update(z, r, weights=torch.zeros(z.shape[0]))

        assert torch.allclose(head.mu, mu_before)
        assert torch.allclose(head.Lambda, lambda_before)
        assert head.n_obs.item() == 0.0

    def test_empty_update_is_noop(self):
        head = BayesianLinearHead(feat_dim=4)
        mu_before = head.mu.clone()
        head.update(torch.zeros(0, 4), torch.zeros(0))
        assert torch.allclose(head.mu, mu_before)
        assert head.n_obs.item() == 0.0


class TestBayesianLinearHeadSampling:
    """Thompson sampling and act()."""

    def test_sample_weights_statistics_match_posterior(self):
        """Empirical mean/var of many samples should approximate mu/diag(Lambda_inv)."""
        feat_dim = 3
        head = BayesianLinearHead(feat_dim=feat_dim, prior_precision=2.0, noise_var=1.0)
        z, r, _, _ = _make_regression_data(n=50, feat_dim=feat_dim, seed=5)
        head.update(z, r)

        torch.manual_seed(0)
        samples = torch.stack([head.sample_weights() for _ in range(4000)])
        emp_mean = samples.mean(dim=0)
        emp_var = samples.var(dim=0)

        assert torch.allclose(emp_mean, head.mu, atol=0.05)
        assert torch.allclose(emp_var, torch.diagonal(head.Lambda_inv), atol=0.05)

    def test_act_matches_manual_augmentation(self):
        feat_dim = 4
        head = BayesianLinearHead(feat_dim=feat_dim)
        z = torch.randn(6, feat_dim)
        w = torch.randn(feat_dim + 1)
        expected = z @ w[:-1] + w[-1]
        assert torch.allclose(head.act(z, w), expected, atol=1e-5)

    def test_deterministic_prediction_uses_mu(self):
        feat_dim = 3
        head = BayesianLinearHead(feat_dim=feat_dim)
        z, r, _, _ = _make_regression_data(n=20, feat_dim=feat_dim, seed=6)
        head.update(z, r)
        mean, _ = head.predict(z)
        assert torch.allclose(head.act(z, head.mu), mean, atol=1e-5)


class TestBayesianLinearHeadUtilities:
    def test_seed_mean_survives_refresh(self):
        """seed_mean should set mu exactly, and it must remain exact after the
        next _refresh_posterior() call (i.e. Lambda_mu was back-filled consistently)."""
        feat_dim = 4
        head = BayesianLinearHead(feat_dim=feat_dim, prior_precision=3.0)
        weight = torch.tensor([1.0, -2.0, 0.5, 0.0])
        bias = 0.25
        head.seed_mean(weight, bias)

        expected_mu = torch.cat([weight, torch.tensor([bias])])
        assert torch.allclose(head.mu, expected_mu)

        head._refresh_posterior()
        assert torch.allclose(head.mu, expected_mu, atol=1e-5)

    def test_set_noise_var_floors_at_positive(self):
        head = BayesianLinearHead(feat_dim=2)
        head.set_noise_var(-5.0)
        assert head.noise_var.item() > 0.0

    def test_reset_restores_prior(self):
        feat_dim = 3
        head = BayesianLinearHead(feat_dim=feat_dim, prior_precision=2.0)
        z, r, _, _ = _make_regression_data(n=10, feat_dim=feat_dim, seed=7)
        head.update(z, r)
        assert head.n_obs.item() > 0.0

        head.reset()
        assert head.n_obs.item() == 0.0
        assert torch.allclose(head.mu, torch.zeros(feat_dim + 1))
        expected_lambda = 2.0 * torch.eye(feat_dim + 1)
        assert torch.allclose(head.Lambda, expected_lambda)

    def test_predict_variance_includes_noise_floor(self):
        """With no observations, predictive variance should be roughly
        prior variance (z^T Lambda_inv z) plus noise_var — never below noise_var."""
        feat_dim = 4
        head = BayesianLinearHead(feat_dim=feat_dim, prior_precision=1.0, noise_var=0.5)
        z = torch.zeros(1, feat_dim)  # z=0 -> only the bias-augmentation column contributes
        _, var = head.predict(z)
        assert var.item() >= 0.5
