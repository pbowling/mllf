"""Bayesian linear regression head for NeuralLinear + Thompson Sampling.

``BayesianLinearHead`` implements the "last layer" of a NeuralLinear model: a
closed-form Bayesian linear regression on a fixed feature vector ``z = phi(x)``
produced by a (frozen or slowly-trained) neural trunk. Unlike the deterministic
``nn.Linear`` it replaces, it maintains a full Gaussian posterior over its weight
vector ``w`` and updates that posterior analytically from observed
``(z, r)`` pairs — no backward pass, no optimizer.

This is the mechanism behind NeuralLinear Thompson Sampling: instead of a policy
network that outputs an action distribution trained by REINFORCE, each bias head
keeps a posterior ``N(mu, Lambda^-1)`` over its regression weights. Acting is done
by drawing one weight sample per decision (Thompson sampling) and predicting
``a = z @ w_sample``; learning is done by folding the observed reward directly into
the posterior via ridge-regression-style rank-k updates.

Reference: Riquelme, Tucker & Snoek, "Deep Bayesian Bandits Showdown" (2018).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BayesianLinearHead(nn.Module):
    """Incremental Bayesian linear regression: ``r ~ N(z @ w, noise_var)``.

    Maintains a Gaussian posterior ``w ~ N(mu, Lambda^-1)`` with a zero-mean,
    isotropic Gaussian prior ``w ~ N(0, (1/prior_precision) I)``. All state is
    stored as buffers (not ``nn.Parameter``s) since it is updated analytically,
    never by autograd/optimizer.step().

    A constant-1 feature is appended internally (standard bias-augmentation
    trick) so the posterior includes an intercept term, letting the head warm
    -start cleanly from a pretrained ``nn.Linear(feat_dim, 1)`` (which has both
    a weight vector and a bias). Callers only ever see/pass the un-augmented
    ``[N, feat_dim]`` features — the augmentation is an implementation detail.

    Args:
        feat_dim: Dimension of the input feature vector ``z`` (excludes the
            internal bias term, which is added automatically).
        prior_precision: Ridge/prior precision ``lambda`` for the weight prior
            ``w ~ N(0, (1/lambda) I)``. Higher = stronger shrinkage toward zero
            and lower initial posterior variance.
        noise_var: Initial observation noise variance ``sigma^2``. Should be
            re-estimated from data (see ``estimate_reward_noise_variance`` in
            ``pretrain_policy.py``) and set via ``set_noise_var`` before online use.
        jitter: Small diagonal term added before Cholesky factorization, purely
            for numerical stability against loss of positive-definiteness from
            floating point error.
    """

    def __init__(self, feat_dim: int, prior_precision: float = 1.0,
                 noise_var: float = 1.0, jitter: float = 1e-6):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self._D = self.feat_dim + 1  # +1 for the appended constant-bias feature
        self.jitter = float(jitter)

        eye = torch.eye(self._D, dtype=torch.float32)
        self.register_buffer('Lambda', prior_precision * eye.clone())
        self.register_buffer('Lambda_mu', torch.zeros(self._D, dtype=torch.float32))
        self.register_buffer('mu', torch.zeros(self._D, dtype=torch.float32))
        self.register_buffer('Lambda_inv', (1.0 / prior_precision) * eye.clone())
        self.register_buffer('n_obs', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('noise_var', torch.tensor(float(noise_var), dtype=torch.float32))
        # Kept only for reset(); not re-read after __init__.
        self._prior_precision = float(prior_precision)

    def _augment(self, z: torch.Tensor) -> torch.Tensor:
        """Append a constant-1 column so the posterior has an intercept term."""
        ones = torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)
        return torch.cat([z, ones], dim=-1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Posterior predictive mean and variance for each row of ``z``.

        Args:
            z: ``[N, feat_dim]`` feature rows (un-augmented).

        Returns:
            mean: ``[N]`` posterior predictive mean (``z_aug @ mu``).
            var: ``[N]`` posterior predictive variance
                (``z_aug^T Lambda_inv z_aug + noise_var``), i.e. includes both
                parameter uncertainty and observation noise.
        """
        z = self._augment(z.to(self.mu.dtype))
        mean = z @ self.mu
        # Batched z^T Lambda_inv z without materializing an [N,N] matrix.
        var = torch.einsum('nd,de,ne->n', z, self.Lambda_inv, z) + self.noise_var
        return mean, var

    def sample_weights(self) -> torch.Tensor:
        """Draw one weight sample from the current posterior (Thompson sampling).

        Returns:
            ``[feat_dim + 1]`` sample ``w ~ N(mu, Lambda_inv)`` (last entry is
            the intercept; combine with :meth:`_augment`-ed features, or use
            ``w[:-1] @ z + w[-1]`` directly).
        """
        D = self._D
        cov = self.Lambda_inv + self.jitter * torch.eye(D, device=self.Lambda_inv.device,
                                                          dtype=self.Lambda_inv.dtype)
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError:
            # Fall back to a symmetrized, more heavily jittered matrix if the
            # cached Lambda_inv has drifted from exact PSD due to float error.
            cov = 0.5 * (cov + cov.T) + 1e-4 * torch.eye(
                D, device=cov.device, dtype=cov.dtype)
            L = torch.linalg.cholesky(cov)
        eps = torch.randn(D, device=self.mu.device, dtype=self.mu.dtype)
        return self.mu + L @ eps

    def act(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply a (possibly sampled) weight vector to un-augmented features.

        Args:
            z: ``[N, feat_dim]`` feature rows (un-augmented).
            w: ``[feat_dim + 1]`` weight vector, e.g. from :meth:`sample_weights`
                or ``self.mu`` directly.

        Returns:
            ``[N]`` predicted values ``z_aug @ w``.
        """
        return self._augment(z.to(w.dtype)) @ w

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, z: torch.Tensor, r: torch.Tensor,
               weights: Optional[torch.Tensor] = None) -> None:
        """Fold a batch of ``(z, r)`` observations into the posterior in closed form.

        Args:
            z: ``[N, feat_dim]`` feature rows (un-augmented).
            r: ``[N]`` observed scalar targets (reward/credit signal).
            weights: Optional ``[N]`` per-row confidence weights in ``[0, 1]``
                (e.g. down-weighting pairs with no observed transitions). Rows
                with weight 0 contribute nothing. Defaults to all-ones.
        """
        if z.numel() == 0:
            return
        z = self._augment(z.to(self.mu.dtype))
        r = r.to(self.mu.dtype)
        if weights is None:
            w = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        else:
            w = weights.to(z.dtype)

        inv_noise = 1.0 / self.noise_var.clamp(min=1e-8)
        zw = z * w.unsqueeze(-1)  # [N, D], row i scaled by weight_i
        self.Lambda = self.Lambda + inv_noise * (zw.T @ z)
        self.Lambda_mu = self.Lambda_mu + inv_noise * (zw.T @ r)
        self.n_obs = self.n_obs + w.sum()

        self._refresh_posterior()

    def _refresh_posterior(self) -> None:
        """Recompute cached ``mu``/``Lambda_inv`` from ``Lambda``/``Lambda_mu``."""
        D = self._D
        eye = torch.eye(D, device=self.Lambda.device, dtype=self.Lambda.dtype)
        Lambda_reg = self.Lambda + self.jitter * eye
        self.Lambda_inv = torch.linalg.inv(Lambda_reg)
        self.mu = self.Lambda_inv @ self.Lambda_mu

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_noise_var(self, sigma2: float) -> None:
        """Set the observation noise variance (typically from pretraining residuals)."""
        self.noise_var.fill_(max(float(sigma2), 1e-6))

    def seed_mean(self, weight: torch.Tensor, bias: float = 0.0,
                  prior_precision: Optional[float] = None,
                  precision_floor: float = 1e-6) -> None:
        """Warm-start the posterior mean from a pretrained ``nn.Linear(feat_dim, 1)``.

        Sets ``mu`` to ``[weight, bias]`` directly and back-fills ``Lambda_mu``
        consistently (``Lambda_mu = Lambda @ mu``) so the seeded mean survives
        exactly through the next ``_refresh_posterior()`` — i.e. this seeds the
        posterior *mean* at the pretrained solution while leaving the posterior
        *uncertainty* at the (uninformed) prior, which is then genuinely
        narrowed by real online observations via :meth:`update`.

        Args:
            weight: ``[feat_dim]`` weight vector (e.g. the deterministic head's
                readout weight row for this bias type).
            bias: Scalar bias/intercept term (e.g. the readout bias entry).
            prior_precision: If given, **rescales the prior/posterior
                covariance** to this value *before* seeding (equivalent to
                calling :meth:`reset` with a different ``prior_precision``,
                then seeding). This matters a great deal: a fixed
                ``prior_precision`` (e.g. the constructor default of 1.0)
                gives every sampled weight component unit variance —
                Thompson-sampling noise with std 1.0 per dimension — which
                completely swamps a well-fitted ``weight`` whose components
                are typically much smaller than 1 (e.g. a BC-trained
                deterministic readout's weights are commonly O(0.01-0.1)).
                Left uncalibrated, ``sample_weights()`` draws are dominated by
                noise rather than the seeded mean, regardless of fit quality
                — this is exactly what caused wildly-scaled, near-random
                online-training actions from an otherwise well-trained
                checkpoint. Pass e.g. ``prior_precision / mean(weight**2)``
                (see ``pretrain_policy.initialize_bayesian_heads_from_pretrained``)
                to scale the prior to this head's own fitted weight
                magnitude instead of an arbitrary absolute constant.
            precision_floor: Minimum value accepted for ``prior_precision``
                — floors it away from ~0 (which would make ``Lambda_inv``
                blow up and sampling numerically unstable).
        """
        weight = weight.to(self.mu.dtype).reshape(-1)
        assert weight.numel() == self.feat_dim, (
            f"seed_mean: expected weight of length {self.feat_dim}, got {weight.numel()}")

        if prior_precision is not None:
            prior_precision = max(float(prior_precision), precision_floor)
            eye = torch.eye(self._D, device=self.Lambda.device, dtype=self.Lambda.dtype)
            self.Lambda = prior_precision * eye
            self.Lambda_inv = (1.0 / prior_precision) * eye
            self._prior_precision = prior_precision  # keeps reset() consistent with this seed

        mu0 = torch.cat([weight, torch.tensor([float(bias)], dtype=self.mu.dtype,
                                                device=self.mu.device)])
        self.mu = mu0
        self.Lambda_mu = self.Lambda @ mu0

    def reset(self) -> None:
        """Reset the posterior back to the prior (discard all observations)."""
        D = self._D
        eye = torch.eye(D, device=self.Lambda.device, dtype=self.Lambda.dtype)
        self.Lambda = self._prior_precision * eye.clone()
        self.Lambda_mu = torch.zeros(D, device=self.Lambda.device, dtype=self.Lambda.dtype)
        self.mu = torch.zeros(D, device=self.Lambda.device, dtype=self.Lambda.dtype)
        self.Lambda_inv = (1.0 / self._prior_precision) * eye.clone()
        self.n_obs = torch.tensor(0.0, device=self.n_obs.device, dtype=self.n_obs.dtype)
