"""Bayesian-head warm start (NeuralLinear + Thompson sampling) from a
behavior-cloned deterministic policy.
"""
import logging
from typing import List, Optional

import numpy as np
import torch

from mllf.cb.policy import UnimolPolicy


log = logging.getLogger(__name__)


# ── NeuralLinear + Thompson Sampling: Bayesian head warm-start ──────────────

def _compute_physical_residual_variance(
    policy_det, graph_cache: List, device, max_runs: int = 200,
    weights_cache: Optional[List] = None,
) -> tuple:
    """Physical-unit ``(target - mean)`` residuals per bias type, from BC data.

    Shared low-level pass used by both ``estimate_reward_noise_variance``
    (observation noise) and ``initialize_bayesian_heads_from_pretrained``
    (exploration-precision calibration) — both need "how much do real bias
    coefficients actually scatter around the deterministic model's
    prediction, per bias type", just for different purposes.

    ``weights_cache`` (the data-hygiene [E][4] pairwise weights, indexed like
    ``graph_cache``) restricts residuals to measured targets and skips runs
    without an entry; without it every target counts.

    Returns:
        (residuals, n_used) — residuals is a list of 4 lists of floats (one
        per bias type, in physical bias-coefficient units), n_used is how
        many cached runs actually contributed.
    """
    policy_det.eval()
    residuals = [[] for _ in range(4)]
    n_used = 0
    with torch.no_grad():
        for run_idx, cached in enumerate(graph_cache):
            if cached is None:
                continue
            if n_used >= max_runs:
                break
            unimol_emb, edge_idx, targets_list = cached
            if not targets_list:
                continue
            keep = None
            if weights_cache is not None:
                w = weights_cache[run_idx] if run_idx < len(weights_cache) else None
                if w is None or not w or not isinstance(w[0], (list, tuple)):
                    continue
                keep = np.array(w) > 0
            try:
                unimol_emb = unimol_emb.to(device)
                edge_idx = edge_idx.to(device)
                targets = torch.tensor(targets_list, dtype=torch.float32, device=device)  # [E, 4]
                mean, _ = policy_det._forward_edges(unimol_emb, edge_idx, edge_type=None)
                if mean.shape != targets.shape:
                    continue
                resid = (targets - mean).detach().cpu().numpy()
                if keep is not None and keep.shape != resid.shape:
                    continue
                for d in range(4):
                    col = resid[:, d] if keep is None else resid[keep[:, d], d]
                    residuals[d].extend(col.tolist())
                n_used += 1
            except Exception:
                continue
    return residuals, n_used


def _compute_typical_z_norm_sq(
    policy_det, graph_cache: List, device, max_runs: int = 200,
) -> List[float]:
    """Estimate ``E[||z||^2]`` per bias type from real BC-pretraining edge features.

    ``z`` is each trunk's feature vector (see ``UnimolPolicy.forward_features``)
    — the input the Bayesian head's linear regression actually sees. Needed to
    correctly calibrate ``BayesianLinearHead``'s prior precision: a precision
    calibrated only from the deterministic weights' own magnitude (as an
    earlier version of this code did) implicitly assumes ``||z|| ~ 1``, which
    is not remotely true for a real trunk on real Uni-Mol embeddings (``||z||``
    of 5-10+ is typical) — so the resulting Thompson-sampling noise can still
    swamp the signal even when the weight-based calibration alone looks tight.
    See ``initialize_bayesian_heads_from_pretrained``.

    Returns:
        List of 4 floats: ``[linear, quadratic, skew, end]`` mean ``||z||^2``.
    """
    policy_det.eval()
    sums = [0.0] * 4
    counts = [0] * 4
    with torch.no_grad():
        n_used = 0
        for cached in graph_cache:
            if cached is None:
                continue
            if n_used >= max_runs:
                break
            unimol_emb, edge_idx, targets_list = cached
            if not targets_list:
                continue
            try:
                unimol_emb = unimol_emb.to(device)
                edge_idx = edge_idx.to(device)
                edge_input = policy_det._build_edge_input(unimol_emb, edge_idx)
                for d in range(4):
                    z = policy_det.edge_mlp.mlps[d].forward_features(edge_input)
                    sums[d] += float(z.pow(2).sum(dim=-1).sum())
                    counts[d] += z.shape[0]
                n_used += 1
            except Exception:
                continue
    return [sums[d] / max(counts[d], 1) for d in range(4)]


def estimate_reward_noise_variance(
    policy_det,
    graph_cache: List,
    device,
    max_runs: int = 200,
    floor: float = 0.05,
    weights_cache: Optional[List] = None,
) -> List[float]:
    """Estimate per-bias-type observation noise variance from BC residuals.

    Runs the trained deterministic policy's forward pass over a sample of
    cached pretraining graphs and computes the residual variance between its
    mean prediction and the actual BC target, per bias type. This becomes each
    ``BayesianLinearHead``'s ``noise_var`` (sigma^2) — NeuralLinear's posterior
    update needs this to correctly weigh how much a new observation should move
    the mean vs. how much predictive uncertainty should remain.

    Args:
        policy_det: Trained deterministic UnimolPolicy (use_bayesian_heads=False).
        graph_cache: Per-run cache of ``(unimol_emb, edge_index, targets_list)``
            or ``None``, as built during the main BC loop (see ``pretrain_with_runs``).
        device: torch device.
        weights_cache: Optional data-hygiene [E][4] pairwise weights (indexed
            like ``graph_cache``); when given, only measured targets count.
        max_runs: Cap on how many cached runs to sample — residual-variance
            estimation doesn't need the full dataset; keeps this fast.
        floor: Minimum returned variance, so a small/lucky sample can't make the
            posterior overconfident with a near-zero sigma^2.

    Returns:
        List of 4 floats: ``[linear, quadratic, skew, end]`` noise variances.
    """
    residuals, n_used = _compute_physical_residual_variance(
        policy_det, graph_cache, device, max_runs, weights_cache=weights_cache)

    # Residuals above are in *physical* bias-coefficient units (both `mean`
    # and `targets` are), but BayesianLinearHead.noise_var is consumed in the
    # posterior's native *pre-activation* space (see UnimolPolicy
    # ._preact_to_action / ._action_to_preact — the posterior is seeded from,
    # and regresses toward, the deterministic readout's pre-softsign+scale
    # output). Convert via the local-linear approximation d(scale*softsign(x))/dx
    # = scale/(1+|x|)^2 -> near x=0 this is ~scale, so Var(physical) ~
    # scale^2 * Var(pre-activation). Exact at the origin, an underestimate of
    # the true pre-activation noise near saturation -- adequate for a warm-
    # start prior hyperparameter, not a claim of statistical precision.
    scale_factors = policy_det.scale_factors.detach().cpu().numpy()

    noise_vars = []
    for d in range(4):
        if len(residuals[d]) >= 5:
            noise_var_physical = max(float(np.var(residuals[d])), floor)
        else:
            noise_var_physical = 1.0  # too little data — fall back to a neutral hyperparameter
        noise_vars.append(noise_var_physical / float(scale_factors[d]) ** 2)
    dim_names = ['linear', 'quadratic', 'skew', 'end']
    log.info('%s', "  Estimated reward noise variance (sigma^2, pre-activation space) "
             f"from {n_used} sampled runs: "
             + ', '.join(f'{n}={v:.6f}' for n, v in zip(dim_names, noise_vars)))
    return noise_vars


def initialize_bayesian_heads_from_pretrained(
    policy_det, bayesian_prior_precision: float = 1.0,
    graph_cache: Optional[List] = None, device=None,
    exploration_fraction: float = 0.3,
    weights_cache: Optional[List] = None,
):
    """Build a ``use_bayesian_heads=True`` sibling of a trained deterministic policy.

    Copies each bias type's trunk (feature extractor) weights 1:1 — the trunk
    architecture is identical between the two modes — and warm-starts each
    ``BayesianLinearHead``'s posterior *mean* at the deterministic head's
    trained readout weight/bias, leaving posterior *uncertainty* at a
    calibrated prior (see below) so it is genuinely narrowed by real online
    observations.

    Args:
        policy_det: Trained deterministic UnimolPolicy (use_bayesian_heads=False).
        bayesian_prior_precision: Fallback precision multiplier, used only
            when ``graph_cache`` is not provided (see below).
        graph_cache: Per-run cache of ``(unimol_emb, edge_index, targets_list)``
            or ``None`` entries, as built during the main BC loop (see
            ``pretrain_with_runs``). Strongly recommended: when given, prior
            precision is calibrated from real data (see "Calibration" below)
            instead of the weaker weight-magnitude-only heuristic.
        device: torch device for the calibration forward passes (defaults to
            ``policy_det``'s own device). Only used when ``graph_cache`` is given.
        weights_cache: Optional data-hygiene [E][4] pairwise weights (indexed
            like ``graph_cache``); when given, calibration uses only measured
            targets.
        exploration_fraction: Only used with ``graph_cache``. Target
            Thompson-sampling exploration variance, in **physical**
            bias-coefficient units, as a fraction of the real physical-space
            residual variance already estimated for that dimension (see
            ``estimate_reward_noise_variance``) — i.e. "let sampling wiggle
            the action by roughly this fraction of how much real bias
            coefficients for this type actually vary." 0.3 gives visible but
            not overwhelming exploration; raise to explore more aggressively,
            lower to hew closer to the deterministic prediction initially.

    Calibration:
        Without ``graph_cache``, precision is set from ``weight`` magnitude
        alone (``bayesian_prior_precision / mean(weight**2)``), implicitly
        assuming trunk features ``z`` have norm ~1. Real Uni-Mol-derived
        trunk features don't (``||z||`` of 5-10+ is typical), and scale
        factors vary hugely across bias types (quadratic's is 520, end's is
        30) — so even a "tight-looking" weight-only calibration can leave
        Thompson-sampling noise, in *physical* units near the origin
        (``scale_factor * raw_noise``), swamping the signal for
        high-scale_factor dimensions. This was caught against a real
        checkpoint: sampled actions differed from the (correct) deterministic
        mean by up to ~9x, including sign flips, despite an apparently
        well-calibrated weight-based precision.

        With ``graph_cache``, precision is instead set so the *physical-unit*
        exploration variance directly targets a fraction of the *physical*
        residual variance real BC data actually shows for that dimension —
        using the real trunk's ``E[||z||^2]`` (``_compute_typical_z_norm_sq``)
        and the real physical-space residual variance
        (``_compute_physical_residual_variance``, shared with
        ``estimate_reward_noise_variance``) instead of a scale-blind guess.

    Returns:
        A new UnimolPolicy with use_bayesian_heads=True, same architecture/
        dims as policy_det, trunks copied, posterior means seeded.
    """

    policy_bayes = UnimolPolicy(
        unimol_dim=policy_det.unimol_dim,
        mlp_out_dim=policy_det.mlp_out_dim,
        use_dual_embeddings=policy_det.use_dual_embeddings,
        env_input=getattr(policy_det, 'env_input', 'dual'),
        site_size_input=getattr(policy_det, 'site_size_input', False),
        use_bayesian_heads=True,
        bayesian_prior_precision=bayesian_prior_precision,
    ).to(next(policy_det.parameters()).device)

    typical_z_norm_sq = None
    residual_vars_physical = None
    if graph_cache is not None:
        device = device or next(policy_det.parameters()).device
        typical_z_norm_sq = _compute_typical_z_norm_sq(policy_det, graph_cache, device)
        residuals, _ = _compute_physical_residual_variance(
            policy_det, graph_cache, device, weights_cache=weights_cache)
        residual_vars_physical = [
            max(float(np.var(residuals[d])), 1e-6) if len(residuals[d]) >= 5 else None
            for d in range(4)
        ]
        log.info('%s', "  Data-driven prior-precision calibration: E[||z||^2]="
                 + ', '.join(f'{v:.2f}' for v in typical_z_norm_sq)
                 + ", physical residual var="
                 + ', '.join(f'{v:.4f}' if v is not None else 'n/a'
                             for v in residual_vars_physical))

    scale_factors = policy_det.scale_factors.detach().cpu().numpy()

    for d in range(policy_det.mlp_out_dim):
        det_mlp = policy_det.edge_mlp.mlps[d]
        bayes_mlp = policy_bayes.edge_mlp.mlps[d]
        bayes_mlp.trunk.load_state_dict(det_mlp.trunk.state_dict())
        # readout: Linear(64, 2) -> row 0 is the mean branch (row 1 is log_std,
        # which has no Bayesian-mode analogue — uncertainty now comes from the
        # posterior instead).
        weight = det_mlp.readout.weight.data[0]  # [64]
        bias = float(det_mlp.readout.bias.data[0])

        if typical_z_norm_sq is not None and residual_vars_physical[d] is not None:
            # Data-driven: target physical-unit exploration variance as a
            # fraction of real physical residual variance, then convert to
            # the raw pre-activation precision that produces it given this
            # head's actual typical ||z||^2 (see docstring "Calibration").
            target_physical_var = exploration_fraction * residual_vars_physical[d]
            target_preact_var = target_physical_var / (float(scale_factors[d]) ** 2)
            adaptive_precision = typical_z_norm_sq[d] / max(target_preact_var, 1e-8)
        else:
            # Fallback: weight-magnitude-only heuristic (see docstring).
            combined = torch.cat([weight, torch.tensor(
                [bias], dtype=weight.dtype, device=weight.device)])
            typical_scale_sq = float(combined.pow(2).mean().clamp(min=1e-6))
            adaptive_precision = bayesian_prior_precision / typical_scale_sq

        bayes_mlp.bayesian_head.seed_mean(weight, bias, prior_precision=adaptive_precision)

    return policy_bayes
