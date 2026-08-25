"""Tests for UnimolPolicy's NeuralLinear + Thompson Sampling mode
(use_bayesian_heads=True): forward_features/predict_uncertainty/
get_actions_thompson/update_bayesian_posteriors, and that the default
(use_bayesian_heads=False) REINFORCE path is unaffected.
"""
import torch

from mllf.cb.policy import UnimolPolicy


def _toy_graph(n_nodes=6, unimol_dim=512, dual=True, seed=0):
    torch.manual_seed(seed)
    emb_dim = 1024 if dual else unimol_dim
    emb = torch.randn(n_nodes, emb_dim)
    # 3 undirected pairs, each represented as one directed edge; edge_type
    # spans all 8 relation slots (2 per bias type) so bias_type = edge_type//2
    # covers all 4 bias types across a handful of edges.
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
                                [1, 0, 3, 2, 5, 4]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    return emb, edge_index, edge_type


class TestBayesianPolicyShapes:
    def test_forward_features_shape(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()
        z = policy.forward_features(emb, edge_index, edge_type)
        assert z.shape == (edge_index.shape[1], 64)

    def test_predict_uncertainty_shapes_and_positive_var(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()
        mean, var = policy.predict_uncertainty(emb, edge_index, edge_type)
        E = edge_index.shape[1]
        assert mean.shape == (E,)
        assert var.shape == (E,)
        assert (var > 0).all()

    def test_get_actions_thompson_shape_and_clip(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()
        actions, mean, var = policy.get_actions_thompson(emb, edge_index, edge_type)
        E = edge_index.shape[1]
        assert actions.shape == (E, 4)
        assert mean.shape == (E,)
        assert var.shape == (E,)
        clip_limits = policy.scale_factors * 1.05
        for d in range(4):
            assert (actions[:, d].abs() <= clip_limits[d] + 1e-4).all()

    def test_get_actions_thompson_only_routed_column_nonzero(self):
        """Non-routed dims stay exactly 0 (same convention as get_actions'
        routed path — only the edge's own bias-type column is meaningful)."""
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()
        actions, _, _ = policy.get_actions_thompson(emb, edge_index, edge_type)
        bias_type = edge_type // 2
        for e in range(edge_index.shape[1]):
            for d in range(4):
                if d != bias_type[e].item():
                    assert actions[e, d].item() == 0.0

    def test_deterministic_uses_posterior_mean(self):
        """deterministic=True should reproduce the posterior mean exactly
        (no sampling noise) across repeated calls."""
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()
        a1, _, _ = policy.get_actions_thompson(emb, edge_index, edge_type, deterministic=True)
        a2, _, _ = policy.get_actions_thompson(emb, edge_index, edge_type, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_thompson_sampling_produces_variation(self):
        """Non-deterministic calls should (almost surely) differ across draws
        when the posterior still has real uncertainty (untrained prior)."""
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()
        torch.manual_seed(1)
        a1, _, _ = policy.get_actions_thompson(emb, edge_index, edge_type)
        a2, _, _ = policy.get_actions_thompson(emb, edge_index, edge_type)
        assert not torch.allclose(a1, a2)


class TestBayesianPolicyLearning:
    def test_update_reduces_uncertainty(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()
        _, var_before = policy.predict_uncertainty(emb, edge_index, edge_type)

        targets = torch.randn(edge_index.shape[1])
        policy.update_bayesian_posteriors(emb, edge_index, edge_type, targets)

        _, var_after = policy.predict_uncertainty(emb, edge_index, edge_type)
        assert (var_after <= var_before + 1e-6).all()
        assert (var_after < var_before).any()

    def test_update_increments_n_obs_only_for_active_heads(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()  # bias_type covers dims 0,1,2 only
        targets = torch.randn(edge_index.shape[1])
        policy.update_bayesian_posteriors(emb, edge_index, edge_type, targets)

        n_obs = [mlp.bayesian_head.n_obs.item() for mlp in policy.edge_mlp.mlps]
        assert n_obs[0] > 0 and n_obs[1] > 0 and n_obs[2] > 0
        assert n_obs[3] == 0.0  # bias type 3 ('end') never appears in this toy graph

    def test_update_pulls_mean_toward_target(self):
        """After many updates on a fixed (z, target) pair, the predicted mean
        for that same edge should move toward the observed target."""
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = _toy_graph()
        target_value = 50.0
        targets = torch.full((edge_index.shape[1],), target_value)

        mean_before, _ = policy.predict_uncertainty(emb, edge_index, edge_type)
        for _ in range(20):
            policy.update_bayesian_posteriors(emb, edge_index, edge_type, targets)
        mean_after, _ = policy.predict_uncertainty(emb, edge_index, edge_type)

        assert (mean_after - target_value).abs().mean() < (mean_before - target_value).abs().mean()


class TestDeterministicPolicyUnaffected:
    """use_bayesian_heads=False (the default) must behave exactly as before."""

    def test_default_policy_has_no_bayesian_heads(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True)
        assert policy.use_bayesian_heads is False
        for mlp in policy.edge_mlp.mlps:
            assert mlp.bayesian_head is None
            assert mlp.readout is not None

    def test_legacy_get_actions_still_works(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True)
        emb, edge_index, edge_type = _toy_graph()
        actions, logp, mean, log_std = policy.get_actions(emb, edge_index, edge_type)
        E = edge_index.shape[1]
        assert actions.shape == (E, 4)
        assert logp.shape == (E,)
        assert mean.shape == (E, 4)
        assert log_std.shape == (E, 4)


class TestBayesianPolicyUnroutedEdgeType:
    """edge_type=None: the *actual* shape training/workflow.py's
    build_graph_and_data always produces (one edge per pair, every bias type
    independently meaningful for every edge -- no routing). This is the path
    online training actually exercises, as opposed to the routed path used by
    pretraining's fully-connected 4x-edges-per-pair graph structure.
    """

    def _toy_graph_unrouted(self, n_nodes=6, seed=0):
        torch.manual_seed(seed)
        emb = torch.randn(n_nodes, 1024)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
                                    [1, 0, 3, 2, 5, 4]], dtype=torch.long)
        return emb, edge_index, None

    def test_forward_features_shape(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = self._toy_graph_unrouted()
        z = policy.forward_features(emb, edge_index, edge_type)
        E = edge_index.shape[1]
        assert z.shape == (E, 4, 64)

    def test_predict_uncertainty_shape(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = self._toy_graph_unrouted()
        mean, var = policy.predict_uncertainty(emb, edge_index, edge_type)
        E = edge_index.shape[1]
        assert mean.shape == (E, 4)
        assert var.shape == (E, 4)
        assert (var > 0).all()

    def test_get_actions_thompson_every_column_is_real(self):
        """Unlike the routed path, no column should be a zero-filled
        placeholder -- every bias type predicts for every edge."""
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = self._toy_graph_unrouted()
        actions, mean, var = policy.get_actions_thompson(emb, edge_index, edge_type)
        E = edge_index.shape[1]
        assert actions.shape == (E, 4)
        assert mean.shape == (E, 4)
        assert var.shape == (E, 4)
        # A freshly-initialized (non-degenerate) posterior sampled at a
        # random z is vanishingly unlikely to land exactly at 0.
        assert (actions != 0).all()

    def test_update_bayesian_posteriors_updates_every_head_with_every_edge(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = self._toy_graph_unrouted()
        E = edge_index.shape[1]
        targets = torch.randn(E, 4)

        policy.update_bayesian_posteriors(emb, edge_index, edge_type, targets)

        n_obs = [mlp.bayesian_head.n_obs.item() for mlp in policy.edge_mlp.mlps]
        assert all(n == float(E) for n in n_obs), (
            "every edge should update every head when edge_type is None"
        )

    def test_deterministic_reproducible_when_unrouted(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = self._toy_graph_unrouted()
        a1, _, _ = policy.get_actions_thompson(emb, edge_index, edge_type, deterministic=True)
        a2, _, _ = policy.get_actions_thompson(emb, edge_index, edge_type, deterministic=True)
        assert torch.allclose(a1, a2)


class TestPreactivationScaling:
    """Regression tests for the softsign+scale_factors transform between the
    posterior's native pre-activation space and physical bias-coefficient
    units (see UnimolPolicy._preact_to_action / ._action_to_preact).

    An earlier version of get_actions_thompson used the posterior's raw
    z @ w output directly as the action -- an effectively-unbounded linear
    regression value used as-is, wildly out of scale with real MSLD bias
    coefficients (this was caught because it drove combos to near-zero
    transitions in a real training run: reward mean collapsed to -0.976,
    vs. 0.1-0.8 for the REINFORCE baseline on comparable pretraining).
    """

    def _toy_graph(self, seed=0):
        torch.manual_seed(seed)
        emb = torch.randn(6, 1024)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
        return emb, edge_index, None

    def test_preact_action_roundtrip(self):
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        for d in range(4):
            x = torch.linspace(-8.0, 8.0, 50)
            y = policy._preact_to_action(x, d)
            x_back = policy._action_to_preact(y, d)
            assert torch.allclose(x, x_back, atol=1e-3)

    def test_actions_always_within_physical_scale_even_for_huge_weights(self):
        """A pathologically large posterior mean must still produce an action
        strictly within (-scale_factors[d], scale_factors[d]) -- this is
        exactly the failure mode the missing transform allowed."""
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = self._toy_graph()

        for d, mlp in enumerate(policy.edge_mlp.mlps):
            mlp.bayesian_head.mu.data.fill_(50.0)  # deliberately huge pre-activation weights

        actions, mean, _ = policy.get_actions_thompson(emb, edge_index, edge_type, deterministic=True)
        for d in range(4):
            limit = policy.scale_factors[d].item()
            assert (actions[:, d].abs() < limit).all(), f"dim {d} exceeded physical scale"
            assert (mean[:, d].abs() < limit).all()

    def test_update_moves_deterministic_action_toward_physical_target(self):
        """After many updates toward a fixed physical-unit target, the
        deterministic (posterior-mean) action for that same context should
        converge close to the target -- in physical units, not raw
        pre-activation units."""
        policy = UnimolPolicy(unimol_dim=512, mlp_out_dim=4,
                               use_dual_embeddings=True, use_bayesian_heads=True)
        emb, edge_index, edge_type = self._toy_graph()
        # Must be comfortably within the *smallest* scale_factor (end: 30) --
        # softsign asymptotes toward but never reaches +/-scale_factor, so a
        # target beyond a dimension's own scale is fundamentally unreachable
        # for that dimension (that's correct behavior, not a bug to test for).
        target_value = 15.0
        E = edge_index.shape[1]
        targets = torch.full((E, 4), target_value)

        for _ in range(50):
            policy.update_bayesian_posteriors(emb, edge_index, edge_type, targets)

        actions, _, _ = policy.get_actions_thompson(emb, edge_index, edge_type, deterministic=True)
        assert (actions - target_value).abs().mean() < 5.0

    def test_noise_var_converted_to_preactivation_space(self):
        """estimate_reward_noise_variance must divide by scale_factors^2, not
        return raw physical-space residual variance (which would be off by
        ~scale_factors^2, badly miscalibrating the posterior update)."""
        from mllf.cb.pretrain_policy import estimate_reward_noise_variance
        policy_det = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True)
        emb = torch.randn(6, 1024)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
        # Targets far outside physical range -> large physical-space residual.
        targets = [[1000.0, 0.0, 0.0, 0.0]] * 6
        graph_cache = [(emb, edge_index, targets)]

        noise_vars = estimate_reward_noise_variance(policy_det, graph_cache, device=torch.device('cpu'))
        scale_linear = policy_det.scale_factors[0].item()
        # A residual of order `scale_linear` in physical space should become
        # order ~1 in pre-activation space (division by scale^2), not stay
        # at the raw physical-space magnitude.
        assert noise_vars[0] < scale_linear


class TestPriorPrecisionCalibration:
    """Regression tests for the prior-precision scale-calibration bug: a
    fixed bayesian_prior_precision=1.0 gives every sampled weight component
    unit variance, which swamps BC-trained weights whose components are
    typically << 1 (and vary hugely in scale across the four bias types).
    Confirmed against a real training run: Thompson-sampled actions were an
    order of magnitude off vs. reference pretraining bias values, even
    though the seeded mean itself (and the softsign+scale transform) were
    correct -- initialize_bayesian_heads_from_pretrained now scales each
    head's prior precision to its own fitted weight magnitude.
    """

    def test_sampled_actions_stay_close_to_seeded_mean(self):
        from mllf.cb.pretrain_policy import initialize_bayesian_heads_from_pretrained

        torch.manual_seed(0)
        det = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True,
                            use_bayesian_heads=False)
        bayes = initialize_bayesian_heads_from_pretrained(det, bayesian_prior_precision=1.0)

        emb = torch.randn(20, 1024)
        edge_index = torch.stack([torch.arange(20), torch.roll(torch.arange(20), 1)])

        for d in range(4):
            head = bayes.edge_mlp.mlps[d].bayesian_head
            z = bayes.edge_mlp.mlps[d].forward_features(bayes._build_edge_input(emb, edge_index))
            mean_det, _ = head.predict(z)

            torch.manual_seed(1)
            samples = torch.stack([head.act(z, head.sample_weights()) for _ in range(20)])

            # Before calibration this ratio was routinely >20x on a
            # lightly-trained network (and worse for well-trained, small-
            # weight heads like 'end'). A well-calibrated prior keeps
            # sampling noise the same order of magnitude as the signal.
            mean_abs = mean_det.abs().mean().clamp(min=1e-8)
            sample_abs = samples.abs().mean()
            assert sample_abs / mean_abs < 10.0, (
                f"dim {d}: Thompson-sampling noise ({sample_abs:.4g}) swamps "
                f"the seeded mean ({mean_abs:.4g}) by >10x"
            )

    def test_data_driven_calibration_keeps_sampling_close_to_mean(self):
        """Regression test for the SECOND calibration bug found against a
        real checkpoint: the weight-magnitude-only heuristic ignored the
        trunk's actual feature scale (real Uni-Mol trunk features have
        ||z|| ~ 5-10+, not ~1) and each dimension's own scale_factor, so even
        a "tight-looking" calibrated precision still let sampled actions
        swing up to ~9x from the (correct) deterministic mean, including sign
        flips, for high-scale_factor dimensions like quadratic (520). The
        graph_cache-driven calibration targets physical-unit exploration
        variance directly against real physical residual scale instead.
        """
        from mllf.cb.pretrain_policy import initialize_bayesian_heads_from_pretrained

        torch.manual_seed(0)
        det = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True,
                            use_bayesian_heads=False)
        # Force small readout weights, matching real BC-trained scale (an
        # untrained/random-init readout is typically already this small, but
        # be explicit so the test doesn't depend on init RNG behavior).
        with torch.no_grad():
            for d in range(4):
                det.edge_mlp.mlps[d].readout.weight.data[0] = torch.randn(64) * 0.02
                det.edge_mlp.mlps[d].readout.bias.data[0] = 0.01

        # Fabricate a graph_cache with REALISTIC-scale embeddings (real
        # Uni-Mol embeddings have norm ~26-38, not ~1 -- see the live-data
        # investigation this test is based on) and targets with plausible
        # physical-space scatter around the deterministic model's own
        # prediction, per bias type.
        physical_budget = [5.0, 8.0, 2.0, 0.5]  # linear, quadratic, skew, end
        graph_cache = []
        for _ in range(30):
            n = 8
            emb = torch.randn(n, 1024) * 1.2  # realistic-order embedding norm
            edge_index = torch.stack([torch.arange(n), torch.roll(torch.arange(n), 1)])
            with torch.no_grad():
                mean, _ = det._forward_edges(emb, edge_index, edge_type=None)
            noise = torch.randn_like(mean) * torch.tensor(physical_budget)
            graph_cache.append((emb, edge_index, (mean + noise).tolist()))

        bayes = initialize_bayesian_heads_from_pretrained(
            det, graph_cache=graph_cache, device=torch.device('cpu'))

        emb_test = torch.randn(10, 1024) * 1.2
        edge_index_test = torch.stack([torch.arange(10), torch.roll(torch.arange(10), 1)])
        actions_det, _, _ = bayes.get_actions_thompson(
            emb_test, edge_index_test, edge_type=None, deterministic=True)

        torch.manual_seed(5)
        diffs = torch.stack([
            (bayes.get_actions_thompson(emb_test, edge_index_test, edge_type=None)[0]
             - actions_det).abs()
            for _ in range(10)
        ])
        dim_names = ['linear', 'quadratic', 'skew', 'end']
        for d, (name, budget) in enumerate(zip(dim_names, physical_budget)):
            mean_diff = diffs[:, :, d].mean().item()
            assert mean_diff < 3 * budget, (
                f"dim {name}: sampled actions deviate from the mean by "
                f"{mean_diff:.2f} on average, expected < {3 * budget} "
                f"(3x the calibration's own physical noise budget)"
            )

    def test_prior_precision_scales_inversely_with_weight_magnitude(self):
        """A head whose fitted weights are 10x smaller should get an
        adaptive prior precision ~100x larger (precision ~ 1/scale^2)."""
        from mllf.cb.pretrain_policy import initialize_bayesian_heads_from_pretrained

        torch.manual_seed(0)
        det = UnimolPolicy(unimol_dim=512, mlp_out_dim=4, use_dual_embeddings=True,
                            use_bayesian_heads=False)
        with torch.no_grad():
            det.edge_mlp.mlps[0].readout.weight.data[0] = torch.randn(64) * 1.0
            det.edge_mlp.mlps[0].readout.bias.data[0] = 0.1
            det.edge_mlp.mlps[1].readout.weight.data[0] = torch.randn(64) * 0.1
            det.edge_mlp.mlps[1].readout.bias.data[0] = 0.01

        bayes = initialize_bayesian_heads_from_pretrained(det, bayesian_prior_precision=1.0)
        precision_0 = bayes.edge_mlp.mlps[0].bayesian_head.Lambda[0, 0].item()
        precision_1 = bayes.edge_mlp.mlps[1].bayesian_head.Lambda[0, 0].item()
        ratio = precision_1 / precision_0
        assert 50.0 < ratio < 200.0, f"expected ~100x tighter precision, got {ratio:.1f}x"
