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
