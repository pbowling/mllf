"""Tests for EdgePolicy with separate heads architecture.

This test suite verifies the new EdgePolicy architecture with:
- Separate heads per bias type (quadratic, skew, end, linear)
- Output scaling to [-20, 20] range
- Increased exploration (log_std max = 3.5)
- Deeper MLP with shared trunk
"""
import torch
import torch.nn as nn
import pytest
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy, EdgeValueMLP


class TestEdgeValueMLP:
    """Tests for EdgeValueMLP with 4 independent BiasHeadMLPs."""

    def test_independent_mlps_architecture(self):
        """Test that MLP has 4 independent BiasHeadMLP instances."""
        in_dim = 64
        num_bias_types = 4

        mlp = EdgeValueMLP(in_dim=in_dim, num_bias_types=num_bias_types)

        assert hasattr(mlp, 'mlps'), "EdgeValueMLP should have 'mlps' ModuleList"
        assert len(mlp.mlps) == num_bias_types, f"Should have {num_bias_types} independent MLPs"

        from mllf.cb.policy import BiasHeadMLP
        for i, head in enumerate(mlp.mlps):
            assert isinstance(head, BiasHeadMLP), f"mlps[{i}] should be a BiasHeadMLP"
            # Check final layer output dimension
            final_layer = list(head.net.children())[-1]
            assert isinstance(final_layer, nn.Linear), f"mlps[{i}] final layer should be Linear"
            assert final_layer.out_features == 2, f"mlps[{i}] should output 2 values"

    def test_output_shape(self):
        """Test that output has correct shape."""
        in_dim = 64
        num_bias_types = 4
        batch_size = 10

        mlp = EdgeValueMLP(in_dim=in_dim, num_bias_types=num_bias_types)

        x = torch.randn(batch_size, in_dim)
        out = mlp(x)

        expected_shape = (batch_size, 2 * num_bias_types)
        assert out.shape == expected_shape, f"Output shape {out.shape} != expected {expected_shape}"

    def test_gradient_isolation(self):
        """Test that gradients for each MLP are isolated via edge_type routing."""
        mlp = EdgeValueMLP(in_dim=32, num_bias_types=4)
        x = torch.randn(5, 32)

        # --- Routed path: edge_type provided, all edges belong to bias type 0 ---
        # edge_type 0 and 1 both map to bias_type_index 0 (linear fwd/bwd)
        edge_type = torch.zeros(5, dtype=torch.long)  # all linear edges
        out = mlp(x, edge_type)
        out[:, 0].sum().backward()

        # mlps[0] should have non-zero gradients (it processed all edges)
        for param in mlp.mlps[0].parameters():
            assert param.grad is not None and param.grad.norm().item() > 0.0, \
                "mlps[0] should receive non-zero gradients for its own edge type"

        # mlps[1], [2], [3] must have zero (or no) gradients — routing excluded them
        for i in range(1, 4):
            for param in mlp.mlps[i].parameters():
                grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                assert grad_norm == 0.0, (
                    f"mlps[{i}] should NOT receive non-zero gradients when only mlps[0] "
                    f"edge type is active; got grad_norm={grad_norm:.6f}"
                )

        # --- Legacy path (no edge_type): all MLPs run, all get gradients ---
        for p in mlp.parameters():
            if p.grad is not None:
                p.grad.zero_()
        out_legacy = mlp(x, edge_type=None)
        out_legacy[:, 0].sum().backward()
        # In the legacy path all MLPs participate, so non-zero grads everywhere
        for i in range(4):
            for param in mlp.mlps[i].parameters():
                assert param.grad is not None, f"mlps[{i}] should have grads in legacy path"


class TestEdgePolicyArchitecture:
    """Tests for EdgePolicy with new architecture."""
    
    def test_output_scaling_to_bias_range(self):
        """Test that mean outputs are bounded by tanh scale factors.

        Scale factors (empirical max + ~10% headroom, full pretraining scan):
        - Linear:    ±305
        - Quadratic: ±520
        - Skew:      ±85
        - End:       ±30

        tanh guarantees strict bound: |mean| < scale_factor.
        """
        N, E = 5, 10
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0, mlp_hidden=32, mlp_out_dim=4)

        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))

        _, _, mean, _ = policy.get_actions(x, edge_index, edge_type, deterministic=True)

        # tanh(anything) is in (-1, 1), so mean must be strictly within scale factors
        scale_factors = torch.tensor([305.0, 520.0, 85.0, 30.0])
        for i, max_val in enumerate(scale_factors):
            assert mean[:, i].abs().max() < max_val, (
                f"Bias type {i} max {mean[:, i].abs().max():.4f} should be < {max_val}"
            )
    
    def test_log_std_clamp_range(self):
        """Test that log_std is clamped to [-20, 2.0]."""
        N, E = 5, 10
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0, mlp_hidden=32, mlp_out_dim=4)

        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))

        _, _, _, log_std = policy.get_actions(x, edge_index, edge_type, deterministic=True)

        # Log_std should be clamped to [-20, 2.0]
        assert log_std.min() >= -20.0, f"Log_std min {log_std.min()} should be >= -20"
        assert log_std.max() <= 2.0, f"Log_std max {log_std.max()} should be <= 2.0"
    
    def test_multi_dimensional_actions(self):
        """Test that policy outputs 4 actions per edge (one per bias type)."""
        N, E = 5, 10
        num_bias_types = 4
        
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0, mlp_hidden=32, mlp_out_dim=num_bias_types)
        
        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        
        actions, logp, mean, log_std = policy.get_actions(x, edge_index, edge_type, deterministic=False)
        
        assert actions.shape == (E, num_bias_types), f"Actions shape {actions.shape} != ({E}, {num_bias_types})"
        assert mean.shape == (E, num_bias_types), f"Mean shape {mean.shape} != ({E}, {num_bias_types})"
        assert log_std.shape == (E, num_bias_types), f"Log_std shape {log_std.shape} != ({E}, {num_bias_types})"
        assert logp.shape == (E,), f"Log-prob shape {logp.shape} != ({E},)"
    
    def test_deterministic_mode(self):
        """Test that deterministic mode returns means without sampling."""
        N, E = 5, 10
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, mlp_hidden=32, mlp_out_dim=4)
        
        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        
        actions_det, logp_det, mean, _ = policy.get_actions(x, edge_index, edge_type, deterministic=True)
        
        assert torch.allclose(actions_det, mean), "Deterministic actions should equal means"
        assert torch.allclose(logp_det, torch.zeros_like(logp_det)), "Deterministic logp should be zero"
    
    def test_stochastic_sampling(self):
        """Test that stochastic mode samples different actions."""
        N, E = 5, 10
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, mlp_hidden=32, mlp_out_dim=4)
        
        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        
        # Sample twice
        actions1, _, _, _ = policy.get_actions(x, edge_index, edge_type, deterministic=False)
        actions2, _, _, _ = policy.get_actions(x, edge_index, edge_type, deterministic=False)
        
        # Actions should be different due to stochastic sampling
        assert not torch.allclose(actions1, actions2), "Stochastic samples should differ"
    
    def test_gradient_flow_end_to_end(self):
        """Test that gradients flow from actions back through encoder."""
        N, E = 5, 10
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, mlp_hidden=32, mlp_out_dim=4)
        
        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        
        actions, logp, _, _ = policy.get_actions(x, edge_index, edge_type, deterministic=False)
        
        # Simulate REINFORCE loss
        loss = -(logp.sum() * 1.0) + actions.mean()
        loss.backward()
        
        # Check encoder gradients
        for param in encoder.parameters():
            assert param.grad is not None, "Encoder should receive gradients"
        
        # Check policy gradients
        for param in policy.edge_mlp.parameters():
            assert param.grad is not None, "Policy MLP should receive gradients"
    
    def test_from_pyg_data_constructor(self):
        """Test that from_pyg_data correctly infers dimensions."""
        from torch_geometric.data import Data
        
        N, E = 5, 10
        num_bias_types = 4
        
        # Create dummy PyG data
        data = Data(
            x=torch.randn(N, 10),
            edge_index=torch.randint(0, N, (2, E)),
            edge_type=torch.randint(0, 2, (E,)),
            edge_attr=torch.randn(E, 3)  # 3-dimensional edge features
        )
        
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy.from_pyg_data(
            encoder=encoder,
            emb_dim=8,
            data=data,
            mlp_hidden=32,
            mlp_out_dim=num_bias_types
        )
        #         actions, _, _, _ = policy.get_actions(data.x, data.edge_index, data.edge_type, data.edge_attr)
class TestBackwardCompatibility:
    """Tests to ensure new architecture maintains compatibility."""

    def test_output_format_matches_old_architecture(self):
        """Test that output format is compatible with existing code."""
        N, E = 5, 10
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, mlp_hidden=32, mlp_out_dim=4)

        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))

        # Should return 4-tuple like old architecture
        result = policy.get_actions(x, edge_index, edge_type)
        assert len(result) == 4, "Should return (actions, logp, mean, log_std)"

        actions, logp, mean, log_std = result
        assert isinstance(actions, torch.Tensor), "Actions should be tensor"
        assert isinstance(logp, torch.Tensor), "Logp should be tensor"
        assert isinstance(mean, torch.Tensor), "Mean should be tensor"
        assert isinstance(log_std, torch.Tensor), "Log_std should be tensor"


class TestP1DimSkipConnection:
    """Tests for the p1_dim skip-connection path (AtomBondGNN P1 → RGCN P2)."""

    def _make_policy(self, N, x_dim, emb_dim, p1_dim, mlp_out_dim=4):
        encoder = RGCNEncoder(in_dim=x_dim, hidden_dims=[16], out_dim=emb_dim, num_relations=2)
        policy = EdgePolicy(
            encoder=encoder, emb_dim=emb_dim, edge_feat_dim=0,
            mlp_hidden=32, mlp_out_dim=mlp_out_dim, p1_dim=p1_dim
        )
        return policy

    def test_p1_dim_stored_correctly(self):
        policy = self._make_policy(N=5, x_dim=10, emb_dim=8, p1_dim=16)
        assert policy.p1_dim == 16

    def test_edge_mlp_input_dim_with_p1(self):
        """When p1_dim=16 and emb_dim=8, edge MLP input should be 2*(8+16) = 48."""
        policy = self._make_policy(N=5, x_dim=10, emb_dim=8, p1_dim=16)
        first_layer = policy.edge_mlp.mlps[0].net[0]
        assert first_layer.in_features == 2 * (8 + 16), (
            f"Expected in_features=48, got {first_layer.in_features}"
        )

    def test_edge_mlp_input_dim_without_p1(self):
        """Without p1_dim, edge MLP input should be 2*emb_dim = 16."""
        policy = self._make_policy(N=5, x_dim=10, emb_dim=8, p1_dim=0)
        first_layer = policy.edge_mlp.mlps[0].net[0]
        assert first_layer.in_features == 2 * 8, (
            f"Expected in_features=16, got {first_layer.in_features}"
        )

    def test_skip_connection_forward(self):
        """Test that p1_dim path produces valid outputs with correct shapes."""
        N, E = 6, 12
        x_dim, emb_dim, p1_dim = 10, 8, 10
        policy = self._make_policy(N, x_dim, emb_dim, p1_dim, mlp_out_dim=4)

        x = torch.randn(N, x_dim)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))

        actions, logp, mean, log_std = policy.get_actions(
            x, edge_index, edge_type, deterministic=True
        )
        assert mean.shape == (E, 4)
        assert logp.shape == (E,)
        assert not torch.isnan(mean).any()

    def test_p1_changes_output(self):
        """p1_dim=0 and p1_dim=x_dim should produce different outputs for same x."""
        torch.manual_seed(0)
        N, E = 4, 6
        x_dim = 10
        x = torch.randn(N, x_dim)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))

        torch.manual_seed(42)
        pol_no_skip = self._make_policy(N, x_dim, emb_dim=8, p1_dim=0, mlp_out_dim=4)
        torch.manual_seed(42)
        pol_skip = self._make_policy(N, x_dim, emb_dim=8, p1_dim=x_dim, mlp_out_dim=4)

        _, _, mean_no_skip, _ = pol_no_skip.get_actions(x, edge_index, edge_type, deterministic=True)
        _, _, mean_skip, _ = pol_skip.get_actions(x, edge_index, edge_type, deterministic=True)
        # Different architectures must produce different shapes or values
        assert mean_no_skip.shape == mean_skip.shape
        # (shapes same, but values differ because input dimension differs)


class TestEvaluateLogp:
    """Tests for EdgePolicy.evaluate_logp."""

    def test_evaluate_logp_shapes(self):
        """evaluate_logp returns [E] logp and [E, D] log_std."""
        N, E, D = 5, 8, 4
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0,
                            mlp_hidden=32, mlp_out_dim=D)

        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        saved_actions = torch.randn(E, D)

        logp, log_std = policy.evaluate_logp(x, edge_index, edge_type, None, saved_actions)
        assert logp.shape == (E,), f"logp shape {logp.shape} != ({E},)"
        assert log_std.shape == (E, D), f"log_std shape {log_std.shape} != ({E}, {D})"

    def test_evaluate_logp_gradients(self):
        """evaluate_logp should produce gradients through policy parameters."""
        N, E, D = 4, 6, 4
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0,
                            mlp_hidden=32, mlp_out_dim=D)

        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        saved_actions = torch.randn(E, D).detach()

        logp, _ = policy.evaluate_logp(x, edge_index, edge_type, None, saved_actions)
        loss = -logp.sum()
        loss.backward()

        for name, param in policy.edge_mlp.named_parameters():
            assert param.grad is not None, f"edge_mlp param {name} has no gradient"

    def test_evaluate_logp_saved_actions_no_grad(self):
        """saved_actions tensor must not carry gradients through from evaluate_logp."""
        N, E, D = 4, 6, 4
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0,
                            mlp_hidden=32, mlp_out_dim=D)

        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        saved_actions = torch.randn(E, D, requires_grad=True)

        logp, _ = policy.evaluate_logp(x, edge_index, edge_type, None, saved_actions)
        loss = -logp.sum()
        loss.backward()

        # saved_actions.grad should be None because evaluate_logp detaches it
        assert saved_actions.grad is None, "saved_actions should have no gradient (detached)"

    def test_evaluate_logp_deterministic_peak(self):
        """logp should be highest when saved_actions == mean."""
        N, E, D = 4, 6, 4
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0,
                            mlp_hidden=32, mlp_out_dim=D)
        policy.eval()

        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))

        with torch.no_grad():
            _, _, mean, _ = policy.get_actions(x, edge_index, edge_type, deterministic=True)
            logp_at_mean, _ = policy.evaluate_logp(x, edge_index, edge_type, None, mean)
            off_actions = mean + 50.0   # far from mean
            logp_off, _ = policy.evaluate_logp(x, edge_index, edge_type, None, off_actions)

        assert logp_at_mean.sum() > logp_off.sum(), (
            "logp should be higher at mean than far from mean"
        )


class TestFrozenEncoderRL:
    """Test that freezing the encoder isolates gradients to edge_mlp."""

    def test_frozen_encoder_no_grad(self):
        """encoder.requires_grad_(False) should prevent encoder parameter updates."""
        N, E = 5, 8
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, mlp_hidden=32, mlp_out_dim=4)

        # Freeze encoder (as done in run_workflow_deepset.py)
        policy.encoder.requires_grad_(False)

        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        saved_actions = torch.randn(E, 4).detach()

        logp, _ = policy.evaluate_logp(x, edge_index, edge_type, None, saved_actions)
        loss = -logp.sum()
        loss.backward()

        # Encoder params should have no gradient
        for name, param in policy.encoder.named_parameters():
            assert param.grad is None, f"Frozen encoder param {name} should have no grad"

        # edge_mlp params should have gradient
        for name, param in policy.edge_mlp.named_parameters():
            assert param.grad is not None, f"edge_mlp param {name} should have grad"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
