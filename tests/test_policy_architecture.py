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
    """Tests for EdgeValueMLP with separate heads."""
    
    def test_separate_heads_architecture(self):
        """Test that MLP has separate heads for each bias type."""
        in_dim = 64
        hidden = 64
        num_bias_types = 4
        
        mlp = EdgeValueMLP(in_dim=in_dim, hidden=hidden, num_bias_types=num_bias_types)
        
        # Check trunk architecture
        assert hasattr(mlp, 'trunk'), "MLP should have shared trunk"
        assert len(mlp.trunk) == 2, "Trunk should have 2 layers (Linear, ReLU)"
        
        # Check separate heads
        assert hasattr(mlp, 'heads'), "MLP should have separate heads"
        assert len(mlp.heads) == num_bias_types, f"Should have {num_bias_types} heads"
        
        # Each head should be a Sequential module that outputs 2 values (mean, log_std)
        for i, head in enumerate(mlp.heads):
            assert isinstance(head, nn.Sequential), f"Head {i} should be Sequential"
            # Check final layer output dimension
            final_layer = list(head.children())[-1]
            assert isinstance(final_layer, nn.Linear), f"Head {i} final layer should be Linear"
            assert final_layer.out_features == 2, f"Head {i} should output 2 values"
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        in_dim = 64
        hidden = 64
        num_bias_types = 4
        batch_size = 10
        
        mlp = EdgeValueMLP(in_dim=in_dim, hidden=hidden, num_bias_types=num_bias_types)
        
        x = torch.randn(batch_size, in_dim)
        out = mlp(x)
        
        expected_shape = (batch_size, 2 * num_bias_types)
        assert out.shape == expected_shape, f"Output shape {out.shape} != expected {expected_shape}"
    
    def test_gradient_flow_through_all_heads(self):
        """Test that gradients flow through all heads."""
        mlp = EdgeValueMLP(in_dim=32, hidden=64, num_bias_types=4)
        
        x = torch.randn(5, 32)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        
        # Check trunk gradients
        for param in mlp.trunk.parameters():
            assert param.grad is not None, "Trunk should receive gradients"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), "Gradients should be non-zero"
        
        # Check all heads receive gradients
        for i, head in enumerate(mlp.heads):
            for param in head.parameters():
                assert param.grad is not None, f"Head {i} should receive gradients"


class TestEdgePolicyArchitecture:
    """Tests for EdgePolicy with new architecture."""
    
    def test_output_scaling_to_bias_range(self):
        """Test that mean outputs are scaled to expected bias coefficient ranges.
        
        Scale factors based on 95th percentile + 20% headroom from pretraining data:
        - Linear: ±79
        - Quadratic: ±163
        - Skew: ±11
        - End: ±7
        """
        N, E = 5, 10
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0, mlp_hidden=32, mlp_out_dim=4)
        
        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        
        _, _, mean, _ = policy.get_actions(x, edge_index, edge_type, deterministic=True)
        
        # Mean should be within expected ranges per bias type
        # [linear, quadratic, skew, end]
        expected_max = torch.tensor([79.0, 163.0, 11.0, 7.0])
        
        for i, max_val in enumerate(expected_max):
            assert mean[:, i].min() >= -max_val, f"Bias type {i} min {mean[:, i].min()} should be >= -{max_val}"
            assert mean[:, i].max() <= max_val, f"Bias type {i} max {mean[:, i].max()} should be <= {max_val}"
    
    def test_increased_exploration_range(self):
        """Test that log_std can reach 3.5 (std up to ~33)."""
        N, E = 5, 10
        encoder = RGCNEncoder(in_dim=10, hidden_dims=[16], out_dim=8, num_relations=2)
        policy = EdgePolicy(encoder=encoder, emb_dim=8, edge_feat_dim=0, mlp_hidden=32, mlp_out_dim=4)
        
        x = torch.randn(N, 10)
        edge_index = torch.randint(0, N, (2, E))
        edge_type = torch.randint(0, 2, (E,))
        
        _, _, _, log_std = policy.get_actions(x, edge_index, edge_type, deterministic=True)
        
        # Log_std should be clamped to [-20, 3.5]
        assert log_std.min() >= -20.0, f"Log_std min {log_std.min()} should be >= -20"
        assert log_std.max() <= 3.5, f"Log_std max {log_std.max()} should be <= 3.5"
    
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
        
        # Should correctly infer edge_feat_dim=3 from data.edge_attr
        actions, _, _, _ = policy.get_actions(data.x, data.edge_index, data.edge_type, data.edge_attr)
        assert actions.shape == (E, num_bias_types), "Actions shape should match num_bias_types"
    
    def test_compatibility_with_workflow(self):
        """Test that policy works with actual workflow functions."""
        from pathlib import Path
        from mllf.cli.workflow import run_quick_epoch_for_combo
        
        combo_dir = 'examples/14benz/generated_combos/comb_0001_site1_1__site1_2'
        
        if Path(combo_dir).exists():
            result = run_quick_epoch_for_combo(combo_dir)
            assert 'reward' in result, "Result should contain reward"
            assert isinstance(result['reward'], (int, float)), "Reward should be numeric"


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
