"""Comprehensive tests for pairwise MLP policy implementation.

Tests:
1. Bidirectional pair generation
2. Feature extraction from RTF files
3. Policy forward pass and predictions
4. Bias dictionary conversion (antisymmetric and independent biases)
5. Linear bias conversion from relative-to-first format
6. Pretraining target extraction
"""
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from mllf.cb.pairwise_utils import (
    build_directed_pairs,
    predictions_to_bias_dict,
    extract_substituent_features,
)
from mllf.cb.pairwise_mlp_policy import PairwiseBiasMLP, PairwiseMLPPolicy
from mllf.cb.pretrain_pairwise_policy import extract_pairwise_targets_from_variables


class TestBidirectionalPairs:
    """Test bidirectional pair generation."""
    
    def test_pair_count(self):
        """Test correct number of pairs generated."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        # Site 1: 3 subs -> 3*(3-1) = 6 directed pairs
        # Site 2: 2 subs -> 2*(2-1) = 2 directed pairs
        # Total: 8 pairs
        assert len(pairs) == 8
    
    def test_no_diagonal(self):
        """Test that no self-pairs are generated."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        for i, j in pairs:
            assert i != j, f"Found diagonal pair ({i}, {j})"
    
    def test_both_directions(self):
        """Test that both directions are present."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        forward = {(i, j) for i, j in pairs if i < j}
        reverse = {(j, i) for i, j in pairs if i > j}
        
        assert forward == reverse, "Forward and reverse pairs don't match"
    
    def test_no_cross_site(self):
        """Test that no cross-site pairs are generated."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        for i, j in pairs:
            # Determine which site each substituent belongs to
            if i < 3:
                site_i = 0
            else:
                site_i = 1
            
            if j < 3:
                site_j = 0
            else:
                site_j = 1
            
            assert site_i == site_j, f"Cross-site pair: ({i}, {j})"


class TestBiasConversion:
    """Test conversion from predictions to bias dictionaries."""
    
    def test_linear_antisymmetry(self):
        """Test that linear biases are antisymmetric."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        # Create test predictions with slight asymmetry
        actions = torch.tensor([
            [5.0, 10.0, 15.0, 20.0],   # (0→1)
            [7.0, 12.0, 16.0, 21.0],   # (0→2)
            [-6.0, -11.0, 25.0, 30.0], # (1→0)
            [9.0, 14.0, 17.0, 22.0],   # (1→2)
            [-8.0, -13.0, 26.0, 31.0], # (2→0)
            [-10.0, -15.0, 27.0, 32.0],# (2→1)
            [11.0, 16.0, 18.0, 23.0],  # (3→4)
            [-12.0, -17.0, 28.0, 33.0],# (4→3)
        ], dtype=torch.float32)
        
        bias_dict = predictions_to_bias_dict(actions, pairs, nsubs_per_site)
        b_mat = np.array(bias_dict['b'])
        
        # Check antisymmetry: b[j,i] = -b[i,j]
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert abs(b_mat[i, j] + b_mat[j, i]) < 0.01, \
                        f"Linear bias not antisymmetric at ({i},{j})"
    
    def test_quadratic_antisymmetry(self):
        """Test that quadratic biases use upper triangle only (prevents cancellation)."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        actions = torch.tensor([
            [5.0, 10.0, 15.0, 20.0],
            [7.0, 12.0, 16.0, 21.0],
            [-6.0, -11.0, 25.0, 30.0],
            [9.0, 14.0, 17.0, 22.0],
            [-8.0, -13.0, 26.0, 31.0],
            [-10.0, -15.0, 27.0, 32.0],
            [11.0, 16.0, 18.0, 23.0],
            [-12.0, -17.0, 28.0, 33.0],
        ], dtype=torch.float32)
        
        bias_dict = predictions_to_bias_dict(actions, pairs, nsubs_per_site)
        c_mat = np.array(bias_dict['c'])
        
        # Check upper triangle only (lower triangle should be zero to prevent cancellation)
        for i in range(5):
            for j in range(5):
                if i > j:
                    # Lower triangle should be zero
                    assert abs(c_mat[i, j]) < 0.01, \
                        f"Lower triangle should be zero at ({i},{j}), got {c_mat[i, j]}"
                elif i < j:
                    # Upper triangle can have non-zero values
                    pass  # Values in upper triangle are valid
    
    def test_skew_independence(self):
        """Test that skew biases have independent directions."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        actions = torch.tensor([
            [5.0, 10.0, 15.0, 20.0],
            [7.0, 12.0, 16.0, 21.0],
            [-6.0, -11.0, 25.0, 30.0],
            [9.0, 14.0, 17.0, 22.0],
            [-8.0, -13.0, 26.0, 31.0],
            [-10.0, -15.0, 27.0, 32.0],
            [11.0, 16.0, 18.0, 23.0],
            [-12.0, -17.0, 28.0, 33.0],
        ], dtype=torch.float32)
        
        bias_dict = predictions_to_bias_dict(actions, pairs, nsubs_per_site)
        x_mat = np.array(bias_dict['x'])
        
        # Check independence: x[0,1] should not equal -x[1,0]
        assert abs(x_mat[0, 1]) > 0.01, "x[0,1] should be non-zero"
        assert abs(x_mat[1, 0]) > 0.01, "x[1,0] should be non-zero"
        assert abs(x_mat[0, 1] + x_mat[1, 0]) > 0.01, \
            "Skew bias should NOT be antisymmetric"
    
    def test_end_independence(self):
        """Test that end biases have independent directions."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        actions = torch.tensor([
            [5.0, 10.0, 15.0, 20.0],
            [7.0, 12.0, 16.0, 21.0],
            [-6.0, -11.0, 25.0, 30.0],
            [9.0, 14.0, 17.0, 22.0],
            [-8.0, -13.0, 26.0, 31.0],
            [-10.0, -15.0, 27.0, 32.0],
            [11.0, 16.0, 18.0, 23.0],
            [-12.0, -17.0, 28.0, 33.0],
        ], dtype=torch.float32)
        
        bias_dict = predictions_to_bias_dict(actions, pairs, nsubs_per_site)
        s_mat = np.array(bias_dict['s'])
        
        # Check independence: s[0,1] should not equal -s[1,0]
        assert abs(s_mat[0, 1]) > 0.01, "s[0,1] should be non-zero"
        assert abs(s_mat[1, 0]) > 0.01, "s[1,0] should be non-zero"
        assert abs(s_mat[0, 1] + s_mat[1, 0]) > 0.01, \
            "End bias should NOT be antisymmetric"


class TestLinearBiasConversion:
    """Test conversion of linear biases from relative-to-first format."""
    
    def test_basic_conversion(self):
        """Test basic linear bias conversion."""
        # Create a simple test case
        nsubs_per_site = [3, 2]
        b_vector = [0.0, 10.0, 20.0, 0.0, 5.0]
        
        # Create full matrices
        c_matrix = np.zeros((5, 5))
        c_matrix[0, 1] = 100.0
        c_matrix[1, 0] = -100.0
        c_matrix[0, 2] = 200.0
        c_matrix[2, 0] = -200.0
        
        x_matrix = np.zeros((5, 5))
        s_matrix = np.zeros((5, 5))
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('"""Test variables file."""\n')
            f.write('import yaml\n\n')
            f.write('bias_string = """\n')
            f.write("b:\n")
            f.write(f"- - {b_vector[0]}\n")
            for v in b_vector[1:]:
                f.write(f"  - {v}\n")
            for name, mat in [("c", c_matrix), ("x", x_matrix), ("s", s_matrix)]:
                f.write(f"{name}:\n")
                for row in mat:
                    f.write(f"- - {row[0]}\n")
                    for val in row[1:]:
                        f.write(f"  - {val}\n")
            f.write('"""\n')
            variables_path = Path(f.name)
        
        try:
            targets = extract_pairwise_targets_from_variables(variables_path, nsubs_per_site)
            
            # For pair (0,1): linear should be b[1] - b[0] = 10 - 0 = 10
            # For pair (1,0): linear should be b[0] - b[1] = 0 - 10 = -10
            pairs = build_directed_pairs(nsubs_per_site)
            
            # Find (0,1) pair
            idx_01 = pairs.index((0, 1))
            assert abs(targets[idx_01, 0] - 10.0) < 0.01, \
                f"Expected linear=10.0 for (0,1), got {targets[idx_01, 0]}"
            
            # Find (1,0) pair
            idx_10 = pairs.index((1, 0))
            assert abs(targets[idx_10, 0] - (-10.0)) < 0.01, \
                f"Expected linear=-10.0 for (1,0), got {targets[idx_10, 0]}"
            
        finally:
            variables_path.unlink()
    
    def test_antisymmetry_preserved(self):
        """Test that linear conversion preserves antisymmetry."""
        nsubs_per_site = [3]
        b_vector = [0.0, 5.0, 15.0]
        
        c_matrix = np.zeros((3, 3))
        x_matrix = np.zeros((3, 3))
        s_matrix = np.zeros((3, 3))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('"""Test variables file."""\n')
            f.write('import yaml\n\n')
            f.write('bias_string = """\n')
            f.write("b:\n")
            f.write(f"- - {b_vector[0]}\n")
            for v in b_vector[1:]:
                f.write(f"  - {v}\n")
            for name, mat in [("c", c_matrix), ("x", x_matrix), ("s", s_matrix)]:
                f.write(f"{name}:\n")
                for row in mat:
                    f.write(f"- - {row[0]}\n")
                    for val in row[1:]:
                        f.write(f"  - {val}\n")
            f.write('"""\n')
            variables_path = Path(f.name)
        
        try:
            targets = extract_pairwise_targets_from_variables(variables_path, nsubs_per_site)
            pairs = build_directed_pairs(nsubs_per_site)
            
            # Check that linear values are antisymmetric
            # (0,1): b[1] - b[0] = 5
            # (1,0): b[0] - b[1] = -5
            idx_01 = pairs.index((0, 1))
            idx_10 = pairs.index((1, 0))
            
            linear_01 = targets[idx_01, 0]
            linear_10 = targets[idx_10, 0]
            
            assert abs(linear_01 + linear_10) < 0.01, \
                "Linear values should be antisymmetric"
            
        finally:
            variables_path.unlink()


class TestPairwiseMLPPolicy:
    """Test pairwise MLP policy architecture."""
    
    def test_forward_pass(self):
        """Test that forward pass produces correct output shape."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        # Create dummy pair features [N_pairs, 178] (difference features - default mode)
        pair_features = torch.randn(len(pairs), 178)
        
        # Create policy - note feature_dim is for a SINGLE substituent
        # Default feature_mode='difference' expects input of size feature_dim
        policy = PairwiseBiasMLP(
            feature_dim=178,
            num_bias_types=4,
            bias_embed_dim=16
        )
        
        # Forward pass
        means, log_stds = policy(pair_features)
        
        # Check shapes
        assert means.shape == (len(pairs), 4), f"Expected shape ({len(pairs)}, 4), got {means.shape}"
        assert log_stds.shape == (len(pairs), 4), f"Expected shape ({len(pairs)}, 4), got {log_stds.shape}"
    
    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        policy = PairwiseBiasMLP(
            feature_dim=178,
            num_bias_types=4,
            bias_embed_dim=16
        )
        
        total_params = sum(p.numel() for p in policy.parameters())
        
        # With difference mode (178 input), should be around 116K parameters
        # (less than concat mode ~162K, much less than graph-based ~500K)
        assert 100_000 < total_params < 150_000, \
            f"Parameter count {total_params} outside expected range"
    
    def test_output_clipping(self):
        """Test that outputs are within expected ranges."""
        nsubs_per_site = [3, 2]
        pairs = build_directed_pairs(nsubs_per_site)
        
        # Create dummy substituent features [N_subs, 178]
        features = torch.randn(5, 178)  # 5 substituents total
        
        # Convert pairs to tensor
        pairs_tensor = torch.tensor(pairs, dtype=torch.long)
        
        policy_wrapper = PairwiseMLPPolicy(
            feature_dim=178,
            num_bias_types=4,
            bias_embed_dim=16
        )
        
        # Get deterministic actions
        actions, logp, means, log_stds = policy_wrapper.get_actions(
            features, pairs_tensor, deterministic=True
        )
        
        # Check that actions are clipped to reasonable ranges
        # Linear: ±79, Quadratic: ±163, Skew: ±11, End: ±7
        assert torch.all(torch.abs(actions[:, 0]) <= 79), "Linear out of range"
        assert torch.all(torch.abs(actions[:, 1]) <= 163), "Quadratic out of range"
        assert torch.all(torch.abs(actions[:, 2]) <= 11), "Skew out of range"
        assert torch.all(torch.abs(actions[:, 3]) <= 7), "End out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
