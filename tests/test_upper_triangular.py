#!/usr/bin/env python3
"""Test that bias matrices are upper-triangular (not antisymmetric) to prevent cancellation."""

import sys
import torch
import yaml
import tempfile
from pathlib import Path

from torch_geometric.data import Data
from mllf.cli.workflow import write_variables_from_actions


def test_upper_triangular_bias_matrices():
    """Test that nonlinear bias matrices only populate upper triangle.
    
    For quadratic (symmetric) matrices, predictions from both edge directions
    (forward and backward) are averaged together. This test verifies that
    forward values are properly stored in the upper triangle.
    """
    
    print("Testing upper-triangular bias matrix generation...")
    print("=" * 60)
    
    # Create a simple 3-node graph with all possible edges
    N = 3
    
    # Edge index: all directed edges (both directions for each pair)
    # For symmetric matrices, both directions will be averaged
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],  # source nodes
        [1, 2, 0, 2, 0, 1]   # target nodes
    ], dtype=torch.long)
    
    # Edge types: 0=quadratic_fwd, 1=quadratic_bwd
    edge_type = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long)
    
    # Node features (doesn't matter for this test)
    x = torch.randn(N, 10)
    
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    # Extras with relation names
    extras = {
        'relation_names': ['quadratic_fwd', 'quadratic_bwd', 'skew_fwd', 'skew_bwd', 
                          'end_fwd', 'end_bwd', 'linear_fwd', 'linear_bwd'],
        'base_relation_map': {
            'quadratic': ('quadratic_fwd', 'quadratic_bwd'),
            'skew': ('skew_fwd', 'skew_bwd'),
            'end': ('end_fwd', 'end_bwd'),
            'linear': ('linear_fwd', 'linear_bwd')
        }
    }
    
    # Create test actions with distinct values for each edge
    # Edge 0 (0->1, quadratic_fwd): 10.0
    # Edge 1 (0->2, quadratic_fwd): 20.0
    # Edge 2 (1->0, quadratic_bwd): 10.0 (will be averaged with fwd)
    # Edge 3 (1->2, quadratic_fwd): 30.0
    # Edge 4 (2->0, quadratic_bwd): 20.0 (will be averaged with fwd)
    # Edge 5 (2->1, quadratic_bwd): 30.0 (will be averaged with fwd)
    # After averaging:
    # (0,1): (10+10)/2 = 10.0
    # (0,2): (20+20)/2 = 20.0
    # (1,2): (30+30)/2 = 30.0
    actions = torch.tensor([10.0, 20.0, 10.0, 30.0, 20.0, 30.0], dtype=torch.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        combo_path = Path(tmpdir)
        
        write_variables_from_actions(
            str(combo_path), data, extras, actions,
            out_name='variables.py',
            bias_clip=1000.0
        )
        
        # Read back and parse
        vars_file = combo_path / 'variables.py'
        vars_content = vars_file.read_text()
        
        exec_globals = {}
        exec(vars_content, exec_globals)
        bias_string = exec_globals['bias_string']
        
        bias_data = yaml.safe_load(bias_string)
        
        c_matrix = bias_data['c']
        
        print("\nGenerated quadratic matrix (c):")
        for i, row in enumerate(c_matrix):
            print(f"  Row {i}: {row}")
        
        print("\nChecking upper-triangular property...")
        
        # Expected: upper triangle populated, lower triangle is zero
        errors = []
        
        # Check upper triangle has values (i < j)
        # Values are averaged from forward and backward edges
        if abs(c_matrix[0][1] - 10.0) > 0.01:
            errors.append(f"c[0][1] should be ~10.0 (average of fwd/bwd), got {c_matrix[0][1]}")
        if abs(c_matrix[0][2] - 20.0) > 0.01:
            errors.append(f"c[0][2] should be ~20.0 (average of fwd/bwd), got {c_matrix[0][2]}")
        if abs(c_matrix[1][2] - 30.0) > 0.01:
            errors.append(f"c[1][2] should be ~30.0 (average of fwd/bwd), got {c_matrix[1][2]}")
        
        # Check lower triangle is zero (i > j)
        if c_matrix[1][0] != 0.0:
            errors.append(f"c[1][0] should be 0.0 (lower triangle), got {c_matrix[1][0]}")
        if c_matrix[2][0] != 0.0:
            errors.append(f"c[2][0] should be 0.0 (lower triangle), got {c_matrix[2][0]}")
        if c_matrix[2][1] != 0.0:
            errors.append(f"c[2][1] should be 0.0 (lower triangle), got {c_matrix[2][1]}")
        
        # Check diagonal is zero
        for i in range(N):
            if c_matrix[i][i] != 0.0:
                errors.append(f"c[{i}][{i}] should be 0.0 (diagonal), got {c_matrix[i][i]}")
        
        if errors:
            print("\n❌ FAILED:")
            for err in errors:
                print(f"  - {err}")
            assert False, "Matrix validation failed:\n" + "\n".join(errors)
        else:
            print("\n✅ SUCCESS: Matrix is upper-triangular")
            print("  ✓ Upper triangle populated")
            print("  ✓ Lower triangle is zero")
            print("  ✓ Diagonal is zero")
            print("  ✓ No antisymmetry (no cancellation)")


if __name__ == '__main__':
    success = test_upper_triangular_bias_matrices()
    sys.exit(0 if success else 1)
