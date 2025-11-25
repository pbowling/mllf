#!/usr/bin/env python3
"""
Test script to verify bias coefficient clipping works correctly.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mllf.cli.workflow import write_variables_from_actions, build_data_and_targets_from_combo

def test_bias_clipping():
    """Test that extreme bias values are clipped to [-1000, 1000]."""
    
    print("Testing bias coefficient clipping...")
    print("="*60)
    
    # Create a simple test case with extreme values
    # Simulate a small graph with 3 nodes
    N = 3
    
    # Create fake PyG data
    from torch_geometric.data import Data
    
    # Simple triangle graph with all edges
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1]
    ], dtype=torch.long)
    
    # 6 edges, each with a relation type (quadratic_fwd, quadratic_bwd, etc.)
    edge_type = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long)
    
    # Node features (doesn't matter for this test)
    x = torch.randn(N, 10)
    
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    # Create extras with relation names
    extras = {
        'relation_names': ['quadratic_fwd', 'quadratic_bwd', 'skew_fwd', 'skew_bwd', 'end_fwd', 'end_bwd', 'linear_fwd', 'linear_bwd'],
        'base_relation_map': {
            'quadratic': ('quadratic_fwd', 'quadratic_bwd'),
            'skew': ('skew_fwd', 'skew_bwd'),
            'end': ('end_fwd', 'end_bwd'),
            'linear': ('linear_fwd', 'linear_bwd')
        }
    }
    
    # Create actions with EXTREME values
    actions = torch.tensor([
        5e16,   # Edge 0->1: extreme positive (should be clipped to 1000)
        -3e15,  # Edge 0->2: extreme negative (should be clipped to -1000)
        500.0,  # Edge 1->0: reasonable value (should stay 500)
        2500.0, # Edge 1->2: moderately large (should be clipped to 1000)
        -1500.0,# Edge 2->0: moderately negative (should be clipped to -1000)
        100.0   # Edge 2->1: reasonable value (should stay 100)
    ], dtype=torch.float32)
    
    # Test with default clip value (1000.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        combo_path = Path(tmpdir)
        
        write_variables_from_actions(
            str(combo_path), data, extras, actions,
            out_name='variables.py',
            bias_clip=1000.0
        )
        
        # Read back and parse the generated file
        vars_file = combo_path / 'variables.py'
        vars_content = vars_file.read_text()
        
        # Extract bias_string
        exec_globals = {}
        exec(vars_content, exec_globals)
        bias_string = exec_globals['bias_string']
        
        # Parse YAML
        bias_data = yaml.safe_load(bias_string)
        
        print("\nGenerated bias coefficients:")
        print(f"b (linear): {bias_data['b']}")
        print(f"c (quadratic) matrix:")
        for i, row in enumerate(bias_data['c']):
            print(f"  Row {i}: {row}")
        
        # Check that all values are within [-1000, 1000]
        all_values = []
        all_values.extend(bias_data['b'][0])  # b is a list with one row
        for matrix_name in ['c', 'x', 's']:
            for row in bias_data[matrix_name]:
                all_values.extend(row)
        
        max_val = max(all_values)
        min_val = min(all_values)
        
        print(f"\nValue range: [{min_val:.2f}, {max_val:.2f}]")
        
        # Verify clipping
        if max_val <= 1000.0 and min_val >= -1000.0:
            print("✓ SUCCESS: All bias coefficients are within [-1000, 1000]")
            return True
        else:
            print("✗ FAILED: Some values exceed the clip range!")
            print(f"  Max value: {max_val}")
            print(f"  Min value: {min_val}")
            return False

if __name__ == '__main__':
    success = test_bias_clipping()
    sys.exit(0 if success else 1)
