"""Integration test for graph_utils with atom type encoding."""
import torch
from mllf.cb.graph import Graph
from mllf.cb.graph_utils import build_pyg_graph_from_mllf_graph


def test_atom_type_vocab_integration():
    """Test that atom type vocabulary is built correctly from a graph."""
    # Create a simple graph with 3 nodes
    g = Graph(3)
    
    # Set metadata for each node with different atom types
    g.set_node_info(0, {
        'total_charge': 0.5,
        'solvent': 'gas',
        'distinct_atom_types': ['CG2R61', 'HGR61']
    })
    
    g.set_node_info(1, {
        'total_charge': -0.3,
        'solvent': 'solv',
        'distinct_atom_types': ['NG2R60', 'HGR61']  # HGR61 is shared
    })
    
    g.set_node_info(2, {
        'total_charge': 0.0,
        'solvent': 'protein',
        'distinct_atom_types': ['OG2D1', 'CG2R61']  # CG2R61 is shared
    })
    
    # Add some edges
    g.add_edge(0, 1, linear=1.0, quadratic=0.5)
    g.add_edge(1, 2, linear=-0.5, skew=0.3)
    
    # Convert to PyG format
    data, extras = build_pyg_graph_from_mllf_graph(g)
    
    # Check that vocab was built
    assert 'atom_type_vocab' in extras, "atom_type_vocab should be in extras"
    vocab = extras['atom_type_vocab']
    
    # Should have 4 unique atom types
    expected_types = {'CG2R61', 'HGR61', 'NG2R60', 'OG2D1'}
    assert set(vocab.keys()) == expected_types, f"Expected {expected_types}, got {set(vocab.keys())}"
    
    # Vocab should be sorted
    assert list(vocab.keys()) == sorted(expected_types), "Vocab should be sorted alphabetically"
    
    # Check indices are sequential
    assert set(vocab.values()) == {0, 1, 2, 3}, "Vocab indices should be 0-3"
    
    # Check node feature dimensions
    num_nodes = 3
    vocab_size = 4
    expected_dim = 4 + vocab_size  # charge + 3 environment flags + 4 atom types
    assert data.x.shape == (num_nodes, expected_dim), f"Expected shape ({num_nodes}, {expected_dim}), got {data.x.shape}"
    
    # Check that node 0 has correct atom type encoding
    # Node 0 has CG2R61 and HGR61
    node0_features = data.x[0]
    assert node0_features[0].item() == 0.5, "Charge should be 0.5"
    assert node0_features[1].item() == 1.0, "is_vacuum should be 1.0"
    assert node0_features[2].item() == 0.0, "is_solvent should be 0.0"
    assert node0_features[3].item() == 0.0, "is_protein should be 0.0"
    
    # Check atom type encoding (CG2R61=index 0, HGR61=index 1, NG2R60=index 2, OG2D1=index 3)
    assert node0_features[4 + vocab['CG2R61']].item() == 1.0, "CG2R61 should be present"
    assert node0_features[4 + vocab['HGR61']].item() == 1.0, "HGR61 should be present"
    assert node0_features[4 + vocab['NG2R60']].item() == 0.0, "NG2R60 should not be present"
    assert node0_features[4 + vocab['OG2D1']].item() == 0.0, "OG2D1 should not be present"
    
    # Check node 1
    node1_features = data.x[1]
    assert node1_features[2].item() == 1.0, "is_solvent should be 1.0"
    assert node1_features[4 + vocab['NG2R60']].item() == 1.0, "NG2R60 should be present"
    assert node1_features[4 + vocab['HGR61']].item() == 1.0, "HGR61 should be present"
    
    # Check node 2
    node2_features = data.x[2]
    assert node2_features[3].item() == 1.0, "is_protein should be 1.0"
    assert node2_features[4 + vocab['OG2D1']].item() == 1.0, "OG2D1 should be present"
    assert node2_features[4 + vocab['CG2R61']].item() == 1.0, "CG2R61 should be present"
    
    print("Integration test passed!")


if __name__ == '__main__':
    test_atom_type_vocab_integration()
