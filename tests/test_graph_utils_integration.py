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
    from mllf.cb.graph import EdgeCoeffs
    g.set_edge(0, 1, EdgeCoeffs(linear=1.0, quadratic=0.5, skew=0.0, end=0.0))
    g.set_edge(1, 2, EdgeCoeffs(linear=-0.5, quadratic=0.0, skew=0.3, end=0.0))
    
    # Convert to PyG format
    data, extras = build_pyg_graph_from_mllf_graph(g)
    
    # Check that vocab was built
    assert 'atom_type_vocab' in extras, "atom_type_vocab should be in extras"
    vocab = extras['atom_type_vocab']
    
    # Vocab should come from toppar files (default includes ~333 types)
    assert len(vocab) > 100, f"Expected >100 atom types from toppar, got {len(vocab)}"
    
    # The atom types used in the graph should be in the vocab
    graph_atom_types = {'CG2R61', 'HGR61', 'NG2R60', 'OG2D1'}
    for atom_type in graph_atom_types:
        assert atom_type in vocab, f"{atom_type} should be in vocabulary"
    
    # Vocab should be sorted
    assert list(vocab.keys()) == sorted(vocab.keys()), "Vocab should be sorted alphabetically"
    
    # Check indices are sequential
    assert set(vocab.values()) == set(range(len(vocab))), "Vocab indices should be sequential"
    
    # Check node feature dimensions
    num_nodes = 3
    vocab_size = len(vocab)
    expected_dim = 4 + vocab_size  # charge + 3 environment flags + vocab_size atom types
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


def test_custom_toppar_configuration():
    """Test that custom toppar configuration is passed through correctly."""
    g = Graph(2)
    
    g.set_node_info(0, {
        'total_charge': 0.5,
        'solvent': 'gas',
        'distinct_atom_types': ['CG2R61', 'HGR61']
    })
    
    g.set_node_info(1, {
        'total_charge': -0.3,
        'solvent': 'solv',
        'distinct_atom_types': ['NG2R60']
    })
    
    from mllf.cb.graph import EdgeCoeffs
    g.set_edge(0, 1, EdgeCoeffs(linear=1.0, quadratic=0.5, skew=0.0, end=0.0))
    
    # Convert with specific toppar files
    data, extras = build_pyg_graph_from_mllf_graph(
        g,
        toppar_files=['top_all36_cgenff.rtf'],
        warn_missing_types=False
    )
    
    # Vocabulary should only include CGenFF types
    vocab = extras['atom_type_vocab']
    assert len(vocab) < 200, "CGenFF-only vocab should be smaller than full vocab"
    
    # Feature dimension should reflect the filtered vocabulary
    expected_dim = 4 + len(vocab)
    assert data.x.shape[1] == expected_dim, \
        f"Feature dimension should be {expected_dim}, got {data.x.shape[1]}"
    
    print("Custom toppar configuration test passed!")


def test_warning_missing_types_integration():
    """Test that missing atom type warnings work in integration."""
    import warnings
    
    g = Graph(2)
    
    # Use protein types that won't be in CGenFF
    g.set_node_info(0, {
        'total_charge': 0.0,
        'solvent': 'protein',
        'distinct_atom_types': ['CT1', 'CT2', 'NH1']  # Protein types
    })
    
    g.set_node_info(1, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61']  # CGenFF type
    })
    
    from mllf.cb.graph import EdgeCoeffs
    g.set_edge(0, 1, EdgeCoeffs(linear=1.0, quadratic=0.0, skew=0.0, end=0.0))
    
    # Should warn when using CGenFF-only vocabulary
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        data, extras = build_pyg_graph_from_mllf_graph(
            g,
            toppar_files=['top_all36_cgenff.rtf'],
            warn_missing_types=True
        )
        
        # Should have at least one warning
        assert len(w) > 0, "Should warn about missing atom types"
        
        # Warning should mention atom types
        warning_text = str(w[0].message).lower()
        assert 'atom type' in warning_text, "Warning should mention atom types"
    
    # Should not warn when disabled
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        data, extras = build_pyg_graph_from_mllf_graph(
            g,
            toppar_files=['top_all36_cgenff.rtf'],
            warn_missing_types=False
        )
        
        # Filter out non-atom-type warnings
        atom_warnings = [warning for warning in w 
                        if 'atom type' in str(warning.message).lower()]
        assert len(atom_warnings) == 0, "Should not warn when disabled"
    
    print("Warning integration test passed!")


if __name__ == '__main__':
    test_atom_type_vocab_integration()
    test_custom_toppar_configuration()
    test_warning_missing_types_integration()
