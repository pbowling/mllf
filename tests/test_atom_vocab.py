"""Tests for atom vocabulary configuration and filtering."""
import pytest
import warnings
from mllf.cb.atom_vocab import (
    get_atom_type_vocab, 
    build_atom_type_vocab_from_toppar,
    parse_toppar_file
)
from mllf.cb.graph import Graph
from mllf.cb.graph_utils import build_pyg_graph_from_mllf_graph


def test_default_vocabulary():
    """Test that default vocabulary loads all toppar files."""
    vocab = get_atom_type_vocab(force_rebuild=True)
    
    # Should have many atom types (default toppar has ~333)
    assert len(vocab) > 100, f"Expected >100 atom types, got {len(vocab)}"
    
    # Should be alphabetically sorted
    keys = list(vocab.keys())
    assert keys == sorted(keys), "Vocabulary should be sorted alphabetically"
    
    # Indices should be sequential
    assert set(vocab.values()) == set(range(len(vocab))), "Indices should be sequential"
    
    # Should contain common CHARMM atom types
    common_types = ['CG2R61', 'HGR61', 'NG2R60', 'OG2D1', 'CT1', 'CT2', 'NH1']
    for atom_type in common_types:
        assert atom_type in vocab, f"{atom_type} should be in default vocabulary"


def test_filtered_vocabulary_single_file():
    """Test vocabulary with single toppar file specified."""
    # Use only CGenFF file
    vocab = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf'],
        force_rebuild=True
    )
    
    # Should have fewer types than default
    default_vocab = get_atom_type_vocab(toppar_files=None, force_rebuild=True)
    assert len(vocab) < len(default_vocab), "Filtered vocab should be smaller"
    
    # Should still be sorted
    keys = list(vocab.keys())
    assert keys == sorted(keys), "Filtered vocabulary should be sorted"
    
    # CGenFF types should be present
    assert 'CG2R61' in vocab, "CG2R61 from CGenFF should be present"
    assert 'HGR61' in vocab, "HGR61 from CGenFF should be present"


def test_filtered_vocabulary_multiple_files():
    """Test vocabulary with multiple toppar files specified."""
    vocab = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf', 'top_all36_prot.rtf'],
        force_rebuild=True
    )
    
    # Should have types from both files
    # CGenFF type
    assert 'CG2R61' in vocab or 'CG251O' in vocab, "Should have CGenFF types"
    # Protein type
    assert 'CT1' in vocab or 'NH1' in vocab or 'C' in vocab, "Should have protein types"
    
    # Should still be sorted
    keys = list(vocab.keys())
    assert keys == sorted(keys), "Multi-file vocabulary should be sorted"


def test_vocabulary_caching():
    """Test that vocabulary is cached and reused."""
    # First call builds vocabulary
    vocab1 = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf'],
        force_rebuild=True
    )
    
    # Second call should return cached version (same object)
    vocab2 = get_atom_type_vocab(toppar_files=['top_all36_cgenff.rtf'])
    assert vocab1 is vocab2, "Should return cached vocabulary"
    
    # Different config should rebuild
    vocab3 = get_atom_type_vocab(
        toppar_files=['top_all36_prot.rtf'],
        force_rebuild=True
    )
    assert vocab1 is not vocab3, "Different config should create new vocabulary"


def test_vocabulary_invalidation():
    """Test that cache is invalidated when config changes."""
    # Build with one config
    vocab1 = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf'],
        force_rebuild=True
    )
    size1 = len(vocab1)
    
    # Change config - should rebuild automatically
    vocab2 = get_atom_type_vocab(
        toppar_files=['top_all36_prot.rtf'],
        force_rebuild=False  # Even without force_rebuild
    )
    size2 = len(vocab2)
    
    # Sizes should be different (different toppar files)
    assert size1 != size2, "Different configs should produce different vocabularies"


def test_missing_toppar_file_warning():
    """Test that warning is issued for missing toppar files."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        vocab = build_atom_type_vocab_from_toppar(
            toppar_files=['nonexistent_file.rtf', 'top_all36_cgenff.rtf']
        )
        
        # Should have issued a warning
        assert len(w) > 0, "Should warn about missing file"
        assert "not found" in str(w[0].message).lower(), "Warning should mention file not found"
        
        # Should still have loaded the valid file
        assert len(vocab) > 0, "Should still load valid files"


def test_missing_atom_types_warning():
    """Test warning when substituent has atom types not in vocabulary."""
    # Create a graph with atom types not in CGenFF
    g = Graph(2)
    
    # Use protein-specific atom types that aren't in CGenFF
    g.set_node_info(0, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CT1', 'NH1', 'C']  # Protein types
    })
    
    g.set_node_info(1, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61']  # CGenFF type
    })
    
    from mllf.cb.graph import EdgeCoeffs
    g.set_edge(0, 1, EdgeCoeffs(linear=1.0, quadratic=0.0, skew=0.0, end=0.0))
    
    # Convert with CGenFF-only vocabulary - should warn about missing types
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        data, extras = build_pyg_graph_from_mllf_graph(
            g,
            toppar_files=['top_all36_cgenff.rtf'],
            warn_missing_types=True
        )
        
        # Should have warned about missing atom types
        assert len(w) > 0, "Should warn about missing atom types"
        warning_msg = str(w[0].message).lower()
        assert "atom type" in warning_msg, "Warning should mention atom types"
        assert "not in the vocabulary" in warning_msg or "not in vocabulary" in warning_msg


def test_no_warning_when_disabled():
    """Test that warnings can be disabled."""
    g = Graph(2)
    
    # Use protein types not in CGenFF
    g.set_node_info(0, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CT1', 'NH1']
    })
    
    g.set_node_info(1, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61']
    })
    
    from mllf.cb.graph import EdgeCoeffs
    g.set_edge(0, 1, EdgeCoeffs(linear=1.0, quadratic=0.0, skew=0.0, end=0.0))
    
    # Convert with warnings disabled
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        data, extras = build_pyg_graph_from_mllf_graph(
            g,
            toppar_files=['top_all36_cgenff.rtf'],
            warn_missing_types=False  # Disabled
        )
        
        # Should not have warned
        atom_warnings = [warning for warning in w 
                        if "atom type" in str(warning.message).lower()]
        assert len(atom_warnings) == 0, "Should not warn when disabled"


def test_vocabulary_consistency_across_graphs():
    """Test that same vocabulary config gives consistent features across graphs."""
    toppar_config = ['top_all36_cgenff.rtf']
    
    # Create two graphs with different atom types
    g1 = Graph(1)
    g1.set_node_info(0, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61', 'HGR61']
    })
    
    g2 = Graph(1)
    g2.set_node_info(0, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['NG2R60', 'OG2D1']
    })
    
    # Convert both
    data1, extras1 = build_pyg_graph_from_mllf_graph(
        g1, toppar_files=toppar_config, warn_missing_types=False
    )
    data2, extras2 = build_pyg_graph_from_mllf_graph(
        g2, toppar_files=toppar_config, warn_missing_types=False
    )
    
    # Feature dimensions should match
    assert data1.x.shape[1] == data2.x.shape[1], \
        "Feature dimensions should be consistent across graphs"
    
    # Vocabularies should be identical
    assert extras1['atom_type_vocab'] == extras2['atom_type_vocab'], \
        "Vocabulary should be consistent across graphs"


def test_parse_toppar_file():
    """Test parsing individual toppar file."""
    import os
    
    # Find a toppar file
    toppar_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'toppar'
    )
    
    if os.path.exists(toppar_dir):
        cgenff_path = os.path.join(toppar_dir, 'top_all36_cgenff.rtf')
        
        if os.path.exists(cgenff_path):
            atom_types = parse_toppar_file(cgenff_path)
            
            # Should have found atom types
            assert len(atom_types) > 0, "Should parse atom types from file"
            assert isinstance(atom_types, set), "Should return a set"
            
            # Should contain expected CGenFF types
            expected = {'CG2R61', 'HGR61', 'NG2R60'}
            # At least some of these should be present
            assert len(expected.intersection(atom_types)) > 0, \
                "Should contain expected CGenFF atom types"


def test_vocabulary_with_empty_distinct_types():
    """Test that nodes with no distinct_atom_types don't cause errors."""
    vocab = get_atom_type_vocab(toppar_files=['top_all36_cgenff.rtf'], force_rebuild=True)
    
    g = Graph(2)
    
    # Node with no distinct types
    g.set_node_info(0, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': []
    })
    
    # Node with types
    g.set_node_info(1, {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61']
    })
    
    from mllf.cb.graph import EdgeCoeffs
    g.set_edge(0, 1, EdgeCoeffs(linear=1.0, quadratic=0.0, skew=0.0, end=0.0))
    
    # Should not raise error
    data, extras = build_pyg_graph_from_mllf_graph(
        g, toppar_files=['top_all36_cgenff.rtf'], warn_missing_types=False
    )
    
    # Node 0 should have all zeros in atom type encoding
    vocab = extras['atom_type_vocab']
    node0_atom_features = data.x[0, 4:]  # Skip first 4 base features
    assert node0_atom_features.sum().item() == 0.0, \
        "Node with no atom types should have all zeros"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
