"""Test node feature encoding for different environment types with element/atom type separation."""
import torch
from mllf.cb.graph_utils import _node_feature_from_meta
from mllf.cb.atom_vocab import get_atom_type_vocab


# Use real CHARMM atom types for testing
def test_solvent_environment():
    """Test that solvent/water environments are encoded correctly."""
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf']
    )
    meta = {
        'total_charge': -0.5,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61', 'NG2R60']  # C and N atoms
    }
    features = _node_feature_from_meta(meta, atom_type_vocab, element_vocab, atom_to_element)
    
    # Expected: [charge, is_solvent, is_protein, <14 elements>, <160 atom types>]
    expected_dim = 3 + len(element_vocab) + len(atom_type_vocab)
    assert features.shape == (expected_dim,), f"Expected {expected_dim} features, got {features.shape}"
    assert features[0].item() == -0.5, "Charge should be -0.5"
    assert features[1].item() == 1.0, "is_solvent should be 1.0 for 'solv'"
    assert features[2].item() == 0.0, "is_protein should be 0.0"
    
    # Check element encoding (C and N should be present)
    element_offset = 3
    assert features[element_offset + element_vocab['C']].item() == 1.0, "Element C should be present"
    assert features[element_offset + element_vocab['N']].item() == 1.0, "Element N should be present"
    assert features[element_offset + element_vocab['H']].item() == 0.0, "Element H should not be present"
    
    # Check atom type encoding
    atom_type_offset = 3 + len(element_vocab)
    assert features[atom_type_offset + atom_type_vocab['CG2R61']].item() == 1.0, "CG2R61 should be present"
    assert features[atom_type_offset + atom_type_vocab['NG2R60']].item() == 1.0, "NG2R60 should be present"
    assert features[atom_type_offset + atom_type_vocab['HGR61']].item() == 0.0, "HGR61 should not be present"


def test_protein_environment():
    """Test that protein environments are encoded correctly."""
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf']
    )
    meta = {
        'total_charge': 0.0,
        'solvent': 'protein',
        'distinct_atom_types': ['CG2R61', 'HGR61', 'NG2R60']  # C, H, N atoms
    }
    features = _node_feature_from_meta(meta, atom_type_vocab, element_vocab, atom_to_element)
    
    expected_dim = 3 + len(element_vocab) + len(atom_type_vocab)
    assert features.shape == (expected_dim,), f"Expected {expected_dim} features, got {features.shape}"
    assert features[0].item() == 0.0, "Charge should be 0.0"
    assert features[1].item() == 0.0, "is_solvent should be 0.0"
    assert features[2].item() == 1.0, "is_protein should be 1.0 for 'protein'"
    
    # Check element encoding (C, H, N should be present)
    element_offset = 3
    assert features[element_offset + element_vocab['C']].item() == 1.0, "Element C should be present"
    assert features[element_offset + element_vocab['H']].item() == 1.0, "Element H should be present"
    assert features[element_offset + element_vocab['N']].item() == 1.0, "Element N should be present"
    assert features[element_offset + element_vocab['O']].item() == 0.0, "Element O should not be present"


def test_alternative_names():
    """Test that alternative names for environments work correctly."""
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf']
    )
    
    # Test 'water' alternative
    meta2 = {'total_charge': 0.0, 'solvent': 'water', 'distinct_atom_types': []}
    features2 = _node_feature_from_meta(meta2, atom_type_vocab, element_vocab, atom_to_element)
    assert features2[1].item() == 1.0, "is_solvent should be 1.0 for 'water'"
    
    # Test 'prot' alternative
    meta3 = {'total_charge': 0.0, 'solvent': 'prot', 'distinct_atom_types': []}
    features3 = _node_feature_from_meta(meta3, atom_type_vocab, element_vocab, atom_to_element)
    assert features3[2].item() == 1.0, "is_protein should be 1.0 for 'prot'"


def test_missing_solvent():
    """Test that missing solvent field results in all zeros for environment."""
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf']
    )
    meta = {
        'total_charge': 1.0,
        'distinct_atom_types': ['CG2R61']
    }
    features = _node_feature_from_meta(meta, atom_type_vocab, element_vocab, atom_to_element)
    
    # All environment indicators should be 0.0
    assert features[1].item() == 0.0, "is_solvent should be 0.0 when solvent is missing"
    assert features[2].item() == 0.0, "is_protein should be 0.0 when solvent is missing"


def test_element_and_atom_type_encoding():
    """Test that elements and atom types are encoded separately as multi-hot vectors."""
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf']
    )
    
    # Test with benzene-like molecule (C and H)
    meta1 = {
        'total_charge': 0.0, 
        'solvent': 'solv', 
        'distinct_atom_types': ['CG2R61', 'HGR61']  # aromatic C and H
    }
    features1 = _node_feature_from_meta(meta1, atom_type_vocab, element_vocab, atom_to_element)
    expected_dim = 3 + len(element_vocab) + len(atom_type_vocab)
    assert features1.shape == (expected_dim,), f"Expected {expected_dim} features, got {features1.shape}"
    
    # Check element encoding (only C and H should be present)
    element_offset = 3
    assert features1[element_offset + element_vocab['C']].item() == 1.0, "Element C should be present"
    assert features1[element_offset + element_vocab['H']].item() == 1.0, "Element H should be present"
    assert features1[element_offset + element_vocab['N']].item() == 0.0, "Element N should not be present"
    
    # Check atom type encoding
    atom_type_offset = 3 + len(element_vocab)
    assert features1[atom_type_offset + atom_type_vocab['CG2R61']].item() == 1.0, "CG2R61 should be present"
    assert features1[atom_type_offset + atom_type_vocab['HGR61']].item() == 1.0, "HGR61 should be present"
    assert features1[atom_type_offset + atom_type_vocab['NG2R60']].item() == 0.0, "NG2R60 should not be present"
    
    # Test with no atom types
    meta2 = {'total_charge': 0.0, 'solvent': 'solv', 'distinct_atom_types': []}
    features2 = _node_feature_from_meta(meta2, atom_type_vocab, element_vocab, atom_to_element)
    assert features2.shape == (expected_dim,), f"Expected {expected_dim} features, got {features2.shape}"
    
    # All element and atom type features should be 0
    for i in range(3, expected_dim):
        assert features2[i].item() == 0.0, f"Feature at index {i} should be 0"
    
    # Test with missing distinct_atom_types
    meta3 = {'total_charge': 0.0, 'solvent': 'solv'}
    features3 = _node_feature_from_meta(meta3, atom_type_vocab, element_vocab, atom_to_element)
    assert features3.shape == (expected_dim,), f"Expected {expected_dim} features, got {features3.shape}"


def test_vocabulary_consistency():
    """Test that same vocabulary produces consistent feature dimensions."""
    # Get vocabulary with specific configuration
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab(
        toppar_files=['top_all36_cgenff.rtf'],
        force_rebuild=True
    )
    
    # Create features for two different nodes
    meta1 = {
        'total_charge': 0.5,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61']
    }
    
    meta2 = {
        'total_charge': -0.5,
        'solvent': 'protein',
        'distinct_atom_types': ['NG2R60', 'OG2D1']
    }
    
    features1 = _node_feature_from_meta(meta1, atom_type_vocab, element_vocab, atom_to_element)
    features2 = _node_feature_from_meta(meta2, atom_type_vocab, element_vocab, atom_to_element)
    
    # Both should have same dimension
    assert features1.shape == features2.shape, \
        "Features with same vocab should have same dimension"
    
    expected_dim = 3 + len(element_vocab) + len(atom_type_vocab)
    assert features1.shape == (expected_dim,), \
        f"Feature dimension should be {expected_dim}"


def test_missing_atom_types_in_vocab():
    """Test behavior when distinct_atom_types contains types not in vocabulary."""
    # Create a very limited vocabulary (just a few types)
    limited_atom_vocab = {'CG2R61': 0, 'HGR61': 1}
    limited_element_vocab = {'C': 0, 'H': 1}
    limited_mapping = {'CG2R61': 'C', 'HGR61': 'H'}
    
    # Try to encode node with types not in limited vocab
    meta = {
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61', 'NG2R60', 'UNKNOWN_TYPE']
    }
    
    features = _node_feature_from_meta(meta, limited_atom_vocab, limited_element_vocab, limited_mapping)
    
    # Should have 3 base + 2 elements + 2 atom types = 7 features
    assert features.shape == (7,), f"Expected 7 features, got {features.shape}"
    
    # CG2R61 should be present
    element_offset = 3
    atom_type_offset = 3 + len(limited_element_vocab)
    assert features[element_offset + limited_element_vocab['C']].item() == 1.0, \
        "Element C should be encoded"
    assert features[atom_type_offset + limited_atom_vocab['CG2R61']].item() == 1.0, \
        "CG2R61 should be encoded"
    
    # NG2R60 and UNKNOWN_TYPE should be silently ignored (not in vocab)
    # All other positions should be 0
    assert features[atom_type_offset + limited_atom_vocab['HGR61']].item() == 0.0, \
        "HGR61 should not be present"


if __name__ == '__main__':
    test_solvent_environment()
    test_protein_environment()
    test_alternative_names()
    test_missing_solvent()
    test_element_and_atom_type_encoding()
    test_vocabulary_consistency()
    test_missing_atom_types_in_vocab()
    print("All tests passed!")
