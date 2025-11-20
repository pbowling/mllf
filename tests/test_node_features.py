"""Test node feature encoding for different environment types."""
import torch
from mllf.cb.graph_utils import _node_feature_from_meta
from mllf.cb.atom_vocab import get_atom_type_vocab


# Use real CHARMM atom types for testing
def test_vacuum_environment():
    """Test that vacuum/gas environments are encoded correctly."""
    vocab = get_atom_type_vocab()
    meta = {
        'total_charge': 1.5,
        'solvent': 'gas',
        'distinct_atom_types': ['CG2R61', 'HGR61', 'NG2R60']
    }
    features = _node_feature_from_meta(meta, vocab)
    
    # Expected: [charge, is_vacuum, is_solvent, is_protein, <333 atom types>]
    expected_dim = 4 + len(vocab)
    assert features.shape == (expected_dim,), f"Expected {expected_dim} features, got {features.shape}"
    assert features[0].item() == 1.5, "Charge should be 1.5"
    assert features[1].item() == 1.0, "is_vacuum should be 1.0 for 'gas'"
    assert features[2].item() == 0.0, "is_solvent should be 0.0"
    assert features[3].item() == 0.0, "is_protein should be 0.0"
    # Multi-hot encoding for atom types
    assert features[4 + vocab['CG2R61']].item() == 1.0, "CG2R61 should be present"
    assert features[4 + vocab['HGR61']].item() == 1.0, "HGR61 should be present"
    assert features[4 + vocab['NG2R60']].item() == 1.0, "NG2R60 should be present"


def test_solvent_environment():
    """Test that solvent/water environments are encoded correctly."""
    vocab = get_atom_type_vocab()
    meta = {
        'total_charge': -0.5,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61', 'NG2R60']
    }
    features = _node_feature_from_meta(meta, vocab)
    
    expected_dim = 4 + len(vocab)
    assert features.shape == (expected_dim,), f"Expected {expected_dim} features, got {features.shape}"
    assert features[0].item() == -0.5, "Charge should be -0.5"
    assert features[1].item() == 0.0, "is_vacuum should be 0.0"
    assert features[2].item() == 1.0, "is_solvent should be 1.0 for 'solv'"
    assert features[3].item() == 0.0, "is_protein should be 0.0"
    # Multi-hot encoding
    assert features[4 + vocab['CG2R61']].item() == 1.0, "CG2R61 should be present"
    assert features[4 + vocab['NG2R60']].item() == 1.0, "NG2R60 should be present"
    assert features[4 + vocab['HGR61']].item() == 0.0, "HGR61 should not be present"


def test_protein_environment():
    """Test that protein environments are encoded correctly."""
    vocab = get_atom_type_vocab()
    meta = {
        'total_charge': 0.0,
        'solvent': 'protein',
        'distinct_atom_types': ['CT1', 'CT2', 'NH1', 'O', 'S']
    }
    features = _node_feature_from_meta(meta, vocab)
    
    expected_dim = 4 + len(vocab)
    assert features.shape == (expected_dim,), f"Expected {expected_dim} features, got {features.shape}"
    assert features[0].item() == 0.0, "Charge should be 0.0"
    assert features[1].item() == 0.0, "is_vacuum should be 0.0"
    assert features[2].item() == 0.0, "is_solvent should be 0.0"
    assert features[3].item() == 1.0, "is_protein should be 1.0 for 'protein'"
    # All specified atom types should be present
    for atom_type in ['CT1', 'CT2', 'NH1', 'O', 'S']:
        assert features[4 + vocab[atom_type]].item() == 1.0, f"{atom_type} should be present"


def test_alternative_names():
    """Test that alternative names for environments work correctly."""
    vocab = get_atom_type_vocab()
    
    # Test 'vacuum' alternative
    meta1 = {'total_charge': 0.0, 'solvent': 'vacuum', 'distinct_atom_types': []}
    features1 = _node_feature_from_meta(meta1, vocab)
    assert features1[1].item() == 1.0, "is_vacuum should be 1.0 for 'vacuum'"
    
    # Test 'water' alternative
    meta2 = {'total_charge': 0.0, 'solvent': 'water', 'distinct_atom_types': []}
    features2 = _node_feature_from_meta(meta2, vocab)
    assert features2[2].item() == 1.0, "is_solvent should be 1.0 for 'water'"
    
    # Test 'prot' alternative
    meta3 = {'total_charge': 0.0, 'solvent': 'prot', 'distinct_atom_types': []}
    features3 = _node_feature_from_meta(meta3, vocab)
    assert features3[3].item() == 1.0, "is_protein should be 1.0 for 'prot'"


def test_missing_solvent():
    """Test that missing solvent field results in all zeros for environment."""
    vocab = get_atom_type_vocab()
    meta = {
        'total_charge': 1.0,
        'distinct_atom_types': ['CG2R61']
    }
    features = _node_feature_from_meta(meta, vocab)
    
    # All environment indicators should be 0.0
    assert features[1].item() == 0.0, "is_vacuum should be 0.0 when solvent is missing"
    assert features[2].item() == 0.0, "is_solvent should be 0.0 when solvent is missing"
    assert features[3].item() == 0.0, "is_protein should be 0.0 when solvent is missing"


def test_atom_type_multihot_encoding():
    """Test that atom types are encoded as multi-hot vectors."""
    vocab = get_atom_type_vocab()
    
    # Test with subset of atom types
    meta1 = {'total_charge': 0.0, 'solvent': 'gas', 'distinct_atom_types': ['CG2R61', 'HGR61']}
    features1 = _node_feature_from_meta(meta1, vocab)
    expected_dim = 4 + len(vocab)
    assert features1.shape == (expected_dim,), f"Expected {expected_dim} features, got {features1.shape}"
    assert features1[4 + vocab['CG2R61']].item() == 1.0, "CG2R61 should be present"
    assert features1[4 + vocab['HGR61']].item() == 1.0, "HGR61 should be present"
    assert features1[4 + vocab['NG2R60']].item() == 0.0, "NG2R60 should not be present"
    
    # Test with no atom types
    meta2 = {'total_charge': 0.0, 'solvent': 'gas', 'distinct_atom_types': []}
    features2 = _node_feature_from_meta(meta2, vocab)
    assert features2.shape == (expected_dim,), f"Expected {expected_dim} features, got {features2.shape}"
    # All atom type features should be 0
    for i in range(4, 4 + len(vocab)):
        assert features2[i].item() == 0.0, f"Atom type at index {i} should be 0"
    
    # Test with missing distinct_atom_types
    meta3 = {'total_charge': 0.0, 'solvent': 'gas'}
    features3 = _node_feature_from_meta(meta3, vocab)
    assert features3.shape == (expected_dim,), f"Expected {expected_dim} features, got {features3.shape}"


if __name__ == '__main__':
    test_vacuum_environment()
    test_solvent_environment()
    test_protein_environment()
    test_alternative_names()
    test_missing_solvent()
    test_atom_type_multihot_encoding()
    print("All tests passed!")
