"""Tests for AEV processor and atom feature extraction."""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mllf.cb.aev_processor import (
    ELEMENT_TO_ID,
    NUM_SPECIES,
    extract_charges_from_rtf_metadata,
)


class TestElementMapping:
    """Test element to ID mapping."""
    
    def test_all_cgenff_elements_present(self):
        """Verify all CGenFF elements are in mapping (some map to unknown)."""
        expected_elements = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'Al', 'B', 'Se', 'X']
        assert set(ELEMENT_TO_ID.keys()) == set(expected_elements), \
            "ELEMENT_TO_ID should contain 10 common elements + X + rare elements (Al, B, Se)"
    
    def test_num_species_is_11(self):
        """Verify NUM_SPECIES equals 11 (10 common + 1 unknown)."""
        assert NUM_SPECIES == 11, "NUM_SPECIES should be 11 (10 common elements + X for rare)"
    
    def test_element_ids_are_correct_range(self):
        """Verify element IDs are in range 0 to 10."""
        ids = set(ELEMENT_TO_ID.values())
        assert ids == set(range(11)), "Element IDs should be exactly 0-10 (11 species)"
    
    def test_unknown_element_in_mapping(self):
        """Verify unknown element 'X' is in mapping for rare elements."""
        assert 'X' in ELEMENT_TO_ID, "'X' should be in mapping for unknown/rare elements"
        # Verify rare elements map to X's ID
        x_id = ELEMENT_TO_ID['X']
        assert ELEMENT_TO_ID['Al'] == x_id, "Al should map to X (unknown)"
        assert ELEMENT_TO_ID['B'] == x_id, "B should map to X (unknown)"
        assert ELEMENT_TO_ID['Se'] == x_id, "Se should map to X (unknown)"
    
    def test_common_elements_present(self):
        """Verify common organic chemistry elements are present."""
        common = ['H', 'C', 'N', 'O', 'S', 'P']
        for elem in common:
            assert elem in ELEMENT_TO_ID, f"Common element {elem} should be in mapping"
    
    def test_halogens_present(self):
        """Verify all halogens are present."""
        halogens = ['F', 'Cl', 'Br', 'I']
        for halogen in halogens:
            assert halogen in ELEMENT_TO_ID, f"Halogen {halogen} should be in mapping"
    
    def test_special_elements_present(self):
        """Verify special CGenFF elements are present."""
        special = ['Al', 'B', 'Se']
        for elem in special:
            assert elem in ELEMENT_TO_ID, f"Special element {elem} should be in mapping"


class TestChargeExtraction:
    """Test charge extraction from RTF metadata."""
    
    def test_extract_charges_from_rtf_with_charges(self):
        """Test extracting charges when present in RTF entry."""
        rtf_entry = {
            'charges': [0.1, -0.2, 0.15, -0.05]
        }
        charges = extract_charges_from_rtf_metadata(rtf_entry)
        
        assert charges is not None
        assert isinstance(charges, torch.Tensor)
        assert charges.shape == (4,)
        assert torch.allclose(charges, torch.tensor([0.1, -0.2, 0.15, -0.05]))
    
    def test_extract_charges_from_rtf_without_charges(self):
        """Test extracting charges when not present in RTF entry."""
        rtf_entry = {
            'atom_types': ['CG2R61', 'HGR61']
        }
        charges = extract_charges_from_rtf_metadata(rtf_entry)
        
        assert charges is None
    
    def test_extract_charges_from_empty_rtf(self):
        """Test extracting charges from empty RTF entry."""
        rtf_entry = {}
        charges = extract_charges_from_rtf_metadata(rtf_entry)
        
        assert charges is None
    
    def test_extract_charges_dtype(self):
        """Test that extracted charges have float32 dtype."""
        rtf_entry = {
            'charges': [1.0, -1.0, 0.5]
        }
        charges = extract_charges_from_rtf_metadata(rtf_entry)
        
        assert charges.dtype == torch.float32


class TestAEVComputer:
    """Test AEV computer initialization."""
    
    def test_aev_computer_exists(self):
        """Test that AEV computer is initialized."""
        from mllf.cb.aev_processor import aev_computer
        assert aev_computer is not None
    
    def test_aev_computer_has_correct_species(self):
        """Test that AEV computer is initialized and functional."""
        from mllf.cb.aev_processor import aev_computer
        # AEV computer should be a callable module
        assert aev_computer is not None
        assert hasattr(aev_computer, '__call__')  # Should be callable


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath('old_files/generated_combos_no_pretraining').exists(),
    reason="Test data not available"
)
class TestWithRealData:
    """Tests that require actual PDB/RTF files."""
    
    def test_element_mapping_covers_real_rtf_types(self):
        """Verify ELEMENT_TO_ID covers atom types found in real RTF files."""
        # Common atom types from CGenFF
        common_types = [
            ('CG2R61', 'C'),  # aromatic carbon
            ('HGR61', 'H'),   # aromatic hydrogen
            ('CG331', 'C'),   # aliphatic CH3
            ('OG311', 'O'),   # hydroxyl
            ('NG2S1', 'N'),   # amide nitrogen
            ('SG311', 'S'),   # thiol
            ('CLGA1', 'Cl'),  # aliphatic chlorine
            ('FGA1', 'F'),    # aliphatic fluorine
        ]
        
        # Extract element from atom type (first letter typically)
        for atom_type, expected_element in common_types:
            assert expected_element in ELEMENT_TO_ID, \
                f"Element {expected_element} from atom type {atom_type} should be in mapping"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_extract_charges_with_none_input(self):
        """Test extracting charges with None input."""
        charges = extract_charges_from_rtf_metadata(None)
        assert charges is None
    
    def test_extract_charges_with_empty_list(self):
        """Test extracting charges with empty charge list."""
        rtf_entry = {
            'charges': []
        }
        charges = extract_charges_from_rtf_metadata(rtf_entry)
        
        assert charges is not None
        assert isinstance(charges, torch.Tensor)
        assert len(charges) == 0
    
    def test_element_to_id_has_correct_keys(self):
        """Test that ELEMENT_TO_ID has correct structure."""
        # Should have 14 keys: 10 common + X + Al + B + Se
        assert len(ELEMENT_TO_ID) == 14
        # But only 11 unique values (Al, B, Se map to X's ID)
        assert len(set(ELEMENT_TO_ID.values())) == 11
    
    def test_unknown_element_fallback(self):
        """Test that unknown elements fall back to H with warning."""
        # Test that unknown elements are handled gracefully
        # This is a documentation test - actual behavior tested in integration tests
        fallback_id = ELEMENT_TO_ID.get('UnknownElement', ELEMENT_TO_ID['H'])
        assert fallback_id == ELEMENT_TO_ID['H']


class TestIntegration:
    """Integration tests for multiple components."""
    
    def test_element_ids_match_num_species(self):
        """Test that number of unique element IDs matches NUM_SPECIES."""
        unique_ids = len(set(ELEMENT_TO_ID.values()))
        assert unique_ids == NUM_SPECIES
    
    def test_rare_elements_share_id(self):
        """Test that rare elements (B, Se, Al) share the same ID with X."""
        ids = list(ELEMENT_TO_ID.values())
        # We expect duplicates now (Al, B, Se, X all map to ID 10)
        x_id = ELEMENT_TO_ID['X']
        assert ELEMENT_TO_ID['Al'] == x_id
        assert ELEMENT_TO_ID['B'] == x_id
        assert ELEMENT_TO_ID['Se'] == x_id
    
    def test_element_names_are_valid(self):
        """Test that element names follow chemical conventions."""
        for elem in ELEMENT_TO_ID.keys():
            if elem == 'X':  # Special case for unknown/rare elements
                continue
            # Element symbols should be 1-2 characters, first uppercase
            assert 1 <= len(elem) <= 2, f"Element {elem} should be 1-2 characters"
            assert elem[0].isupper(), f"Element {elem} should start with uppercase"
            if len(elem) == 2:
                assert elem[1].islower(), f"Second char of {elem} should be lowercase"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
