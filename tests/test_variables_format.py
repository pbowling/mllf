"""Test analysis of variables file formats.

Verifies understanding of ALF variables file structure:
- Linear biases stored relative to first substituent
- Quadratic biases stored antisymmetrically (implied)
- Skew and end biases stored with independent directions
"""
import pytest
from pathlib import Path

from mllf.file_handling.read_bias_coeff import parse_old


class TestVariablesFileFormat:
    """Test understanding of variables file format from real ALF output."""
    
    @pytest.fixture
    def variables85_data(self):
        """Load variables85.inp from the checked-in samples directory."""
        inp_file = Path(__file__).parent / 'samples' / 'variables85.inp'
        if not inp_file.exists():
            pytest.skip("variables85.inp not found")
        return parse_old(str(inp_file))
    
    def test_linear_format(self, variables85_data):
        """Test that linear biases are relative to first sub."""
        lams = variables85_data['lams']
        
        # Check that first sub at each site is 0.00
        # Site 1: lams1s1 should be 0
        assert 'lams1s1' in lams
        assert abs(lams['lams1s1']) < 0.01, "First sub should be reference (0.00)"
        
        # Site 2: lams2s1 should be 0
        assert 'lams2s1' in lams
        assert abs(lams['lams2s1']) < 0.01, "First sub should be reference (0.00)"
    
    def test_quadratic_storage(self, variables85_data):
        """Test that quadratic coefficients are stored in upper triangle only."""
        cs = variables85_data['cs']
        
        # Count total entries
        total_entries = len(cs)
        
        # Check that reverse directions are NOT present
        # If cs1s1s1s2 exists, cs1s2s1s1 should NOT exist
        has_forward = 'cs1s1s1s2' in cs
        has_reverse = 'cs1s2s1s1' in cs
        
        if has_forward:
            assert not has_reverse, \
                "Quadratic should only store forward direction (upper triangle)"
    
    def test_skew_bidirectional(self, variables85_data):
        """Test that skew coefficients are stored in both directions."""
        xs = variables85_data['xs']
        
        # Count should be approximately 2x upper triangle (both directions)
        total_entries = len(xs)
        
        # Check that both directions exist for at least one pair
        has_forward = 'xs1s1s1s2' in xs
        has_reverse = 'xs1s2s1s1' in xs
        
        if has_forward:
            assert has_reverse, \
                "Skew should store both directions"
    
    def test_end_bidirectional(self, variables85_data):
        """Test that end coefficients are stored in both directions."""
        ss = variables85_data['ss']
        
        # Check that both directions exist for at least one pair
        has_forward = 'ss1s1s1s2' in ss
        has_reverse = 'ss1s2s1s1' in ss
        
        if has_forward:
            assert has_reverse, \
                "End should store both directions"
    
    def test_skew_not_antisymmetric(self, variables85_data):
        """Test that skew coefficients are NOT antisymmetric."""
        xs = variables85_data['xs']
        
        # Find a pair where both directions exist
        if 'xs1s1s1s2' in xs and 'xs1s2s1s1' in xs:
            fwd = xs['xs1s1s1s2']
            rev = xs['xs1s2s1s1']
            
            # They should NOT satisfy fwd = -rev
            assert abs(fwd + rev) > 0.01, \
                "Skew biases should NOT be antisymmetric"
    
    def test_end_not_antisymmetric(self, variables85_data):
        """Test that end coefficients are NOT antisymmetric."""
        ss = variables85_data['ss']
        
        # Find a pair where both directions exist
        if 'ss1s1s1s2' in ss and 'ss1s2s1s1' in ss:
            fwd = ss['ss1s1s1s2']
            rev = ss['ss1s2s1s1']
            
            # They should NOT satisfy fwd = -rev
            assert abs(fwd + rev) > 0.01, \
                "End biases should NOT be antisymmetric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
