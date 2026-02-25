"""Tests for PDB file reading with CGenFF validation and spatial filtering."""
import pytest
import torch
import tempfile
import os
from pathlib import Path

from mllf.file_handling.read_pdb import (
    parse_pdb_file,
    extract_site_number,
    find_duplicate_atoms,
    remove_duplicate_atoms,
    combine_pdb_files,
    calculate_min_distance,
    find_nearby_pdbs,
    find_reference_subs_from_other_sites,
    parse_pdb_dir
)
from mllf.file_handling.read_rtf import parse_rtf_file


def format_pdb_line(atom_num, atom_name, res_name, res_num, x, y, z, seg='SUB'):
    """Format a PDB ATOM line with proper column alignment."""
    return f"ATOM{atom_num:>7} {atom_name:<4} {res_name:<3}  {res_num:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00      {seg}\n"


@pytest.fixture
def sample_pdb_content():
    """Sample PDB content with various atom types."""
    lines = [
        format_pdb_line(1, 'C001', 'SUB', 1, 1.234, 2.345, 3.456),
        format_pdb_line(2, 'H002', 'SUB', 1, 2.234, 3.345, 4.456),
        format_pdb_line(3, 'N003', 'SUB', 1, 3.234, 4.345, 5.456),
        format_pdb_line(4, 'O004', 'SUB', 1, 4.234, 5.345, 6.456),
        format_pdb_line(5, 'CL05', 'SUB', 1, 5.234, 6.345, 7.456),
        'END\n'
    ]
    return ''.join(lines)


@pytest.fixture
def sample_pdb_with_bromine():
    """Sample PDB with bromine atom that might be misidentified."""
    lines = [
        format_pdb_line(1, 'C001', 'SUB', 1, 1.000, 0.000, 0.000),
        format_pdb_line(2, 'BR02', 'SUB', 1, 2.500, 0.000, 0.000),
        format_pdb_line(3, 'H003', 'SUB', 1, 0.500, 1.000, 0.000),
        'END\n'
    ]
    return ''.join(lines)


@pytest.fixture
def sample_rtf_data():
    """Sample RTF data for CGenFF validation."""
    return {
        'atom_types': ['CG2R61', 'BRGR1', 'HGR61'],
        'charges': [-0.115, 0.100, 0.015],
        'total_charge': 0.0
    }


class TestPDBParsing:
    """Test basic PDB file parsing."""
    
    def test_parse_pdb_basic(self, tmp_path, sample_pdb_content):
        """Test parsing basic PDB file without validation."""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(sample_pdb_content)
        
        coords, elements = parse_pdb_file(str(pdb_file))
        
        assert len(coords) == 5
        assert len(elements) == 5
        assert elements == ['C', 'H', 'N', 'O', 'Cl']
        assert coords[0] == [1.234, 2.345, 3.456]
    
    def test_parse_pdb_with_cgenff_validation(self, tmp_path, sample_pdb_with_bromine, sample_rtf_data):
        """Test CGenFF validation catches element misidentification."""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(sample_pdb_with_bromine)
        
        # Without validation, BR might be parsed incorrectly
        coords_no_val, elements_no_val = parse_pdb_file(str(pdb_file))
        
        # With validation, should use CGenFF atom types
        coords_val, elements_val = parse_pdb_file(str(pdb_file), rtf_data=sample_rtf_data)
        
        assert len(coords_val) == 3
        assert len(elements_val) == 3
        # CGenFF should override: BRGR1 -> Br
        assert elements_val[1] == 'Br'
    
    def test_two_letter_elements(self, tmp_path):
        """Test parsing two-letter element symbols (Cl, Br, Al, Se)."""
        lines = [
            format_pdb_line(1, 'CL01', 'SUB', 1, 1.000, 2.000, 3.000),
            format_pdb_line(2, 'BR02', 'SUB', 1, 4.000, 5.000, 6.000),
            format_pdb_line(3, 'AL03', 'SUB', 1, 7.000, 8.000, 9.000),
            format_pdb_line(4, 'SE04', 'SUB', 1, 10.000, 11.000, 12.000),
            'END\n'
        ]
        pdb_content = ''.join(lines)
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content)
        
        coords, elements = parse_pdb_file(str(pdb_file))
        
        assert elements == ['Cl', 'Br', 'Al', 'Se']
    
    def test_single_letter_elements(self, tmp_path):
        """Test parsing single-letter elements (H, C, N, O, F, S, P, B, I)."""
        lines = [
            format_pdb_line(1, 'H001', 'SUB', 1, 1.000, 0.000, 0.000),
            format_pdb_line(2, 'C002', 'SUB', 1, 2.000, 0.000, 0.000),
            format_pdb_line(3, 'N003', 'SUB', 1, 3.000, 0.000, 0.000),
            format_pdb_line(4, 'O004', 'SUB', 1, 4.000, 0.000, 0.000),
            format_pdb_line(5, 'F005', 'SUB', 1, 5.000, 0.000, 0.000),
            format_pdb_line(6, 'S006', 'SUB', 1, 6.000, 0.000, 0.000),
            format_pdb_line(7, 'P007', 'SUB', 1, 7.000, 0.000, 0.000),
            format_pdb_line(8, 'I008', 'SUB', 1, 8.000, 0.000, 0.000),
            'END\n'
        ]
        pdb_content = ''.join(lines)
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content)
        
        coords, elements = parse_pdb_file(str(pdb_file))
        
        assert elements == ['H', 'C', 'N', 'O', 'F', 'S', 'P', 'I']
    
    def test_rcsb_pdb_format(self, tmp_path):
        """Test parsing standard RCSB PDB format with explicit element symbols."""
        # Real RCSB PDB format with element symbols in columns 76-77
        rcsb_content = """ATOM      1  N   MET A   1      43.982  -3.258   9.163  1.00 28.71           N  
ATOM      2  CA  MET A   1      43.434  -1.917   9.134  1.00 22.52           C  
ATOM      3  C   MET A   1      42.006  -1.966   9.640  1.00 22.85           C  
ATOM      4  O   MET A   1      41.334  -2.969   9.468  1.00 26.86           O  
ATOM      5  CB  MET A   1      43.582  -1.397   7.675  1.00 20.59           C  
ATOM      6  CG  MET A   1      42.903  -0.084   7.444  1.00 42.42           C  
ATOM      7  SD  MET A   1      44.006   1.250   7.952  1.00 39.82           S  
ATOM      8  CE  MET A   1      45.481   0.757   7.112  1.00 37.41           C  
END
"""
        pdb_file = tmp_path / "rcsb.pdb"
        pdb_file.write_text(rcsb_content)
        
        coords, elements = parse_pdb_file(str(pdb_file))
        
        assert len(coords) == 8
        assert elements == ['N', 'C', 'C', 'O', 'C', 'C', 'S', 'C']
        # Check first coordinate
        assert abs(coords[0][0] - 43.982) < 0.001
        assert abs(coords[0][1] - (-3.258)) < 0.001
        assert abs(coords[0][2] - 9.163) < 0.001
        # Check negative coordinate
        assert abs(coords[1][1] - (-1.917)) < 0.001


class TestSiteExtraction:
    """Test site number extraction from filenames."""
    
    def test_extract_site_number(self):
        """Test extracting site number from various filename patterns."""
        assert extract_site_number("site1_sub2_frag.pdb") == 1
        assert extract_site_number("site5_sub10_pres.rtf") == 5
        assert extract_site_number("/path/to/site3_sub1.pdb") == 3
        assert extract_site_number("SITE2_sub1.pdb") == 2  # Case insensitive
    
    def test_extract_site_number_no_match(self):
        """Test extracting site number when pattern not found."""
        assert extract_site_number("core.pdb") is None
        assert extract_site_number("protein.pdb") is None
        assert extract_site_number("random_file.txt") is None


class TestDuplicateDetection:
    """Test duplicate atom detection and removal."""
    
    def test_find_duplicate_atoms_exact_match(self):
        """Test finding exact duplicate coordinates."""
        coords1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        coords2 = [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0], [4.0, 5.0, 6.0]]
        
        duplicates = find_duplicate_atoms(coords1, coords2)
        
        assert duplicates == [0, 2]
    
    def test_find_duplicate_atoms_within_tolerance(self):
        """Test finding duplicates within tolerance."""
        coords1 = [[1.0, 2.0, 3.0]]
        coords2 = [[1.00005, 2.00005, 3.00005]]  # Within 1e-4 A
        
        duplicates = find_duplicate_atoms(coords1, coords2, tolerance=1e-4)
        
        assert len(duplicates) == 1
        assert 0 in duplicates
    
    def test_find_duplicate_atoms_no_duplicates(self):
        """Test when no duplicates exist."""
        coords1 = [[1.0, 2.0, 3.0]]
        coords2 = [[10.0, 20.0, 30.0]]
        
        duplicates = find_duplicate_atoms(coords1, coords2)
        
        assert duplicates == []
    
    def test_remove_duplicate_atoms(self):
        """Test removing duplicate atoms from coords and elements."""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        elements = ['C', 'H', 'N']
        core_coords = [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]
        
        filtered_coords, filtered_elements = remove_duplicate_atoms(
            coords, elements, core_coords
        )
        
        assert len(filtered_coords) == 1
        assert len(filtered_elements) == 1
        assert filtered_coords[0] == [4.0, 5.0, 6.0]
        assert filtered_elements[0] == 'H'
    
    def test_remove_duplicate_atoms_no_overlap(self):
        """Test removing duplicates when there are none."""
        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        elements = ['C', 'H']
        core_coords = [[10.0, 20.0, 30.0]]
        
        filtered_coords, filtered_elements = remove_duplicate_atoms(
            coords, elements, core_coords
        )
        
        assert filtered_coords == coords
        assert filtered_elements == elements


class TestSpatialFiltering:
    """Test spatial filtering and distance calculations."""
    
    def test_calculate_min_distance_simple(self):
        """Test minimum distance calculation."""
        coords1 = [[0.0, 0.0, 0.0]]
        coords2 = [[3.0, 4.0, 0.0]]
        
        min_dist = calculate_min_distance(coords1, coords2)
        
        assert abs(min_dist - 5.0) < 1e-6  # 3-4-5 triangle
    
    def test_calculate_min_distance_multiple_atoms(self):
        """Test minimum distance with multiple atoms."""
        coords1 = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        coords2 = [[2.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
        
        min_dist = calculate_min_distance(coords1, coords2)
        
        assert abs(min_dist - 2.0) < 1e-6  # Closest pair is (0,0,0) to (2,0,0)
    
    def test_calculate_min_distance_empty(self):
        """Test distance calculation with empty coordinates."""
        coords1 = []
        coords2 = [[1.0, 2.0, 3.0]]
        
        min_dist = calculate_min_distance(coords1, coords2)
        
        assert min_dist == float('inf')
    
    def test_find_nearby_pdbs(self, tmp_path):
        """Test finding PDB files within cutoff distance."""
        # Create target PDB
        target_content = format_pdb_line(1, 'C001', 'SUB', 1, 0.000, 0.000, 0.000) + 'END\n'
        target_pdb = tmp_path / "target.pdb"
        target_pdb.write_text(target_content)
        
        # Create nearby PDB (distance = 2.0 A)
        nearby_content = format_pdb_line(1, 'C001', 'SUB', 1, 2.000, 0.000, 0.000) + 'END\n'
        nearby_pdb = tmp_path / "nearby.pdb"
        nearby_pdb.write_text(nearby_content)
        
        # Create far PDB (distance = 20.0 A)
        far_content = format_pdb_line(1, 'C001', 'SUB', 1, 20.000, 0.000, 0.000) + 'END\n'
        far_pdb = tmp_path / "far.pdb"
        far_pdb.write_text(far_content)
        
        candidates = [str(nearby_pdb), str(far_pdb)]
        nearby = find_nearby_pdbs(str(target_pdb), candidates, cutoff=5.0)
        
        assert len(nearby) == 1
        assert str(nearby_pdb) in nearby
    
    def test_find_reference_subs_from_other_sites(self, tmp_path):
        """Test finding reference substituents from other sites."""
        # Create site1_sub1 (target)
        site1_sub1 = tmp_path / "site1_sub1_frag.pdb"
        site1_sub1.write_text(format_pdb_line(1, 'C001', 'SUB', 1, 0.000, 0.000, 0.000) + 'END\n')
        
        # Create site2_sub1 (nearby, should be included)
        site2_sub1 = tmp_path / "site2_sub1_frag.pdb"
        site2_sub1.write_text(format_pdb_line(1, 'C001', 'SUB', 1, 3.000, 0.000, 0.000) + 'END\n')
        
        # Create site3_sub1 (far, should be excluded)
        site3_sub1 = tmp_path / "site3_sub1_frag.pdb"
        site3_sub1.write_text(format_pdb_line(1, 'C001', 'SUB', 1, 20.000, 0.000, 0.000) + 'END\n')
        
        # Create site1_sub2 (same site, should be excluded)
        site1_sub2 = tmp_path / "site1_sub2_frag.pdb"
        site1_sub2.write_text(format_pdb_line(1, 'C001', 'SUB', 1, 1.000, 0.000, 0.000) + 'END\n')
        
        nearby_refs = find_reference_subs_from_other_sites(
            str(site1_sub1), str(tmp_path), cutoff=5.0
        )
        
        # Should find only site2_sub1 (different site, within cutoff)
        assert len(nearby_refs) == 1
        assert str(site2_sub1) in nearby_refs


class TestPDBDirectoryParsing:
    """Test parsing all PDB files in a directory."""
    
    def test_parse_pdb_dir(self, tmp_path):
        """Test parsing all PDB files in a directory."""
        # Create multiple PDB files
        pdb1 = tmp_path / "file1.pdb"
        content1 = ''.join([
            format_pdb_line(1, 'C001', 'SUB', 1, 1.000, 2.000, 3.000),
            format_pdb_line(2, 'H002', 'SUB', 1, 2.000, 3.000, 4.000),
            'END\n'
        ])
        pdb1.write_text(content1)
        
        pdb2 = tmp_path / "file2.pdb"
        content2 = format_pdb_line(1, 'N003', 'SUB', 1, 5.000, 6.000, 7.000) + 'END\n'
        pdb2.write_text(content2)
        
        results = parse_pdb_dir(str(tmp_path))
        
        assert 'file1' in results
        assert 'file2' in results
        assert results['file1']['num_atoms'] == 2
        assert results['file2']['num_atoms'] == 1
        assert results['file1']['elements'] == ['C', 'H']
        assert results['file2']['elements'] == ['N']
    
    def test_parse_pdb_dir_with_pattern(self, tmp_path):
        """Test parsing directory with specific pattern."""
        # Create PDB files
        (tmp_path / "site1_sub1.pdb").write_text(
            format_pdb_line(1, 'C001', 'SUB', 1, 1.000, 2.000, 3.000) + 'END\n'
        )
        (tmp_path /  "site1_sub2.pdb").write_text(
            format_pdb_line(1, 'H002', 'SUB', 1, 2.000, 3.000, 4.000) + 'END\n'
        )
        (tmp_path / "core.pdb").write_text(
            format_pdb_line(1, 'N003', 'SUB', 1, 5.000, 6.000, 7.000) + 'END\n'
        )
        
        # Parse only site*_sub* files
        results = parse_pdb_dir(str(tmp_path), pattern='site*_sub*.pdb')
        
        assert len(results) == 2
        assert 'site1_sub1' in results
        assert 'site1_sub2' in results
        assert 'core' not in results


class TestCombinePDBFiles:
    """Test combining multiple PDB files."""
    
    def test_combine_pdb_files(self, tmp_path):
        """Test combining multiple PDB files into single coordinate list."""
        pdb1 = tmp_path / "file1.pdb"
        content1 = ''.join([
            format_pdb_line(1, 'C001', 'SUB', 1, 1.000, 2.000, 3.000),
            format_pdb_line(2, 'H002', 'SUB', 1, 2.000, 3.000, 4.000),
            'END\n'
        ])
        pdb1.write_text(content1)
        
        pdb2 = tmp_path / "file2.pdb"
        content2 = format_pdb_line(1, 'N003', 'SUB', 1, 5.000, 6.000, 7.000) + 'END\n'
        pdb2.write_text(content2)
        
        coords, elements, atom_counts = combine_pdb_files([str(pdb1), str(pdb2)])
        
        assert len(coords) == 3  # 2 from pdb1 + 1 from pdb2
        assert len(elements) == 3
        assert elements == ['C', 'H', 'N']
        assert atom_counts == [2, 1]
    
    def test_combine_pdb_files_empty_list(self):
        """Test combining empty list of PDB files."""
        coords, elements, atom_counts = combine_pdb_files([])
        
        assert coords == []
        assert elements == []
        assert atom_counts == []


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_parse_pdb_nonstandard_spacing(self, tmp_path):
        """Test parsing PDB with non-standard spacing (uses regex fallback)."""
        # This has irregular spacing that's too short for fixed columns
        nonstandard_pdb = tmp_path / "nonstandard.pdb"
        nonstandard_pdb.write_text("""ATOM 1 C001 SUB 1 1.000 2.000 3.000
ATOM 2 H002 SUB 1 4.000 5.000 6.000
END
""")
        
        coords, elements = parse_pdb_file(str(nonstandard_pdb))
        
        # Should successfully parse using regex fallback
        assert len(coords) == 2
        assert elements == ['C', 'H']
        assert coords[0] == [1.000, 2.000, 3.000]
        assert coords[1] == [4.000, 5.000, 6.000]
    
    def test_parse_pdb_negative_coordinates(self, tmp_path):
        """Test parsing PDB with negative coordinates."""
        pdb_content = ''.join([
            format_pdb_line(1, 'C001', 'SUB', 1, -1.234, -2.345, -3.456),
            format_pdb_line(2, 'H002', 'SUB', 1, 1.000, -1.000, 0.500),
            'END\n'
        ])
        pdb_file = tmp_path / "negative.pdb"
        pdb_file.write_text(pdb_content)
        
        coords, elements = parse_pdb_file(str(pdb_file))
        
        assert len(coords) == 2
        assert coords[0] == [-1.234, -2.345, -3.456]
        assert coords[1] == [1.000, -1.000, 0.500]
    
    def test_parse_pdb_nonexistent_file(self):
        """Test parsing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            parse_pdb_file("/nonexistent/path/file.pdb")
    
    def test_parse_pdb_empty_file(self, tmp_path):
        """Test parsing empty PDB file."""
        empty_pdb = tmp_path / "empty.pdb"
        empty_pdb.write_text("")
        
        coords, elements = parse_pdb_file(str(empty_pdb))
        
        assert coords == []
        assert elements == []
    
    def test_parse_pdb_no_atom_lines(self, tmp_path):
        """Test parsing PDB with no ATOM lines."""
        no_atoms = tmp_path / "no_atoms.pdb"
        no_atoms.write_text("""HEADER    TEST PDB
REMARK    No atom lines
END
""")
        
        coords, elements = parse_pdb_file(str(no_atoms))
        
        assert coords == []
        assert elements == []
    
    def test_parse_pdb_malformed_lines_raises_error(self, tmp_path):
        """Test that completely unparseable PDB raises ValueError."""
        malformed_pdb = tmp_path / "malformed.pdb"
        malformed_pdb.write_text("""ATOM garbage data here
ATOM more garbage
ATOM even more bad data
END
""")
        
        # Should raise ValueError when no atoms can be parsed
        # Also expect warnings for each failed line
        with pytest.warns(UserWarning, match="Failed to parse PDB line"):
            with pytest.raises(ValueError, match="CRITICAL: Failed to parse any atoms"):
                parse_pdb_file(str(malformed_pdb))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
