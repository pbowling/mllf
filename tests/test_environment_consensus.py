#!/usr/bin/env python3
"""Tests for environment consensus atom management.

Environment consensus provides stable, consistent environment representations
across pretraining and online training by identifying atoms that appear in all
substituents' environments at a given site.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Optional, Set
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mllf.cb.environment_consensus import (
    save_consensus_atoms,
    load_consensus_atoms,
)


class TestConsensusAtomsSerialization:
    """Test saving and loading consensus atoms."""
    
    def test_save_consensus_atoms_creates_file(self):
        """Verify save_consensus_atoms creates JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            consensus_dict = {
                'site1': {(0, 100, 'A', 'OW'), (1, 101, 'A', 'HW1'), (2, 101, 'A', 'HW2')},
                'site2': None,
            }
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            
            assert save_path.exists(), "Consensus file should be created"
            assert save_path.name == "environment_consensus.json", "Should use default filename"
    
    def test_save_consensus_atoms_custom_filename(self):
        """Verify save_consensus_atoms respects custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            consensus_dict = {'site1': {(0, 100, 'A', 'OW')}}
            
            save_path = save_consensus_atoms(
                consensus_dict, run_dir, 
                filename="custom_consensus.json"
            )
            
            assert save_path.name == "custom_consensus.json", "Should use custom filename"
    
    def test_save_consensus_atoms_stores_correct_format(self):
        """Verify consensus atoms are stored in correct JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            consensus_dict = {
                'site1': {(0, 100, 'A', 'OW'), (1, 101, 'A', 'HW1')},
            }
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            
            # Load and verify JSON structure
            with open(save_path) as f:
                data = json.load(f)
            
            assert 'site1' in data, "Should have site1 key"
            assert isinstance(data['site1'], list), "Consensus atoms should be stored as list"
            assert len(data['site1']) == 2, "Should have 2 atoms"
            # Atoms should be lists (tuples serialized to JSON)
            assert all(isinstance(atom, list) for atom in data['site1']), "Each atom should be a list"
    
    def test_save_consensus_atoms_handles_none_values(self):
        """Verify save_consensus_atoms handles None and empty consensus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            consensus_dict = {
                'site1': {(0, 100, 'A', 'OW')},
                'site2': None,
                'site3': set(),
            }
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            
            with open(save_path) as f:
                data = json.load(f)
            
            assert data['site1'] is not None, "site1 should have atoms"
            assert data['site2'] is None, "site2 should be None"
            # Empty set is falsy, so it's saved as None (same as missing consensus)
            assert data['site3'] is None, "Empty set should serialize to None (no consensus)"
    
    def test_load_consensus_atoms_returns_dict(self):
        """Verify load_consensus_atoms returns proper dict structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            original = {
                'site1': {(0, 100, 'A', 'OW'), (1, 101, 'A', 'HW1')},
                'site2': None,
            }
            
            save_consensus_atoms(original, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            assert isinstance(loaded, dict), "Should return dict"
            assert 'site1' in loaded, "Should have site1 key"
            assert 'site2' in loaded, "Should have site2 key"
    
    def test_load_consensus_atoms_restores_sets(self):
        """Verify load_consensus_atoms restores sets from lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            original = {
                'site1': {(0, 100, 'A', 'OW'), (1, 101, 'A', 'HW1'), (2, 101, 'A', 'HW2')},
            }
            
            save_consensus_atoms(original, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            assert isinstance(loaded['site1'], set), "Should restore as set"
            assert loaded['site1'] == original['site1'], "Should restore exact atoms"
    
    def test_load_consensus_atoms_preserves_none(self):
        """Verify load_consensus_atoms preserves None values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            original = {
                'site1': {(0, 100, 'A', 'OW')},
                'site2': None,
            }
            
            save_consensus_atoms(original, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            assert loaded['site2'] is None, "Should preserve None values"    
    def test_save_consensus_atoms_treats_empty_set_as_none(self):
        """Verify save_consensus_atoms treats empty sets as None (no consensus)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            original = {
                'site1': {(0, 100, 'A', 'OW')},
                'site2': set(),  # Empty set
            }
            
            save_consensus_atoms(original, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            # Empty set should be saved/loaded as None
            assert loaded['site2'] is None, "Empty set should be treated as None (no consensus)"    
    def test_save_load_roundtrip(self):
        """Verify save/load roundtrip preserves all consensus data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            original = {
                'site1': {(0, 100, 'A', 'OW'), (1, 101, 'A', 'HW1'), (2, 101, 'A', 'HW2')},
                'site2': {(0, 150, 'B', 'CL')},
                'site3': None,
            }
            
            save_consensus_atoms(original, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            assert loaded == original, "Roundtrip should preserve all data"
    
    def test_load_consensus_atoms_returns_empty_if_not_found(self):
        """Verify load_consensus_atoms returns empty dict if file not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            loaded = load_consensus_atoms(run_dir)
            
            assert loaded == {}, "Should return empty dict if file not found"
    
    def test_load_consensus_atoms_custom_filename(self):
        """Verify load_consensus_atoms respects custom filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            original = {'site1': {(0, 100, 'A', 'OW')}}
            
            save_consensus_atoms(
                original, run_dir,
                filename="my_consensus.json"
            )
            
            loaded = load_consensus_atoms(
                run_dir,
                filename="my_consensus.json"
            )
            
            assert loaded == original, "Should load with custom filename"


class TestConsensusAtomProperties:
    """Test properties of consensus atoms."""
    
    def test_consensus_atom_tuple_structure(self):
        """Verify consensus atoms have correct tuple structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # Atom ID format: (file_index, resnum, chain, atomname)
            atom = (42, 100, 'A', 'OW')
            consensus_dict = {'site1': {atom}}
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            
            with open(save_path) as f:
                data = json.load(f)
            
            # After JSON roundtrip
            saved_atom = data['site1'][0]
            assert len(saved_atom) == 4, "Atom should have 4 components"
            assert saved_atom[0] == 42, "First element should be file_index"
            assert saved_atom[1] == 100, "Second element should be resnum"
            assert saved_atom[2] == 'A', "Third element should be chain"
            assert saved_atom[3] == 'OW', "Fourth element should be atomname"
    
    def test_consensus_atoms_are_unique(self):
        """Verify sets eliminate duplicate atoms in consensus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # Sets automatically deduplicate
            atoms = {(0, 100, 'A', 'OW'), (0, 100, 'A', 'OW'), (1, 101, 'A', 'HW1')}
            consensus_dict = {'site1': atoms}
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            # After roundtrip, should only have 2 unique atoms
            assert len(loaded['site1']) == 2, "Duplicates should be eliminated by set"
    
    def test_consensus_atom_limit_respected(self):
        """Verify consensus respects atom limit (max 256)."""
        # This test documents the expected constraint that consensus atoms
        # should be capped at 256 (Uni-Mol model limit), though the save/load
        # functions don't enforce this — it's done in build_environment_consensus()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create 300 atoms (exceeds Uni-Mol limit of 256)
            atoms = {(i, 100 + i, 'A', f'C{i}') for i in range(300)}
            consensus_dict = {'site1': atoms}
            
            # Save/load should preserve all atoms
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            # The save/load functions don't truncate, but the builder should
            assert len(loaded['site1']) == 300, "Save/load should preserve all atoms"


class TestConsensusFileStructure:
    """Test file structure and JSON formatting."""
    
    def test_consensus_json_is_valid(self):
        """Verify saved consensus file is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            consensus_dict = {
                'site1': {(0, 100, 'A', 'OW')},
                'site2': None,
            }
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            
            # Should be parseable as valid JSON
            try:
                with open(save_path) as f:
                    json.load(f)
            except json.JSONDecodeError:
                pytest.fail("Saved consensus should be valid JSON")
    
    def test_consensus_json_is_readable(self):
        """Verify consensus JSON is human-readable (indented)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            consensus_dict = {'site1': {(0, 100, 'A', 'OW')}}
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            
            content = save_path.read_text()
            # Should have indentation for readability
            assert '\n' in content, "JSON should be formatted with newlines"
            assert '  ' in content, "JSON should be indented"
    
    def test_consensus_atoms_are_sorted(self):
        """Verify consensus atoms are sorted for consistency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # Atoms in random order
            atoms = {(2, 102, 'A', 'HW2'), (0, 100, 'A', 'OW'), (1, 101, 'A', 'HW1')}
            consensus_dict = {'site1': atoms}
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            # Roundtrip through JSON
            save_path2 = save_consensus_atoms(loaded, run_dir, filename="consensus2.json")
            
            # Two JSON files should be identical (sorted order)
            content1 = save_path.read_text()
            content2 = save_path2.read_text()
            
            # Should be deterministic
            assert content1 == content2, "Consensus JSON should be deterministic (sorted)"


class TestConsensusEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_consensus_dict(self):
        """Verify handling of empty consensus dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            consensus_dict = {}
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            assert loaded == {}, "Empty dict should roundtrip as empty"
    
    def test_many_sites_in_consensus(self):
        """Verify handling of many sites in consensus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            consensus_dict = {
                f'site{i}': {(0, 100 + i, 'A', f'A{i}')} if i % 2 == 0 else None
                for i in range(10)
            }
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            assert len(loaded) == 10, "Should handle many sites"
            for i in range(10):
                if i % 2 == 0:
                    assert loaded[f'site{i}'] is not None, f"site{i} should have atoms"
                else:
                    assert loaded[f'site{i}'] is None, f"site{i} should be None"
    
    def test_special_characters_in_atomname(self):
        """Verify handling of special characters in atom names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # Atom names can have special characters from CHARMM
            atoms = {
                (0, 100, 'A', "CG2'"),
                (1, 101, 'A', 'HG1"'),
                (2, 102, 'A', 'O1+'),
            }
            consensus_dict = {'site1': atoms}
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            assert loaded['site1'] == atoms, "Should handle special characters in atom names"
    
    def test_large_residue_numbers(self):
        """Verify handling of large residue numbers (CHARMM wrapping)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # CHARMM PDB can have wrapped residue numbers (100000+)
            atoms = {
                (0, 99999, 'A', 'OW'),
                (1, 100000, 'B', 'HW1'),
            }
            consensus_dict = {'site1': atoms}
            
            save_path = save_consensus_atoms(consensus_dict, run_dir)
            loaded = load_consensus_atoms(run_dir)
            
            assert loaded['site1'] == atoms, "Should handle large residue numbers"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
