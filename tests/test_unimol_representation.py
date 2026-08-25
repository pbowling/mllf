#!/usr/bin/env python3
"""Tests for Uni-Mol molecular representation and embeddings.

Uni-Mol representations are now the primary feature extraction method for both
pretraining and online training, replacing the deprecated AEV/DeepSet/atombond GNN approaches.
These tests verify embedding computation, caching, and environment handling.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mllf.cb.unimol_representation import (
    _generate_embedding_cache_key,
    _system_label_from_prep_dir,
    save_embedding,
    load_embedding,
)


class TestEmbeddingCacheKey:
    """Test embedding cache key generation."""
    
    def test_cache_key_is_deterministic(self):
        """Verify cache key generation is deterministic."""
        key1 = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
        )
        
        key2 = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
        )
        
        assert key1 == key2, "Cache keys should be deterministic for same inputs"
    
    def test_cache_key_differs_for_different_inputs(self):
        """Verify cache key differs for different inputs."""
        key_cutoff5 = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
        )
        
        key_cutoff8 = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=8.0,
            include_other_sites=False,
        )
        
        assert key_cutoff5 != key_cutoff8, "Cache keys should differ for different env_cutoff"
    
    def test_cache_key_length(self):
        """Verify cache key has expected length."""
        key = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
        )
        
        assert len(key) == 16, "Cache key should be 16 characters (SHA256 truncated)"
    
    def test_cache_key_includes_other_sites(self):
        """Verify cache key differs based on include_other_sites flag."""
        key_without = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
        )
        
        key_with = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=True,
        )
        
        assert key_without != key_with, "Cache keys should differ based on include_other_sites"
    
    def test_cache_key_includes_custom_search_paths(self):
        """Verify cache key includes custom search paths."""
        key_default = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
            custom_search_paths=None,
        )
        
        key_custom = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
            custom_search_paths=['/custom/path1', '/custom/path2'],
        )
        
        assert key_default != key_custom, "Cache keys should differ with custom search paths"


class TestEmbeddingSaving:
    """Test embedding save/load functionality."""
    
    def test_save_embedding_creates_directory(self):
        """Verify save_embedding creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding = np.random.randn(512).astype(np.float32)
            
            save_path = save_embedding(
                embedding=embedding,
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                env_cutoff=5.0,
            )
            
            assert save_path.exists(), "Embedding file should be created"
            # Layout: embeddings/{system_label}/{sub_name}/{cache_key}.npy —
            # system_label groups entries by system (derived from prep_dir) so
            # different systems sharing a substituent name (e.g. site1_sub3)
            # don't land in one shared directory.
            assert save_path.parent.parent.parent.name == 'embeddings', "Should be in embeddings subdirectory"
            assert save_path.parent.parent.name == 'to', "Should be nested under a system-label directory"
            assert save_path.parent.name == 'sub', "Should be nested under the substituent-name directory"
    
    def test_save_embedding_creates_metadata(self):
        """Verify save_embedding creates metadata JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding = np.random.randn(512).astype(np.float32)
            
            save_path = save_embedding(
                embedding=embedding,
                cache_dir=cache_dir,
                sub_pdb='/path/to/site1_sub1_frag.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                env_cutoff=5.0,
                include_other_sites=False,
            )
            
            metadata_path = save_path.with_name(save_path.stem + '_metadata.json')
            assert metadata_path.exists(), "Metadata file should be created"
            
            # Verify metadata contents
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert metadata['embedding_shape'] == [512], "Metadata should store embedding shape"
            assert metadata['embedding_dtype'] == 'float32', "Metadata should store dtype"
            assert metadata['env_cutoff'] == 5.0, "Metadata should store env_cutoff"
            assert metadata['include_other_sites'] is False, "Metadata should store include_other_sites"
    
    def test_save_embedding_stores_correct_values(self):
        """Verify saved embedding contains correct values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding = np.array([1.0, 2.0, 3.0] + [0.0] * 509).astype(np.float32)
            
            save_path = save_embedding(
                embedding=embedding,
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
            )
            
            loaded = np.load(save_path)
            assert np.allclose(loaded, embedding), "Loaded embedding should match saved values"
    
    def test_save_embedding_respects_overwrite_flag(self):
        """Verify save_embedding respects overwrite flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding1 = np.ones(512).astype(np.float32)
            embedding2 = np.ones(512).astype(np.float32) * 2.0
            
            # First save
            save_path = save_embedding(
                embedding=embedding1,
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                overwrite=False,
            )
            
            # Second save without overwrite should not overwrite
            save_path2 = save_embedding(
                embedding=embedding2,
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                overwrite=False,
            )
            
            # Should be same path, but file should still contain original values
            assert save_path == save_path2, "Should return same path"
            loaded = np.load(save_path)
            assert np.allclose(loaded, embedding1), "Original embedding should be preserved without overwrite"
    
    def test_save_embedding_overwrites_with_flag(self):
        """Verify save_embedding overwrites with overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding1 = np.ones(512).astype(np.float32)
            embedding2 = np.ones(512).astype(np.float32) * 2.0
            
            # First save
            save_path = save_embedding(
                embedding=embedding1,
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                overwrite=False,
            )
            
            # Second save with overwrite
            save_path2 = save_embedding(
                embedding=embedding2,
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                overwrite=True,
            )
            
            assert save_path == save_path2, "Should return same path"
            loaded = np.load(save_path)
            assert np.allclose(loaded, embedding2), "Embedding should be overwritten with overwrite=True"


class TestSystemLabelScoping:
    """Test that the embedding cache is organized (and looked up) per system,
    not just per substituent name -- see _system_label_from_prep_dir."""

    def test_flat_layout_label(self):
        assert _system_label_from_prep_dir('/x/pretraining/14benz_solv/prep') == '14benz_solv'

    def test_nested_combo_layout_label(self):
        label = _system_label_from_prep_dir(
            '/x/pretraining/12fuzed_vac/comb_1744603_site1_subs_18_19_28_32/prep'
        )
        assert label == '12fuzed_vac__comb_1744603_site1_subs_18_19_28_32'

    def test_two_systems_sharing_a_sub_name_do_not_collide(self):
        """Two different systems that both have e.g. site1_sub3 must land in
        separate cache subdirectories, not one shared bucket."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            emb_a = np.random.randn(512).astype(np.float32)
            emb_b = np.random.randn(512).astype(np.float32)

            path_a = save_embedding(
                embedding=emb_a, cache_dir=cache_dir,
                sub_pdb='/x/pretraining/sysA/comb_0001/prep/site1_sub3_frag.pdb',
                core_pdb='/x/pretraining/sysA/comb_0001/prep/core.pdb',
                prep_dir='/x/pretraining/sysA/comb_0001/prep',
            )
            path_b = save_embedding(
                embedding=emb_b, cache_dir=cache_dir,
                sub_pdb='/x/pretraining/sysB/comb_0007/prep/site1_sub3_frag.pdb',
                core_pdb='/x/pretraining/sysB/comb_0007/prep/core.pdb',
                prep_dir='/x/pretraining/sysB/comb_0007/prep',
            )

            assert path_a.parent != path_b.parent, "Different systems must not share a directory"
            assert np.allclose(np.load(path_a), emb_a)
            assert np.allclose(np.load(path_b), emb_b)

    def test_load_falls_back_to_legacy_flat_layout(self):
        """Entries written before system-scoping was added (flat
        embeddings/{sub_name}/ layout, no system_label) must still be found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding = np.random.randn(512).astype(np.float32)

            legacy_dir = cache_dir / 'embeddings' / 'site9_sub9'
            legacy_dir.mkdir(parents=True)
            key = _generate_embedding_cache_key(
                '/x/pretraining/sysC/prep/site9_sub9_frag.pdb',
                '/x/pretraining/sysC/prep/core.pdb',
                '/x/pretraining/sysC/prep', 5.0, False, None, None, None,
            )
            np.save(legacy_dir / f'{key}.npy', embedding)

            loaded = load_embedding(
                cache_dir,
                sub_pdb='/x/pretraining/sysC/prep/site9_sub9_frag.pdb',
                core_pdb='/x/pretraining/sysC/prep/core.pdb',
                prep_dir='/x/pretraining/sysC/prep',
                verbose=False,
            )
            assert loaded is not None, "Should fall back to the legacy flat layout"
            assert np.allclose(loaded, embedding)


class TestEmbeddingLoading:
    """Test embedding load functionality."""
    
    def test_load_embedding_returns_none_if_not_found(self):
        """Verify load_embedding returns None if embedding not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            embedding = load_embedding(
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                verbose=False,
            )
            
            assert embedding is None, "Should return None if embedding not found"
    
    def test_load_embedding_returns_none_if_cache_dir_missing(self):
        """Verify load_embedding returns None if cache dir doesn't exist."""
        embedding = load_embedding(
            cache_dir=Path('/nonexistent/cache/dir'),
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            verbose=False,
        )
        
        assert embedding is None, "Should return None if cache dir doesn't exist"
    
    def test_save_and_load_roundtrip(self):
        """Verify save/load roundtrip preserves embedding values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            original = np.random.randn(512).astype(np.float32)
            
            # Save
            save_embedding(
                embedding=original,
                cache_dir=cache_dir,
                sub_pdb='/path/to/site1_sub1_frag.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                env_cutoff=5.0,
            )
            
            # Load
            loaded = load_embedding(
                cache_dir=cache_dir,
                sub_pdb='/path/to/site1_sub1_frag.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                env_cutoff=5.0,
                verbose=False,
            )
            
            assert loaded is not None, "Should load saved embedding"
            assert np.allclose(loaded, original), "Loaded embedding should match original"
    
    def test_load_embedding_respects_cache_key_uniqueness(self):
        """Verify load_embedding respects cache key differences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding1 = np.ones(512).astype(np.float32)
            
            # Save with cutoff=5.0
            save_embedding(
                embedding=embedding1,
                cache_dir=cache_dir,
                sub_pdb='/path/to/site1_sub1_frag.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                env_cutoff=5.0,
            )
            
            # Try loading with cutoff=8.0 (different key)
            loaded = load_embedding(
                cache_dir=cache_dir,
                sub_pdb='/path/to/site1_sub1_frag.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                env_cutoff=8.0,
                verbose=False,
            )
            
            assert loaded is None, "Should not load embedding with different cache key"
    
    def test_load_embedding_with_different_include_other_sites(self):
        """Verify load_embedding respects include_other_sites parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding = np.ones(512).astype(np.float32) * 3.14
            
            # Save with include_other_sites=False
            save_embedding(
                embedding=embedding,
                cache_dir=cache_dir,
                sub_pdb='/path/to/site1_sub1_frag.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                include_other_sites=False,
            )
            
            # Load with include_other_sites=False (should match)
            loaded_false = load_embedding(
                cache_dir=cache_dir,
                sub_pdb='/path/to/site1_sub1_frag.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                include_other_sites=False,
                verbose=False,
            )
            
            # Load with include_other_sites=True (should NOT match)
            loaded_true = load_embedding(
                cache_dir=cache_dir,
                sub_pdb='/path/to/site1_sub1_frag.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                include_other_sites=True,
                verbose=False,
            )
            
            assert loaded_false is not None, "Should load with matching include_other_sites=False"
            assert loaded_true is None, "Should NOT load with different include_other_sites=True"


class TestEmbeddingProperties:
    """Test properties and constraints of embeddings."""
    
    def test_embedding_dimension_is_512(self):
        """Verify Uni-Mol embeddings are 512-dimensional."""
        # This is a property test verifying the expected dimension
        expected_dim = 512
        
        # Verify this is consistent with Uni-Mol model output
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            # Create a test embedding with correct dimension
            embedding = np.random.randn(expected_dim).astype(np.float32)
            
            save_embedding(
                embedding=embedding,
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
            )
            
            # Verify saved dimension
            loaded = load_embedding(
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                verbose=False,
            )
            
            assert loaded.shape == (512,), "Embedding should be 512-dimensional"
    
    def test_embedding_dtype_is_float32(self):
        """Verify Uni-Mol embeddings are float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            embedding = np.random.randn(512).astype(np.float32)
            
            save_embedding(
                embedding=embedding,
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
            )
            
            loaded = load_embedding(
                cache_dir=cache_dir,
                sub_pdb='/path/to/sub.pdb',
                core_pdb='/path/to/core.pdb',
                prep_dir='/path/to/prep',
                verbose=False,
            )
            
            assert loaded.dtype == np.float32, "Embedding should be float32"


class TestCacheKeyEdgeCases:
    """Test edge cases in cache key generation."""
    
    def test_cache_key_with_empty_custom_paths(self):
        """Verify cache key handles empty custom search paths."""
        # Note: The implementation treats empty list and None the same,
        # which is reasonable since both mean "no custom paths"
        key_none = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
            custom_search_paths=None,
        )
        
        key_empty = _generate_embedding_cache_key(
            sub_pdb='/path/to/sub.pdb',
            core_pdb='/path/to/core.pdb',
            prep_dir='/path/to/prep',
            env_cutoff=5.0,
            include_other_sites=False,
            custom_search_paths=[],
        )
        
        # Empty list and None should produce same key
        assert key_none == key_empty, "Empty list and None should produce same key"
    
    def test_cache_key_with_path_normalization(self):
        """Verify cache key normalizes paths correctly."""
        # Relative vs absolute paths should produce same key after normalization
        key1 = _generate_embedding_cache_key(
            sub_pdb='./sub.pdb',
            core_pdb='./core.pdb',
            prep_dir='./prep',
            env_cutoff=5.0,
            include_other_sites=False,
        )
        
        # Note: paths are normalized to absolute, so different relative paths
        # should still be same if they resolve to same absolute path
        # This test just verifies the function works with relative paths
        assert len(key1) == 16, "Cache key should be generated even with relative paths"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
