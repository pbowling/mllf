"""Tests for DeepSet pretraining components."""
import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path

from mllf.cb.deepset_autoencoder import (
    DeepSetEncoder,
    DeepSetDecoder,
    DeepSetAutoencoder,
    PretrainedDeepSet,
    create_autoencoder,
    load_pretrained_deepset
)
from mllf.cb.train_deepset_autoencoder import AtomFeatureDataset
from mllf.cb.deepset_pretraining_dataset import (
    detect_core_pdb,
    detect_protein_pdb,
    extract_charges_from_rtf,
    load_system_metadata,
)
from mllf.cb.aev_processor import (
    detect_minimized_pdb,
    extract_environment_atoms_from_minimized,
)
# Both protein and solvent extraction use the same underlying function;
# keep descriptive local names so test class/method bodies read clearly.
extract_protein_atoms_from_minimized = extract_environment_atoms_from_minimized
extract_solvent_atoms_from_minimized = extract_environment_atoms_from_minimized


class TestDeepSetEncoder:
    """Test DeepSet encoder architecture."""
    
    def test_encoder_initialization(self):
        """Test encoder initializes with correct dimensions."""
        encoder = DeepSetEncoder(input_dim=2289, hidden_dim=256, embedding_dim=64)
        
        assert encoder.input_dim == 2289
        assert encoder.hidden_dim == 256
        assert encoder.embedding_dim == 64
    
    def test_encoder_forward_pass(self):
        """Test encoder forward pass produces correct output shape."""
        encoder = DeepSetEncoder(input_dim=2289, hidden_dim=256, embedding_dim=64)
        
        # Create sample input [num_atoms, 2289]
        num_atoms = 25
        atom_features = torch.randn(num_atoms, 2289)
        
        with torch.no_grad():
            embeddings = encoder(atom_features)
        
        assert embeddings.shape == (num_atoms, 64)
        assert not torch.isnan(embeddings).any()
    
    def test_encoder_single_atom(self):
        """Test encoder handles single atom."""
        encoder = DeepSetEncoder(input_dim=2289, hidden_dim=256, embedding_dim=64)
        
        atom_features = torch.randn(1, 2289)
        
        with torch.no_grad():
            embeddings = encoder(atom_features)
        
        assert embeddings.shape == (1, 64)
    
    def test_encoder_variable_sizes(self):
        """Test encoder handles variable numbers of atoms."""
        encoder = DeepSetEncoder()
        
        for num_atoms in [1, 10, 50, 100]:
            atom_features = torch.randn(num_atoms, 2289)
            
            with torch.no_grad():
                embeddings = encoder(atom_features)
            
            assert embeddings.shape == (num_atoms, 64)


class TestDeepSetDecoder:
    """Test DeepSet decoder architecture."""
    
    def test_decoder_initialization(self):
        """Test decoder initializes with correct dimensions."""
        decoder = DeepSetDecoder(embedding_dim=64, hidden_dim=256, output_dim=2289)
        
        assert decoder.embedding_dim == 64
        assert decoder.hidden_dim == 256
        assert decoder.output_dim == 2289
    
    def test_decoder_forward_pass(self):
        """Test decoder forward pass produces correct output shape."""
        decoder = DeepSetDecoder(embedding_dim=64, hidden_dim=256, output_dim=2289)
        
        num_atoms = 25
        embeddings = torch.randn(num_atoms, 64)
        
        with torch.no_grad():
            reconstructed = decoder(embeddings)
        
        assert reconstructed.shape == (num_atoms, 2289)
        assert not torch.isnan(reconstructed).any()


class TestDeepSetAutoencoder:
    """Test complete autoencoder architecture."""
    
    def test_autoencoder_initialization(self):
        """Test autoencoder initializes correctly."""
        autoencoder = DeepSetAutoencoder(input_dim=2289, hidden_dim=256, embedding_dim=64)
        
        assert autoencoder.input_dim == 2289
        assert autoencoder.hidden_dim == 256
        assert autoencoder.embedding_dim == 64
        assert hasattr(autoencoder, 'encoder')
        assert hasattr(autoencoder, 'decoder')
    
    def test_autoencoder_forward_pass(self):
        """Test autoencoder forward pass returns both embedding and reconstruction."""
        autoencoder = DeepSetAutoencoder()
        
        num_atoms = 25
        atom_features = torch.randn(num_atoms, 2289)
        
        with torch.no_grad():
            output = autoencoder(atom_features)
        
        assert 'embedding' in output
        assert 'reconstruction' in output
        assert output['embedding'].shape == (num_atoms, 64)
        assert output['reconstruction'].shape == (num_atoms, 2289)
    
    def test_autoencoder_encode_only(self):
        """Test encoding without reconstruction."""
        autoencoder = DeepSetAutoencoder()
        
        num_atoms = 25
        atom_features = torch.randn(num_atoms, 2289)
        
        with torch.no_grad():
            embeddings = autoencoder.encode(atom_features)
        
        assert embeddings.shape == (num_atoms, 64)
    
    def test_autoencoder_reconstruction_shape_matches(self):
        """Test reconstruction has same shape as input."""
        autoencoder = DeepSetAutoencoder()
        
        for num_atoms in [1, 10, 50]:
            atom_features = torch.randn(num_atoms, 2289)
            
            with torch.no_grad():
                output = autoencoder(atom_features)
            
            assert output['reconstruction'].shape == atom_features.shape
    
    def test_save_encoder(self, tmp_path):
        """Test saving encoder to file."""
        autoencoder = DeepSetAutoencoder(input_dim=2289, hidden_dim=256, embedding_dim=64)
        encoder_path = tmp_path / "encoder.pt"
        
        autoencoder.save_encoder(str(encoder_path))
        
        assert encoder_path.exists()
        
        # Load and verify contents
        checkpoint = torch.load(encoder_path)
        assert 'state_dict' in checkpoint
        assert 'input_dim' in checkpoint
        assert 'hidden_dim' in checkpoint
        assert 'embedding_dim' in checkpoint
        assert checkpoint['input_dim'] == 2289
        assert checkpoint['embedding_dim'] == 64
    
    def test_factory_function(self):
        """Test create_autoencoder factory function."""
        autoencoder = create_autoencoder(input_dim=2289, hidden_dim=256, embedding_dim=64)
        
        assert isinstance(autoencoder, DeepSetAutoencoder)
        assert autoencoder.input_dim == 2289
        assert autoencoder.embedding_dim == 64


class TestPretrainedDeepSet:
    """Test pretrained DeepSet with max-pooling."""
    
    def test_load_pretrained_encoder(self, tmp_path):
        """Test loading pretrained encoder."""
        # Create and save an encoder
        autoencoder = DeepSetAutoencoder(input_dim=2289, hidden_dim=256, embedding_dim=64)
        encoder_path = tmp_path / "encoder.pt"
        autoencoder.save_encoder(str(encoder_path))
        
        # Load as PretrainedDeepSet
        pretrained = PretrainedDeepSet(str(encoder_path), freeze_weights=True)
        
        assert pretrained.input_dim == 2289
        assert pretrained.embedding_dim == 64
        assert pretrained.frozen is True
    
    def test_pretrained_forward_with_pooling(self, tmp_path):
        """Test pretrained model performs max-pooling."""
        # Create and save encoder
        autoencoder = DeepSetAutoencoder()
        encoder_path = tmp_path / "encoder.pt"
        autoencoder.save_encoder(str(encoder_path))
        
        # Load pretrained
        pretrained = PretrainedDeepSet(str(encoder_path))
        
        num_atoms = 25
        atom_features = torch.randn(num_atoms, 2289)
        
        with torch.no_grad():
            pooled_embedding = pretrained(atom_features)
        
        # Output should be pooled to single vector
        assert pooled_embedding.shape == (64,)
        assert not torch.isnan(pooled_embedding).any()
    
    def test_pretrained_frozen_weights(self, tmp_path):
        """Test frozen weights don't require gradients."""
        autoencoder = DeepSetAutoencoder()
        encoder_path = tmp_path / "encoder.pt"
        autoencoder.save_encoder(str(encoder_path))
        
        pretrained = PretrainedDeepSet(str(encoder_path), freeze_weights=True)
        
        # Check that parameters don't require gradients
        for param in pretrained.encoder.parameters():
            assert param.requires_grad is False
    
    def test_pretrained_unfrozen_weights(self, tmp_path):
        """Test unfrozen weights allow gradients."""
        autoencoder = DeepSetAutoencoder()
        encoder_path = tmp_path / "encoder.pt"
        autoencoder.save_encoder(str(encoder_path))
        
        pretrained = PretrainedDeepSet(str(encoder_path), freeze_weights=False)
        
        # Check that parameters require gradients
        for param in pretrained.encoder.parameters():
            assert param.requires_grad is True
    
    def test_freeze_unfreeze_methods(self, tmp_path):
        """Test freeze() and unfreeze() methods."""
        autoencoder = DeepSetAutoencoder()
        encoder_path = tmp_path / "encoder.pt"
        autoencoder.save_encoder(str(encoder_path))
        
        pretrained = PretrainedDeepSet(str(encoder_path), freeze_weights=False)
        assert pretrained.frozen is False
        
        # Freeze
        pretrained.freeze()
        assert pretrained.frozen is True
        for param in pretrained.encoder.parameters():
            assert param.requires_grad is False
        
        # Unfreeze
        pretrained.unfreeze()
        assert pretrained.frozen is False
        for param in pretrained.encoder.parameters():
            assert param.requires_grad is True
    
    def test_factory_function_load(self, tmp_path):
        """Test load_pretrained_deepset factory function."""
        autoencoder = DeepSetAutoencoder()
        encoder_path = tmp_path / "encoder.pt"
        autoencoder.save_encoder(str(encoder_path))
        
        pretrained = load_pretrained_deepset(str(encoder_path), freeze_weights=True)
        
        assert isinstance(pretrained, PretrainedDeepSet)
        assert pretrained.frozen is True


class TestAutoencoderGradients:
    """Test gradient flow through autoencoder."""
    
    def test_encoder_gradients_flow(self):
        """Test gradients flow through encoder."""
        encoder = DeepSetEncoder()
        
        num_atoms = 10
        atom_features = torch.randn(num_atoms, 2289, requires_grad=True)
        
        embeddings = encoder(atom_features)
        loss = embeddings.sum()
        loss.backward()
        
        assert atom_features.grad is not None
        assert not torch.isnan(atom_features.grad).any()
    
    def test_autoencoder_gradients_flow(self):
        """Test gradients flow through full autoencoder."""
        autoencoder = DeepSetAutoencoder()
        
        num_atoms = 10
        atom_features = torch.randn(num_atoms, 2289, requires_grad=True)
        
        output = autoencoder(atom_features)
        loss = torch.nn.functional.mse_loss(output['reconstruction'], atom_features)
        loss.backward()
        
        assert atom_features.grad is not None
        assert not torch.isnan(atom_features.grad).any()
    
    def test_model_parameters_trainable(self):
        """Test autoencoder parameters are trainable."""
        autoencoder = DeepSetAutoencoder()
        
        params = list(autoencoder.parameters())
        assert len(params) > 0
        
        for param in params:
            assert param.requires_grad


class TestAtomFeatureDataset:
    """Test PyTorch dataset for atom features."""
    
    def test_dataset_loading(self, tmp_path):
        """Test loading dataset from .pt file."""
        # Create sample data
        num_atoms = 100
        features = torch.randn(num_atoms, 2289)
        
        data_path = tmp_path / "test_data.pt"
        torch.save({
            'features': features,
            'system_name': 'test_system',
            'num_atoms': num_atoms,
            'feature_dim': 2289,
        }, data_path)
        
        # Load dataset
        dataset = AtomFeatureDataset(str(data_path))
        
        assert len(dataset) == num_atoms
        assert dataset.feature_dim == 2289
        assert dataset.system_name == 'test_system'
    
    def test_dataset_getitem(self, tmp_path):
        """Test dataset __getitem__ returns correct features."""
        num_atoms = 50
        features = torch.randn(num_atoms, 2289)
        
        data_path = tmp_path / "test_data.pt"
        torch.save({
            'features': features,
            'system_name': 'test',
            'feature_dim': 2289,
        }, data_path)
        
        dataset = AtomFeatureDataset(str(data_path))
        
        # Test random access
        idx = 10
        feature = dataset[idx]
        
        assert feature.shape == (2289,)
        assert torch.allclose(feature, features[idx])


class TestPretrainingDatasetHelpers:
    """Test helper functions for pretraining dataset generation."""
    
    def test_detect_core_pdb_exists(self, tmp_path):
        """Test detecting core PDB when it exists."""
        core_pdb = tmp_path / "core.pdb"
        core_pdb.write_text("ATOM      1  C001 CORE    1       1.0   2.0   3.0  1.00  0.00           C\nEND\n")
        
        detected = detect_core_pdb(tmp_path)
        
        assert detected == core_pdb
    
    def test_detect_core_pdb_not_exists(self, tmp_path):
        """Test detecting core PDB when it doesn't exist."""
        detected = detect_core_pdb(tmp_path)
        
        assert detected is None
    
    def test_detect_protein_pdb_single_candidate(self, tmp_path):
        """Test detecting protein PDB with single candidate."""
        # Create protein PDB
        protein_pdb = tmp_path / "protein.pdb"
        protein_pdb.write_text("ATOM      1  CA  ALA     1       1.0   2.0   3.0  1.00  0.00           C\nEND\n")
        
        # Create core PDB (should be excluded)
        core_pdb = tmp_path / "core.pdb"
        core_pdb.write_text("ATOM      1  C001 CORE    1       1.0   2.0   3.0  1.00  0.00           C\nEND\n")
        
        detected = detect_protein_pdb(tmp_path)
        
        assert detected == protein_pdb
    
    def test_detect_protein_pdb_excludes_substituents(self, tmp_path):
        """Test protein detection excludes substituent PDB files."""
        # Create protein PDB
        protein_pdb = tmp_path / "protein.pdb"
        protein_pdb.write_text("ATOM      1  CA  ALA     1       1.0   2.0   3.0  1.00  0.00           C\nEND\n")
        
        # Create substituent PDB (should be excluded)
        sub_pdb = tmp_path / "site1_sub1_frag.pdb"
        sub_pdb.write_text("ATOM      1  C001 SUB     1       1.0   2.0   3.0  1.00  0.00           C\nEND\n")
        
        detected = detect_protein_pdb(tmp_path)
        
        assert detected == protein_pdb
    
    def test_detect_protein_pdb_not_exists(self, tmp_path):
        """Test protein detection returns None when no protein found."""
        # Create only core PDB
        core_pdb = tmp_path / "core.pdb"
        core_pdb.write_text("ATOM      1  C001 CORE    1       1.0   2.0   3.0  1.00  0.00           C\nEND\n")
        
        detected = detect_protein_pdb(tmp_path)
        
        assert detected is None
    
    def test_extract_charges_from_rtf(self, tmp_path):
        """Test extracting charges from RTF file."""
        # Create RTF file
        rtf_content = """* Test RTF
*
RESI TEST     0.00
ATOM C1   CG2R61 -0.115
ATOM H1   HGR61   0.115
END
"""
        rtf_path = tmp_path / "test_pres.rtf"
        rtf_path.write_text(rtf_content)
        
        charges = extract_charges_from_rtf(rtf_path, "test.pdb")
        
        assert charges is not None
        assert len(charges) == 2
        assert torch.allclose(charges, torch.tensor([-0.115, 0.115]))


class TestEdgeCasesPretraining:
    """Test edge cases in pretraining components."""
    
    def test_encoder_empty_input_handles_gracefully(self):
        """Test encoder with empty input."""
        encoder = DeepSetEncoder()
        empty_input = torch.randn(0, 2289)
        
        # PyTorch Linear layers handle empty input gracefully
        output = encoder(empty_input)
        assert output.shape == (0, 64)
    
    def test_autoencoder_mismatched_dimensions(self):
        """Test autoencoder with mismatched input dimension."""
        autoencoder = DeepSetAutoencoder(input_dim=2289)
        
        # Wrong input dimension
        wrong_input = torch.randn(10, 1000)  # Should be 2289
        
        with pytest.raises(RuntimeError):
            autoencoder(wrong_input)
    
    def test_pretrained_deterministic_pooling(self, tmp_path):
        """Test max-pooling produces deterministic results."""
        autoencoder = DeepSetAutoencoder()
        encoder_path = tmp_path / "encoder.pt"
        autoencoder.save_encoder(str(encoder_path))
        
        pretrained = PretrainedDeepSet(str(encoder_path))
        pretrained.eval()
        
        atom_features = torch.randn(25, 2289)
        
        # Multiple forward passes should give same result
        with torch.no_grad():
            out1 = pretrained(atom_features)
            out2 = pretrained(atom_features)
        
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Fixture prep directories checked in under examples/cb/ – available to all
# contributors after cloning the repository.
# ---------------------------------------------------------------------------
_EXAMPLES_CB    = Path(__file__).parent.parent / 'examples' / 'cb'
_ABL_PREP       = _EXAMPLES_CB / 'protein_abl'
_BENZ_SOLV_PREP = _EXAMPLES_CB / 'solvent_14benz'


class TestDetectMinimizedPdb:
    """Unit tests for detect_minimized_pdb."""

    def test_returns_path_when_file_exists(self, tmp_path):
        """Returns Path when minimized.pdb is present."""
        (tmp_path / 'minimized.pdb').write_text('ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n')
        result = detect_minimized_pdb(tmp_path)
        assert result == tmp_path / 'minimized.pdb'

    def test_returns_none_when_absent(self, tmp_path):
        """Returns None when minimized.pdb is not present."""
        assert detect_minimized_pdb(tmp_path) is None

    def test_detects_real_minimized_pdb(self):
        """Finds the actual minimized.pdb in the ABL protein prep directory."""
        result = detect_minimized_pdb(_ABL_PREP)
        assert result is not None
        assert result.name == 'minimized.pdb'
        assert result.exists()


class TestExtractProteinAtomsFromMinimized:
    """Unit tests for extract_protein_atoms_from_minimized."""

    def _write_pdb(self, path: Path, atoms):
        """Helper: write a minimal PDB file from a list of (x, y, z, elem) tuples."""
        lines = []
        for i, (x, y, z, elem) in enumerate(atoms, start=1):
            lines.append(
                f'ATOM{i:>7}  CA  ALA A{i:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2}\n'
            )
        lines.append('END\n')
        path.write_text(''.join(lines))

    # ------------------------------------------------------------------
    # Synthetic data tests
    # ------------------------------------------------------------------

    def test_returns_none_for_empty_minimized(self, tmp_path):
        """Returns None when minimized.pdb has no ATOM lines."""
        (tmp_path / 'minimized.pdb').write_text('REMARK empty\nEND\n')
        self._write_pdb(tmp_path / 'core.pdb',   [(0.0, 0.0, 0.0, 'C')])
        self._write_pdb(tmp_path / 'sub.pdb',    [(1.0, 0.0, 0.0, 'C')])
        result = extract_protein_atoms_from_minimized(
            minimized_pdb=tmp_path / 'minimized.pdb',
            sub_pdb=tmp_path / 'sub.pdb',
            core_pdb=tmp_path / 'core.pdb',
        )
        assert result is None

    def test_returns_none_when_no_atoms_within_cutoff(self, tmp_path):
        """Returns None when all protein atoms are beyond the AEV cutoff."""
        # protein atom at 50 Å from the substituent
        self._write_pdb(tmp_path / 'minimized.pdb', [(0.0, 0.0, 0.0, 'C'), (50.0, 0.0, 0.0, 'N')])
        self._write_pdb(tmp_path / 'core.pdb',      [(0.0, 0.0, 0.0, 'C')])
        self._write_pdb(tmp_path / 'sub.pdb',       [(1.0, 0.0, 0.0, 'C')])
        result = extract_protein_atoms_from_minimized(
            minimized_pdb=tmp_path / 'minimized.pdb',
            sub_pdb=tmp_path / 'sub.pdb',
            core_pdb=tmp_path / 'core.pdb',
            aev_cutoff=5.1,
        )
        assert result is None

    def test_ligand_atoms_excluded(self, tmp_path):
        """Core and sub atoms from minimized.pdb are not returned."""
        # minimized has: sub atom (1,0,0), core atom (0,0,0), protein atom (3,0,0)
        self._write_pdb(tmp_path / 'minimized.pdb', [(0.0, 0.0, 0.0, 'C'),
                                                      (1.0, 0.0, 0.0, 'C'),
                                                      (3.0, 0.0, 0.0, 'N')])
        self._write_pdb(tmp_path / 'core.pdb', [(0.0, 0.0, 0.0, 'C')])
        self._write_pdb(tmp_path / 'sub.pdb',  [(1.0, 0.0, 0.0, 'C')])
        coords, elements = extract_protein_atoms_from_minimized(
            minimized_pdb=tmp_path / 'minimized.pdb',
            sub_pdb=tmp_path / 'sub.pdb',
            core_pdb=tmp_path / 'core.pdb',
            aev_cutoff=5.1,
            duplicate_tolerance=0.5,
        )
        # Only the protein atom (3,0,0) should survive
        assert len(coords) == 1
        assert abs(coords[0][0] - 3.0) < 1e-3
        assert elements[0] == 'N'

    def test_tolerance_handles_minimization_shift(self, tmp_path):
        """Ligand atoms that moved ≤0.5 Å during minimization are still excluded."""
        shift = 0.25  # Å – well within the 0.5 Å tolerance
        self._write_pdb(tmp_path / 'minimized.pdb',
                        [(shift, 0.0, 0.0, 'C'),       # core atom, slightly shifted
                         (1.0 + shift, 0.0, 0.0, 'C'), # sub atom, slightly shifted
                         (3.0, 0.0, 0.0, 'N')])         # protein atom, unmoved
        self._write_pdb(tmp_path / 'core.pdb', [(0.0, 0.0, 0.0, 'C')])
        self._write_pdb(tmp_path / 'sub.pdb',  [(1.0, 0.0, 0.0, 'C')])
        coords, elements = extract_protein_atoms_from_minimized(
            minimized_pdb=tmp_path / 'minimized.pdb',
            sub_pdb=tmp_path / 'sub.pdb',
            core_pdb=tmp_path / 'core.pdb',
            aev_cutoff=5.1,
            duplicate_tolerance=0.5,
        )
        assert len(coords) == 1
        assert elements[0] == 'N'

    def test_other_site_subs_excluded_when_prep_dir_provided(self, tmp_path):
        """Other-site substituent atoms in minimized.pdb are excluded when prep_dir is given."""
        # Place core + target sub + other-site sub in minimized, plus a protein atom
        self._write_pdb(tmp_path / 'minimized.pdb',
                        [(0.0, 0.0, 0.0, 'C'),   # core
                         (1.0, 0.0, 0.0, 'C'),   # target sub
                         (2.0, 0.0, 0.0, 'C'),   # other-site sub (site2_sub1)
                         (4.0, 0.0, 0.0, 'N')])  # protein
        self._write_pdb(tmp_path / 'core.pdb',             [(0.0, 0.0, 0.0, 'C')])
        self._write_pdb(tmp_path / 'site1_sub1_frag.pdb',  [(1.0, 0.0, 0.0, 'C')])  # target
        self._write_pdb(tmp_path / 'site2_sub1_frag.pdb',  [(2.0, 0.0, 0.0, 'C')])  # other site

        # Without prep_dir: other-site sub atom (2,0,0) leaks into protein context
        coords_no_dir, _ = extract_protein_atoms_from_minimized(
            minimized_pdb=tmp_path / 'minimized.pdb',
            sub_pdb=tmp_path / 'site1_sub1_frag.pdb',
            core_pdb=tmp_path / 'core.pdb',
            aev_cutoff=5.1,
            prep_dir=None,
        )
        assert len(coords_no_dir) == 2  # other-site sub atom + protein atom

        # With prep_dir: other-site sub atom is also excluded
        coords_with_dir, elems_with_dir = extract_protein_atoms_from_minimized(
            minimized_pdb=tmp_path / 'minimized.pdb',
            sub_pdb=tmp_path / 'site1_sub1_frag.pdb',
            core_pdb=tmp_path / 'core.pdb',
            aev_cutoff=5.1,
            prep_dir=tmp_path,
        )
        assert len(coords_with_dir) == 1   # only the protein atom survives
        assert elems_with_dir[0] == 'N'

    def test_all_returned_atoms_are_within_cutoff(self, tmp_path):
        """Verify every returned coordinate is within aev_cutoff of the substituent."""
        aev_cutoff = 5.1
        # Place protein atoms at various distances from sub (at origin)
        atoms = [
            (0.0, 0.0, 0.0, 'C'),   # sub itself – excluded by duplicate check
            (2.0, 0.0, 0.0, 'N'),   # 2 Å – within cutoff
            (4.0, 0.0, 0.0, 'O'),   # 4 Å – within cutoff
            (6.0, 0.0, 0.0, 'N'),   # 6 Å – beyond cutoff
            (10.0, 0.0, 0.0, 'N'),  # 10 Å – beyond cutoff
        ]
        self._write_pdb(tmp_path / 'minimized.pdb', atoms)
        self._write_pdb(tmp_path / 'core.pdb', [(99.0, 0.0, 0.0, 'C')])  # far away
        self._write_pdb(tmp_path / 'sub.pdb',  [(0.0, 0.0, 0.0, 'C')])
        coords, _ = extract_protein_atoms_from_minimized(
            minimized_pdb=tmp_path / 'minimized.pdb',
            sub_pdb=tmp_path / 'sub.pdb',
            core_pdb=tmp_path / 'core.pdb',
            aev_cutoff=aev_cutoff,
        )
        sub_pos = np.array([0.0, 0.0, 0.0])
        for coord in coords:
            dist = np.linalg.norm(np.array(coord) - sub_pos)
            assert dist <= aev_cutoff, f'Atom at {coord} is {dist:.2f} Å from sub (> {aev_cutoff} Å cutoff)'

    # ------------------------------------------------------------------
    # Real-data tests against the ABL protein wildtype group1 run2 system
    # ------------------------------------------------------------------

    def test_real_data_returns_protein_atoms(self):
        """Returns a non-empty result for a real protein prep directory."""
        minimized_pdb = _ABL_PREP / 'minimized.pdb'
        core_pdb      = _ABL_PREP / 'core.pdb'
        sub_pdb       = _ABL_PREP / 'site1_sub1_frag.pdb'

        result = extract_protein_atoms_from_minimized(
            minimized_pdb=minimized_pdb,
            sub_pdb=sub_pdb,
            core_pdb=core_pdb,
            aev_cutoff=5.1,
            prep_dir=_ABL_PREP,
        )
        assert result is not None, 'Expected protein atoms within 5.1 Å of site1_sub1'
        coords, elements = result
        assert len(coords) > 0
        assert len(coords) == len(elements)
        print(f'\n  site1_sub1: {len(coords)} protein atoms within 5.1 Å')

    def test_real_data_atoms_within_cutoff(self):
        """All returned atoms are within the AEV cutoff of the substituent."""
        from mllf.file_handling.read_pdb import parse_pdb_file as _parse
        aev_cutoff = 5.1
        sub_coords, _ = _parse(str(_ABL_PREP / 'site1_sub1_frag.pdb'))
        sub_arr = np.array(sub_coords)

        coords, _ = extract_protein_atoms_from_minimized(
            minimized_pdb=_ABL_PREP / 'minimized.pdb',
            sub_pdb=_ABL_PREP / 'site1_sub1_frag.pdb',
            core_pdb=_ABL_PREP / 'core.pdb',
            aev_cutoff=aev_cutoff,
            prep_dir=_ABL_PREP,
        )
        prot_arr = np.array(coords)
        # Per-atom minimum distance to any sub atom
        min_dists = np.sqrt(((prot_arr[:, None, :] - sub_arr[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        assert (min_dists <= aev_cutoff + 1e-6).all(), \
            f'Max distance to sub: {min_dists.max():.3f} Å, expected ≤ {aev_cutoff} Å'

    def test_real_data_no_ligand_atoms_in_result(self):
        """Returned coordinates do not contain core or substituent atoms."""
        from mllf.file_handling.read_pdb import parse_pdb_file as _parse
        core_coords, _ = _parse(str(_ABL_PREP / 'core.pdb'))
        sub_coords,  _ = _parse(str(_ABL_PREP / 'site1_sub1_frag.pdb'))
        ligand_coords = np.array(core_coords + sub_coords)

        coords, _ = extract_protein_atoms_from_minimized(
            minimized_pdb=_ABL_PREP / 'minimized.pdb',
            sub_pdb=_ABL_PREP / 'site1_sub1_frag.pdb',
            core_pdb=_ABL_PREP / 'core.pdb',
            aev_cutoff=5.1,
            prep_dir=_ABL_PREP,
        )
        prot_arr = np.array(coords)
        # Minimum distance from any returned atom to any ligand atom must be > 0.5 Å
        min_dists = np.sqrt(
            ((prot_arr[:, None, :] - ligand_coords[None, :, :]) ** 2).sum(axis=2)
        ).min(axis=1)
        too_close = (min_dists < 0.5).sum()
        assert too_close == 0, (
            f'{too_close} returned atoms are within 0.5 Å of a ligand atom – '
            f'suggests ligand atoms were not excluded correctly'
        )

    def test_real_data_count_reasonable(self):
        """Atom count is positive and plausibly protein."""
        coords, elements = extract_protein_atoms_from_minimized(
            minimized_pdb=_ABL_PREP / 'minimized.pdb',
            sub_pdb=_ABL_PREP / 'site1_sub1_frag.pdb',
            core_pdb=_ABL_PREP / 'core.pdb',
            aev_cutoff=5.1,
            prep_dir=_ABL_PREP,
        )
        # Expect at least a few heavy atoms within 5.1 Å of the ligand
        assert len(coords) >= 5, f'Unexpectedly few protein atoms: {len(coords)}'
        # Should not exceed the total nearby atoms included in the trimmed fixture
        assert len(coords) <= 330, f'More atoms than the trimmed fixture contains: {len(coords)}'
        print(f'\n  Protein context atoms within 5.1 Å: {len(coords)}')
        print(f'  Element breakdown: { {e: elements.count(e) for e in set(elements)} }')

    def test_real_data_consistent_across_subs(self):
        """Different substituents at site1 each get a valid protein context."""
        results = {}
        for sub_idx in range(1, 7):  # sub1 through sub6
            sub_pdb = _ABL_PREP / f'site1_sub{sub_idx}_frag.pdb'
            if not sub_pdb.exists():
                continue
            result = extract_protein_atoms_from_minimized(
                minimized_pdb=_ABL_PREP / 'minimized.pdb',
                sub_pdb=sub_pdb,
                core_pdb=_ABL_PREP / 'core.pdb',
                aev_cutoff=5.1,
                prep_dir=_ABL_PREP,
            )
            assert result is not None, f'site1_sub{sub_idx} returned None'
            coords, _ = result
            assert len(coords) >= 5, f'site1_sub{sub_idx}: too few protein atoms ({len(coords)})'
            results[sub_idx] = len(coords)

        print(f'\n  Protein context atom counts per substituent: {results}')
        # All subs at the same site should get a similar number of nearby protein atoms
        counts = list(results.values())
        assert max(counts) - min(counts) < max(counts) * 0.5, (
            f'Atom counts vary too widely across substituents: {results}'
        )


class TestExtractSolventAtomsFromMinimized:
    """Unit tests for extract_solvent_atoms_from_minimized.

    The function is a thin wrapper over extract_protein_atoms_from_minimized;
    these tests focus on the solvent-system-specific behaviour (water molecules
    returned, ligand atoms excluded) and real-data sanity checks.
    """

    # -----------------------------------------------------------------------
    # Synthetic tests (in-memory PDB files)
    # -----------------------------------------------------------------------

    def _write_pdb(self, tmpdir, name, atoms):
        """Write a minimal PDB file.  atoms = [(x,y,z,elem,resname)]"""
        lines = []
        for i, (x, y, z, elem, resname) in enumerate(atoms, start=1):
            lines.append(
                f'ATOM  {i:5d}  {elem:<4s}{resname:<4s}    1    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00      SEG {elem:<2s}'
            )
        lines.append('END')
        p = Path(tmpdir) / name
        p.write_text('\n'.join(lines))
        return p

    def test_returns_none_for_empty_minimized(self):
        """Returns None when minimized.pdb has no atoms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            min_pdb  = self._write_pdb(tmpdir, 'minimized.pdb', [])
            sub_pdb  = self._write_pdb(tmpdir, 'sub.pdb',  [(0, 0, 0, 'C', 'LIG')])
            core_pdb = self._write_pdb(tmpdir, 'core.pdb', [(0, 1, 0, 'C', 'LIG')])
            result = extract_solvent_atoms_from_minimized(
                minimized_pdb=min_pdb,
                sub_pdb=sub_pdb,
                core_pdb=core_pdb,
            )
            assert result is None

    def test_returns_none_when_no_water_within_cutoff(self):
        """Returns None when all non-ligand atoms in minimized.pdb are beyond the cutoff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Sub at origin, water far away
            min_pdb = self._write_pdb(tmpdir, 'minimized.pdb', [
                (0,  0, 0, 'C', 'LIG'),   # core atom (excluded)
                (0, 50, 0, 'O', 'TIP3'),  # water outside cutoff
                (0, 51, 0, 'H', 'TIP3'),
            ])
            sub_pdb  = self._write_pdb(tmpdir, 'sub.pdb',  [(0, 0, 0, 'C', 'LIG')])
            core_pdb = self._write_pdb(tmpdir, 'core.pdb', [(0, 0, 0, 'C', 'LIG')])
            result = extract_solvent_atoms_from_minimized(
                minimized_pdb=min_pdb,
                sub_pdb=sub_pdb,
                core_pdb=core_pdb,
                aev_cutoff=5.1,
            )
            assert result is None

    def test_water_atoms_returned_within_cutoff(self):
        """Water oxygen and hydrogens within the AEV cutoff are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            min_pdb = self._write_pdb(tmpdir, 'minimized.pdb', [
                (0, 0, 0, 'C', 'LIG'),   # core (excluded)
                (3, 0, 0, 'O', 'TIP3'),  # water O within 5.1 Å
                (3, 1, 0, 'H', 'TIP3'),  # water H within 5.1 Å
                (3, -1, 0, 'H', 'TIP3'), # water H within 5.1 Å
            ])
            sub_pdb  = self._write_pdb(tmpdir, 'sub.pdb',  [(0, 0, 0, 'C', 'LIG')])
            core_pdb = self._write_pdb(tmpdir, 'core.pdb', [(0, 0, 0, 'C', 'LIG')])
            result = extract_solvent_atoms_from_minimized(
                minimized_pdb=min_pdb,
                sub_pdb=sub_pdb,
                core_pdb=core_pdb,
                aev_cutoff=5.1,
            )
            assert result is not None
            coords, elements = result
            assert len(coords) == 3
            assert 'O' in elements
            assert elements.count('H') == 2

    def test_ligand_atoms_excluded_from_result(self):
        """Ligand atoms (core + sub) are never returned, even when within cutoff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            min_pdb = self._write_pdb(tmpdir, 'minimized.pdb', [
                (0,   0, 0, 'C', 'LIG'),  # core atom: excluded
                (0.1, 0, 0, 'C', 'LIG'),  # sub atom: excluded (tiny shift after minimisation)
                (3,   0, 0, 'O', 'TIP3'), # water O within cutoff: kept
            ])
            sub_pdb  = self._write_pdb(tmpdir, 'sub.pdb',  [(0,   0, 0, 'C', 'LIG')])
            core_pdb = self._write_pdb(tmpdir, 'core.pdb', [(0,   0, 0, 'C', 'LIG')])
            coords, elements = extract_solvent_atoms_from_minimized(
                minimized_pdb=min_pdb,
                sub_pdb=sub_pdb,
                core_pdb=core_pdb,
                aev_cutoff=5.1,
                duplicate_tolerance=0.5,
            )
            assert len(coords) == 1
            assert elements == ['O']

    # -----------------------------------------------------------------------
    # Real-data tests (14benz solvent system)
    # -----------------------------------------------------------------------

    def test_real_data_returns_solvent_atoms(self):
        """extract_solvent_atoms_from_minimized returns a non-empty tuple for site1_sub1."""
        result = extract_solvent_atoms_from_minimized(
            minimized_pdb=_BENZ_SOLV_PREP / 'minimized.pdb',
            sub_pdb=_BENZ_SOLV_PREP / 'site1_sub1_frag.pdb',
            core_pdb=_BENZ_SOLV_PREP / 'core.pdb',
            aev_cutoff=5.1,
            prep_dir=_BENZ_SOLV_PREP,
        )
        assert result is not None, 'Expected solvent atoms within 5.1 Å but got None'
        coords, elements = result
        assert len(coords) > 0
        assert len(coords) == len(elements)
        print(f'\n  site1_sub1: {len(coords)} solvent atoms within 5.1 Å')

    def test_real_data_atoms_within_cutoff(self):
        """All returned atoms are within 5.1 Å of site1_sub1."""
        from mllf.file_handling.read_pdb import parse_pdb_file as _parse
        sub_coords, _ = _parse(str(_BENZ_SOLV_PREP / 'site1_sub1_frag.pdb'))
        result = extract_solvent_atoms_from_minimized(
            minimized_pdb=_BENZ_SOLV_PREP / 'minimized.pdb',
            sub_pdb=_BENZ_SOLV_PREP / 'site1_sub1_frag.pdb',
            core_pdb=_BENZ_SOLV_PREP / 'core.pdb',
            aev_cutoff=5.1,
            prep_dir=_BENZ_SOLV_PREP,
        )
        assert result is not None
        coords, _ = result
        sub_arr  = np.array(sub_coords)
        prot_arr = np.array(coords)
        diff = prot_arr[:, None, :] - sub_arr[None, :, :]
        min_dists = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
        assert (min_dists <= 5.1 + 1e-6).all(), (
            f'{(min_dists > 5.1).sum()} atom(s) beyond 5.1 Å cutoff'
        )

    def test_real_data_no_ligand_atoms_in_result(self):
        """No returned atom is within 0.5 Å of any ligand (core or sub) atom."""
        from mllf.file_handling.read_pdb import parse_pdb_file as _parse
        sub_coords,  _ = _parse(str(_BENZ_SOLV_PREP / 'site1_sub1_frag.pdb'))
        core_coords, _ = _parse(str(_BENZ_SOLV_PREP / 'core.pdb'))
        lig_arr = np.array(sub_coords + core_coords)
        result = extract_solvent_atoms_from_minimized(
            minimized_pdb=_BENZ_SOLV_PREP / 'minimized.pdb',
            sub_pdb=_BENZ_SOLV_PREP / 'site1_sub1_frag.pdb',
            core_pdb=_BENZ_SOLV_PREP / 'core.pdb',
            aev_cutoff=5.1,
            prep_dir=_BENZ_SOLV_PREP,
        )
        assert result is not None
        coords, _ = result
        solv_arr = np.array(coords)
        diff = solv_arr[:, None, :] - lig_arr[None, :, :]
        min_dists = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
        close = (min_dists < 0.5).sum()
        assert close == 0, f'{close} returned atom(s) are within 0.5 Å of a ligand atom'

    def test_real_data_consistent_across_subs(self):
        """All 11 substituents in the 14benz solvent system return valid solvent contexts."""
        results = {}
        subs = [
            ('site1', range(1, 7)), ('site2', range(1, 6))
        ]
        for site, sub_range in subs:
            for sub_idx in sub_range:
                sub_pdb = _BENZ_SOLV_PREP / f'{site}_sub{sub_idx}_frag.pdb'
                if not sub_pdb.exists():
                    continue
                result = extract_solvent_atoms_from_minimized(
                    minimized_pdb=_BENZ_SOLV_PREP / 'minimized.pdb',
                    sub_pdb=sub_pdb,
                    core_pdb=_BENZ_SOLV_PREP / 'core.pdb',
                    aev_cutoff=5.1,
                    prep_dir=_BENZ_SOLV_PREP,
                )
                key = f'{site}_sub{sub_idx}'
                assert result is not None, f'{key} returned None'
                coords, _ = result
                assert len(coords) > 0, f'{key} returned empty context'
                results[key] = len(coords)

        print(f'\n  Solvent context atom counts: {results}')
        # Every substituent should have a non-empty solvent context
        for key, count in results.items():
            assert count > 0, f'{key}: empty solvent context'
        # Counts should be broadly similar across substituents at the same site
        site1_counts = [v for k, v in results.items() if k.startswith('site1')]
        site2_counts = [v for k, v in results.items() if k.startswith('site2')]
        for site_counts in (site1_counts, site2_counts):
            if len(site_counts) > 1:
                assert max(site_counts) - min(site_counts) < max(site_counts) * 0.8, (
                    f'Solvent atom counts vary too widely: {site_counts}'
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
