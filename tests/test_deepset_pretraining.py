"""Tests for DeepSet pretraining components."""
import pytest
import torch
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
