"""Tests for DeepSet feature extractor."""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mllf.cb.deepset import DeepSetFeatureExtractor
from mllf.cb.aev_processor import NUM_SPECIES


class TestDeepSetInitialization:
    """Test DeepSet model initialization."""
    
    def test_default_initialization(self):
        """Test DeepSet with default parameters."""
        model = DeepSetFeatureExtractor()
        
        assert model.include_charge is True
        assert model.include_atom_id is True
        assert model.num_atom_types == 11  # Default: 10 common + 1 unknown
    
    def test_custom_initialization(self):
        """Test DeepSet with custom parameters."""
        model = DeepSetFeatureExtractor(
            aev_length=2000,
            num_atom_types=20,
            embedding_dim=128,
            hidden_dim=512,
            include_charge=False,
            include_atom_id=False
        )
        
        assert model.include_charge is False
        assert model.include_atom_id is False
        assert model.num_atom_types == 20
    
    def test_model_has_mlp(self):
        """Test that model has atom_mlp attribute."""
        model = DeepSetFeatureExtractor()
        assert hasattr(model, 'atom_mlp')
        assert isinstance(model.atom_mlp, torch.nn.Sequential)
    
    def test_mlp_input_dimension_calculation(self):
        """Test that MLP input dimension is calculated correctly."""
        # With default: AEV (2288) + charge (1) + atom_id one-hot (11) = 2300
        model = DeepSetFeatureExtractor(
            aev_length=2288,
            num_atom_types=11,
            include_charge=True,
            include_atom_id=True
        )
        
        # First layer should accept 2300 features
        first_layer = model.atom_mlp[0]
        assert isinstance(first_layer, torch.nn.Linear)
        assert first_layer.in_features == 2288 + 1 + 11  # 2300
    
    def test_mlp_output_dimension(self):
        """Test that MLP output dimension matches embedding_dim."""
        embedding_dim = 64
        model = DeepSetFeatureExtractor(embedding_dim=embedding_dim)
        
        # Last layer should output embedding_dim features
        last_layer = model.atom_mlp[-1]
        assert isinstance(last_layer, torch.nn.Linear)
        assert last_layer.out_features == embedding_dim


class TestDeepSetForward:
    """Test DeepSet forward pass."""
    
    def test_forward_with_all_features(self):
        """Test forward pass with AEV, charges, and atom IDs."""
        model = DeepSetFeatureExtractor(
            aev_length=2288,
            num_atom_types=11,
            embedding_dim=64,
            include_charge=True,
            include_atom_id=True
        )
        
        num_atoms = 25
        aevs = torch.randn(num_atoms, 2288)
        charges = torch.randn(num_atoms)
        atom_ids = torch.randint(0, 11, (num_atoms,))
        
        with torch.no_grad():
            embedding = model(aevs, charges, atom_ids)
        
        assert embedding.shape == torch.Size([64])
        assert not torch.isnan(embedding).any()
    
    def test_forward_without_charge(self):
        """Test forward pass without charges."""
        model = DeepSetFeatureExtractor(
            aev_length=2288,
            num_atom_types=11,
            embedding_dim=64,
            include_charge=False,
            include_atom_id=True
        )
        
        num_atoms = 15
        aevs = torch.randn(num_atoms, 2288)
        atom_ids = torch.randint(0, 11, (num_atoms,))
        
        with torch.no_grad():
            embedding = model(aevs, charges=None, atom_ids=atom_ids)
        
        assert embedding.shape == torch.Size([64])
    
    def test_forward_without_atom_id(self):
        """Test forward pass without atom IDs."""
        model = DeepSetFeatureExtractor(
            aev_length=2288,
            num_atom_types=11,
            embedding_dim=64,
            include_charge=True,
            include_atom_id=False
        )
        
        num_atoms = 20
        aevs = torch.randn(num_atoms, 2288)
        charges = torch.randn(num_atoms)
        
        with torch.no_grad():
            embedding = model(aevs, charges=charges, atom_ids=None)
        
        assert embedding.shape == torch.Size([64])
    
    def test_forward_aev_only(self):
        """Test forward pass with AEVs only."""
        model = DeepSetFeatureExtractor(
            aev_length=2288,
            num_atom_types=11,
            embedding_dim=64,
            include_charge=False,
            include_atom_id=False
        )
        
        num_atoms = 10
        aevs = torch.randn(num_atoms, 2288)
        
        with torch.no_grad():
            embedding = model(aevs, charges=None, atom_ids=None)
        
        assert embedding.shape == torch.Size([64])
    
    def test_permutation_invariance(self):
        """Test that output is permutation invariant (up to numerical precision)."""
        model = DeepSetFeatureExtractor(
            aev_length=100,
            num_atom_types=11,
            embedding_dim=32,
            include_charge=True,
            include_atom_id=True
        )
        model.eval()  # Disable dropout for deterministic behavior
        
        num_atoms = 10
        aevs = torch.randn(num_atoms, 100)
        charges = torch.randn(num_atoms)
        atom_ids = torch.randint(0, 11, (num_atoms,))
        
        # Compute embedding
        with torch.no_grad():
            emb1 = model(aevs, charges, atom_ids)
        
        # Permute atoms
        perm = torch.randperm(num_atoms)
        aevs_perm = aevs[perm]
        charges_perm = charges[perm]
        atom_ids_perm = atom_ids[perm]
        
        with torch.no_grad():
            emb2 = model(aevs_perm, charges_perm, atom_ids_perm)
        
        # Embeddings should be identical (within numerical tolerance)
        assert torch.allclose(emb1, emb2, atol=1e-5)
    
    def test_variable_size_handling(self):
        """Test that model handles variable-sized inputs."""
        model = DeepSetFeatureExtractor(embedding_dim=64)
        
        # Test with different sizes
        sizes = [5, 10, 25, 50]
        
        for num_atoms in sizes:
            aevs = torch.randn(num_atoms, 2288)
            charges = torch.randn(num_atoms)
            atom_ids = torch.randint(0, 11, (num_atoms,))
            
            with torch.no_grad():
                embedding = model(aevs, charges, atom_ids)
            
            # All should produce same output size
            assert embedding.shape == torch.Size([64])


class TestDeepSetErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_charges_when_required(self):
        """Test that error is raised when charges are required but not provided."""
        model = DeepSetFeatureExtractor(include_charge=True)
        
        aevs = torch.randn(10, 1920)
        atom_ids = torch.randint(0, 11, (10,))
        
        with pytest.raises(ValueError, match="charges must be provided"):
            model(aevs, charges=None, atom_ids=atom_ids)
    
    def test_missing_atom_ids_when_required(self):
        """Test that error is raised when atom IDs are required but not provided."""
        model = DeepSetFeatureExtractor(include_atom_id=True)
        
        aevs = torch.randn(10, 1920)
        charges = torch.randn(10)
        
        with pytest.raises(ValueError, match="atom_ids must be provided"):
            model(aevs, charges=charges, atom_ids=None)
    
    def test_charge_shape_handling(self):
        """Test that charges can be 1D or 2D."""
        model = DeepSetFeatureExtractor()
        model.eval()  # Disable dropout for deterministic behavior
        
        num_atoms = 10
        aevs = torch.randn(num_atoms, 2288)
        atom_ids = torch.randint(0, 11, (num_atoms,))
        
        # Test with 1D charges
        charges_1d = torch.randn(num_atoms)
        with torch.no_grad():
            emb1 = model(aevs, charges_1d, atom_ids)
        
        # Test with 2D charges
        charges_2d = charges_1d.unsqueeze(1)
        with torch.no_grad():
            emb2 = model(aevs, charges_2d, atom_ids)
        
        assert torch.allclose(emb1, emb2, atol=1e-5)
    
    def test_single_atom_substituent(self):
        """Test handling of single-atom substituent (edge case)."""
        model = DeepSetFeatureExtractor(embedding_dim=64)
        
        aevs = torch.randn(1, 2288)
        charges = torch.randn(1)
        atom_ids = torch.randint(0, 11, (1,))
        
        with torch.no_grad():
            embedding = model(aevs, charges, atom_ids)
        
        assert embedding.shape == torch.Size([64])
    
    def test_empty_substituent_raises_error(self):
        """Test that empty substituent raises appropriate error."""
        model = DeepSetFeatureExtractor()
        
        # Empty tensors
        aevs = torch.randn(0, 2288)
        charges = torch.randn(0)
        atom_ids = torch.randint(0, 11, (0,))
        
        # Max pooling on empty tensor should raise an IndexError
        with pytest.raises(IndexError, match="Expected reduction dim 0 to have non-zero size"):
            with torch.no_grad():
                embedding = model(aevs, charges, atom_ids)


class TestDeepSetGradients:
    """Test gradient flow through DeepSet."""
    
    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = DeepSetFeatureExtractor(embedding_dim=32)
        
        num_atoms = 10
        aevs = torch.randn(num_atoms, 2288, requires_grad=True)
        charges = torch.randn(num_atoms, requires_grad=True)
        atom_ids = torch.randint(0, 11, (num_atoms,))
        
        embedding = model(aevs, charges, atom_ids)
        loss = embedding.sum()
        loss.backward()
        
        # Check gradients exist
        assert aevs.grad is not None
        assert charges.grad is not None
        assert not torch.isnan(aevs.grad).any()
        assert not torch.isnan(charges.grad).any()
    
    def test_model_parameters_have_gradients(self):
        """Test that model parameters can be optimized."""
        model = DeepSetFeatureExtractor()
        
        # Check that model has trainable parameters
        params = list(model.parameters())
        assert len(params) > 0
        
        for param in params:
            assert param.requires_grad


class TestDeepSetWithNumSpecies:
    """Test DeepSet with NUM_SPECIES constant."""
    
    def test_default_uses_num_species(self):
        """Test that default num_atom_types matches NUM_SPECIES."""
        model = DeepSetFeatureExtractor()
        assert model.num_atom_types == NUM_SPECIES == 11
    
    def test_one_hot_dimension_matches_num_species(self):
        """Test that one-hot encoding dimension matches NUM_SPECIES."""
        model = DeepSetFeatureExtractor(num_atom_types=NUM_SPECIES)
        
        num_atoms = 5
        aevs = torch.randn(num_atoms, 2288)
        charges = torch.randn(num_atoms)
        atom_ids = torch.randint(0, NUM_SPECIES, (num_atoms,))
        
        with torch.no_grad():
            embedding = model(aevs, charges, atom_ids)
        
        assert embedding.shape == torch.Size([64])


class TestAtomBondGNN:
    """Tests for AtomBondGNN — bond-topology-aware substituent encoder."""

    def _model(self, embedding_dim=64, hidden_dim=64, **kw):
        from mllf.cb.deepset import AtomBondGNN
        return AtomBondGNN(
            aev_length=64,       # small for fast tests
            num_atom_types=11,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            **kw,
        )

    def test_initialization(self):
        model = self._model()
        assert model.embedding_dim == 64
        assert model.include_charge is True
        assert model.include_atom_id is True
        # Dual-stream architecture: sub_gin_layers and core_gin_layers replaced gin1/gin2
        assert hasattr(model, 'sub_gin_layers') and hasattr(model, 'core_gin_layers')
        # Attentional pooling replaced simple pool
        assert hasattr(model, 'gate_nn') and hasattr(model, 'pool_nn')

    def test_atom_mlp_shim_out_features(self):
        """atom_mlp[-1].out_features must equal embedding_dim for graph_utils compat."""
        model = self._model(embedding_dim=32)
        assert model.atom_mlp[-1].out_features == 32

    def test_output_shape_with_edges(self):
        """Forward with explicit bond edges returns [embedding_dim]."""
        model = self._model(embedding_dim=32)
        N = 6
        aevs = torch.randn(N, 64)
        charges = torch.randn(N)
        atom_ids = torch.randint(0, 11, (N,))
        # bidirectional bond edges
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)

        with torch.no_grad():
            emb = model(aevs, charges, atom_ids, bond_edge_index=edge_index)

        assert emb.shape == torch.Size([32])
        assert not torch.isnan(emb).any()

    def test_output_shape_no_edges(self):
        """Forward without bond edges falls back to self-loops, still returns [embedding_dim]."""
        model = self._model(embedding_dim=32)
        N = 5
        aevs = torch.randn(N, 64)
        charges = torch.randn(N)
        atom_ids = torch.randint(0, 11, (N,))

        with torch.no_grad():
            emb = model(aevs, charges, atom_ids, bond_edge_index=None)

        assert emb.shape == torch.Size([32])
        assert not torch.isnan(emb).any()

    def test_output_shape_single_atom(self):
        """Single-atom substituent should still produce valid embedding."""
        model = self._model(embedding_dim=32)
        aevs = torch.randn(1, 64)
        charges = torch.randn(1)
        atom_ids = torch.randint(0, 11, (1,))

        with torch.no_grad():
            emb = model(aevs, charges, atom_ids)

        assert emb.shape == torch.Size([32])

    def test_missing_charges_raises(self):
        model = self._model(include_charge=True)
        aevs = torch.randn(4, 64)
        atom_ids = torch.randint(0, 11, (4,))
        with pytest.raises(ValueError, match="charges must be provided"):
            model(aevs, charges=None, atom_ids=atom_ids)

    def test_missing_atom_ids_raises(self):
        model = self._model(include_atom_id=True)
        aevs = torch.randn(4, 64)
        charges = torch.randn(4)
        with pytest.raises(ValueError, match="atom_ids must be provided"):
            model(aevs, charges=charges, atom_ids=None)

    def test_no_charge_no_id(self):
        """Model with both flags False should accept AEVs only."""
        from mllf.cb.deepset import AtomBondGNN
        model = AtomBondGNN(
            aev_length=64, num_atom_types=11, embedding_dim=32,
            hidden_dim=64, include_charge=False, include_atom_id=False
        )
        aevs = torch.randn(4, 64)
        with torch.no_grad():
            emb = model(aevs)
        assert emb.shape == torch.Size([32])

    def test_gradient_flow(self):
        """Gradients should flow back through GINConv and input_proj."""
        model = self._model(embedding_dim=32, hidden_dim=64)
        N = 4
        aevs = torch.randn(N, 64, requires_grad=True)
        charges = torch.randn(N)
        atom_ids = torch.randint(0, 11, (N,))

        emb = model(aevs, charges, atom_ids)
        emb.sum().backward()

        assert aevs.grad is not None
        assert not torch.isnan(aevs.grad).any()

    def test_bond_edge_attr_accepted_but_unused(self):
        """bond_edge_attr should be accepted without error (GINConv ignores it)."""
        model = self._model(embedding_dim=32)
        N = 4
        aevs = torch.randn(N, 64)
        charges = torch.randn(N)
        atom_ids = torch.randint(0, 11, (N,))
        edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]], dtype=torch.long)
        edge_attr = torch.ones(4, 1)  # bond-type weights — not consumed

        with torch.no_grad():
            emb = model(aevs, charges, atom_ids,
                       bond_edge_index=edge_index, bond_edge_attr=edge_attr)
        assert emb.shape == torch.Size([32])

    def test_variable_atom_counts(self):
        """Embedding dimension should be constant regardless of atom count."""
        model = self._model(embedding_dim=32)
        model.eval()

        for N in [1, 3, 10, 25]:
            aevs = torch.randn(N, 64)
            charges = torch.randn(N)
            atom_ids = torch.randint(0, 11, (N,))
            with torch.no_grad():
                emb = model(aevs, charges, atom_ids)
            assert emb.shape == torch.Size([32]), f"Failed for N={N}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
