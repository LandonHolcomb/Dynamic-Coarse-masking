"""
Tests for quantum utility functions.
"""

import pytest
import torch
import torch.nn.functional as F

from dcm_msr.quantum_utils import (
    create_pure_state_density_matrix,
    create_ensemble_density_matrix,
    swap_test_fidelity,
    batch_swap_test_scores,
)


class TestPureStateDensityMatrix:
    """Tests for create_pure_state_density_matrix."""
    
    def test_basic_pure_state(self):
        """Test creating a density matrix from a simple state."""
        state = torch.tensor([1.0, 0.0])
        dm = create_pure_state_density_matrix(state)
        
        expected = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        assert torch.allclose(dm, expected, atol=1e-6)
    
    def test_normalized_output(self):
        """Test that output has trace 1 (normalized)."""
        state = torch.randn(8)
        dm = create_pure_state_density_matrix(state)
        
        trace = torch.trace(dm)
        assert torch.isclose(trace, torch.tensor(1.0), atol=1e-6)
    
    def test_hermitian_property(self):
        """Test that density matrix is Hermitian."""
        state = torch.randn(16)
        dm = create_pure_state_density_matrix(state)
        
        assert torch.allclose(dm, dm.T, atol=1e-6)
    
    def test_positive_semidefinite(self):
        """Test that density matrix is positive semi-definite."""
        state = torch.randn(8)
        dm = create_pure_state_density_matrix(state)
        
        eigenvalues = torch.linalg.eigvalsh(dm)
        assert torch.all(eigenvalues >= -1e-6)
    
    def test_pure_state_property(self):
        """Test that Tr(ρ²) = 1 for pure states."""
        state = torch.randn(8)
        dm = create_pure_state_density_matrix(state)
        
        purity = torch.trace(dm @ dm)
        assert torch.isclose(purity, torch.tensor(1.0), atol=1e-6)
    
    def test_batched_input(self):
        """Test with batched input."""
        batch_size = 4
        dim = 8
        states = torch.randn(batch_size, dim)
        
        dms = create_pure_state_density_matrix(states)
        
        assert dms.shape == (batch_size, dim, dim)
        
        # Check trace for each batch
        for i in range(batch_size):
            trace = torch.trace(dms[i])
            assert torch.isclose(trace, torch.tensor(1.0), atol=1e-6)


class TestEnsembleDensityMatrix:
    """Tests for create_ensemble_density_matrix."""
    
    def test_uniform_mixture(self):
        """Test uniform mixture of states."""
        states = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        dm = create_ensemble_density_matrix(states)
        
        # Uniform mixture of |0⟩ and |1⟩ should give I/2
        expected = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        assert torch.allclose(dm, expected, atol=1e-6)
    
    def test_weighted_mixture(self):
        """Test weighted mixture of states."""
        states = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        weights = torch.tensor([0.75, 0.25])
        dm = create_ensemble_density_matrix(states, weights)
        
        expected = torch.tensor([[0.75, 0.0], [0.0, 0.25]])
        assert torch.allclose(dm, expected, atol=1e-6)
    
    def test_trace_one(self):
        """Test that ensemble has trace 1."""
        n_states = 5
        dim = 8
        states = torch.randn(n_states, dim)
        
        dm = create_ensemble_density_matrix(states)
        trace = torch.trace(dm)
        
        assert torch.isclose(trace, torch.tensor(1.0), atol=1e-6)
    
    def test_mixed_state_purity(self):
        """Test that purity Tr(ρ²) < 1 for mixed states."""
        states = torch.randn(4, 8)
        dm = create_ensemble_density_matrix(states)
        
        purity = torch.trace(dm @ dm)
        assert purity < 1.0
    
    def test_batched_input(self):
        """Test with batched input."""
        batch = 3
        n_states = 4
        dim = 8
        states = torch.randn(batch, n_states, dim)
        
        dms = create_ensemble_density_matrix(states)
        
        assert dms.shape == (batch, dim, dim)


class TestSwapTestFidelity:
    """Tests for swap_test_fidelity."""
    
    def test_identical_states(self):
        """Test fidelity of identical states is 1."""
        state = torch.randn(8)
        dm = create_pure_state_density_matrix(state)
        
        fidelity = swap_test_fidelity(dm, dm)
        assert torch.isclose(fidelity, torch.tensor(1.0), atol=1e-6)
    
    def test_orthogonal_states(self):
        """Test fidelity of orthogonal states is 0."""
        state1 = torch.tensor([1.0, 0.0])
        state2 = torch.tensor([0.0, 1.0])
        
        dm1 = create_pure_state_density_matrix(state1)
        dm2 = create_pure_state_density_matrix(state2)
        
        fidelity = swap_test_fidelity(dm1, dm2)
        assert torch.isclose(fidelity, torch.tensor(0.0), atol=1e-6)
    
    def test_symmetric(self):
        """Test that fidelity is symmetric."""
        state1 = torch.randn(8)
        state2 = torch.randn(8)
        
        dm1 = create_pure_state_density_matrix(state1)
        dm2 = create_pure_state_density_matrix(state2)
        
        f12 = swap_test_fidelity(dm1, dm2)
        f21 = swap_test_fidelity(dm2, dm1)
        
        assert torch.isclose(f12, f21, atol=1e-6)
    
    def test_bounded_zero_one(self):
        """Test that fidelity is in [0, 1]."""
        for _ in range(10):
            state1 = torch.randn(8)
            state2 = torch.randn(8)
            
            dm1 = create_pure_state_density_matrix(state1)
            dm2 = create_pure_state_density_matrix(state2)
            
            fidelity = swap_test_fidelity(dm1, dm2)
            assert 0.0 <= fidelity <= 1.0 + 1e-6
    
    def test_full_quantum_fidelity(self):
        """Test full quantum fidelity calculation."""
        state1 = torch.randn(4)
        state2 = torch.randn(4)
        
        dm1 = create_pure_state_density_matrix(state1)
        dm2 = create_pure_state_density_matrix(state2)
        
        # For pure states, classical and quantum fidelity should match
        classical = swap_test_fidelity(dm1, dm2, classical_approximation=True)
        quantum = swap_test_fidelity(dm1, dm2, classical_approximation=False)
        
        assert torch.isclose(classical, quantum, atol=1e-3)


class TestBatchSwapTestScores:
    """Tests for batch_swap_test_scores."""
    
    def test_basic_batch(self):
        """Test basic batched SWAP test scoring."""
        batch = 2
        heads = 4
        num_windows = 8
        dim = 16
        
        query_dm = torch.randn(batch, heads, dim, dim)
        # Make symmetric for valid density matrices
        query_dm = (query_dm + query_dm.transpose(-1, -2)) / 2
        
        key_ensembles = torch.randn(batch, heads, num_windows, dim, dim)
        key_ensembles = (key_ensembles + key_ensembles.transpose(-1, -2)) / 2
        
        scores = batch_swap_test_scores(query_dm, key_ensembles)
        
        assert scores.shape == (batch, heads, num_windows)
    
    def test_scores_bounded(self):
        """Test that scores are in valid range."""
        batch = 2
        heads = 2
        num_windows = 4
        dim = 8
        
        # Create valid density matrices
        states = torch.randn(batch, heads, dim)
        query_dm = create_pure_state_density_matrix(states)
        
        key_states = torch.randn(batch, heads, num_windows, 3, dim)
        key_ensembles = create_ensemble_density_matrix(key_states)
        
        scores = batch_swap_test_scores(query_dm, key_ensembles)
        
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1 + 1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
