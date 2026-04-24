import pytest
import numpy as np
from core_physics.h7_wrapper import CKernelWrapper, covariance_from_circuit_probs

def test_covariance_deduction():
    # Mock counts for 3 qubits
    counts = {
        '000': 100,
        '111': 100,
        '010': 50
    }
    n_qubits = 3
    mu, cov, cov_inv = covariance_from_circuit_probs(counts, n_qubits)
    
    assert mu.shape == (3,)
    assert cov.shape == (3, 3)
    assert cov_inv.shape == (3, 3)
    # Check if symmetric
    assert np.allclose(cov, cov.T)

def test_update_weights_compatibility():
    # Mock parameters
    q_weights = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    counts = {'000': 10, '111': 10}
    n_qubits = 3
    energy = -1.5
    lr = 0.01
    eps = 1e-4
    
    # This should not crash (handles 3x3 -> 4x4 padding internally)
    norm_grad, mahal_dist, cov_inv = CKernelWrapper.update_weights(
        q_weights, counts, n_qubits, energy, lr, eps
    )
    
    assert isinstance(norm_grad, float)
    assert isinstance(mahal_dist, float)
    assert cov_inv.shape == (3, 3)
    # Weights should remain normalized
    assert np.isclose(np.linalg.norm(q_weights), 1.0)
