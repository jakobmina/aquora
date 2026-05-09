import pytest
import numpy as np
from h7_metriplectic_os.covariance_asymmetry import quantum_to_covariance_inverse

def test_quantum_to_covariance_inverse_uniform():
    """Test with uniform distribution. Torsion should be 0, Precision should be uniform/diagonal."""
    # Uniform probabilities for 3 qubits
    probs = {i: 0.125 for i in range(8)}
    
    precision_matrix = quantum_to_covariance_inverse(probs, n_qubits=3, alpha=1.0)
    
    # Precision matrix should be symmetric
    assert np.allclose(precision_matrix, precision_matrix.T)
    
    # For a uniform distribution of independent coin flips, 
    # Covariance is diagonal (0.25). 
    # Precision is inverse of Covariance -> diagonal (4.0).
    # Since we apply pseudo-inverse and regularization it might vary slightly,
    # but the off-diagonal elements should be exactly 0 (no conditional dependence).
    
    # Check off-diagonal elements
    for i in range(3):
        for j in range(3):
            if i != j:
                assert abs(precision_matrix[i, j]) < 1e-5, f"Off-diagonal element ({i},{j}) must be 0 for independent vars"

def test_quantum_to_covariance_inverse_asymmetric_torsion():
    """Test with highly asymmetric distribution to ensure torsion creates strong dependencies."""
    # Strong correlation between q0 and q2
    # States where q0 == q2 have higher probability
    probs = {
        0: 0.25,  # 000 (q0=0, q2=0)
        1: 0.05,  # 001 (q0=1, q2=0)
        2: 0.05,  # 010 (q0=0, q2=0) - wait, binary is q2,q1,q0 usually but let's just test structural asymmetry
        3: 0.05,  # 011
        4: 0.05,  # 100
        5: 0.25,  # 101 (q0=1, q2=1)
        6: 0.05,  # 110
        7: 0.25,  # 111 (q0=1, q2=1)
    }
    
    # Normalize
    total = sum(probs.values())
    probs = {k: v/total for k, v in probs.items()}
    
    # Apply torsion
    precision_matrix = quantum_to_covariance_inverse(probs, n_qubits=3, alpha=1.0)
    
    # Should have non-zero off-diagonal elements, proving conditional dependence was captured
    # specifically since the distribution is asymmetric, some off-diagonals must be non-zero
    off_diagonals = [precision_matrix[i, j] for i in range(3) for j in range(3) if i != j]
    assert any(abs(val) > 1e-4 for val in off_diagonals), "Precision matrix must capture dependencies for asymmetric states"
    
def test_quantum_to_covariance_inverse_torsion_limits():
    """Test the limits of alpha (torsion coupling)."""
    probs = {
        0: 0.4,
        1: 0.1,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.4,
        7: 0.1,
    }
    
    # Zero torsion
    prec_no_torsion = quantum_to_covariance_inverse(probs, n_qubits=3, alpha=0.0)
    
    # Full torsion
    prec_torsion = quantum_to_covariance_inverse(probs, n_qubits=3, alpha=1.0)
    
    # They should be different because torsion modulates the probabilities
    assert not np.allclose(prec_no_torsion, prec_torsion), "Torsion must modulate the precision matrix"

