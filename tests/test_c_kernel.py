import pytest
import numpy as np
import math

from metriplex_bridge import MetriplexCKernel

@pytest.fixture
def kernel():
    return MetriplexCKernel(lib_path="core_physics/libmetriplex_core.so")

def test_golden_operator_c(kernel):
    """Test that C implementation of Golden Operator matches Python math."""
    PHI = 1.618033988749895
    for n in [0.0, 1.0, 2.0, 3.5]:
        val_c = kernel.get_golden_operator(n)
        val_py = math.cos(math.pi * n) * math.cos(math.pi * PHI * n)
        assert np.isclose(val_c, val_py)

def test_compute_lagrangian(kernel):
    """Test Lagrangian calculation and Rule 1.3 (no singularities)."""
    # Create dummy state
    N = 4
    real_arr = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    imag_arr = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state = kernel.allocate_state(real_arr, imag_arr)
    
    # Create dummy Hermitian H and S matrices
    H = np.eye(N, dtype=np.float64) * 2.0 # Energy
    S = np.zeros((N, N), dtype=np.float64) # Zero entropy (purely conservative system)
    
    L_symp, L_metr = kernel.compute_lagrangian(state, H, S)
    
    # L_metr should hit the 1e-5 floor (Rule 1.3)
    assert L_metr == 1e-5
    # L_symp should be non-zero and positive
    assert L_symp > 0.1

def test_phase_evolution_reversibility(kernel):
    """Test Phase Evolution and Reversibility Rule (Rule 1)."""
    N = 2
    real_arr = np.array([1.0, 0.0], dtype=np.float64)
    imag_arr = np.array([0.0, 0.0], dtype=np.float64)
    state = kernel.allocate_state(real_arr, imag_arr)
    
    # Evolve forward in time (conservative only)
    dt = 1.0
    L_symp = math.pi / 2.0 # 90 degree rotation
    L_metr = 0.0 # No dissipation
    
    kernel.evolve_phase(state, dt, L_symp, L_metr)
    
    assert np.isclose(state.real_parts[0], 0.0, atol=1e-5)
    assert np.isclose(state.imag_parts[0], 1.0, atol=1e-5)
    
    # Evolve backward in time
    kernel.evolve_phase(state, -dt, L_symp, L_metr)
    
    # Should return to original state
    assert np.isclose(state.real_parts[0], 1.0, atol=1e-5)
    assert np.isclose(state.imag_parts[0], 0.0, atol=1e-5)
