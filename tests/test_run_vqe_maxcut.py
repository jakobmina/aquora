import pytest
import numpy as np
import sys
import os

# Append current path to sys.path to import run_vqe_maxcut
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_vqe_maxcut import MetriplecticMaxCut

@pytest.fixture
def base_edges():
    return [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 4, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]

def test_golden_operator_modulation(base_edges):
    """
    Test that the Golden Operator (O_n) correctly modulates the edge weights
    and never collapses to zero (Regla 2.1).
    """
    system = MetriplecticMaxCut(base_edges, phi=1.6180339887)
    
    for i, (u, v, weight) in enumerate(system.modulated_edges):
        n = i + 1
        expected_On = np.cos(np.pi * n) * np.cos(np.pi * 1.6180339887 * n)
        if abs(expected_On) < 1e-5:
            expected_On = 1e-5
            
        # The weight should be the original weight (1.0) * expected_On
        assert np.isclose(weight, 1.0 * expected_On)
        assert abs(weight) >= 1e-5  # No flat vacuums

def test_compute_lagrangian_signature_and_values(base_edges):
    """
    Test that compute_lagrangian returns both L_symp and L_metr (Regla 3.1)
    and prevents pure states (Regla 1.3).
    """
    system = MetriplecticMaxCut(base_edges)
    
    # Test with typical values
    psi = np.array([0.5] * len(base_edges))
    rho = 1.0
    v = np.array([0.1] * len(base_edges))
    
    L_symp, L_metr = system.compute_lagrangian(psi, rho, v)
    
    assert L_symp != 0
    assert L_metr != 0
    assert isinstance(L_symp, float)
    assert isinstance(L_metr, float)
    
    # Test with values that would produce zero to check rule 1.3 prevention
    psi_zero = np.zeros(len(base_edges))
    v_zero = np.zeros(len(base_edges))
    
    L_symp_zero, L_metr_zero = system.compute_lagrangian(psi_zero, rho, v_zero)
    
    assert L_symp_zero == 1e-5
    assert L_metr_zero == 1e-5

def test_visualize_dynamics_runs(base_edges):
    """
    Test that the visualization method runs without errors and produces the output file.
    """
    system = MetriplecticMaxCut(base_edges)
    
    # Remove file if exists to ensure it's generated
    if os.path.exists('metriplectic_dynamics.png'):
        os.remove('metriplectic_dynamics.png')
        
    system.visualize_dynamics(steps=5) # Run a short simulation
    
    assert os.path.exists('metriplectic_dynamics.png')
