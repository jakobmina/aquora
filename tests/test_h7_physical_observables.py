import pytest
import numpy as np
import math
from run_vqe_maxcut import MetriplecticMaxCut
from h7_quaternion import compute_h7_pair_overlaps, H7QuaternionMapper, H7_AMPLITUDES
from metriplex_bridge import MetriplexEndianBridge, golden_operator

def test_parity_classification_logic():
    """
    Validate that odd n is fermionic and even n is bosonic.
    This is the core of our Second Quantization mapping.
    """
    # Test cases: (n, expected_type)
    test_cases = [
        (1, "fermionic"),
        (2, "bosonic"),
        (3, "fermionic"),
        (4, "bosonic"),
        (5, "fermionic"),
        (6, "bosonic"),
    ]
    
    for n, expected in test_cases:
        # Check in Bridge
        bridge = MetriplexEndianBridge()
        p = bridge.oracle._occupation_to_momentum((n, 0, 0))
        h7_state = (p - 1) % 8
        expected_type = "fermionic" if h7_state % 2 != 0 else "bosonic"
        
        # Report should match our physics rule
        report = bridge.full_state_report(occupation=(n, 0, 0))
        assert report['particle_type'] == expected_type

def test_vacuum_overlap_symmetry():
    """
    W_pair = O(n) + O(7-n)
    Test that W_pair is symmetric for the H7 pairs (0,7), (1,6), (2,5), (3,4).
    """
    phi = 0.618
    overlaps = compute_h7_pair_overlaps(phi)
    assert len(overlaps) == 4
    
    # Manually calculate and compare
    for n in range(4):
        o_n = golden_operator(n, phi)
        o_inv = golden_operator(7 - n, phi)
        expected_w = o_n + o_inv
        assert np.isclose(overlaps[n], expected_w, atol=1e-7)

def test_non_locality_violation():
    """
    Verify that O(n1 + n2) != O(n1) + O(n2)
    This confirms the non-linearity of the quasiperiodic vacuum.
    """
    phi = 0.618
    o1 = golden_operator(1, phi)
    o2 = golden_operator(2, phi)
    o3 = golden_operator(3, phi)
    
    # Linear expectation: o1 + o2 = o3
    # Non-local reality: o1 + o2 != o3
    assert not np.isclose(o1 + o2, o3, atol=1e-3)

def test_quaternion_chirality():
    """
    Ensure that the Metriplectic analysis correctly identifies chirality
    in non-Abelian states (n=1, 3, etc).
    """
    mapper = H7QuaternionMapper(H7_AMPLITUDES)
    phi = 0.618
    report = mapper.analyze(phi_param=phi)
    
    # Chirality should be non-zero for the standard H7 configuration
    assert report['chirality'] > 0
    assert report['is_non_abelian'] is True
    
    # Check that Lagrangian components are positive (magnitudes)
    assert report['L_symp'] > 0
    assert report['L_metr'] > 0

if __name__ == "__main__":
    pytest.main([__file__])
