import pytest
import numpy as np
from h7_framework import (
    MetriplexVQESolver,
    SolverConfig,
    OptimizationMode,
    QuaternionMetrics
)

def test_quaternion_normalization():
    """Test quaternion normalization preserves unit norm."""
    q = np.array([3.0, 4.0, 0.0, 0.0])
    q_norm = QuaternionMetrics.normalize(q)

    norm = np.linalg.norm(q_norm)
    assert np.isclose(norm, 1.0), f"Norm {norm} ≠ 1.0"


def test_euler_round_trip():
    """Test quaternion ↔ Euler conversion is invertible."""
    q_original = np.array([0.8, 0.2, 0.3, 0.5])
    q_original = QuaternionMetrics.normalize(q_original)

    # Forward: quaternion → Euler
    euler = QuaternionMetrics.quaternion_to_euler(q_original)

    # Backward: Euler → quaternion
    q_reconstructed = QuaternionMetrics.euler_to_quaternion(*euler)
    q_reconstructed = QuaternionMetrics.normalize(q_reconstructed)

    # Check fidelity (dot product should be ±1)
    fidelity = abs(np.dot(q_original, q_reconstructed))
    assert np.isclose(fidelity, 1.0, atol=1e-6), f"Fidelity {fidelity} ≠ 1.0"


def test_ansatz_circuit():
    """Test that ansatz circuit builds without errors."""
    config = SolverConfig(n_qubits=3)
    solver = MetriplexVQESolver(config)

    euler_angles = (0.1, 0.2, 0.3)
    qc = solver.build_ansatz(euler_angles)

    assert qc.num_qubits == 3, f"Wrong qubit count: {qc.num_qubits}"
    assert len(qc) > 0, "Circuit is empty"


def test_config_modes():
    """Test that OptimizationMode enum works correctly."""
    # Valid modes should work
    config1 = SolverConfig(mode=OptimizationMode.MOLECULAR)
    config2 = SolverConfig(mode=OptimizationMode.GENERIC)

    assert config1.mode == OptimizationMode.MOLECULAR
    assert config2.mode == OptimizationMode.GENERIC

    # Invalid mode should raise AttributeError
    with pytest.raises(AttributeError):
        OptimizationMode.INVALID_MODE


def test_config_defaults():
    """Test that SolverConfig has reasonable defaults."""
    config = SolverConfig()

    assert config.n_qubits >= 2
    assert config.base_epsilon > 0
    assert config.learning_rate > 0
    assert config.target_bond_length > 0
