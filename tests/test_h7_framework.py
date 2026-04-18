import pytest
import numpy as np
from qiskit.quantum_info import Statevector
from h7_framework import (
    run_quantum_experiment,
    interpret_measured_bitstring,
    QuaternionMetrics,
    MetriplexVQESolver,
    SolverConfig,
    OptimizationMode
)

def test_interpret_measured_bitstring():
    res = interpret_measured_bitstring('101', 'little')
    assert res['decimal_value'] == 5
    assert res['convention'] == 'little'

def test_run_quantum_experiment():
    qc, psi, probs_dict, particle_type = run_quantum_experiment(1, 1.618)
    assert particle_type in ["fermionic", "bosonic", "unknown"]
    assert isinstance(psi, Statevector)

def test_quaternion_metrics():
    q = np.array([1.0, 1.0, 1.0, 1.0])
    q_norm = QuaternionMetrics.normalize(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0)
    
    euler = QuaternionMetrics.quaternion_to_euler(q_norm)
    assert len(euler) == 3

def test_metriplex_vqe_solver():
    config = SolverConfig() # Just a dummy param setting test
    solver = MetriplexVQESolver(config)
    history = solver.train_loop(epochs=2, verbose=False)
    assert 'energy' in history
    assert len(history['energy']) == 2
