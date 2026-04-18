import numpy as np
from h7_framework import (
    MetriplexVQESolver,
    SolverConfig,
    OptimizationMode,
    QuaternionMetrics
)

# =============================================================================
# EXAMPLE 1: Basic H2 Molecular Optimization
# =============================================================================

def example_h2_basic():
    """
    Simplest usage: optimize H2 from stretched geometry.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic H2 Optimization")
    print("="*80)

    # Use default config for H2
    config = SolverConfig(
        n_qubits=3,
        target_bond_length=0.74,  # Ångströms
        learning_rate=0.05
    )

    solver = MetriplexVQESolver(config)

    # Start with stretched molecule
    history = solver.train_loop(
        initial_bond_length=1.5,
        epochs=30,
        verbose=True
    )

    # Print final results
    final_params = solver.get_final_params()
    print("Final parameters:")
    print(f"  Quaternion: {final_params['quaternion']}")
    print(f"  Euler angles: {final_params['euler_angles']}")
    print(f"  Final energy: {history['energy'][-1]:.6f}")
    print(f"  Final bond: {history['bond_length'][-1]:.4f} Å")


# =============================================================================
# EXAMPLE 2: Custom Configuration with LiH Molecule
# =============================================================================

def example_lih_custom():
    """
    Advanced usage: custom configuration for LiH molecule.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom LiH Optimization")
    print("="*80)

    # LiH equilibrium bond length ~1.64 Ångströms
    config = SolverConfig(
        n_qubits=4,  # More qubits for larger molecule
        target_bond_length=1.64,
        learning_rate=0.03,  # Smaller learning rate for stability
        entropy_scaling=0.15,
        base_epsilon=2e-4,
        covariance_momentum=0.95  # Higher momentum (inertia)
    )

    solver = MetriplexVQESolver(config)

    # Optimize from compressed state
    history = solver.train_loop(
        initial_bond_length=1.0,
        epochs=50,
        verbose=True
    )

    # Analyze convergence
    energies = history['energy']
    print(f"Convergence analysis:")
    print(f"  Initial energy: {energies[0]:.6f}")
    print(f"  Final energy: {energies[-1]:.6f}")
    print(f"  Energy improvement: {energies[0] - energies[-1]:.6f}")
    print(f"  Entropy (final): {history['entropy'][-1]:.4f}")


# =============================================================================
# EXAMPLE 3: Quaternion Mathematics
# =============================================================================

def example_quaternion_operations():
    """
    Demonstrate quaternion utilities for SU(2) operations.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Quaternion Operations")
    print("="*80)

    # Identity quaternion (no rotation)
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    print(f"Identity quaternion: {q_identity}")

    # Convert to Euler angles
    euler_id = QuaternionMetrics.quaternion_to_euler(q_identity)
    print(f"  → Euler angles: {euler_id} (should be ~[0, 0, 0])")

    # Random quaternion
    q_random = np.random.randn(4)
    q_random = QuaternionMetrics.normalize(q_random)
    print(f"Random quaternion (normalized): {q_random}")
    print(f"  Norm: {np.linalg.norm(q_random):.6f} (should be 1.0)")

    # Convert to Euler
    euler_random = QuaternionMetrics.quaternion_to_euler(q_random)
    print(f"  → Euler angles: {euler_random}")

    # Convert back to quaternion
    q_reconstructed = QuaternionMetrics.euler_to_quaternion(*euler_random)
    q_reconstructed = QuaternionMetrics.normalize(q_reconstructed)
    print(f"  → Reconstructed quaternion: {q_reconstructed}")

    # Check round-trip fidelity
    fidelity = np.dot(q_random, q_reconstructed)
    print(f"  Fidelity (⟨q|q'⟩): {fidelity:.6f} (should be ~1.0)")


# =============================================================================
# EXAMPLE 4: Monitoring Training History
# =============================================================================

def example_training_analysis():
    """
    Run optimization and analyze detailed training history.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Training History Analysis")
    print("="*80)

    config = SolverConfig(n_qubits=3, learning_rate=0.05)
    solver = MetriplexVQESolver(config)

    history = solver.train_loop(
        initial_bond_length=1.5,
        epochs=50,
        verbose=False  # Quiet mode
    )

    # Analyze energy convergence
    energies = np.array(history['energy'])
    print(f"Energy statistics:")
    print(f"  Mean: {np.mean(energies):.6f}")
    print(f"  Std dev: {np.std(energies):.6f}")
    print(f"  Min: {np.min(energies):.6f}")
    print(f"  Max: {np.max(energies):.6f}")

    # Convergence rate
    energy_diffs = np.diff(energies)
    print(f"Convergence:")
    print(f"  Avg energy change per epoch: {np.mean(energy_diffs):.6f}")
    print(f"  Last 10 epochs avg change: {np.mean(energy_diffs[-10:]):.6f}")

    # Entropy trends
    entropies = np.array(history['entropy'])
    print(f"Entanglement (von Neumann entropy):")
    print(f"  Initial: {entropies[0]:.4f}")
    print(f"  Final: {entropies[-1]:.4f}")
    print(f"  Peak: {np.max(entropies):.4f} (at epoch {np.argmax(entropies)})")

    # Regularization parameter evolution
    epsilons = np.array(history['epsilon'])
    print(f"Tikhonov regularization (ε):")
    print(f"  Min: {np.min(epsilons):.8f}")
    print(f"  Max: {np.max(epsilons):.8f}")
    print(f"  Current: {epsilons[-1]:.8f}")

    # Gradient norms
    gradients = np.array(history['norm_gradient'])
    print(f"Gradient norms:")
    print(f"  Initial: {gradients[0]:.6f}")
    print(f"  Final: {gradients[-1]:.6f}")
    print(f"  Trend: {'Decreasing' if gradients[-1] < gradients[0] else 'Unstable'}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("METRIPLEX VQE SOLVER - EXAMPLES")
    print("="*80)

    # Run examples
    print("--- RUNNING EXAMPLES ---")
    example_h2_basic()
    example_quaternion_operations()
    example_training_analysis()
