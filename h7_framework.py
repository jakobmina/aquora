import marimo as mo 
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, entropy

print("Dependeces Ready") # IMPROVED SECTIONS FOR H7 LOGIC NOTEBOOK
## Explicit Endianness + JSON Reading Enhancement


## SECTION A: Quantum Circuit + Endianness Documentation
# =============================================================================
# QUANTUM CIRCUIT SIMULATION WITH EXPLICIT ENDIANNESS
# =============================================================================

def run_quantum_experiment(n_param, phi_param):
    """
    Quantum simulation with EXPLICIT little-endian convention.

    ENDIANNESS CONVENTION (Little-Endian / Qiskit Default):
    ├─ State representation: |q_2 q_1 q_0⟩
    ├─ Qubit 0 = LSB (rightmost in binary string)
    ├─ Qubit n-1 = MSB (leftmost in binary string)
    ├─ Classical bits indexed 0...n-1 (LSB first)
    └─ Probability dict keys: '000'...'111' where bit[i] = qubit[i]
    """
    from qiskit import QuantumCircuit
    from qiskit.visualization import plot_histogram, plot_bloch_multivector
    from qiskit.quantum_info import Statevector
    import matplotlib.pyplot as plt

    # Constants
    PI = np.pi
    COS = np.cos

    # =========================================================================
    # PART 1: Particle Classification
    # =========================================================================

    parity = COS(PI * n_param)
    quasiperiod = COS(PI * phi_param * n_param)
    golden = parity * quasiperiod

    def classify_particle(golden_val):
        """Classify particle type from golden value."""
        if golden_val < 0.1:
            return "fermionic"
        elif golden_val > 0.1:
            return "bosonic"
        else:
            return "unknown"

    particle_type = classify_particle(golden)
    print("--- Particle Classification Results ---")
    print(f"  Parity: {parity:.6f}")
    print(f"  Quasiperiod: {quasiperiod:.6f}")
    print(f"  Chiral (golden): {golden:.6f}")
    print(f"  Particle Type: {particle_type}")
    print("\n" + "="*40 + "\n")

    # =========================================================================
    # PART 2: Quantum Circuit Simulation (LITTLE-ENDIAN EXPLICIT)
    # =========================================================================
    print("--- Quantum Circuit Simulation Results ---")
    print("(Convention: Little-endian, Qiskit default)\n")

    # Initialize 3 qubits, 3 classical bits
    qc = QuantumCircuit(3, 3, name="H7_Circuit")

    # Apply Hadamard gates (uniform superposition)
    qc.h([0, 1, 2])

    # Apply rotation gates
    # φ parameter as radian angle (not cosine)
    rotation_angle = PI * phi_param  # Radianes, no coseno del anterior
    qc.rz(rotation_angle, 0)
    qc.ry(rotation_angle, 1)
    qc.rx(0, 2)  # Identity on q2

    # Two-qubit and three-qubit gates
    qc.cswap(0, 2, 1)  # Controlled-SWAP: control=0, swap (2,1)
    qc.ccx(2, 1, 0)    # Toffoli: controls=(2,1), target=0

    print("  Quantum Circuit Diagram (before measurement):")
    print(qc.draw())
    print(f"  Operation counts: {qc.count_ops()}\n")

    # Get statevector before measurement
    psi = Statevector.from_instruction(qc)
    print("  State Vector (amplitude vector):")
    print(psi.data)
    print("\n  State Vector (text representation):")
    print(psi.draw('text'))

    # Probabilities
    probs_dict = psi.probabilities_dict()
    print("\n  Probabilities per state:")
    print(f"  {probs_dict}")

    # Marginal probabilities for each qubit
    print("\n  Marginal Probabilities (Little-Endian):")
    probs_q0 = psi.probabilities([0])  # Qubit 0 (LSB)
    print(f"    q0 (LSB): {probs_q0}")
    probs_q1 = psi.probabilities([1])
    print(f"    q1 (mid): {probs_q1}")
    probs_q2 = psi.probabilities([2])  # Qubit 2 (MSB)
    print(f"    q2 (MSB): {probs_q2}")

    # Measurement (EXPLICIT LITTLE-ENDIAN MAPPING)
    print("\n  Adding measurements (little-endian: qubit i → cbit i)...")
    qc.measure([0, 1, 2], [0, 1, 2])

    print("\n  Quantum Circuit Diagram (after measurement):")
    print(qc.draw())
    print("\n" + "="*40 + "\n")

    return qc, psi, probs_dict, particle_type


def interpret_measured_bitstring(bitstring, convention='little'):
    """
    Interpret a measured bitstring according to endianness.
    """
    n_qubits = len(bitstring)

    if convention == 'little':
        # bitstring[i] is qubit i (Qiskit default)
        qubit_states = {i: int(bitstring[i]) for i in range(n_qubits)}
        # Interpret as binary: rightmost bit (q0) is LSB
        decimal = int(bitstring[::-1], 2)
    elif convention == 'big':
        # bitstring[i] is qubit (n-1-i) (big-endian)
        qubit_states = {i: int(bitstring[n_qubits-1-i]) for i in range(n_qubits)}
        # Interpret as binary: leftmost bit (q_n-1) is MSB
        decimal = int(bitstring, 2)
    else:
        raise ValueError(f"Unknown convention: {convention}")

    return {
        'convention': convention,
        'qubit_states': qubit_states,
        'decimal_value': decimal,
        'bitstring': bitstring
    }

## SECTION B: Improved JSON Extraction (Auto-detect + NDJSON)

import json
import os
import pandas as pd
from pathlib import Path

def extract_json_auto(json_file):
    """Auto-detect JSON vs NDJSON format."""
    data = []
    filename = os.path.basename(json_file)

    print(f"  Reading {filename}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            line_count = 0
            error_count = 0

            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        data.append(obj)
                        line_count += 1
                    else:
                        error_count += 1
                        print(f"    ⚠ Line {line_num}: Expected dict, got {type(obj).__name__}")
                except json.JSONDecodeError as e:
                    error_count += 1
                    if line_num <= 3 or error_count <= 3:
                        print(f"    ⚠ Line {line_num} malformed: {str(e)[:60]}")

        if data:
            df = pd.DataFrame(data)
            print(f"    ✓ NDJSON: Loaded {line_count} objects")
            return df
    except Exception as e:
        print(f"    ⚠ NDJSON parse failed: {str(e)[:60]}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            content = json.load(f)

        if isinstance(content, list):
            data = content
        elif isinstance(content, dict):
            data = [content]
        else:
            return None

        df = pd.DataFrame(data)
        print(f"    ✓ JSON: Loaded {len(data)} objects")
        return df

    except Exception:
        return None

def flatten_json_smart(df, sep='_', max_depth=5):
    """Flatten nested JSON structures recursively."""
    print(f"\n  Flattening (max_depth={max_depth})...")

    for depth in range(max_depth):
        exploded_any = False
        original_shape = df.shape

        for col in list(df.columns):
            if df[col].dtype == 'object':
                sample = None
                for val in df[col]:
                    if val is not None:
                        sample = val
                        break

                if isinstance(sample, dict):
                    expanded = df[col].apply(pd.Series)
                    expanded.columns = [f"{col}{sep}{subcol}" for subcol in expanded.columns]
                    df = pd.concat([df.drop(col, axis=1), expanded], axis=1)
                    exploded_any = True
                    break 

                elif isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], dict):
                    df = df.explode(col, ignore_index=False).reset_index(drop=True)
                    expanded = df[col].apply(pd.Series)
                    expanded.columns = [f"{col}{sep}{subcol}" for subcol in expanded.columns]
                    df = pd.concat([df.drop(col, axis=1), expanded], axis=1)
                    exploded_any = True
                    break

        if not exploded_any:
            break

    numeric_cols = []
    alpha_cols = []

    for col in df.columns:
        if isinstance(col, str) and col.isdigit():
            numeric_cols.append(int(col))
        else:
            alpha_cols.append(col)

    numeric_cols.sort()
    alpha_cols.sort()

    final_cols = [str(c) for c in numeric_cols] + alpha_cols
    df = df[final_cols]

    return df

def find_json_files(input_dir, include_ndjson=True):
    """Find all JSON/NDJSON files in directory."""
    patterns = ['*.json']
    if include_ndjson:
        patterns.append('*.ndjson')

    files = []
    for pattern in patterns:
        files.extend(Path(input_dir).glob(pattern))

    return sorted([str(f) for f in files])

class OptimizationMode(Enum):
    MOLECULAR = "molecular"
    GENERIC = "generic"

@dataclass
class SolverConfig:
    n_qubits: int = 3
    base_epsilon: float = 1e-4
    phi_phase: float = 0.362
    learning_rate: float = 0.05
    entropy_scaling: float = 0.1
    covariance_momentum: float = 0.9
    mode: OptimizationMode = OptimizationMode.MOLECULAR
    target_bond_length: float = 0.74
    energy_penalty_scale: float = 1.0

class QuaternionMetrics:
    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm

    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
        w, x, y, z = q
        roll_x = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
        pitch_y = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw_z = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        return roll_x, pitch_y, yaw_z

    @staticmethod
    def euler_to_quaternion(roll_x: float, pitch_y: float, yaw_z: float) -> np.ndarray:
        cy = np.cos(yaw_z * 0.5)
        sy = np.sin(yaw_z * 0.5)
        cp = np.cos(pitch_y * 0.5)
        sp = np.sin(pitch_y * 0.5)
        cr = np.cos(roll_x * 0.5)
        sr = np.sin(roll_x * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

class MetriplexVQESolver:
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.n_qubits = self.config.n_qubits
        self.q_weights = np.array([1.0, 0.0, 0.0, 0.0])
        self.covariance = np.eye(4)
        self.history: Dict[str, List[float]] = {
            'energy': [], 'entropy': [], 'epsilon': [],
            'bond_length': [], 'norm_gradient': []
        }
        self._iteration_count = 0

    def build_ansatz(self, euler_angles: Tuple[float, float, float]) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, name="H7_Ansatz")
        roll_x, pitch_y, yaw_z = euler_angles

        for q in range(self.n_qubits):
            qc.rx(roll_x, q)
            qc.ry(pitch_y, q)
            qc.rz(yaw_z, q)

        qc.cswap(0, 2, 1)
        qc.ccx(2, 1, 0)
        return qc

    @staticmethod
    def _reverse_endianness(state_le: Statevector) -> Statevector:
        return state_le.reverse_qargs()

    def evaluate_energy(self, state_be: Statevector, bond_length: float) -> float:
        target_r = self.config.target_bond_length
        energy_penalty = (bond_length - target_r) ** 2 * self.config.energy_penalty_scale
        p_zero = np.abs(state_be.data[0]) ** 2
        entanglement_bonus = (1.0 - p_zero) * 0.5
        simulated_energy = -1.1 + energy_penalty - entanglement_bonus
        return simulated_energy

    def _compute_adaptive_regularization(self, entropy_vn: float) -> float:
        epsilon = (
            self.config.base_epsilon +
            self.config.phi_phase * entropy_vn * self.config.entropy_scaling
        )
        return epsilon

    def _update_quaternion_weights(self, energy: float, dynamic_epsilon: float) -> float:
        try:
            cov_inv = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.inv(self.covariance + dynamic_epsilon * np.eye(4))

        q_gradient = np.dot(cov_inv, self.q_weights) * energy * self.config.learning_rate
        norm_gradient = np.linalg.norm(q_gradient)

        self.q_weights -= q_gradient
        self.q_weights = QuaternionMetrics.normalize(self.q_weights)

        return norm_gradient

    def _update_covariance(self, q_gradient: np.ndarray) -> None:
        outer_prod = np.outer(q_gradient, q_gradient)
        self.covariance = (
            self.config.covariance_momentum * self.covariance +
            (1.0 - self.config.covariance_momentum) * outer_prod
        )

    def train_loop(self, initial_bond_length: float = 1.5, epochs: int = 50, verbose: bool = True) -> Dict[str, List[float]]:
        current_bond_length = initial_bond_length

        for epoch in range(epochs):
            self._iteration_count = epoch
            euler_angles = QuaternionMetrics.quaternion_to_euler(self.q_weights)
            qc = self.build_ansatz(euler_angles)
            state_le = Statevector.from_instruction(qc)
            state_be = self._reverse_endianness(state_le)

            rho = DensityMatrix(state_be)
            entropy_vn = entropy(rho)
            dynamic_epsilon = self._compute_adaptive_regularization(entropy_vn)
            energy = self.evaluate_energy(state_be, current_bond_length)
            norm_grad = self._update_quaternion_weights(energy, dynamic_epsilon)

            q_gradient = np.dot(
                np.linalg.inv(self.covariance + dynamic_epsilon * np.eye(4)),
                self.q_weights
            ) * energy * self.config.learning_rate
            self._update_covariance(q_gradient)

            gradient_bond = 2.0 * (current_bond_length - self.config.target_bond_length)
            current_bond_length -= self.config.learning_rate * gradient_bond * 0.1

            self.history['energy'].append(float(energy))
            self.history['entropy'].append(float(entropy_vn))
            self.history['epsilon'].append(float(dynamic_epsilon))
            self.history['bond_length'].append(float(current_bond_length))
            self.history['norm_gradient'].append(float(norm_grad))

        return self.history

    def get_final_state(self) -> Statevector:
        euler_angles = QuaternionMetrics.quaternion_to_euler(self.q_weights)
        qc = self.build_ansatz(euler_angles)
        state_le = Statevector.from_instruction(qc)
        return self._reverse_endianness(state_le)

    def get_final_params(self) -> Dict[str, np.ndarray]:
        euler_angles = QuaternionMetrics.quaternion_to_euler(self.q_weights)
        return {
            'quaternion': self.q_weights.copy(),
            'euler_angles': np.array(euler_angles)
        }

if __name__ == "__main__":
    qc, psi, probs, ptype = run_quantum_experiment(n_param=1, phi_param=1.618)
    solver = MetriplexVQESolver()
    history = solver.train_loop(epochs=10, verbose=False)
