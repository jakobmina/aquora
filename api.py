from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import os
import random

# Import our custom kernel wrapper and H7 framework
from core_physics.h7_wrapper import CKernelWrapper
from h7_framework import SolverConfig, MetriplexVQESolver
from metriplex_bridge import MetriplexEndianBridge

app = FastAPI(title="H7 QNN Experiment API")

# Mount static files for UI
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# State for the API
class AppState:
    def __init__(self):
        self.config = SolverConfig(n_qubits=3, target_bond_length=0.74, learning_rate=0.05)
        self.solver = MetriplexVQESolver(self.config)
        self.bridge = MetriplexEndianBridge()
        self.epochs = 0
        self.current_bond_length = 1.5
        self.entropy = 0.0
        self.energy = 0.0
        self.norm_grad = 0.0
        self.dynamic_epsilon = self.config.base_epsilon
        self.golden_ratio = 0.0
        self.hex_state_big_endian = "0000"
        self.hex_state_little_endian = "0000"
        self.L_symp = 0.0
        self.L_metr = 0.0
        self.drift = 0.0
        self.particle_type = "unknown"
        self.chirality = 0.0
        self.pair_overlaps = [0.0, 0.0, 0.0, 0.0]
        self.logs = ["[INIT] H7 QNN v0.1 — API initialized"]

state = AppState()

@app.get("/")
def read_root():
    return FileResponse("ui/templates/index.html")

@app.post("/api/reset")
def reset_state():
    state.__init__()
    state.logs.append("[RESET] System state reset to initial conditions")
    return get_metrics()

@app.post("/api/roll")
def roll_dice():
    # Generate random occupation
    n0 = random.randint(0, 2)
    n1 = random.randint(0, 2)
    n2 = random.randint(0, 2)
    occupation = (n0, n1, n2)
    
    # Use the bridge as the single source of truth
    report = state.bridge.full_state_report(occupation)
    
    state.golden_ratio = report['o_n']
    state.particle_type = report['particle_type']
    
    # H7 overlaps from the bridge
    # For UI display, we still use the 4-pair array from compute_h7_pair_overlaps
    from h7_quaternion import compute_h7_pair_overlaps, H7QuaternionMapper, H7_AMPLITUDES
    phi = 0.618
    state.pair_overlaps = compute_h7_pair_overlaps(phi).tolist()
    
    # But we update the current state's log with the bridge's specific overlap
    state.logs.append(f"[BRIDGE] Overlap for p={report['momentum_p']}: {report['vacuum_overlap']:.4f}")

    mapper = H7QuaternionMapper(H7_AMPLITUDES)
    h7_rep = mapper.analyze(phi_param=phi)
    state.chirality = h7_rep["chirality"]

    state.hex_state_big_endian = report['hex_uint16']
    state.hex_state_little_endian = state.hex_state_big_endian[2:] + state.hex_state_big_endian[:2]

    state.L_symp = report['L_symp']
    state.L_metr = report['L_metr']
    state.drift = report['ratio'] - 2.618034
    
    state.logs.append(f"[ROLL] Type={state.particle_type}, Hex={state.hex_state_big_endian}")
    return get_metrics()

@app.post("/api/epoch")
def run_epoch():
    # Run a single epoch using the C kernel
    from core_physics.h7_wrapper import CKernelWrapper
    euler_angles = CKernelWrapper.quaternion_to_euler(state.solver.q_weights)
    qc = state.solver.build_ansatz(euler_angles)
    
    from qiskit.quantum_info import Statevector, DensityMatrix, entropy
    state_le = Statevector.from_instruction(qc)
    state_be = state.solver._reverse_endianness(state_le)
    
    rho = DensityMatrix(state_be)
    state.entropy = float(entropy(rho))
    state.dynamic_epsilon = float(state.solver._compute_adaptive_regularization(state.entropy))
    state.energy = float(state.solver.evaluate_energy(state_be, state.current_bond_length))
    
    # Use C Kernel for update
    state.norm_grad = float(CKernelWrapper.update_weights(
        state.solver.q_weights, 
        state.solver.covariance, 
        state.energy, 
        state.config.learning_rate, 
        state.dynamic_epsilon
    ))
    
    # Update covariance
    q_inv = np.linalg.inv(state.solver.covariance + state.dynamic_epsilon * np.eye(4))
    q_gradient = np.dot(q_inv, state.solver.q_weights) * state.energy * state.config.learning_rate
    state.solver._update_covariance(q_gradient)
    
    # Update bond length
    gradient_bond = 2.0 * (state.current_bond_length - state.config.target_bond_length)
    state.current_bond_length -= state.config.learning_rate * gradient_bond * 0.1
    
    state.epochs += 1
    
    # Bridge Lagrangian
    L_symp, L_metr = state.bridge.compute_lagrangian()
    state.L_symp = L_symp
    state.L_metr = L_metr
    state.drift = state.norm_grad - state.entropy
    
    state.logs.append(f"[{state.epochs}] Epoch complete: E={state.energy:.4f}, |grad|={state.norm_grad:.4f}")
    if len(state.logs) > 10:
        state.logs.pop(0)
        
    return get_metrics()

@app.get("/api/metrics")
def get_metrics():
    return {
        "epochs": state.epochs,
        "energy": state.energy,
        "entropy": state.entropy,
        "drift": state.drift,
        "golden_ratio": state.golden_ratio,
        "mahalanobis": state.norm_grad,
        "bond_length": state.current_bond_length,
        "L_symp": state.L_symp,
        "L_metr": state.L_metr,
        "hex_be": state.hex_state_big_endian,
        "hex_le": state.hex_state_little_endian,
        "particle_type": state.particle_type,
        "chirality": state.chirality,
        "pair_overlaps": state.pair_overlaps,
        "logs": state.logs
    }
