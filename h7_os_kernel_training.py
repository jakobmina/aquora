"""
h7_os_kernel_training.py
========================
Entrenamiento del Kernel del OS usando Entropía Cuántica Real (20Q).

Este script cierra el bucle:
1. Extrae 721 bits del job 'which-pink-counter'.
2. Mapea estos bits a un espacio de parámetros del kernel (OS State).
3. Usa H7QNNBridge para aplicar el Gradiente Natural Cuántico.
4. Valida la integridad del entrenamiento con el Oráculo Bayesiano H7.

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import numpy as np
import math
import json
import os
from h7_bayesian_oracle import run_extraction_pipeline, H7BayesianOracle
from h7_qnn_bridge import H7QNNBridge, run_h7_qnn_pipeline
from scipy.linalg import inv
import matplotlib.pyplot as plt

def bits_to_floats(bitstream: str, n: int) -> np.ndarray:
    """Convierte un bitstream en un vector de floats en [-1, 1]."""
    # Usamos chunks de 8 bits para cada float
    floats = []
    chunk_size = 8
    for i in range(0, len(bitstream) - chunk_size + 1, chunk_size):
        chunk = bitstream[i:i+chunk_size]
        val = int(chunk, 2) / (2**chunk_size - 1)
        floats.append(val * 2 - 1)
        if len(floats) >= n: break
    return np.array(floats)

def train_os_kernel():
    print("="*70)
    print("  H7 OS KERNEL TRAINING — Quantum Entropy Integration")
    print("="*70)

    # 1. Extracción de bits reales (Sincronizado con 80Q)
    job_id = "sim_h7_80q"
    if os.path.exists("h7_outputs/active_80q_job.txt"):
        with open("h7_outputs/active_80q_job.txt", "r") as f:
            job_id = f.read().strip()
            
    print(f"📡 Usando entropía del Job: {job_id}")
    extraction = run_extraction_pipeline(job_id)
    if not extraction:
        print("❌ Error: No se pudo obtener entropía del job.")
        return

    bits = extraction["bits"]
    n_phys = extraction["n_qubits"]
    
    # 2. Preparar targets de entrenamiento desde los bits
    # Queremos entrenar un componente de d=n_phys dimensiones
    n_samples = 50
    targets = bits_to_floats(bits, n_samples)
    print(f"\n🎯 Generados {len(targets)} targets de entrenamiento desde el bitstream cuántico.")

    # 3. Cargar la covarianza del bridge (usamos la física del job)
    # En un sistema real, cargaríamos Σ desde el archivo .json generado por run_vqe_h7_full.py
    # Para esta demo, simulamos la Σ física de 20Q coherente con los resultados previos.
    Sigma_h7 = np.eye(n_phys) * 0.5
    # Añadimos estructura topológica (diagonal áurea)
    for i in range(n_phys):
        Sigma_h7[i, i] *= abs(math.cos(math.pi * i * (1+math.sqrt(5))/2))

    bridge = H7QNNBridge(Sigma_h7, lambda_metr=0.1)
    oracle = H7BayesianOracle(n_phys)

    # 4. Loop de Entrenamiento Metripléctico
    print("\n🚀 Iniciando entrenamiento del Kernel (Natural Gradient)...")
    weights = np.zeros(n_phys)
    learning_rate = 0.001  # Reducido para evitar explosión
    damping = 1e-2         # Regularización Metripléctica (Regla 1.3)
    history = []

    # X de entrenamiento (simulando sensores del OS)
    X_train = np.random.randn(n_samples, n_phys)

    for epoch in range(15):
        # Gradiente Euclidiano (MSE)
        preds = X_train @ weights
        error = preds - targets
        grad_eucl = (X_train.T @ error) / n_samples
        
        # Corrección H7: GRADIENTE NATURAL con damping metripléctico
        # ∇_nat = (G + λI)⁻¹ ∇_eucl
        Sigma_reg = Sigma_h7 + damping * np.eye(n_phys)
        grad_nat = inv(Sigma_reg) @ grad_eucl
        
        # Update
        weights -= learning_rate * grad_nat
        
        # Monitoreo Bayesiano
        oracle.update(X_train, targets, sigma2=0.1)
        integrity = oracle.get_integrity(X_train)
        loss = np.mean(error**2)
        
        history.append((loss, integrity))
        status = "🟩" if integrity > 0.1 else "🟨" if integrity > 0.01 else "🟥"
        print(f"  Epoch {epoch+1:02d} | Loss: {loss:.6f} | I_H7: {integrity:.6f} | {status}")

    # 5. Resultado Final
    print("\n" + "="*70)
    print(f"  Entrenamiento Finalizado | Pérdida Final: {history[-1][0]:.6f}")
    print(f"  Integridad H7 Final: {history[-1][1]:.6f}")
    print(f"  Estado del Kernel: {'ESTABLE (LAMINAR)' if history[-1][1] > 0.36 else 'INESTABLE'}")
    print("="*70)

    # Generar reporte visual (opcional)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([h[0] for h in history], 'b-o')
    plt.title("Pérdida (MSE)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([h[1] for h in history], 'g-o')
    plt.axhline(y=0.3623, color='r', linestyle='--', label='O_n Integrity')
    plt.title("Integridad H7")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs("h7_outputs", exist_ok=True)
    plt.savefig("h7_outputs/h7_kernel_training_report.png")
    print("\n📈 Reporte visual guardado: h7_outputs/h7_kernel_training_report.png")

if __name__ == "__main__":
    train_os_kernel()
