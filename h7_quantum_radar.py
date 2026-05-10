"""
h7_quantum_radar.py
===================
Prototipo de Radar Cuántico H7 (Gobernanza de Entorno).

Este módulo:
1. Emite 'pulsos' (ejecución de micro-kernels H7).
2. Mide la 'reflexión' (jitter de latencia del sistema local).
3. Utiliza el H7-Classifier (sembrado con 20Q-Entropy) para detectar
   estados de coherencia en el ruido del hardware.

Analía: Cada latido del CPU es una sonda; el jitter es la interferencia
del entorno. El Radar H7 detecta si el sistema está en 'Flujo Laminar'.

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import numpy as np
import time
import math
from h7_classifier_prototype import H7Classifier
from h7_bayesian_oracle import run_extraction_pipeline, O_n
import matplotlib.pyplot as plt

def h7_quantum_pulse(n_qubits: int = 3):
    """
    Simula un pulso cuántico donde el jitter del sistema modula la fase.
    Cada 'latido' provoca un colapso de información en la asimetría.
    """
    # 1. Capturar el 'ruido' ambiental (el jitter)
    t1 = time.perf_counter_ns()
    time.sleep(0.0001)
    t2 = time.perf_counter_ns()
    jitter = (t2 - t1) % 1000 / 1000.0  # Normalizado 0-1
    
    # 2. El 'Circuito': Superposición modulada por el jitter
    # En un sistema real, esto sería un Job en q3as
    n_states = 2**n_qubits
    phi = 2 * np.pi * jitter  # Fase inducida por el hardware
    
    # Generar amplitudes con asimetría cuántica
    # La fase PHI interfiere con la base H7 (Golden Ratio)
    theta = np.linspace(0, 2*np.pi, n_states)
    amplitudes = np.cos(theta + phi) + 1j * np.sin(theta * 1.618)
    probabilities = np.abs(amplitudes)**2
    probabilities /= np.sum(probabilities)  # Normalización (Colapso)
    
    # 3. Extraer la Asimetría (La Distancia Implícita)
    # A = P_max - P_min
    asymmetry = np.max(probabilities) - np.min(probabilities)
    
    return asymmetry, probabilities

def collect_radar_echoes(n_pulses: int = 200):
    """Recolecta los ecos del radar (latencia del sistema)."""
    print(f"📡 Emitiendo {n_pulses} pulsos de radar al sistema...")
    echoes = []
    for _ in range(n_pulses):
        asym, _ = h7_quantum_pulse()
        echoes.append(asym)
        # Pequeño jitter aleatorio de espera para no saturar el scheduler
        time.sleep(0.001)
    
    # Normalizar ecos (nanosegundos a escala relativa)
    echoes = np.array(echoes, dtype=float)
    mean = np.mean(echoes)
    std = np.std(echoes)
    # Z-score normalization para el clasificador
    return (echoes - mean) / (std + 1e-12)

def run_quantum_radar():
    print("="*70)
    print("  H7 QUANTUM RADAR — Environmental Noise Detection")
    print("="*70)

    # 1. Obtener la 'Firma Cuántica' (20Q Entropy)
    job_id = "which-pink-counter"
    extraction = run_extraction_pipeline(job_id)
    if not extraction: return
    bits = extraction["bits"]

    # 2. Recolectar Ecos Reales (Jitter del CPU)
    n_features = 10
    raw_echoes = collect_radar_echoes(n_pulses=300)
    
    # 3. Preparar Ventanas de Radar
    # Cada ventana de 10 ecos es una 'observación' del radar
    X_radar = []
    for i in range(0, len(raw_echoes) - n_features, 5):
        X_radar.append(raw_echoes[i:i+n_features])
    X_radar = np.array(X_radar)
    
    print(f"📊 Radar activo: {len(X_radar)} ventanas de escaneo capturadas.")

    # 4. Inicializar Clasificador como 'Detector de Radar'
    # Entrenamos el detector para distinguir entre:
    # Clase 0: Ruido de Fondo (Normalizado)
    # Clase 1: 'Target' Detectado (Simulamos una anomalía en los ecos)
    detector = H7Classifier(n_features, bits)
    
    # Creamos un target sintético: inyectamos un 'pulso áureo' en algunas ventanas
    y_radar = []
    for i in range(len(X_radar)):
        if i % 7 == 0: # El target aparece cada 7 ventanas
            X_radar[i] += 0.5 * np.array([O_n(j) for j in range(n_features)])
            y_radar.append(1)
        else:
            y_radar.append(0)
    y_radar = np.array(y_radar)

    # 5. Escaneo y Entrenamiento en Tiempo Real
    print("\n🔍 Escaneando entorno en busca de coherencia H7...")
    losses = []
    for epoch in range(30):
        loss = detector.train_step(X_radar, y_radar, lr=0.05)
        losses.append(loss)
    
    # 6. Detección
    probas = detector.predict_proba(X_radar)
    detections = (probas > 0.6).astype(int)
    
    print(f"\n✅ Escaneo completado.")
    print(f"   Objetivos detectados: {np.sum(detections)}")
    print(f"   Falsa alarma (noise): {np.sum(detections[y_radar==0])}")
    print(f"   Detecciones Reales:  {np.sum(detections[y_radar==1])}/{np.sum(y_radar)}")

    # Visualización del Radar
    plt.figure(figsize=(12, 5))
    plt.plot(probas, 'b-', alpha=0.6, label='Firma del Radar (Proba)')
    plt.scatter(np.where(y_radar==1)[0], probas[y_radar==1], color='red', marker='x', label='Target H7')
    plt.axhline(y=0.6, color='r', linestyle='--', label='Umbral de Detección')
    plt.title("H7 Quantum Radar: Detección de Coherencia en Jitter de CPU")
    plt.xlabel("Ventana de Tiempo")
    plt.ylabel("Probabilidad de Coherencia")
    plt.legend()
    plt.grid(True)
    
    import os
    os.makedirs("h7_outputs", exist_ok=True)
    plt.savefig("h7_outputs/h7_quantum_radar_echoes.png")
    print("\n📈 Visualización del Radar guardada: h7_outputs/h7_quantum_radar_echoes.png")
    print("="*70)

if __name__ == "__main__":
    run_quantum_radar()
