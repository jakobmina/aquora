"""
h7_auto_governor.py
===================
Gobernador Automático Metripléctico (Auto-Governor).

Este sistema cierra el bucle de control:
1. Monitorea el radar cuántico (Jitter del sistema).
2. Si la integridad H7 cae o la turbulencia sube, aplica una
   fuerza disipativa (frenado) para estabilizar el kernel.
3. Si el sistema es laminar, libera recursos (fuerza conservativa).

Regla 1.3: Prohibición de singularidades (ni explosión ni muerte térmica).

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import numpy as np
import time
import math
from h7_quantum_radar import h7_pulse, collect_radar_echoes
from h7_classifier_prototype import H7Classifier
from h7_bayesian_oracle import run_extraction_pipeline, O_n
import matplotlib.pyplot as plt

class H7AutoGovernor:
    def __init__(self, bitstream: str, n_features: int = 10):
        self.detector = H7Classifier(n_features, bitstream)
        self.load_factor = 1.0  # El parámetro a regular (1.0 = 100% throughput)
        self.history = []
        self.n_features = n_features

    def regulate(self, echo_window: np.ndarray):
        """Ajusta el load_factor basado en la coherencia detectada."""
        # Predecir probabilidad de coherencia (Laminosidad)
        X = echo_window.reshape(1, -1)
        laminarity = self.detector.predict_proba(X)[0]
        
        # Lógica de Gobernanza Metripléctica:
        # Si laminarity > 0.6 -> El sistema está sano -> Podemos subir carga
        # Si laminarity < 0.4 -> El sistema está turbulento -> Debemos bajar carga
        
        target_load = laminarity * 1.5 # El factor áureo de carga
        
        # Suavizado (damping metripléctico)
        alpha = 0.2
        self.load_factor = (1 - alpha) * self.load_factor + alpha * target_load
        
        # Límites de seguridad (Regla 1.3)
        self.load_factor = np.clip(self.load_factor, 0.1, 2.0)
        
        return laminarity, self.load_factor

def run_governor_live(cycles: int = 40):
    print("="*70)
    print("  H7 AUTO-GOVERNOR — Thermodynamic Closed-Loop Control")
    print("="*70)

    # 1. Cargar Prior Cuántico (20Q)
    job_id = "which-pink-counter"
    ext = run_extraction_pipeline(job_id)
    if not ext: return
    
    governor = H7AutoGovernor(ext["bits"])
    n_feat = governor.n_features

    print(f"\n🧠 Gobernador inicializado. Load Factor inicial: {governor.load_factor}")
    print("🔄 Iniciando ciclo de regulación en tiempo real...")
    
    reg_history = []
    
    for i in range(cycles):
        # Capturar ventana de radar (10 ecos)
        echoes = []
        for _ in range(n_feat):
            echoes.append(h7_pulse())
            time.sleep(0.005) # Simular carga variable
        
        # Normalizar ventana
        echo_window = np.array(echoes, dtype=float)
        echo_window = (echo_window - np.mean(echo_window)) / (np.std(echo_window) + 1e-12)
        
        # Regulación
        lam, load = governor.regulate(echo_window)
        reg_history.append((lam, load))
        
        # Visual de consola
        bar_len = int(load * 20)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        status = "LAMINAR" if lam > 0.5 else "TURBULENTO"
        print(f"Cycle {i+1:02d} | {status:<10} | Lam: {lam:.3f} | Load: {load:.3f} | [{bar}]")
        
        # Simular interferencia externa cada 15 ciclos
        if i % 15 == 0 and i > 0:
            print("⚠️ [SISTEMA] Inyectando Interferencia de Red...")
            time.sleep(0.1) # Causar jitter masivo

    # 5. Visualización del Bucle Cerrado
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot([h[0] for h in reg_history], 'g-', label='Laminosidad (Radar)')
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.title("Feedback del Radar Cuántico")
    plt.legend(); plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot([h[1] for h in reg_history], 'b-', label='Load Factor (Regulación)')
    plt.title("Gobernador Metripléctico: Ajuste de Carga")
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("h7_auto_governor_loop.png")
    print("\n📈 Bucle de regulación guardado: h7_auto_governor_loop.png")
    print("="*70)

if __name__ == "__main__":
    run_governor_live()
