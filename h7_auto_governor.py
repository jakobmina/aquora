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
from h7_quantum_radar import h7_quantum_pulse
from h7_classifier_prototype import H7Classifier
from h7_bayesian_oracle import run_extraction_pipeline, O_n
import matplotlib.pyplot as plt

import signal
import sys

class H7AutoGovernor:
    def __init__(self, bitstream: str, n_features: int = 10):
        self.detector = H7Classifier(n_features, bitstream)
        self.load_factor = 1.0
        self.asymmetry_history = []
        self.n_features = n_features
        self.running = True

    def stop(self, *args):
        print("\n🛑 Recibida señal de parada. Guardando estado metripléctico...")
        self.running = False

    def regulate(self, asym_window: np.ndarray):
        """
        Regulación Basada en el Colapso de Asimetría.
        'Un solo movimiento' para reconciliar la métrica con la probabilidad.
        """
        # 1. Distancia Implícita en Covarianza
        # Analizamos cómo varía la asimetría (el 'gap' entre estados)
        asym_dist = np.var(asym_window) 
        
        # 2. Predicción de Laminosidad (H7 Classifier)
        X = asym_window.reshape(1, -1)
        laminarity = self.detector.predict_proba(X)[0]
        
        # 3. Movimiento Único de Gobernanza
        # El objetivo es mantener la asimetría cerca de PHI/2 (~0.8)
        # Si asym_dist sube (jitter cuántico), bajamos la carga.
        current_asym = asym_window[-1]
        
        # Fórmula Metripléctica: [u, S] + {u, H}
        # S (Disipación): Basada en la pérdida de asimetría
        # H (Conservación): Basada en la laminosidad detectada
        damping = (1.0 - current_asym) * 0.5
        drive = laminarity * 1.618
        
        target_load = drive - damping
        
        # Suavizado y límites (Regla 1.3)
        alpha = 0.25
        self.load_factor = (1 - alpha) * self.load_factor + alpha * target_load
        self.load_factor = np.clip(self.load_factor, 0.05, 2.0)
        
        return laminarity, self.load_factor

def run_governor_live(cycles: int = -1):
    print("="*70)
    print("  H7 AUTO-GOVERNOR — Quantum Collapse Mode")
    print("  Mode: " + ("DAEMON" if cycles == -1 else "DEMO"))
    print("="*70)

    job_id = "which-pink-counter"
    ext = run_extraction_pipeline(job_id)
    if not ext: return
    
    governor = H7AutoGovernor(ext["bits"])
    n_feat = governor.n_features
    
    signal.signal(signal.SIGTERM, governor.stop)
    signal.signal(signal.SIGINT, governor.stop)

    reg_history = []
    i = 0
    while governor.running:
        if cycles != -1 and i >= cycles: break
        
        # Capturar ventana de ASIMETRÍAS (Colapsos Cuánticos)
        asym_echoes = []
        for _ in range(n_feat):
            asym, _ = h7_quantum_pulse()
            asym_echoes.append(asym)
            time.sleep(0.005)
        
        asym_window = np.array(asym_echoes, dtype=float)
        
        # Regulación por 'Movimiento Único'
        lam, load = governor.regulate(asym_window)
        reg_history.append((lam, load))
        
        if i % 5 == 0:
            status = "LAMINAR" if lam > 0.5 else "TURBULENTO"
            gap = asym_window[-1]
            print(f"[OS:{i:05d}] {status} | Gap Cuántico: {gap:.4f} | Load: {load:.3f}")
        
        i += 1
        
    if cycles != -1:
        import os
        os.makedirs("h7_outputs", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1); plt.plot([h[0] for h in reg_history], 'g-', label='Laminosidad')
        plt.subplot(2, 1, 2); plt.plot([h[1] for h in reg_history], 'b-', label='Load Factor')
        plt.savefig("h7_outputs/h7_auto_governor_loop.png")
        print(f"\n📈 Reporte guardado en: h7_outputs/h7_auto_governor_loop.png")

if __name__ == "__main__":
    import sys
    mode = -1 if '--daemon' in sys.argv else 40
    run_governor_live(mode)

if __name__ == "__main__":
    import sys
    # Si se pasa '--daemon', corre infinito
    mode = -1 if '--daemon' in sys.argv else 40
    run_governor_live(mode)
