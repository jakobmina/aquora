"""
h7_classifier_prototype.py
==========================
Prototipo de Clasificador H7-AI.

Este modelo utiliza:
1. Pesos iniciales 'sembrados' por los 721 bits de entropía cuántica (20Q).
2. Optimización vía Gradiente Natural Cuántico (H7QNNBridge).
3. Gobernanza Bayesiana para rechazar señales fuera de la variedad H7.

Misión: Clasificar señales de sensores (Laminar vs Turbulento) con mínima data.

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import numpy as np
import math
from h7_bayesian_oracle import run_extraction_pipeline, H7BayesianOracle, PHI
from h7_qnn_bridge import H7QNNBridge
from scipy.linalg import inv
import matplotlib.pyplot as plt

class H7Classifier:
    def __init__(self, n_features: int, bitstream: str):
        self.n = n_features
        # 1. Sembrar pesos desde la entropía cuántica
        self.weights = self._seed_weights(bitstream)
        self.bias = 0.0
        
        # 2. Inicializar Bridge y Oracle
        # Usamos una covarianza H7 estructurada como prior
        sigma_prior = np.eye(n_features)
        for i in range(n_features):
            sigma_prior[i,i] *= abs(math.cos(math.pi * i * PHI))
            
        self.bridge = H7QNNBridge(sigma_prior, lambda_metr=0.1)
        self.oracle = H7BayesianOracle(n_features)
        
    def _seed_weights(self, bits: str) -> np.ndarray:
        """Transforma el bitstream cuántico en el estado inicial de la IA."""
        # Tomamos trozos de 8 bits para generar pesos en [-1, 1]
        w = []
        for i in range(0, len(bits) - 8, 8):
            val = int(bits[i:i+8], 2) / 255.0
            w.append(val * 2 - 1)
            if len(w) >= self.n: break
        
        # Si faltan bits, rellenamos con O_n
        while len(w) < self.n:
            w.append(math.cos(len(w) * math.pi * PHI))
            
        return np.array(w[:self.n])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoide Metripléctica."""
        z = X @ self.weights + self.bias
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01):
        """Entrenamiento con Gradiente Natural H7."""
        N = X.shape[0]
        probs = self.predict_proba(X)
        error = probs - y
        
        # Gradiente Euclidiano (BCE)
        grad_w = (X.T @ error) / N
        grad_b = np.mean(error)
        
        # APLICAR GRADIENTE NATURAL H7
        # Esto corrige la dirección del aprendizaje según la métrica cuántica
        grad_w_nat = self.bridge.natural_gradient(grad_w)
        
        # Update
        self.weights -= lr * grad_w_nat
        self.bias -= lr * grad_b
        
        # Actualizar Oráculo para integridad
        self.oracle.update(X, y, sigma2=0.1)
        return np.mean(error**2)

def generate_sensor_data(n: int, d: int):
    """Genera datos de sensores: Clase 0 (Laminar) vs Clase 1 (Turbulento)."""
    # Clase 0: Señales suaves, coherentes
    X0 = np.random.randn(n // 2, d) * 0.5
    y0 = np.zeros(n // 2)
    
    # Clase 1: Señales ruidosas, alta varianza
    X1 = np.random.randn(n // 2, d) + 2.0
    y1 = np.ones(n // 2)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    
    # Shuffle
    idx = np.random.permutation(n)
    return X[idx], y[idx]

def run_prototype():
    print("="*70)
    print("  H7-CLASSIFIER PROTOTYPE — Quantum Seeded AI")
    print("="*70)

    # 1. Obtener Entropía Real
    job_id = "which-pink-counter"
    extraction = run_extraction_pipeline(job_id)
    if not extraction: return
    bits = extraction["bits"]

    # 2. Configurar Dataset de Sensores
    n_features = 10
    X, y = generate_sensor_data(n=100, d=n_features)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # 3. Inicializar H7-Classifier
    clf = H7Classifier(n_features, bits)
    print(f"\n🧠 Pesos iniciales sembrados por 20Q-Entropy. Norma: {np.linalg.norm(clf.weights):.4f}")

    # 4. Entrenamiento
    history_loss = []
    history_int  = []
    
    print("\n🚀 Entrenando con Gradiente Natural H7...")
    for epoch in range(20):
        loss = clf.train_step(X_train, y_train, lr=0.1)
        integrity = clf.oracle.get_integrity(X_train)
        
        history_loss.append(loss)
        history_int.append(integrity)
        
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d} | Loss: {loss:.6f} | Integrity: {integrity:.6f}")

    # 5. Evaluación
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\n🎯 Precisión Final en Test: {accuracy*100:.1f}%")
    print(f"   Estado de Integridad: {'COHERENTE' if history_int[-1] > 0.3 else 'DEGRADADO'}")

    # Visualización
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, 'r-', label='H7-BCE Loss')
    plt.title("Convergencia Metripléctica")
    plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_int, 'g-', label='H7-Integrity')
    plt.title("Integridad del OS")
    plt.legend(); plt.grid(True)
    
    plt.savefig("h7_classifier_results.png")
    print("\n📈 Resultados guardados en: h7_classifier_results.png")
    print("="*70)

if __name__ == "__main__":
    run_prototype()
