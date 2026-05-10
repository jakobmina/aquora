"""
h7_bayesian_oracle.py
=====================
Oráculo Bayesiano H7 + Extractor de Bits Cuánticos (QRNG).

Este módulo implementa:
1. H7BitExtractor: Extrae bits de alta calidad desde las asimetrías de probabilidad del VQE.
2. H7BayesianOracle: Inferencia bayesiana para gobernar el flujo de información del OS.

Ecuaciones de Extracción (H7):
  - Asimetría (A) = |P(0...0) - P(1...1)| / (P(0...0) + P(1...1))
  - Entropía Útil (S) = -∑ P(i) log2 P(i) * O_n(i)
  - Extracción de bits vía Corrector de Von Neumann sobre estados entrelazados.

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import numpy as np
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.linalg import inv
from q3as import Client, Credentials

# ============================================================
# CONSTANTES H7
# ============================================================
PHI = (1 + math.sqrt(5)) / 2
O_n_integrity = 0.3623748900804798

def O_n(n: int) -> float:
    """Operador Áureo — Regla 2.1."""
    val = math.cos(math.pi * n) * math.cos(math.pi * PHI * n)
    return val if abs(val) > 1e-5 else 1e-5

# ============================================================
# BLOQUE 1: EXTRACTOR DE BITS (H7 BIT EXTRACTOR)
# ============================================================

class H7BitExtractor:
    """
    Transforma las asimetrías de probabilidad del VQE en datos útiles.
    Implementa un pipeline de purificación de entropía basado en O_n.
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.raw_counts = {}
        self.extracted_bits = ""
        
    def ingest_counts(self, counts: Dict[str, int]):
        """Ingesta de counts desde q3as."""
        self.raw_counts = counts
        return self
    
    def extract(self) -> str:
        """
        Algoritmo de Extracción H7:
        1. Ordena estados por probabilidad.
        2. Aplica filtro de laminosidad (O_n).
        3. Corrección de Von Neumann: 
           Compara pares de estados (s1, s2).
           - 01 -> 0
           - 10 -> 1
           - 00 o 11 -> Descartar
        """
        if not self.raw_counts:
            return ""
            
        # Convertir a probabilidades moduladas
        total = sum(self.raw_counts.values())
        probs = {k: v/total for k, v in self.raw_counts.items()}
        
        # Generar secuencia raw basada en el orden de los estados
        sorted_states = sorted(probs.keys(), key=lambda x: probs[x], reverse=True)
        
        raw_bitstream = "".join(sorted_states)
        
        # Purificación via Von Neumann
        bits = []
        for i in range(0, len(raw_bitstream) - 1, 2):
            pair = raw_bitstream[i:i+2]
            if pair == "01":
                bits.append("0")
            elif pair == "10":
                bits.append("1")
                
        self.extracted_bits = "".join(bits)
        return self.extracted_bits

    def compute_asymmetry_metrics(self) -> Dict:
        """Calcula métricas de asimetría y entropía útil."""
        if not self.raw_counts:
            return {}
            
        total = sum(self.raw_counts.values())
        probs = np.array(list(self.raw_counts.values())) / total
        
        # Entropía de Shannon
        shannon_s = -np.sum(probs * np.log2(probs + 1e-12))
        
        # Entropía H7 (Modulada por O_n)
        h7_s = 0.0
        for i, p in enumerate(probs):
            h7_s += -p * np.log2(p + 1e-12) * abs(O_n(i + 1))
            
        # Asimetría Global
        p_max = np.max(probs)
        p_min = np.min(probs)
        asymmetry = (p_max - p_min) / (p_max + p_min + 1e-12)
        
        return {
            "shannon_entropy": round(shannon_s, 4),
            "h7_entropy": round(h7_s, 4),
            "global_asymmetry": round(asymmetry, 6),
            "bits_per_qubit": round(shannon_s / self.n_qubits, 4),
            "data_quality": "HIGH" if h7_s > O_n_integrity else "LOW"
        }

# ============================================================
# BLOQUE 2: ORÁCULO BAYESIANO H7
# ============================================================

class H7BayesianOracle:
    """
    Gobierna el sistema usando inferencia bayesiana.
    Utiliza el posterior para validar la integridad del OS.
    """
    def __init__(self, n_features: int):
        self.n = n_features
        self.mu_prior = np.zeros(n_features)
        self.sigma_prior = np.eye(n_features)
        self.mu_post = np.zeros(n_features)
        self.sigma_post = np.eye(n_features)
        
    def update(self, X: np.ndarray, y: np.ndarray, sigma2: float = 0.1):
        """Update bayesiano con cierre conjugado."""
        N = X.shape[0]
        prec_prior = inv(self.sigma_prior)
        
        # Precision posterior
        prec_post = prec_prior + (1.0/sigma2) * (X.T @ X)
        self.sigma_post = inv(prec_post)
        
        # Media posterior
        self.mu_post = self.sigma_post @ (prec_prior @ self.mu_prior + (1.0/sigma2) * (X.T @ y))
        
        return self.mu_post, self.sigma_post

    def get_integrity(self, X: np.ndarray) -> float:
        """Calcula I_H7 basada en la distancia de Mahalanobis."""
        if self.mu_post is None: return 0.0
        
        prec_post = inv(self.sigma_post + 1e-6*np.eye(self.n))
        scores = []
        for i, x in enumerate(X):
            diff = x - self.mu_post
            d2 = diff @ prec_post @ diff
            on = abs(O_n(i + 1))
            scores.append(math.exp(-d2 / (2.0 * on * self.n)))
            
        return float(np.mean(scores))

# ============================================================
# FUNCIONES DE INTEGRACIÓN
# ============================================================

def run_extraction_pipeline(job_name: str, credentials_path: str = "credentials.json", cache_file: str = "h7_outputs/h7_quantum_entropy.json"):
    """
    Pipeline completo con Persistencia:
    1. Intenta recuperar de cache local.
    2. Si no hay cache o falla, intenta q3as.
    3. Si q3as falla, usa el cache como fallback.
    """
    print(f"\n🚀 Iniciando Extractor de Bits H7 para Job: {job_name}")
    print("="*60)
    
    # Intentar cargar de cache primero para velocidad
    try:
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            if cached_data.get("job_name") == job_name:
                print(f"📦 [CACHE] Recuperados {len(cached_data['bits'])} bits de entropía local.")
                return cached_data
    except:
        pass

    try:
        client = Client(Credentials.load(credentials_path))
        job = client.get_job(job_name)
        result = job.result()
        
        counts = getattr(result, "meas_counts", {})
        if not counts:
            counts = getattr(result, "sampled", {})
            
        if not counts:
            raise ValueError("No se encontraron counts en el resultado.")
            
        n_qubits = len(list(counts.keys())[0])
        extractor = H7BitExtractor(n_qubits)
        extractor.ingest_counts(counts)
        
        bits = extractor.extract()
        metrics = extractor.compute_asymmetry_metrics()
        
        # Guardar en Cache
        payload = {
            "job_name": job_name,
            "bits": bits,
            "metrics": metrics,
            "n_qubits": n_qubits
        }
        with open(cache_file, 'w') as f:
            json.dump(payload, f)
            
        print(f"✅ Bits extraídos y persistidos en {cache_file}.")
        return payload
        
    except Exception as e:
        print(f"❌ Error en el pipeline (Cloud): {e}")
        # FALLBACK FINAL
        try:
            with open(cache_file, 'r') as f:
                print("⚠️ Usando FALLBACK de entropía persistida...")
                return json.load(f)
        except:
            print("💀 No hay cache disponible. Fallo total.")
            return None

if __name__ == "__main__":
    # Test local con datos simulados si no hay job activo
    job_id = "which-pink-counter"
    res = run_extraction_pipeline(job_id)
    
    if not res:
        print("\n🛠 Modo Simulación (Testing Extractor Logic)...")
        # Simular counts asimétricos de 20 qubits
        sim_counts = {"".join(["1" if i%3==0 else "0" for i in range(20)]): 500,
                      "".join(["0" if i%2==0 else "1" for i in range(20)]): 300,
                      "0"*20: 100,
                      "1"*20: 100}
        
        ext = H7BitExtractor(20)
        ext.ingest_counts(sim_counts)
        print(f"Bits: {ext.extract()[:64]}...")
        print(f"Métricas: {ext.compute_asymmetry_metrics()}")
