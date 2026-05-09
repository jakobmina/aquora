"""
h7_cascade_qiskit.py
====================
Implementación de la Cascada Topológica H7 (12 Qubits) utilizando Qiskit.
"""

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import eigh
import json
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ============================================================================
# CONSTANTES FUNDAMENTALES H7
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2
O_n_integrity = 0.3624
DRIFT_072 = 7 - 2 * np.pi
PHI_SQUARED = PHI ** 2

# ============================================================================
# CLASE PRINCIPAL: CASCADA H7 (12 QUBITS) EN QISKIT
# ============================================================================

class CascadaH7Qiskit:
    def __init__(self, verbose=True):
        self.num_qubits = 12
        self.verbose = verbose
        
        self.bloques = {
            'B1_Atmosfera': [0, 1, 2],
            'B2_MantoSup':  [3, 4, 5],
            'B3_MantoInf':  [6, 7, 8],
            'B4_Nucleo':    [9, 10, 11]
        }
        
        self.circuit = QuantumCircuit(self.num_qubits)
        self.operaciones = []
        self.midpoints = {}
        
        if self.verbose:
            self._print_header()
            
    def _print_header(self):
        print("\n" + "=" * 80)
        print("H7 CASCADA TOPOLÓGICA - 12 QUBIT (QISKIT IMPLEMENTATION)")
        print("=" * 80)
        print(f"\nEstructura de bloques:")
        for name, qubits in self.bloques.items():
            print(f"  {name}: qubits {qubits}")
        print()

    def preparar_mar_de_dirac(self):
        if self.verbose:
            print("\n[PASO 1] Preparación del Mar de Dirac (Superposición H⊗12)")
            print("-" * 80)
        
        for i in range(self.num_qubits):
            self.circuit.h(i)
        self.operaciones.append("H⊗12")

    def inyectar_fase_cuaternionica(self):
        if self.verbose:
            print("\n[PASO 2] Inyección de Fase Cuaterniónica (Frontera q0)")
            print("-" * 80)
            print(f"  Parámetro: O_n_integrity = {O_n_integrity:.6f}")
            
        q_frontera = self.bloques['B1_Atmosfera'][0]
        self.circuit.ry(O_n_integrity, q_frontera)
        self.circuit.rz(O_n_integrity, q_frontera)
        self.circuit.rx(O_n_integrity, q_frontera)
        self.operaciones.extend(["RY(q0)", "RZ(q0)", "RX(q0)"])

    def _procesar_bloque_interno(self, nombre_bloque, q_control, q_target1, q_target2):
        if self.verbose:
            print(f"\n  [{nombre_bloque}] Procesamiento Interno")
            print(f"    Control: q{q_control}, Targets: q{q_target1}, q{q_target2}")
            
        # Qiskit CSWAP: cswap(control, target1, target2)
        # Note: In original code, it was _apply_cswap(control, target2, target1) 
        # which swaps target1 and target2, so order of targets doesn't matter.
        self.circuit.cswap(q_control, q_target2, q_target1)
        self.circuit.ccx(q_target2, q_target1, q_control)
        
        self.operaciones.extend([f"CSWAP({q_control},{q_target2},{q_target1})", 
                                 f"CCX({q_target2},{q_target1},{q_control})"])

    def _crear_puente_topologico(self, nombre, q_salida_prev, q_entrada_next, q_aux):
        if self.verbose:
            print(f"\n  [{nombre}] Puente Topológico")
            print(f"    q{q_salida_prev} (salida) → q{q_entrada_next} (entrada)")
            
        # _apply_cswap(q_salida_prev, q_aux, q_entrada_next)
        self.circuit.cswap(q_salida_prev, q_aux, q_entrada_next)
        self.operaciones.append(f"CSWAP_PUENTE({q_salida_prev},{q_aux},{q_entrada_next})")

    def ejecutar_flujo_laminar(self):
        if self.verbose:
            print("\n[PASO 3] Ejecución del Flujo Laminar (Cascada 4-Bloques)")
            print("=" * 80)
            
        # BLOQUE 1
        if self.verbose: print("\n[B1_ATMOSFERA] Turbulencia Alta (q0-q2)")
        self._procesar_bloque_interno("B1", 0, 1, 2)
        
        # PUENTE B1 -> B2
        if self.verbose: print("\n[PUENTE B1→B2] Enrutador de Mínima Acción")
        self._crear_puente_topologico("B1→B2", 2, 3, 5)
        
        # BLOQUE 2
        if self.verbose: print("\n[B2_MANTO_SUP] Filtro Primario (q3-q5)")
        self._procesar_bloque_interno("B2", 3, 4, 5)
        
        # PUENTE B2 -> B3
        if self.verbose: print("\n[PUENTE B2→B3] Enrutador de Mínima Acción")
        self._crear_puente_topologico("B2→B3", 5, 6, 8)
        
        # BLOQUE 3
        if self.verbose: print("\n[B3_MANTO_INF] Filtro Secundario (q6-q8)")
        self._procesar_bloque_interno("B3", 6, 7, 8)
        
        # PUENTE B3 -> B4
        if self.verbose: print("\n[PUENTE B3→B4] Enrutador de Mínima Acción")
        self._crear_puente_topologico("B3→B4", 8, 9, 11)
        
        # BLOQUE 4
        if self.verbose: print("\n[B4_NUCLEO] Sol de Cristal (q9-q11)")
        self._procesar_bloque_interno("B4", 9, 10, 11)
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("Cascada completada exitosamente")

    def extraer_topologia(self):
        if self.verbose:
            print("\n[PASO 4] Extracción de Topología (Statevector Qiskit)")
            print("-" * 80)
            
        self.statevector = Statevector(self.circuit)
        self.amplitudes = np.asarray(self.statevector.data)
        
        probs = np.abs(self.amplitudes)**2
        
        counts = {}
        for state_idx in range(2**self.num_qubits):
            if probs[state_idx] > 1e-8:
                binary = format(state_idx, f'0{self.num_qubits}b')
                counts[binary] = float(probs[state_idx])
                
        if self.verbose:
            print(f"  Estados con amplitud > 1e-8: {len(counts)}")
            print(f"  Entropía de von Neumann: {-np.sum(probs[probs > 1e-8] * np.log2(probs[probs > 1e-8] + 1e-10)):.6f}")
            print(f"  Amplitud máxima: {np.max(np.abs(self.amplitudes)):.6f}")
            
        return counts

    def get_statevector_data(self):
        return self.amplitudes


# ============================================================================
# DECODIFICADOR COVARIANZA
# ============================================================================

class QuantumCovarianceDecoder12Q:
    def __init__(self, statevector_data, verbose=True):
        self.statevector = statevector_data
        self.dim = 2**12
        self.verbose = verbose
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("DECODIFICADOR SINGLE-PASS - 12 QUBITS (QISKIT DATA)")
            print("=" * 80)

    def decodificar_asimetria_global(self):
        if self.verbose:
            print("\n[1] DECODIFICACIÓN DE ASIMETRÍA GLOBAL")
            print("-" * 80)
            
        probs = np.abs(self.statevector)**2
        
        coherentes = 0.0
        ruido = 0.0
        
        for state_idx in range(self.dim):
            q0_bit = (state_idx >> 0) & 1
            q11_bit = (state_idx >> 11) & 1
            
            xor_val = q0_bit ^ q11_bit
            prob = probs[state_idx]
            
            if xor_val == 0:
                coherentes += prob
            else:
                ruido += prob
                
        ratio_cascada = coherentes / (ruido + 1e-10)
        
        if self.verbose:
            print(f"  Amplitud coherente (q0⊕q11=0): {coherentes:.6f}")
            print(f"  Amplitud ruido (q0⊕q11=1):      {ruido:.6f}")
            print(f"  Ratio cascada: {ratio_cascada:.6f}")
            
        return {
            'coherentes': coherentes,
            'ruido': ruido,
            'ratio': ratio_cascada
        }

    def calcular_matriz_covarianza(self):
        if self.verbose:
            print("\n[2] CÁLCULO DE MATRIZ DE COVARIANZA (12×12)")
            print("-" * 80)
            
        probs = np.abs(self.statevector)**2
        
        bits_matrix = np.array([
            [(state_idx >> i) & 1 for i in range(12)]
            for state_idx in range(self.dim)
        ], dtype=float)
        
        mu = np.sum(bits_matrix.T * probs, axis=1)
        
        cov = np.zeros((12, 12))
        for k in range(self.dim):
            if probs[k] > 1e-12:
                diff = bits_matrix[k] - mu
                cov += probs[k] * np.outer(diff, diff)
                
        if self.verbose:
            print(f"  Media (qubits): {mu}")
            print(f"  Determinante: {np.linalg.det(cov):.6e}")
            print(f"  Traza: {np.trace(cov):.6f}")
            
        return cov, mu

    def calcular_h7_signature(self, ratio_cascada, cov_matrix):
        if self.verbose:
            print("\n[3] CÁLCULO DE FIRMA H7")
            print("-" * 80)
            
        tensor_drift = (ratio_cascada - O_n_integrity) / PHI
        firma_h7 = abs(tensor_drift)
        
        eigvals, _ = eigh(cov_matrix + np.eye(12) * 1e-8)
        cond_number = np.max(eigvals) / (np.min(eigvals) + 1e-10)
        
        umbral_tolerancia = 0.5
        if firma_h7 < umbral_tolerancia:
            estado = "SEGURO"
            mensaje = "Fase contenida. Datos seguros para memoria profunda."
        else:
            estado = "ALERTA"
            mensaje = "Choque entrópico detectado. Truncando proceso."
            
        if self.verbose:
            print(f"  Tensor drift: {tensor_drift:.6f}")
            print(f"  Firma H7: {firma_h7:.6f}")
            print(f"  Número condición: {cond_number:.6f}")
            print(f"  Estado: [{estado}] {mensaje}")
            
        return {
            'tensor_drift': float(tensor_drift),
            'firma_h7': float(firma_h7),
            'condition_number': float(cond_number),
            'estado': estado,
            'mensaje': mensaje,
        }

    def mahalanobis_metric(self, cov_matrix, punto1_idx, punto2_idx):
        bits1 = np.array([(punto1_idx >> i) & 1 for i in range(12)])
        bits2 = np.array([(punto2_idx >> i) & 1 for i in range(12)])
        
        try:
            inv_cov = np.linalg.inv(cov_matrix + np.eye(12)*1e-8)
            dist = mahalanobis(bits1, bits2, inv_cov)
            return float(dist)
        except:
            return np.inf

    def analisis_completo(self):
        asimetria = self.decodificar_asimetria_global()
        cov, mu = self.calcular_matriz_covarianza()
        h7 = self.calcular_h7_signature(asimetria['ratio'], cov)
        
        if self.verbose:
            print("\n[4] EJEMPLOS DE DISTANCIA MAHALANOBIS")
            print("-" * 80)
            d_extremos = self.mahalanobis_metric(cov, 0, 2**12 - 1)
            print(f"  d_M(|00...0⟩, |11...1⟩) = {d_extremos:.6f}")
            
        return {
            'asimetria': asimetria,
            'covarianza': cov.tolist(),
            'h7_signature': h7,
        }

def graficar_matriz_covarianza(cov_matrix, filepath='h7_qiskit_covariance.png'):
    """Genera un mapa de calor de la matriz de covarianza."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cov_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Covarianza')
    plt.title('Matriz de Covarianza H7 (12 Qubits)')
    plt.xlabel('Qubits')
    plt.ylabel('Qubits')
    plt.xticks(np.arange(12), [f'q{i}' for i in range(12)])
    plt.yticks(np.arange(12), [f'q{i}' for i in range(12)])
    
    # Grid boundaries to show blocks
    for i in [2.5, 5.5, 8.5]:
        plt.axhline(i, color='black', linewidth=2, linestyle='--')
        plt.axvline(i, color='black', linewidth=2, linestyle='--')
        
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"\n[GRÁFICA] Matriz de covarianza guardada en {filepath}")


def ejecutar_pipeline_h7_cascada():
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " H7 CASCADA TOPOLÓGICA - 12 QUBIT PIPELINE (QISKIT) ".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    
    cascada = CascadaH7Qiskit(verbose=True)
    cascada.preparar_mar_de_dirac()
    cascada.inyectar_fase_cuaternionica()
    cascada.ejecutar_flujo_laminar()
    counts = cascada.extraer_topologia()
    
    decodificador = QuantumCovarianceDecoder12Q(cascada.get_statevector_data(), verbose=True)
    resultados = decodificador.analisis_completo()
    
    # Save the covariance plot
    cov_matrix = np.array(resultados['covarianza'])
    graficar_matriz_covarianza(cov_matrix)
    
    export = {
        "pipeline": "H7 Cascada Topológica 12-Qubits (Qiskit)",
        "operaciones": cascada.operaciones,
        "mediciones_topologia": counts,
        "decodificador_resultados": resultados,
    }
    
    with open('h7_cascada_qiskit_resultados.json', 'w') as f:
        json.dump(export, f, indent=2, default=str)
        
    print("\n" + "=" * 80)
    print("✓ Pipeline Qiskit completado")
    print(f"✓ Resultados exportados a: h7_cascada_qiskit_resultados.json")
    print("=" * 80)
    
    return cascada, decodificador, resultados

if __name__ == "__main__":
    ejecutar_pipeline_h7_cascada()
