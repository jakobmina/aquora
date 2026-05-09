"""
h7_cascade_maxcut.py
====================
Aplica la topología de la Cascada H7 (12 qubits) a un problema MaxCut, 
encuentra el estado fundamental exacto y analiza la asimetría resultante.
"""

import numpy as np
import networkx as nx
import json
from qiskit_optimization.applications import Maxcut
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Re-usar el decodificador de la implementación anterior
from h7_cascade_qiskit import QuantumCovarianceDecoder12Q, graficar_matriz_covarianza

# Constantes H7
PHI = (1 + np.sqrt(5)) / 2
O_n_integrity = 0.3624

def build_h7_cascade_graph():
    """Construye el grafo de 12 nodos basado en la cascada H7."""
    edges = [
        # B1_Atmosfera
        (0, 1), (1, 2), (0, 2),
        # Puente B1 -> B2
        (2, 3), (3, 5), (2, 5),
        # B2_MantoSup (omitiendo (3,5) que ya está en el puente)
        (3, 4), (4, 5),
        # Puente B2 -> B3
        (5, 6), (6, 8), (5, 8),
        # B3_MantoInf
        (6, 7), (7, 8),
        # Puente B3 -> B4
        (8, 9), (9, 11), (8, 11),
        # B4_Nucleo
        (9, 10), (10, 11)
    ]
    
    # Modulación con Operador Dorado O_n
    G = nx.Graph()
    G.add_nodes_from(range(12))
    
    for i, (u, v) in enumerate(edges):
        n = i + 1
        # O_n = cos(pi * n) * cos(pi * phi * n)
        o_n = float(np.cos(np.pi * n) * np.cos(np.pi * PHI * n))
        
        # Prevenir vacío plano
        if abs(o_n) < 1e-5:
            o_n = 1e-5
            
        G.add_edge(u, v, weight=o_n)
        
    return G

def plot_graph(G, filepath="h7_maxcut_graph.png"):
    """Dibuja el grafo MaxCut."""
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    
    # Normalizar colores para pesos positivos/negativos
    edge_colors = ['red' if w < 0 else 'blue' for w in weights]
    widths = [abs(w)*3 for w in weights]
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, edge_color=edge_colors, width=widths)
    plt.title("Topología H7 Cascada Modulada por O_n")
    plt.savefig(filepath)
    plt.close()
    print(f"\n[GRÁFICA] Grafo guardado en {filepath}")

def ejecutar_maxcut_h7():
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " MAXCUT TOPOLÓGICO H7 - 12 QUBITS ".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    
    # 1. Construir el Grafo
    G = build_h7_cascade_graph()
    plot_graph(G)
    
    # 2. Generar el Hamiltoniano de Ising para MaxCut
    maxcut = Maxcut(G)
    qp = maxcut.to_quadratic_program()
    qubitOp, offset = qp.to_ising()
    
    print(f"\n[1] HAMILTONIANO ISING")
    print("-" * 80)
    print(f"  Nodos: {G.number_of_nodes()}")
    print(f"  Aristas: {G.number_of_edges()}")
    print(f"  Offset: {offset:.4f}")
    
    # 3. Resolver Estado Fundamental Exacto
    print(f"\n[2] RESOLUCIÓN DEL ESTADO FUNDAMENTAL (scipy eigsh)")
    print("-" * 80)
    
    # Extraer la matriz hermitiana sparse (qubitOp is a SparsePauliOp)
    matrix = qubitOp.to_matrix(sparse=True)
    
    # Buscar el eigenvector más bajo (which='SA' -> Smallest Algebraic)
    eigenvalues, eigenvectors = eigsh(matrix, k=1, which='SA')
    
    energy = float(eigenvalues[0])
    opt_state = eigenvectors[:, 0]
    
    maxcut_value = energy + offset
    print(f"  Energía fundamental (Ising): {energy:.6f}")
    print(f"  Valor MaxCut: {maxcut_value:.6f}")
    
    amplitudes = np.asarray(opt_state)
    probs = np.abs(amplitudes)**2
    max_prob_idx = np.argmax(probs)
    best_bitstr = format(max_prob_idx, '012b')
    print(f"  Estado más probable: |{best_bitstr}> (Prob: {probs[max_prob_idx]:.4f})")
    
    # 4. Análisis con el Decodificador H7
    decodificador = QuantumCovarianceDecoder12Q(amplitudes, verbose=True)
    resultados = decodificador.analisis_completo()
    
    cov_matrix = np.array(resultados['covarianza'])
    graficar_matriz_covarianza(cov_matrix, filepath='h7_maxcut_covariance.png')
    
    export = {
        "pipeline": "H7 MaxCut Topológico 12-Qubits",
        "energy": energy,
        "maxcut_value": maxcut_value,
        "best_state": best_bitstr,
        "decodificador_resultados": resultados,
    }
    
    with open('h7_maxcut_resultados.json', 'w') as f:
        json.dump(export, f, indent=2, default=str)
        
    print("\n" + "=" * 80)
    print("✓ Pipeline MaxCut completado")
    print(f"✓ Resultados exportados a: h7_maxcut_resultados.json")
    print("=" * 80)
    
    # Fake result object for test compatibility
    class MockResult:
        def __init__(self, energy):
            self.energy = energy
    
    return MockResult(energy), resultados

if __name__ == "__main__":
    ejecutar_maxcut_h7()

if __name__ == "__main__":
    ejecutar_maxcut_h7()
