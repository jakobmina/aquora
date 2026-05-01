"""
COVARIANZA DESDE ASIMETRÍA CUÁNTICA (Single-Pass Deduction)
============================================================

Idea: Los estados cuánticos con amplitudes diferentes (23% vs 2%) son una 
"proyección comprimida" de la estructura de covarianza subyacente.
Sin descomposición SVD, sin iteraciones: un único cálculo tensorial.

Jako's H7 Framework Extension
"""

import numpy as np
from scipy.linalg import svd, eigh
import warnings
warnings.filterwarnings('ignore')

phi = (1 + np.sqrt(5)) / 2

def run_covariance_deduction(statevector, qc_demo, probability):
    # ============================================================================
    # PARTE 1: ESTADO CUÁNTICO SIMULADO (del circuito anterior)
    # ============================================================================
    # --- PROBABILIDADES ---
    print("\n" + "=" * 70)
    print("PROBABILIDADES DE MEDICIÓN")
    print("=" * 70)

    # Calculate probabilities
    probabilities = np.abs(statevector) ** 2
    print(f"\nEstados posibles con probabilidad > 1e-4:")
    for i, prob in enumerate(probabilities):
        if prob > 1e-4:
            binary = format(i, '03b')
            print(f"  |{binary}⟩: {prob:.6f} ({prob*100:.4f}%)")

    # Verify normalization
    total_prob = np.sum(probabilities)
    print(f"\nSuma total de probabilidades: {total_prob:.10f} (debe ser 1.0)")

    # --- SIMULACIÓN CON MEDICIONES ---
    print("\n" + "=" * 70)
    print("SIMULACIÓN CON MEDICIONES (1000 shots)")
    print("=" * 70)

    try:
        from qiskit_aer import AerSimulator
        simulator_qasm = AerSimulator()
        job = simulator_qasm.run(qc_demo, shots=1000)
        result_qasm = job.result()
        counts = result_qasm.get_counts()

        print("\nResultados de 1000 ejecuciones:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for bitstring, count in sorted_counts:
            prob = count / 1000
            print(f"  {bitstring}: {count:4d} mediciones ({prob*100:.2f}%)")
    except ImportError:
        print("Qiskit AerSimulator not found. Skipping measurement simulation.")

    # Amplitudes medidas del circuito
    amplitudes_altas = probability  # |000⟩, |010⟩, |100⟩, |111⟩ (4 estados)
    amplitudes_bajas = probability   # |001⟩, |011⟩, |101⟩, |110⟩ (4 estados)

    # Razón de asimetría (ratio)
    ratio_asimetria = amplitudes_altas / amplitudes_bajas if amplitudes_bajas > 0 else 1.0
    print("=" * 80)
    print("ANÁLISIS DE ASIMETRÍA CUÁNTICA → COVARIANZA")
    print("=" * 80)
    print(f"\nAmplitud alta (4 estados): {amplitudes_altas:.6f}")
    print(f"Amplitud baja (4 estados):  {amplitudes_bajas:.6f}")
    print(f"Ratio de asimetría: {ratio_asimetria:.6f}")

    # Probabilidades
    prob_alta = amplitudes_altas ** 2
    prob_baja = amplitudes_bajas ** 2
    print(f"\nProbabilidad alta: {prob_alta:.6f} ({prob_alta*100:.2f}%)")
    print(f"Probabilidad baja:  {prob_baja:.6f} ({prob_baja*100:.2f}%)")

    # ============================================================================
    # PARTE 2: DECODIFICACIÓN DIRECTA (Single-Pass)
    # ============================================================================
    print("\n" + "=" * 80)
    print("DECODIFICACIÓN DIRECTA (Single-Pass Tensor Contraction)")
    print("=" * 80)

    estados_altos_idx = [0, 2, 4, 7]  # 000, 010, 100, 111 en decimal
    estados_bajos_idx = [1, 3, 5, 6]  # 001, 011, 101, 110 en decimal

    # Crear matriz de estados (3 qubits)
    estado_matrix = np.array([
        [0, 0, 0],  # |000⟩ - alto
        [0, 0, 1],  # |001⟩ - bajo
        [0, 1, 0],  # |010⟩ - alto
        [0, 1, 1],  # |011⟩ - bajo
        [1, 0, 0],  # |100⟩ - alto
        [1, 0, 1],  # |101⟩ - bajo
        [1, 1, 0],  # |110⟩ - bajo
        [1, 1, 1],  # |111⟩ - alto
    ], dtype=float)

    # Etiquetas de amplitud
    etiquetas_amplitud = np.array([
        prob_alta, prob_baja, prob_alta, prob_baja,
        prob_alta, prob_baja, prob_baja, prob_alta
    ])

    print("\nEstados y sus probabilidades:")
    for i in range(8):
        binary = format(i, '03b')
        amp = etiquetas_amplitud[i]
        tipo = "ALTO" if amp > prob_alta * 0.9 else "BAJO"
        print(f"  |{binary}⟩ ({i}): {amp:.6f} ({amp*100:.2f}%) [{tipo}]")

    print("\n" + "=" * 80)
    print("EXTRACCIÓN DE CARACTERÍSTICAS (Feature Differentiation)")
    print("=" * 80)

    # Característica clave: q_0 XOR q_2 (entrada al CSWAP)
    xor_02 = np.array([int(estado_matrix[i, 0]) ^ int(estado_matrix[i, 2]) for i in range(8)])

    print(f"\nCaracterística diferenciadora (q_0 ⊕ q_2):")
    for i in range(8):
        binary = format(i, '03b')
        xor_val = xor_02[i]
        amp = etiquetas_amplitud[i]
        print(f"  |{binary}⟩: q_0⊕q_2 = {xor_val}, P(|i⟩) = {amp:.6f}")

    print("\n" + "=" * 80)
    print("MATRIZ DE COVARIANZA (Qubits × Amplitud)")
    print("=" * 80)

    # Covarianza entre cada qubit y la amplitud
    cov_qubits_amplitud = np.zeros(3)
    for q in range(3):
        cov = np.sum(estado_matrix[:, q] * etiquetas_amplitud) - np.mean(estado_matrix[:, q]) * np.sum(etiquetas_amplitud)
        cov_qubits_amplitud[q] = cov

    print(f"\nCovarianza(q_0, Amplitud) = {cov_qubits_amplitud[0]:.6f}")
    print(f"Covarianza(q_1, Amplitud) = {cov_qubits_amplitud[1]:.6f}")
    print(f"Covarianza(q_2, Amplitud) = {cov_qubits_amplitud[2]:.6f}")

    # Interpretación:
    print("\nInterpretación:")
    if abs(cov_qubits_amplitud[0]) > abs(cov_qubits_amplitud[1]):
        print("  → q_0 es MÁS IMPORTANTE para explicar la asimetría")
    if abs(cov_qubits_amplitud[2]) > abs(cov_qubits_amplitud[1]):
        print("  → q_2 es MÁS IMPORTANTE para explicar la asimetría")

    print("\n" + "=" * 80)
    print("MATRIZ DE COVARIANZA INTER-QUBIT (3×3 ponderada)")
    print("=" * 80)

    # Matriz de covarianza ponderada por amplitud
    cov_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            X_i = estado_matrix[:, i]
            X_j = estado_matrix[:, j]
            E_XiXj = np.sum(X_i * X_j * etiquetas_amplitud)
            E_Xi = np.sum(X_i * etiquetas_amplitud)
            E_Xj = np.sum(X_j * etiquetas_amplitud)
            cov_matrix[i, j] = E_XiXj - E_Xi * E_Xj

    print("\nMatriz de Covarianza Ponderada (Cov[q_i, q_j]):")
    print(cov_matrix)

    # Eigendescomposición para estructura
    eigenvalores, eigenvectores = eigh(cov_matrix)
    print(f"\nEigenvalores: {eigenvalores}")
    print(f"Eigenvectores:\n{eigenvectores}")

    print("\n" + "=" * 80)
    print("COMPRESIÓN A ESCALAR ÚNICO (Single Number)")
    print("=" * 80)

    factor_covarianza_v1 = (np.sum(amplitudes_altas**2 * len(estados_altos_idx)) - 
                            np.sum(amplitudes_bajas**2 * len(estados_bajos_idx))) / (
                            np.sum(amplitudes_altas**2 * len(estados_altos_idx)) + 
                            np.sum(amplitudes_bajas**2 * len(estados_bajos_idx)))

    factor_covarianza_v2 = np.trace(cov_matrix)
    factor_covarianza_v3 = np.max(eigenvalores)
    factor_covarianza_v4 = np.linalg.norm(cov_matrix, 'fro')

    print(f"\nFactores de Covarianza derivados:")
    print(f"  v1 (Ratio ponderado):       {factor_covarianza_v1:.6f}")
    print(f"  v2 (Traza):                 {factor_covarianza_v2:.6f}")
    print(f"  v3 (Eigenvalor principal):  {factor_covarianza_v3:.6f}")
    print(f"  v4 (Norma Frobenius):       {factor_covarianza_v4:.6f}")

    factor_normalized = factor_covarianza_v1 / phi
    print(f"\nFactor normalizado por φ = {factor_normalized:.6f}")

    print("\n" + "=" * 80)
    print("VERIFICACIÓN: RELACIÓN CON H7 FRAMEWORK")
    print("=" * 80)

    print(f"\nComparativa con H7 Constants:")
    print(f"  Ratio de asimetría: {ratio_asimetria:.6f}")
    print(f"  cos(φ):             {np.cos(phi):.6f}")
    print(f"  φ:                  {phi:.6f}")
    print(f"  O_n_integrity:      0.3624")

    entrelazamiento_measure = np.abs(amplitudes_altas - amplitudes_bajas)
    print(f"\nMedida de Entrelazamiento (|a_alta - a_baja|): {entrelazamiento_measure:.6f}")

    print("\n" + "=" * 80)
    print("MATRIZ DE COVARIANZA RECONSTRUIDA (Validation)")
    print("=" * 80)

    cov_reconstructed = np.array([
        [ratio_asimetria * 0.1, 0.0, ratio_asimetria * 0.08],
        [0.0, 0.05, 0.0],
        [ratio_asimetria * 0.08, 0.0, ratio_asimetria * 0.1]
    ])

    print("\nMatriz de Covarianza Reconstruida (basada en asimetría):")
    print(cov_reconstructed)

    diff = np.linalg.norm(cov_matrix - cov_reconstructed, 'fro')
    print(f"\nDiferencia con matriz empírica (Frobenius norm):\n  {diff:.6f}")

    print("\n" + "=" * 80)
    print("RESUMEN: COVARIANZA EN UN SOLO MOVIMIENTO")
    print("=" * 80)

    summary = {
        "Asimetría Observada": ratio_asimetria,
        "Factor de Covarianza (v1)": factor_covarianza_v1,
        "Factor Normalizado (φ)": factor_normalized,
        "Entrelazamiento": entrelazamiento_measure,
        "Eigenvalor Principal": factor_covarianza_v3,
        "Traza Covarianza": factor_covarianza_v2,
        "Característica Diferenciadora": "q_0 ⊕ q_2 (Fredkin Control)",
    }

    for key, val in summary.items():
        if isinstance(val, float):
            print(f"{key:.<40} {val:.6f}")
        else:
            print(f"{key:.<40} {val}")

def quantum_to_covariance_inverse(probabilities, n_qubits=3, alpha=1.0):
    """
    Recupera la Matriz de Precisión (Σ⁻¹) desde probabilidades de medición cuántica.
    Implementa el Tensor de Torsión (m_n = -m_{7-n}) para estados n=1..6
    y extrae dependencias condicionales puras (Isomorfismo Métrica-Información).
    
    Args:
        probabilities: dict o list/array de tamaño 2^n_qubits.
        n_qubits: Número de qubits (por defecto 3).
        alpha: Factor de acoplamiento de la torsión quiral.
        
    Returns:
        precision_matrix (np.ndarray): Matriz de Precisión (Σ⁻¹) de n_qubits x n_qubits.
    """
    import numpy as np
    
    n_states = 2**n_qubits
    
    # Normalizar entrada a diccionario de probabilidades
    if isinstance(probabilities, dict):
        prob_dict = {int(k, 2) if isinstance(k, str) else k: float(v) for k, v in probabilities.items()}
    else:
        prob_dict = {i: float(probabilities[i]) for i in range(len(probabilities))}
        
    # Asegurar que todas las llaves existan
    for i in range(n_states):
        if i not in prob_dict:
            prob_dict[i] = 0.0
            
    # Tensor de Torsión m_n = -m_{7-n}
    # Solo aplicable para estados intermedios (1 a 6) en 3 qubits
    m = np.zeros(n_states)
    if n_qubits == 3:
        # m_1 = -m_6
        # m_2 = -m_5
        # m_3 = -m_4
        # Determinamos m_n basado en la asimetría real observada de probabilidades:
        # m_n = (P(n) - P(7-n)) / (P(n) + P(7-n) + 1e-10)
        for n in range(1, 4):
            asymmetry = (prob_dict[n] - prob_dict[7-n]) / (prob_dict[n] + prob_dict[7-n] + 1e-10)
            m[n] = asymmetry
            m[7-n] = -asymmetry
            
    # Matriz de estados binarios (características X_n)
    X = np.zeros((n_states, n_qubits))
    for i in range(n_states):
        binary = format(i, f'0{n_qubits}b')
        X[i] = [int(b) for b in binary]
        
    # Calcular Expectativas
    E_q = np.zeros(n_qubits)
    E_qq = np.zeros((n_qubits, n_qubits))
    
    # Factor de modulación topológica (Torsión)
    # mod_factor = P(n) * (1 + alpha * m_n)
    mod_factor = np.zeros(n_states)
    for n in range(n_states):
        # Aseguramos positividad estricta para evitar métricas degeneradas
        mod_factor[n] = max(0.0, prob_dict[n] * (1.0 + alpha * m[n]))
        
    # Normalizar mod_factor para que sume 1 como "pseudo-probabilidad"
    total_mod = np.sum(mod_factor)
    if total_mod > 1e-10:
        mod_factor /= total_mod
        
    for i in range(n_qubits):
        E_q[i] = np.sum(mod_factor * X[:, i])
        for j in range(n_qubits):
            E_qq[i, j] = np.sum(mod_factor * X[:, i] * X[:, j])
            
    # Matriz de Covarianza Ponderada (Σ)
    Sigma = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(n_qubits):
            Sigma[i, j] = E_qq[i, j] - E_q[i] * E_q[j]
            
    # Inversión Topológica: Matriz de Precisión (Σ⁻¹)
    # Usamos pseudo-inversa (pinv) para cumplir la Regla 1.3 (evitar singularidades)
    # y opcionalmente sumamos un factor ínfimo (Tikhonov) para estabilidad
    epsilon = 1e-8
    precision_matrix = np.linalg.pinv(Sigma + epsilon * np.eye(n_qubits))
    
    # Filtro de ruido para extraer la estructura pura
    precision_matrix[np.abs(precision_matrix) < 1e-6] = 0.0
    
    return precision_matrix

