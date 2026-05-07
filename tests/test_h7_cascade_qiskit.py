import pytest
import numpy as np
from qiskit import QuantumCircuit
from h7_cascade_qiskit import CascadaH7Qiskit, QuantumCovarianceDecoder12Q

def test_circuit_creation():
    """Prueba que el circuito de 12 qubits se cree correctamente."""
    cascada = CascadaH7Qiskit(verbose=False)
    assert cascada.circuit.num_qubits == 12
    assert len(cascada.bloques) == 4

def test_mar_de_dirac():
    """Prueba que el mar de Dirac aplique 12 compuertas H."""
    cascada = CascadaH7Qiskit(verbose=False)
    cascada.preparar_mar_de_dirac()
    
    # 12 H gates expected
    count_h = sum(1 for instr in cascada.circuit.data if instr.operation.name == 'h')
    assert count_h == 12

def test_pipeline_execution_and_decoding():
    """Prueba el pipeline completo y el decodificador para asegurar que no hay singularidades."""
    cascada = CascadaH7Qiskit(verbose=False)
    cascada.preparar_mar_de_dirac()
    cascada.inyectar_fase_cuaternionica()
    cascada.ejecutar_flujo_laminar()
    cascada.extraer_topologia()
    
    statevector_data = cascada.get_statevector_data()
    assert len(statevector_data) == 4096
    
    decodificador = QuantumCovarianceDecoder12Q(statevector_data, verbose=False)
    resultados = decodificador.analisis_completo()
    
    # Verificar salidas del decodificador
    assert 'asimetria' in resultados
    assert 'covarianza' in resultados
    assert 'h7_signature' in resultados
    
    firma = resultados['h7_signature']
    assert 'tensor_drift' in firma
    assert 'firma_h7' in firma
    assert 'estado' in firma
    
    # Check that covariance matrix has 12x12 shape
    cov_matrix = np.array(resultados['covarianza'])
    assert cov_matrix.shape == (12, 12)
    
    # The firm must be a finite float
    assert np.isfinite(firma['firma_h7'])
