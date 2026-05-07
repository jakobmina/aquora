import pytest
import numpy as np
from h7_cascade_maxcut import build_h7_cascade_graph, ejecutar_maxcut_h7

def test_graph_creation():
    """Prueba que el grafo MaxCut de la cascada se cree correctamente."""
    G = build_h7_cascade_graph()
    assert G.number_of_nodes() == 12
    assert G.number_of_edges() == 18
    
    # Verificar que los pesos están modulados (ninguno debe ser 1.0 genérico, a menos que n=0 pero empezamos en n=1)
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        assert weight != 1.0
        assert abs(weight) >= 1e-5

def test_maxcut_pipeline():
    """Prueba que la ejecución del MaxCut topológico devuelva el resultado correcto sin romper el pipeline."""
    result, resultados = ejecutar_maxcut_h7()
    
    assert result is not None
    assert 'energy' in dir(result) or 'eigenvalue' in dir(result)
    
    firma = resultados['h7_signature']
    assert 'firma_h7' in firma
    assert 'estado' in firma
    assert 'tensor_drift' in firma
    
    cov_matrix = np.array(resultados['covarianza'])
    assert cov_matrix.shape == (12, 12)
