import pytest
import json
from unittest.mock import MagicMock, patch
import numpy as np
from vertex_h7_bridge import VertexH7Bridge

@pytest.fixture
def mock_bridge():
    with patch('vertex_h7_bridge.vertexai.init'):
        with patch('vertex_h7_bridge.GenerativeModel') as MockModel:
            bridge = VertexH7Bridge(project_id="test-project")
            # Setup mock model
            mock_model_instance = MockModel.return_value
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "rho": 0.8,
                "v": 0.9,
                "reasoning": "Test reasoning for stability"
            })
            mock_model_instance.generate_content.return_value = mock_response
            bridge.model = mock_model_instance
            return bridge

def test_bridge_initialization(mock_bridge):
    assert mock_bridge.model is not None
    assert mock_bridge.config.to_dict()['temperature'] == 0.7

def test_compute_informational_lagrangian(mock_bridge):
    rho, v = 0.8, 0.5
    l_symp, l_metr = mock_bridge.compute_informational_lagrangian(rho, v)
    assert l_symp == 0.8
    assert l_metr == 0.125

def test_get_physical_intent(mock_bridge):
    intent = mock_bridge._get_physical_intent("test prompt")
    assert intent["rho"] == 0.8
    assert intent["v"] == 0.9

def test_run_controlled_vqe_approved(mock_bridge):
    with patch('vertex_h7_bridge.MetriplecticMaxCut') as MockVQE:
        mock_vqe_instance = MockVQE.return_value
        mock_vqe_instance.run.return_value = {
            "h7_state": "EQUILIBRIUM",
            "chirality": 0.1234
        }
        result = mock_bridge.run_controlled_vqe("stable prompt")
        assert result["governance"] == "APPROVED"

def test_run_controlled_vqe_rejected(mock_bridge):
    with patch('vertex_h7_bridge.MetriplecticMaxCut') as MockVQE:
        mock_vqe_instance = MockVQE.return_value
        mock_vqe_instance.run.return_value = {
            "h7_state": "DESTRUCTIVE",
            "chirality": 0.9999
        }
        # Mocking the model response for this specific test
        mock_bridge.model.generate_content.return_value.text = json.dumps({
            "rho": 1.0,
            "v": -1.0,
            "reasoning": "Total collapse"
        })
        result = mock_bridge.run_controlled_vqe("unstable prompt")
        assert result["governance"] == "REJECTED"
