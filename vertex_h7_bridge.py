import os
import json
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from run_vqe_maxcut import MetriplecticMaxCut
from h7_quaternion import H7QuaternionMapper, H7_AMPLITUDES

class VertexH7Bridge:
    def __init__(self, project_id="cuasiperiodico", location="us-central1"):
        """
        Inicializa el puente de inteligencia física entre Vertex AI y el framework H7.
        """
        try:
            vertexai.init(project=project_id, location=location)
            # Usamos gemini-1.5-flash para baja latencia en el bucle de control
            self.model = GenerativeModel("gemini-1.5-flash")
            self.h7_mapper = H7QuaternionMapper(H7_AMPLITUDES)
            print(f"[BRIDGE] Conectado a Vertex AI en proyecto: {project_id}")
        except Exception as e:
            print(f"[ERROR] No se pudo conectar a Vertex AI: {e}")
            self.model = None

    def _get_physical_intent(self, prompt: str):
        """
        Interpreta el prompt del usuario como un par (rho, v) en el espacio de fase.
        """
        if not self.model:
            return {"rho": 0.5, "v": 0.5}

        instruction = """
        Eres un Oráculo de Información Cuántica. Tu objetivo es mapear la intención del usuario a dos parámetros:
        1. rho (densidad): 0.0 (inacción) a 1.0 (saturación de datos).
        2. v (velocidad): -1.0 (destrucción) a 1.0 (construcción/evolución).
        
        Devuelve estrictamente un JSON plano con el formato: {"rho": float, "v": float, "reasoning": "breve explicacion física"}
        """
        
        try:
            response = self.model.generate_content(
                f"{instruction}\n\nProblema: {prompt}",
                generation_config=GenerationConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[BRIDGE] Error en inferencia: {e}")
            return {"rho": 0.5, "v": 0.0, "reasoning": "fallback por error"}

    def run_controlled_vqe(self, user_prompt: str):
        """
        Ejecuta el pipeline VQE gobernado por la intención física de Vertex AI.
        """
        # 1. Fase de Intención
        intent = self._get_physical_intent(user_prompt)
        rho, v = intent["rho"], intent["v"]
        
        print(f"\n[PHASE 1] Intención detectada: {intent['reasoning']}")
        print(f"       -> Parámetros H7: rho={rho:.4f}, v={v:.4f}")

        # 2. Fase de Validación Metripléctica
        # Mapeamos rho/v a n/phi para el pipeline VQE existente
        n_param = float(np.clip(rho * 4, 1.0, 4.0))
        phi_param = float(np.clip(0.362 + (v + 1) * 0.128, 0.362, 0.618))

        # Definimos unos edges base para el problema MaxCut (Grafo Triángulo)
        base_edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]

        try:
            vqe_system = MetriplecticMaxCut(
                edges=base_edges, 
                n_param=n_param, 
                phi_param=phi_param
            )
            result = vqe_system.run()
            status = result.get("h7_state", "UNKNOWN")
            chirality = result.get("chirality", 0.0)
        except Exception as e:
            print(f"[ERROR] Error al ejecutar el pipeline VQE: {e}")
            result = {}
            status = "ERROR"
            chirality = 0.0

        # 3. Fase de Gobernanza
        print(f"[PHASE 2] Estado H7: {status}")
        print(f"       -> Quiralidad: {chirality:.4f}")

        if status == "EQUILIBRIUM":
            print("[PHASE 3] El sistema es estable. Procediendo con el resultado cognitivo.")
        else:
            print(f"[WARNING] El sistema está en estado {status}. Se requiere amortiguación informacional.")

        return {
            "intent": intent,
            "h7_metrics": result,
            "governance": "APPROVED" if status == "EQUILIBRIUM" else "REJECTED"
        }

if __name__ == "__main__":
    # Test simple del puente
    bridge = VertexH7Bridge()
    test_prompt = "Optimiza la red de sensores para detectar rupturas de paridad en el vacío cuántico."
    final_state = bridge.run_controlled_vqe(test_prompt)
    print("\n--- Resultado de la Gobernanza ---")
    print(json.dumps(final_state, indent=2))
