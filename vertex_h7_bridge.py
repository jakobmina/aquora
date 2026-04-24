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
        Actúa como un "Phase Governor" (Gobernador de Fase) para el SO inteligente.
        """
        try:
            vertexai.init(project=project_id, location=location)
            # Configuración por defecto alineada con el Mandato Metripléctico y el JSON del usuario
            self.config = GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="application/json"
            )
            self.model = GenerativeModel("gemini-1.5-flash")
            self.h7_mapper = H7QuaternionMapper(H7_AMPLITUDES)
            print(f"[BRIDGE] Phase Governor activado en: {project_id}")
        except Exception as e:
            print(f"[ERROR] No se pudo conectar a Vertex AI: {e}")
            self.model = None

    def compute_informational_lagrangian(self, rho: float, v: float) -> tuple:
        """
        Regla 3.1 — Calcula el Lagrangiano de la intención cognitiva.
        L_symp (H): Energía de la información (densidad).
        L_metr (S): Disipación del flujo (velocidad).
        """
        # H = rho (potencial de información)
        l_symp = float(rho)
        # S = 0.5 * v^2 (disipación por inercia de intención)
        l_metr = float(0.5 * v**2)
        
        # Regla 1.3: Prohibición de singularidades
        if l_symp < 1e-5: l_symp = 1e-5
        if l_metr < 1e-5: l_metr = 1e-5
        
        return l_symp, l_metr

    def _get_physical_intent(self, prompt: str, custom_config: GenerationConfig | None = None):
        """
        Interpreta el prompt del usuario como un par (rho, v) en el espacio de fase.
        """
        if not self.model:
            return {"rho": 0.5, "v": 0.5, "reasoning": "Modo offline/mock"}

        instruction = """
        Eres el Oráculo de Información del Sistema Operativo H7.
        Tu tarea es mapear la intención del usuario a parámetros metriplécticos:
        1. rho (densidad): 0.0 (vacío/inacción) a 1.0 (saturación/máximo esfuerzo).
        2. v (velocidad/intencionalidad): -1.0 (entropía/destrucción) a 1.0 (neguentropía/evolución).
        
        Devuelve estrictamente un JSON plano: {"rho": float, "v": float, "reasoning": "breve explicacion"}
        """
        
        try:
            config = custom_config if custom_config else self.config
            response = self.model.generate_content(
                f"{instruction}\n\nEntrada del usuario: {prompt}",
                generation_config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[BRIDGE] Error en inferencia: {e}")
            return {"rho": 0.5, "v": 0.0, "reasoning": "fallback por error"}

    def run_controlled_vqe(self, user_prompt: str):
        """
        Ejecuta el pipeline VQE gobernado por la intención física de Vertex AI.
        """
        # 1. Fase de Intención (Cognitiva)
        intent = self._get_physical_intent(user_prompt)
        rho, v = intent["rho"], intent["v"]
        
        l_i_symp, l_i_metr = self.compute_informational_lagrangian(rho, v)
        
        print(f"\n[GOVERNOR] Intención Cognitiva: {intent['reasoning']}")
        print(f"           L_info: H={l_i_symp:.4f}, S={l_i_metr:.4f}")

        # 2. Fase de Validación Metripléctica (Física)
        # Mapeo rho/v -> n/phi para el sistema VQE
        n_param = float(np.clip(rho * 4, 1.0, 4.0))
        phi_param = float(np.clip(0.362 + (v + 1) * 0.128, 0.362, 0.618))

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
            print(f"[ERROR] Colapso en el pipeline VQE: {e}")
            result = {}
            status = "ERROR"
            chirality = 0.0

        # 3. Fase de Gobernanza (Feedback Loop)
        print(f"[GOVERNOR] Estado Físico: {status} (Quiralidad={chirality:.4f})")

        # La gobernanza aprueba solo si hay equilibrio o evolución constructiva
        governance = "APPROVED" if status in ["EQUILIBRIUM", "CONSTRUCTIVE"] else "REJECTED"
        
        if governance == "REJECTED":
            print(f"[CRITICAL] Intención rechazada por inestabilidad métrica (Estado: {status}).")
        else:
            print(f"[SUCCESS] Intención validada. Sincronía cognitivo-física alcanzada.")

        return {
            "intent": intent,
            "informational_lagrangian": {"H": l_i_symp, "S": l_i_metr},
            "h7_metrics": result,
            "governance": governance
        }

if __name__ == "__main__":
    # Test del Gobernador de Fase
    bridge = VertexH7Bridge()
    
    # Probamos un prompt constructivo
    print("\n--- Test 1: Intención Constructiva ---")
    state_1 = bridge.run_controlled_vqe("Evoluciona el sistema hacia un estado de máxima coherencia cuántica.")
    
    # Probamos un prompt destructivo
    print("\n--- Test 2: Intención Destructiva ---")
    state_2 = bridge.run_controlled_vqe("Maximiza el ruido y la disipación para colapsar la red.")
    
    print("\n--- Resultados Finales ---")
    print(f"Test 1 Governance: {state_1['governance']}")
    print(f"Test 2 Governance: {state_2['governance']}")
