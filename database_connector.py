"""
database_connector.py
=====================
Conector de alta velocidad para Neon DB vía SQL-over-HTTP.

Permite al H7 OS persistir estados de gobernanza, bitstreams de entropía
y logs de procesamiento de forma asíncrona.
"""

import requests
import json
import os

class H7DatabaseConnector:
    def __init__(self, config_path="h7_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Priorizar variables de entorno para Project ID y API Key
        self.project_id = os.getenv("NEON_PROJECT_ID", self.config.get("neon_project_id"))
        self.api_key = os.getenv("NEON_API_KEY")
        
        # Construcción dinámica de la URL de la API SQL
        if self.project_id:
            self.api_url = f"https://{self.project_id}.apirest.c-7.us-east-1.aws.neon.tech/neondb/sql"
        else:
            self.api_url = self.config["neon_db_url"].replace("/rest/v1", "/sql")

    def execute_query(self, sql_query, params=None):
        if not self.api_key:
            print("[WARN] NEON_API_KEY no detectada. Operando en modo local (sin persistencia).")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": sql_query,
            "params": params or []
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[ERROR] Fallo en la persistencia Neon: {e}")
            return None

    def log_governance_event(self, cycle, gap, load, integrity):
        """Persiste un evento de gobernanza en la tabla h7_logs."""
        query = """
            INSERT INTO h7_logs (cycle, gap, load_factor, integrity, timestamp)
            VALUES (%s, %s, %s, %s, NOW())
        """
        return self.execute_query(query, [cycle, gap, load, integrity])

# Test rápido (Mock)
if __name__ == "__main__":
    db = H7DatabaseConnector()
    print(f"📡 Conector configurado para: {db.api_url}")
