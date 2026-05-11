"""
database_connector.py
=====================
Conector nativo de alto rendimiento para Postgres (Neon) vía Connection Pooler.

Permite al H7 OS persistir estados de gobernanza, bitstreams de entropía
y logs de procesamiento de forma síncrona mediante psycopg2.
"""

import psycopg2
import os
from datetime import datetime

class H7DatabaseConnector:
    def __init__(self, config_path="h7_config.json"):
        # Carga manual de .env
        self._load_env()
        self.db_url = os.getenv("DATABASE_URL")
        self.conn = None
        
    def _load_env(self):
        """Lee el archivo .env manualmente e inyecta en os.environ."""
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value.strip("'").strip('"')

    def _get_connection(self):
        if self.conn is None or self.conn.closed:
            if not self.db_url:
                return None
            try:
                self.conn = psycopg2.connect(self.db_url)
                self.conn.autocommit = True
            except Exception as e:
                print(f"❌ Error conectando a Postgres: {e}")
                return None
        return self.conn

    def execute_query(self, query, params=None):
        conn = self._get_connection()
        if not conn:
            print("[WARN] Operando en modo local (DATABASE_URL no configurada).")
            return None
        
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                return True
        except Exception as e:
            print(f"❌ Error en query: {e}")
            return None

    def log_task(self, name, status, hex_signature):
        query = "INSERT INTO h7_tasks (name, status, hex_signature, timestamp) VALUES (%s, %s, %s, %s)"
        return self.execute_query(query, [name, status, hex_signature, datetime.now()])

    def log_governance_event(self, cycle, gap, load, integrity):
        query = "INSERT INTO h7_logs (cycle, gap, load_factor, integrity, timestamp) VALUES (%s, %s, %s, %s, %s)"
        return self.execute_query(query, [cycle, gap, load, integrity, datetime.now()])

    def auto_setup_tables(self):
        print("🗄️ Verificando tablas en Neon (Connection Pooler)...")
        q_tasks = """
            CREATE TABLE IF NOT EXISTS h7_tasks (
                id SERIAL PRIMARY KEY,
                name TEXT,
                status TEXT,
                hex_signature TEXT,
                timestamp TIMESTAMP
            );
        """
        q_logs = """
            CREATE TABLE IF NOT EXISTS h7_logs (
                id SERIAL PRIMARY KEY,
                cycle INTEGER,
                gap FLOAT,
                load_factor FLOAT,
                integrity FLOAT,
                timestamp TIMESTAMP
            );
        """
        self.execute_query(q_tasks)
        self.execute_query(q_logs)
        print("✅ Infraestructura Postgres lista.")

if __name__ == "__main__":
    db = H7DatabaseConnector()
    db.auto_setup_tables()
