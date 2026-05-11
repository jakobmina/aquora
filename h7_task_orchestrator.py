"""
h7_task_orchestrator.py
=======================
Orquestador de Tareas Local (El 'n8n' del Host).

Este motor ejecuta flujos de procesamiento basados en el estado del kernel:
1. Lee tareas de un archivo YAML/JSON.
2. Consulta al H7 Auto-Governor (Laminosidad).
3. Ejecuta tareas concurrentes solo si I_H7 > Umbral.
4. Si hay turbulencia, pausa las tareas de fondo para liberar CPU.

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import json
import subprocess
import time
import os
from database_connector import H7DatabaseConnector

class H7TaskOrchestrator:
    def __init__(self, config_path="h7_config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.db = H7DatabaseConnector(config_path)
        self.db.auto_setup_tables() # Asegurar infraestructura
        self.queue = []
        self.running_tasks = []

    def add_task(self, name, command, priority=1):
        self.queue.append({
            "name": name,
            "command": command,
            "priority": priority,
            "status": "PENDING"
        })
        print(f"➕ Tarea añadida a la cola: {name}")

    def check_integrity(self):
        """Lee el último estado H7 y valida si es seguro ejecutar tareas."""
        try:
            with open("h7_outputs/h7_cascade_80q_latest.json", "r") as f:
                state = json.load(f)
            
            # Mapeo de integridad desde las métricas de la firma
            metrics = state.get("metrics", {})
            integrity = metrics.get("h7_entropy", 0.0)
            
            # Normalización (si > 1, es estable)
            self.current_signature = state.get("hex_signature", "UNKNOWN")
            
            if integrity > self.config.get("min_integrity", 0.3623):
                return True, integrity
            return False, integrity
        except Exception:
            return False, 0.0

    def run_step(self):
        is_stable, integrity = self.check_integrity()
        threshold = self.config["h7_governance"]["min_integrity_to_process"]

        if not is_stable:
            print(f"⚠️ [TURBULENCIA] Integridad ({integrity:.4f}) < Umbral ({threshold}). Pausando orquestador.")
            return

        if self.queue:
            task = self.queue.pop(0)
            print(f"🚀 [EJECUCIÓN] Iniciando: {task['name']} (Integridad: {integrity:.4f})")
            
            # Obtener firma del último estado
            hex_sig = "N/A"
            if os.path.exists("h7_outputs/h7_cascade_80q_latest.json"):
                 with open("h7_outputs/h7_cascade_80q_latest.json", "r") as f:
                     hex_sig = json.load(f).get("hex_signature", "N/A")

            try:
                # Ejecución en host local
                process = subprocess.Popen(task['command'], shell=True)
                self.running_tasks.append({"name": task['name'], "proc": process})
                
                # Registrar inicio en Neon con la firma topológica
                query = "INSERT INTO h7_tasks (name, status, hex_signature) VALUES (%s, 'RUNNING', %s)"
                self.db.execute_query(query, [task['name'], hex_sig])
            except Exception as e:
                print(f"❌ Error al ejecutar {task['name']}: {e}")

    def monitor(self):
        while True:
            self.run_step()
            # Limpiar tareas terminadas
            self.running_tasks = [t for t in self.running_tasks if t["proc"].poll() is None]
            time.sleep(2)

if __name__ == "__main__":
    orchestrator = H7TaskOrchestrator()
    
    # Ejemplo de tareas de procesamiento local
    orchestrator.add_task("Compilar C-Kernel", "make -C core_physics/ clean && make -C core_physics/")
    orchestrator.add_task("Respaldo de Salidas", "tar -czf h7_outputs_backup.tar.gz h7_outputs/")
    orchestrator.add_task("Limpiar Temporales", "rm -rf /tmp/h7_scratch_*")
    
    print("\n🛰️ Orquestador H7 iniciado. Monitoreando laminosidad del host...")
    orchestrator.monitor()
