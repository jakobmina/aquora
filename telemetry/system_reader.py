"""
telemetry/system_reader.py
=========================
Lector de métricas REALES del sistema Linux.
Provee los observables físicos para el Mandato Metripléctico.
"""

import psutil
import os

class H7SystemTelemetry:
    """Lee métricas REALES del sistema Linux para alimentar el Oracle."""
    
    def get_cpu_metrics(self):
        """Obtiene el porcentaje de uso de CPU."""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_pressure(self):
        """Lee Pressure Stall Information (PSI) de memoria."""
        try:
            if os.path.exists('/proc/pressure/memory'):
                with open('/proc/pressure/memory', 'r') as f:
                    lines = f.readlines()
                    # Extraer 'some' avg10
                    # Ejemplo: some avg10=0.00 avg60=0.00 avg300=0.00 total=0
                    some_line = lines[0].split()
                    avg10 = float(some_line[1].split('=')[1])
                    return avg10
            else:
                # Fallback a uso de RAM convencional
                return psutil.virtual_memory().percent
        except Exception as e:
            print(f"⚠️ Error leyendo PSI: {e}")
            return psutil.virtual_memory().percent
    
    def get_io_activity(self):
        """Lee actividad de disco."""
        try:
            counters = psutil.disk_io_counters()
            return {
                'read_bytes': counters.read_bytes if counters else 0,
                'write_bytes': counters.write_bytes if counters else 0,
            }
        except:
            return {'read_bytes': 0, 'write_bytes': 0}
    
    def get_process_workload(self):
        """Identifica procesos con alta carga."""
        workloads = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                # Solo reportar procesos con carga significativa
                if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 5.0:
                    workloads.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return workloads

if __name__ == "__main__":
    telemetry = H7SystemTelemetry()
    print(f"CPU: {telemetry.get_cpu_metrics()}%")
    print(f"Mem Pressure (PSI): {telemetry.get_memory_pressure()}")
    print(f"IO: {telemetry.get_io_activity()}")
