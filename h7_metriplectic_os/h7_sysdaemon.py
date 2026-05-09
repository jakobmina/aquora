"""
h7_sysdaemon.py
================
Demonio de telemetría a nivel de sistema. Monitorea los recursos del host
(CPU, Memoria, I/O) y los alimenta al H7 Bayesian Oracle para calcular
la métrica de integridad predictiva en tiempo real.
Actúa como un "hypervisor termodinámico".
"""

import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
from h7_bayesian_oracle import H7BayesianOracle, φ

class H7SysDaemon:
    def __init__(self, tick_rate=1.0, history_size=20):
        self.tick_rate = tick_rate
        self.history_size = history_size
        self.oracle = H7BayesianOracle(n_features=4)
        
        # Buffer de inicialización
        self.buffer_X = []
        self.buffer_y = []
        
        # Ultimos stats para derivadas (velocidad de cambio)
        self.last_net = psutil.net_io_counters()
        self.last_disk = psutil.disk_io_counters()
        self.last_time = time.time()
        
        print("\n" + "="*80)
        print("🚀 [virtqemud-h7] Iniciando Demonio del Sistema H7")
        print(f"📡 Tick Rate: {self.tick_rate} Hz")
        print("="*80)

    def _get_system_state(self):
        """Lee el estado del hardware y retorna features y un target sintético termodinámico."""
        now = time.time()
        dt = now - self.last_time
        if dt == 0: dt = 1e-6
        
        # 1. CPU (Componente activa / Disipativa)
        cpu_percent = psutil.cpu_percent(interval=None) / 100.0
        
        # 2. RAM (Componente de memoria / Simpléctica)
        mem = psutil.virtual_memory()
        ram_percent = mem.percent / 100.0
        
        # 3. Disk I/O (Fricción de estado)
        disk = psutil.disk_io_counters()
        disk_bytes = (disk.read_bytes + disk.write_bytes) - (self.last_disk.read_bytes + self.last_disk.write_bytes)
        disk_rate = (disk_bytes / dt) / (1024 * 1024) # MB/s
        disk_rate_norm = np.clip(disk_rate / 100.0, 0, 1) # Normalizado aprox
        
        # 4. Network I/O (Flujo de entrelazamiento)
        net = psutil.net_io_counters()
        net_bytes = (net.bytes_sent + net.bytes_recv) - (self.last_net.bytes_sent + self.last_net.bytes_recv)
        net_rate = (net_bytes / dt) / (1024 * 1024) # MB/s
        net_rate_norm = np.clip(net_rate / 10.0, 0, 1) # Normalizado aprox
        
        self.last_disk = disk
        self.last_net = net
        self.last_time = now
        
        # X: Features (4 dimensiones)
        x = np.array([cpu_percent, ram_percent, disk_rate_norm, net_rate_norm])
        
        # y: "Energía del sistema" observada. 
        # Combinamos las métricas usando pesos áureos para simular una carga global termodinámica
        # La CPU genera entropía, la red genera intercambio, etc.
        y_obs = (cpu_percent * φ) + (ram_percent / φ) + (disk_rate_norm * φ) + (net_rate_norm)
        
        return x, y_obs

    def run(self):
        print("Recolectando fase de inicialización (Mar de Dirac)...")
        # Fase 1: Recolección del buffer
        for _ in range(self.history_size):
            x, y = self.get_state_safe()
            self.buffer_X.append(x)
            self.buffer_y.append(y)
            time.sleep(self.tick_rate)
            
        print("Entrenando Oráculo Base...")
        df_X = pd.DataFrame(self.buffer_X, columns=["cpu", "ram", "disk", "net"])
        series_y = pd.Series(self.buffer_y, name="energy")
        self.oracle.fit_conjugate_gaussian(df_X, series_y)
        
        print("\n" + "="*80)
        print("🔮 [virtqemud-h7] Gobernador Activo - Monitoreo Termodinámico")
        print("="*80)
        print(f"{'TIMESTAMP':<12} | {'CPU %':<8} | {'RAM %':<8} | {'PRED MEAN':<10} | {'INTEGRITY':<10} | {'STATE'}")
        print("-" * 80)
        
        # Fase 2: Bucle Infinito
        try:
            while True:
                x, y = self.get_state_safe()
                
                # Deslizar ventana (mantener history size)
                self.buffer_X.pop(0)
                self.buffer_y.pop(0)
                self.buffer_X.append(x)
                self.buffer_y.append(y)
                
                df_X_current = pd.DataFrame(self.buffer_X, columns=["cpu", "ram", "disk", "net"])
                series_y_current = pd.Series(self.buffer_y, name="energy")
                
                # Actualizar el prior del oráculo con la nueva ventana (dinámica métrica)
                self.oracle.fit_conjugate_gaussian(df_X_current, series_y_current)
                
                # Calcular integridad sobre la observación actual
                x_df = pd.DataFrame([x], columns=["cpu", "ram", "disk", "net"])
                y_series = pd.Series([y])
                
                integrity = self.oracle.compute_h7_integrity_metric(x_df, y_series)
                pred = self.oracle.predict_posterior_predictive(x_df)
                pred_mean = pred['mean'][0]
                
                # Evaluar estado termodinámico
                if integrity > 0.8:
                    state = "🟩 FLUJO LAMINAR"
                elif integrity > 0.4:
                    state = "🟨 FLUJO TRANSICIONAL"
                else:
                    state = "🟥 TURBULENCIA ENTRÓPICA"
                    
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                print(f"{timestamp:<12} | {x[0]*100:>5.1f}% | {x[1]*100:>5.1f}% | {pred_mean:>8.3f}   | {integrity:>8.5f} | {state}")
                
                time.sleep(self.tick_rate)
                
        except KeyboardInterrupt:
            print("\n[virtqemud-h7] Gobernador Detenido por el usuario.")

    def get_state_safe(self):
        return self._get_system_state()

# helper para el sleep
def append_sleep(rate):
    time.sleep(rate)

def main():
    daemon = H7SysDaemon(tick_rate=0.5, history_size=10) # 0.5s para la demo
    daemon.run()

if __name__ == "__main__":
    main()
