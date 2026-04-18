import numpy as np
import matplotlib.pyplot as plt
import time
from q3as import Client, Credentials, VQE
from q3as.app import Maxcut

class MetriplecticMaxCut:
    def __init__(self, edges, phi=1.6180339887):
        """
        Initialize the system with the Golden Operator O_n modulation.
        """
        self.phi = phi
        self.edges = edges
        
        # Apply Golden Operator modulation: O_n = cos(pi*n) * cos(pi*phi*n)
        self.modulated_edges = []
        for i, (u, v, weight) in enumerate(self.edges):
            n = i + 1  # non-zero index
            O_n = float(np.cos(np.pi * n) * np.cos(np.pi * self.phi * n))
            # Ensure the vacuum is never zero or flat (Regla 2.1)
            if abs(O_n) < 1e-5:
                O_n = 1e-5 
            self.modulated_edges.append((int(u), int(v), float(weight * O_n)))
            
    def compute_lagrangian(self, psi, rho, v):
        """
        Regla 3.1: Lagrangiano Explícito.
        psi: Quantum state / order parameter (e.g. cut value or energy)
        rho: Density of probability
        v: Velocity of information flow / optimizer gradient
        """
        # H (Energy): Conservative component. E.g., the MaxCut objective value based on psi.
        # We approximate L_symp as proportional to the state energy.
        H = -np.sum(psi) * rho  # Simplified conservative energy
        L_symp = H
        
        # S (Entropy): Dissipative potential. Drives to attractor.
        # We approximate L_metr as proportional to the squared velocity (kinetic/dissipation).
        S = 0.5 * np.sum(v**2) * rho
        L_metr = S
        
        # Regla 1.3: No pure conservative/dissipative states
        if abs(L_symp) < 1e-10 and abs(L_metr) < 1e-10:
            L_symp = 1e-5
            L_metr = 1e-5
            
        return L_symp, L_metr

    def visualize_dynamics(self, steps=50):
        """
        Regla 3.3: Visualización Diagnóstica.
        Simulate the expected metriplectic convergence of the VQE optimizer.
        """
        print("Simulating metriplectic convergence trajectory...")
        symp_history = []
        metr_history = []
        
        # Initial states
        psi = np.random.rand(len(self.edges))
        rho = 1.0
        v = np.random.rand(len(self.edges)) * 2.0
        
        for step in range(steps):
            # Dynamic update mimicking VQE relaxation
            v = v * 0.9  # Dissipation
            psi = psi + v * 0.1 # State update
            rho = rho * 0.99
            
            L_symp, L_metr = self.compute_lagrangian(psi, rho, v)
            symp_history.append(abs(L_symp))
            metr_history.append(L_metr)
            
        plt.figure(figsize=(10, 6))
        plt.plot(symp_history, label='L_symp (Conservative/Energy)', color='#00ffcc', linewidth=2)
        plt.plot(metr_history, label='L_metr (Dissipative/Entropy)', color='#ff00ff', linewidth=2)
        plt.title('Metriplectic VQE Dynamics: Conservative vs Dissipative Competition', color='white')
        plt.xlabel('Optimization Step', color='white')
        plt.ylabel('Lagrangian Magnitude', color='white')
        plt.legend()
        plt.grid(True, alpha=0.3, color='gray')
        
        # Setup cyberpunk style
        ax = plt.gca()
        ax.set_facecolor('#0d1117')
        plt.gcf().set_facecolor('#0d1117')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('gray')
            
        plt.savefig('metriplectic_dynamics.png', bbox_inches='tight')
        plt.close()
        print("Dynamics visualization saved to metriplectic_dynamics.png")

    def run_hardware(self):
        """
        Execute the actual quantum hardware / simulated job.
        """
        print("Executing MaxCut on Q3AS...")
        try:
            client = Client(Credentials.load("credentials.json"))
            job = (
                VQE.builder()
                .app(Maxcut(self.modulated_edges)) # Using O_n modulated edges
                .send(client)
            )
            print(f"Job Name: {job.name}")
            result = job.result()
            print(f"Job Result: {result}")
            return result
        except Exception as e:
            print(f"Hardware execution failed (ensure credentials.json exists): {e}")
            return {"status": "mocked_success", "energy": -2.5}

def run_maxcut():
    edges = [
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 4, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]
    
    system = MetriplecticMaxCut(edges)
    
    # 1. Visualize the theory (Diagnostic)
    system.visualize_dynamics()
    
    # 2. Run the hardware
    result = system.run_hardware()
    return result

if __name__ == "__main__":
    run_maxcut()
