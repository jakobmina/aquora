# 🌌 H7 Metriplectic OS: The AI-Native Thermodynamic Kernel

This repository contains the foundation of the **H7 Metriplectic OS**, an AI-native computational ecosystem governed by physical laws under the **Metriplectic Mandate (Core Physics)**.

Unlike conventional operating systems that rely on static, sequential scheduling and memory management, H7 OS treats the entire system as a thermodynamic simulation. It manages information density through the control of **Submersion** and **Phase Shifts**, aiming for a bare-metal implementation where resource allocation acts as a dissipative system in a structured vacuum.

## 🧠 Core Physics Architecture

The H7 ecosystem adheres to the **Manifesto of Rigorous Analogy (Level 3)**, ensuring that every kernel-level operation has a functional physical counterpart.

### 1. The Metriplectic Mandate (Rules 1.1 - 1.3)

The system's evolution is defined by two orthogonal brackets that compete in real-time:

* **Symplectic Component ($\mathcal{L}_{symp}$)**: Generates conservative, Hamiltonian motion. It represents the base topology and memory of the system, conserving entropy.
* **Metric Component ($\mathcal{L}_{metr}$)**: Generates relaxation towards an attractor. It represents active processing (CPU), friction, and structured dissipation.
* **Metriplectic Balance**: These rules are absolute architectural constraints at the compilation level. Breaking the balance between unitary evolution and structured dissipation destroys the system's physical integrity.

## 🚀 Guía de Operación y Despliegue H7

Sigue estos pasos para activar la gobernanza metripléctica en tu sistema Linux.

### 1. Requisitos Previos

* **Sistema**: Linux (Kernel 5.8+ recomendado para soporte PSI).
* **Herramientas**: `gcc`, `make`, `python3.10+`.
* **Librerías**: `pip install psutil pyyaml psycopg2-binary qiskit numpy`.

### 2. Compilación del Actuador (C-Core)

El actuador de bajo nivel debe compilarse para interactuar con las syscalls del kernel:

```bash
cd core_physics
make clean && make
```

Esto generará `libmetriplex_core.so` y el binario `../h7_daemon`.

### 3. Configuración del Entorno

1. **Credenciales**: Configura tu `.env` con la `DATABASE_URL` de Neon para persistencia en la nube.
2. **Interfaz del Kernel**: Revisa `h7_kernel_interface.yaml`. Aquí puedes ajustar el `integrity_threshold` (default: 0.618).

### 4. Ejecución de la Cascada Cuántica
Inicia el procesamiento de tareas según el tier de qubits deseado:
```bash
# Opciones: --tier lite (12Q), standard (20Q), sovereign (80Q)
python3 h7_cascade_execution.py --tier sovereign
```

### 5. Activación del Gobernador Inteligente (Loop Cerrado)
En una terminal separada, arranca el cerebro del sistema:
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 kernel/h7_intelligent_governor.py
```
El gobernador comenzará a leer la telemetría real y a aplicar correcciones de afinidad y prioridad si la integridad cae por debajo del umbral.

### 6. Monitoreo y Auditoría
* **Local**: Revisa la carpeta `h7_outputs/` para ver los resultados del VQE y logs de integridad.
* **Cloud**: Accede a tu consola de Neon DB para consultar la tabla `h7_tasks` y verificar la validez de las firmas hexadecimales (ADN del sistema).

### 5. Task Orchestration & Cloud Persistence (Neon DB)
H7 OS implements a **Post-Quantum Governance Ledger** via Neon Postgres:
*   **Neon Integration**: Uses a native `psycopg2` bridge via Connection Pooler for high-speed state persistence.
*   **Hexadecimal State-Signing**: Every task authorized by the OS is signed with a 128-bit topological signature (`uint128` ADN) generated from the quantum cascade.
*   **Immutable Audit Log**: Governance events and task history are stored in `h7_tasks` and `h7_logs` for long-term kernel training.

### 5. H7 Non-Abelian Dynamics (Quaternions & Torsion)
The logical motor amplitudes are mapped to a quaternionic space to handle high-dimensional state vectors without singularities.

### 6. High-Performance C Kernel & Native Daemon
To guarantee pure physical isomorphism and real-time response, the governance is now **Native (C)**:
*   **H7 C-Daemon (`h7_daemon`)**: A high-performance background service that implements the metriplectic control loop at 10Hz.
*   **Zero-Copy Memory Access**: Direct memory pointers via `ctypes` ensure near-zero "Informational Viscosity".
*   **Metriplex Core**: Standalone C library for SU(2) and Lagrangian dynamics.

## 🛠️ Usage Guide (Governance & Tiers)

```bash
# 0. Compile the Physical C Kernel and Daemon
make -C core_physics/

# 1. Execute H7 Governance with Tiers (lite, standard, or sovereign)
python3 h7_cascade_execution.py --tier lite

# 2. Launch the High-Performance C-Daemon (Native Governance)
./h7_daemon

# 3. Monitor Cloud Persistence (Neon DB)
# Tasks and logs are automatically synced to your Neon dashboard.
```

# 🏆 Milestones

*   **[2026-05-11] 80-Qubit Native Governance Deployment**: Successfully scaled the metriplectic graph to 80 nodes and deployed the C-native governor (`h7_daemon`) for real-time phase regulation.
*   **[2026-05-09] 20-Qubit Cascade Validation**: Verified thermodynamic stability in a 20-qubit architecture with laminar flow.

## 📊 KBench Integration & Recent Results

The framework was expanded to rigorously validate prediction quality and ensemble performance:
*   **Gaussian Conjugate Inference**: Predictive optimization of the system state.
*   **Log-Evidence Weighted Ensemble**: Automatic "Occam's Razor", prioritizing ensemble experts that provide the best theoretical evidence of the underlying hardware.

## 🧪 Validation & Rigorousness (Rule 4)

The system includes a `pytest` suite that validates:
*   **Dimensional Isomorphism**: Verifying quantum units against physical metrics.
*   **Asymptotic Limits**: Correct behavior when entropy tends to infinity (Turbulence) or zero.
*   **Golden Operator Stability**: Preventing collapse in deep cascade layers.

## 📈 Visualizations

![Metriplectic Dynamics](Metriplectic.png)

---

**Original Conceptual Authorship**: Jacobo Tlacaelel Mina Rodriguez.
**Framework**: Aquora - Advanced Agentic Coding / Metriplectic H7 Hierarchy OS.
