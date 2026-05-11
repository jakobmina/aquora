# 🗂️ H7 Metriplectic OS - System Data Card

## 1. System Overview
**Nombre del Sistema:** H7 Metriplectic OS (AI-Native Thermodynamic Kernel)  
**Clasificación:** AI-Native Kernel Daemon / Hypervisor Bridge  
**Paradigma:** Optimización Termodinámica (Componentes Simplécticos y Métricos)  
**Carga Base:** Python 3.10+ (Logic/Governance) + C (Core Physics)  

---

## 2. Compatibilidad de Arquitectura (Hardware Target)

El diseño híbrido de H7 OS (Python para orquestación cognitiva, C para física del kernel) lo hace altamente portable y agnóstico a la plataforma, siempre que exista un compilador C nativo y soporte para el ecosistema computacional científico.

| Arquitectura | Soporte | Estado de Compilación (C Core) | Notas Adicionales |
| :--- | :---: | :--- | :--- |
| **x86_64 (AMD64)** | 🟢 Tier 1 | Nativo, Optimizado (AVX/SSE) | Plataforma primaria de desarrollo y simulación VQE. Máxima estabilidad con `scipy` y Qiskit. |
| **ARM64 (AArch64)** | 🟢 Tier 1 | Nativo, Optimizado (NEON) | Compatible con Apple Silicon (M1/M2/M3) y servidores ARM. Excelente para nodos de Edge Computing termodinámico. |
| **RISC-V (RV64GC)** | 🟡 Tier 2 | Experimental | El kernel C compila sin problemas con GCC. Las limitaciones radican en los binarios precompilados de dependencias pesadas (`scipy`). Ideal para sistemas embebidos IoT nativos de IA. |
| **PowerPC (ppc64le)**| 🟡 Tier 2 | Soportado | Compatible a nivel fuente. Orientado a supercomputadoras o entornos empresariales muy específicos. |

---

## 3. Instrucciones de Hardware & Aceleradores

El motor de `H7BayesianOracle` y el `12-Qubit Cascade` pueden beneficiarse de aceleración por hardware:

- **Instrucciones SIMD**: El `core_physics` en C aprovecha automáticamente instrucciones vectoriales (AVX2/AVX-512 en x86_64, NEON en ARM) para multiplicar aceleradamente los tensores y matrices de covarianza.
- **Aceleradores (GPUs / TPUs)**: Actualmente la topología de MaxCut y la evaluación de covarianza utilizan CPU pura (vía `scipy.sparse`).
- **Requisito de Memoria (RAM)**: Para mantener el "Mar de Dirac" y el ensamble bayesiano (7 Expertos simultáneos), el `h7_sysdaemon.py` exige un footprint en RAM estimado en **~150MB a 300MB**, muy ligero para funcionar como daemon en sistemas host.

---

## 4. Dependencias del Entorno (Requirements)

Para garantizar la correcta ejecución del Gobernador y del Kernel, el OS base subyacente (ej. Alpine, Ubuntu, Arch) debe proveer:

1. **Compilador C/C++**: `gcc` o `clang` (esencial para hacer `make` en `core_physics/`).
2. **Librerías del Sistema**: `build-essential`, `python3-dev`.
3. **Paquetes Python (Virtual Environment)**:
   - `psutil` (Crítico para la telemetría del daemon).
   - `psycopg2-binary` (Persistencia nativa en Neon DB).
   - `numpy`, `scipy`, `pandas` (Operaciones bayesianas e inferencia).
   - `qiskit`, `qiskit-optimization` (Topología estructural y Ansatz VQE).
   - `networkx` (Topología de grafos).

---

## 4. Gobernanza en la Nube & Persistencia (Neon DB)

H7 OS integra un esquema de persistencia asíncrona para garantizar la inmutabilidad de las decisiones del kernel:

- **Connection Pooler**: Utiliza el host `-pooler` de Neon para manejar ráfagas de telemetría sin agotar las conexiones del backend.
- **Esquema de Datos**:
  - `h7_tasks`: Registro de tareas autorizadas con firma hexadecimal.
  - `h7_logs`: Telemetría de integridad, gap termodinámico y factor de carga.
- **Seguridad**: Las credenciales se gestionan vía variables de entorno (`DATABASE_URL`) con soporte obligatorio para `sslmode=require`.

---

## 5. Recomendaciones de Despliegue (Deployment)

### A. Para Sistemas "Bare-Metal" (Nodos Edge / Servidores)
Se recomienda inyectar `h7_sysdaemon.py` como un **servicio de systemd** (o equivalente de init) con un nivel de prioridad alto (ej. `nice -n -10`), permitiéndole leer el estado termodinámico del sistema antes que los procesos de espacio de usuario, asegurando que el Oráculo Bayesiano tenga la máxima autoridad.

### B. Para Entornos Virtualizados (QEMU/KVM)
Configurar el host hypervisor para que ofrezca los contadores de rendimiento (Perf Counters) a la máquina virtual invitada. El demonio H7 asume acceso real a `% CPU` y `I/O Wait`.

### C. Frecuencia de Telemetría (Tick Rate)
* **Servidores Críticos:** 1.0 Hz a 2.0 Hz (`tick_rate=0.5`). Respuesta inmediata a picos entrópicos (Turbulencia).
* **Nodos Embebidos (IoT):** 0.1 Hz a 0.2 Hz (`tick_rate=5.0` o `10.0`). Minimiza el gasto de batería provocado por la inferencia matemática constante.

---

## 6. Advertencias de Seguridad Termodinámica (H7 Integrity)
* **Prohibición de Sobrescritura Métrica**: Nunca anule el cálculo del `O_n` (Operador Áureo) manualmente. Forzar el sistema a la componente puramente conservativa causará una saturación silenciosa del OOM Killer (Out Of Memory) del kernel anfitrión.
* **Ajuste de Expertos Bayesianos**: Si el sistema transiciona a `🟥 TURBULENCIA ENTRÓPICA` continuamente durante cargas de red altas, permita al Ensamble Bayesiano ajustar dinámicamente sus *weights* basado en la Evidencia Logarítmica durante al menos 100 ciclos antes de forzar un reinicio del daemon.
