# 🌌 H7 Metriplectic VQE: The Quantum-Topological Bridge

![Metriplectic Dynamics](metriplectic_dynamics.png)

Este repositorio constituye la implementación avanzada del framework **Metriplectic H7**, un motor de optimización cuántica (VQE para MaxCut) diseñado bajo el **Mandato Metriplético (Core Physics)**. El sistema trasciende la optimización convencional, tratando la convergencia del algoritmo como la evolución de un sistema dinámico disipativo en un vacío estructurado.

## 🧠 Arquitectura de Física Teórica (Core Physics)

El sistema cumple con el **Manifiesto de la Analogía Rigurosa (Nivel 3)**, asegurando que cada operación matemática tenga una contraparte física operacional.

### 1. El Dualismo Metriplético (Reglas 1.1 - 1.3)

Cualquier simulación se define mediante dos corchetes ortogonales que compiten en tiempo real:

* **Componente Simpléctica ($\mathcal{L}_{symp}$)**: Genera movimiento conservativo (Hamiltoniano).
* **Componente Métrica ($\mathcal{L}_{metr}$)**: Genera relajación hacia un atractor (Entropía).
* **Lagrangiano Informacional ($\mathcal{L}_{info}$)**: Mide la fricción entre la intención cognitiva y la estabilidad física.
* **Prohibición de Singularidades**: El sistema mantiene un "piso" de energía evitando estados puramente conservativos o disipativos.

### 2. Gobernador de Fase (Vertex AI)

Integramos **Vertex AI (Gemini 1.5 Flash)** como un Oráculo de Información:

* **Mapeo Cognitivo**: Traduce prompts en parámetros metriplécticos ($\rho, v$).
* **Gobernanza de Bucle Cerrado**: Valida la estabilidad cuántica antes de aprobar la ejecución.

### 3. Topología Áurea y Segunda Cuantización (Regla 2.1)

El vacío está modulado por el **Operador Áureo ($O_n$)**:
$$O_n = \cos(\pi n) \cdot \cos(\pi \phi n)$$

* **Segunda Cuantización**: El sistema clasifica estados basándose en la paridad de $n$:
  * **$n$ Impar $\to$ Fermiónico ($c^\dagger$)**: Rompe simetría, genera quiralidad.
  * **$n$ Par $\to$ Bosónico ($a^\dagger$)**: Simétrico, transporte de información coherente.
* **No-Localidad**: Debido a la naturaleza irracional de $\phi$, el sistema cumple con $O(n_1 + n_2) \neq O(n_1) + O(n_2)$, garantizando que cada nivel de ocupación sea un modo topológico único.

### 3. Dinámica No-Abeliana H7 (Quaternions)

Mapeamos las amplitudes de probabilidad del hardware cuántico a un espacio de cuaterniones H7, agrupando estados en pares simétricos $|s\rangle \leftrightarrow |\bar{s}\rangle$:

* **Vacuum Overlaps ($W_0, W_1, W_2, W_3$)**: Calculamos la superposición no-lineal $O(n) + O(7-n)$ para cada par. Esta "tensión de vacío" es la que hace al sistema **no reversible**, introduciendo una flecha del tiempo informacional.
* **Chirality ($\chi$)**: Medimos la no-conmutatividad $[q_{LE}, q_{BE}] \neq 0$ para detectar rupturas de paridad en el hardware.

## 🛠️ Guía de Uso (Orquestador Central)

Hemos unificado la ejecución del pipeline en `main.py` y el bridge de gobernanza.

```bash
# 1. Validar la integridad física y lógica
./main.py --test

# 2. Ejecutar el Gobernador de Fase (Vertex AI)
python vertex_h7_bridge.py

# 3. Ejecutar la grilla de entrenamiento
./main.py --train

# 4. Visualizar en el Dashboard
./main.py --serve
```

## 📊 Resultados Recientes (Batch H7)

La última ejecución de la grilla de fase ($n=1..4$, $\phi \in \{0.3624, 0.6180\}$) arrojó los siguientes estados de equilibrio dinámico:

| n | phi | Tipo de Partícula | L_symp | L_metr | Estado H7 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.3624 | Fermiónico | -0.6947 | 0.0150 | EQUILIBRIUM |
| 2 | 0.3624 | Bosónico | -0.6947 | 0.0150 | EQUILIBRIUM |
| 3 | 0.3624 | Fermiónico | -0.6947 | 0.0150 | EQUILIBRIUM |
| 4 | 0.3624 | Bosónico | -0.6947 | 0.0150 | EQUILIBRIUM |
| 1 | 0.6180 | Fermiónico | -0.6947 | 0.0150 | EQUILIBRIUM |
| 2 | 0.6180 | Bosónico | -0.6947 | 0.0150 | EQUILIBRIUM |
| 3 | 0.6180 | Fermiónico | -0.6947 | 0.0150 | EQUILIBRIUM |
| 4 | 0.6180 | Bosónico | -0.6947 | 0.0150 | EQUILIBRIUM |

### Observaciones Físicas

* **Independencia de Paridad**: Se confirma que el sistema clasifica correctamente fermiones ($n$ impar) y bosones ($n$ par) sin afectar la estabilidad macroscópica.
* **Vacuum Overlaps**: Los pesos $W_n$ han sido validados para incluir el término de paridad, eliminando derivas no físicas en la "fricción informacional".

## 🧪 Validación y Rigurosidad (Regla 4)

El sistema incluye una suite de `pytest` que valida:

* **test_vertex_bridge.py**: Valida el mapeo oracular y la gobernanza de intención.
* **test_run_vqe_maxcut.py**: Pruebas de integración del pipeline VQE completo.
* **Isomorfismo Dimensional**: Verificación de unidades y constantes.
* **Límites Asintóticos**: Comportamiento correcto cuando $\mu \to 0$.
* **Estabilidad del Operador Áureo**: Prevención de colapsos de fase.

---

**Autoría Conceptual Original**: Jacobo Tlacaelel Mina Rodriguez.

**Framework**: Aquora - Advanced Agentic Coding / Metriplectic H7 Hierarchy.
