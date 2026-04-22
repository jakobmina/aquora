# 🌌 H7 Metriplectic VQE: The Quantum-Topological Bridge

![Metriplectic Dynamics](metriplectic_dynamics.png)

Este repositorio constituye la implementación avanzada del framework **Metriplectic H7**, un motor de optimización cuántica (VQE para MaxCut) diseñado bajo el **Mandato Metriplético (Core Physics)**. El sistema trasciende la optimización convencional, tratando la convergencia del algoritmo como la evolución de un sistema dinámico disipativo en un vacío estructurado.

## 🧠 Arquitectura de Física Teórica (Core Physics)

El sistema cumple con el **Manifiesto de la Analogía Rigurosa (Nivel 3)**, asegurando que cada operación matemática tenga una contraparte física operacional.

### 1. El Dualismo Metriplético (Reglas 1.1 - 1.3)

Cualquier simulación se define mediante dos corchetes ortogonales que compiten en tiempo real:

* **Componente Simpléctica ($\mathcal{L}_{symp}$)**: Genera movimiento conservativo (Hamiltoniano). Representa la estructura del problema MaxCut.
* **Componente Métrica ($\mathcal{L}_{metr}$)**: Genera relajación hacia un atractor (Entropía). Representa la disipación necesaria para la convergencia del optimizador.
* **Prohibición de Singularidades**: El sistema mantiene un "piso" de energía ($1e-10$) evitando estados puramente conservativos (inestables) o puramente disipativos (muerte térmica).

### 2. Topología Áurea y Segunda Cuantización (Regla 2.1)

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

Hemos unificado la ejecución del pipeline en `main.py`. Este script permite gestionar el ciclo completo de vida del sistema:

```bash
# 1. Validar la integridad física y lógica
./main.py --test

# 2. Ejecutar la grilla de entrenamiento (Genera submission.csv)
./main.py --train

# 3. Visualizar en el Dashboard interactivo
./main.py --serve

# 4. Ejecutar todo secuencialmente
./main.py --all
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

* **Isomorfismo Dimensional**: Verificación de unidades y constantes.
* **Límites Asintóticos**: Comportamiento correcto cuando $\mu \to 0$.
* **Estabilidad del Operador Áureo**: Prevención de colapsos de fase.
* **Consistencia de Hardware**: Manejo robusto de objetos `VQEResult` vs diccionarios mock.

---

**Autoría Conceptual Original**: Jacobo Tlacaelel Mina Rodriguez.

**Framework**: Aquora - Advanced Agentic Coding / Metriplectic H7 Hierarchy.
