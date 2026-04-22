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

## 🚀 Pipeline de Producción

### Ejecución de Submission Batch

Para generar un dataset completo de 8 puntos de fase (n=1..4, phi=[0.362, 0.618]) para la competencia:

```bash
python generate_submission.py --out submission.csv --credentials credentials.json
```

### Ejecución Individual

Para análisis profundo de un solo punto de fase:

```bash
python run_vqe_maxcut.py
```

## 🧪 Validación y Rigurosidad (Regla 4)

El sistema incluye una suite de `pytest` que valida:

* **Isomorfismo Dimensional**: Verificación de unidades y constantes.
* **Límites Asintóticos**: Comportamiento correcto cuando $\mu \to 0$.
* **Estabilidad del Operador Áureo**: Prevención de colapsos de fase.

Ejecutar tests:

```bash
python -m pytest tests/
```

---
**Desarrollado bajo los principios del Manifiesto de la Analogía Rigurosa.**
*Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.  Mina Rodriguez, J. T. (2025) “El Marco de la Analogía Rigurosa: Una Guía para Validar Mapeos Físico-Matemáticos,” Primera parte .
