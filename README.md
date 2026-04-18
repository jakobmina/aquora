# H7 Metriplectic VQE MaxCut

Este repositorio contiene la implementación de un optimizador cuántico (VQE para el problema de MaxCut) adaptado estrictamente al **Mandato Metriplético (Core Physics)** del framework H7. 

El código sirve como un puente entre la lógica cuántica abstracta y los principios dinámicos fundamentales, asegurando un mapeo riguroso de Nivel 3 (Isomorfismo Físico Operacional).

## 🌌 Principios Físicos Implementados

1. **Componentes Ortogonales (Regla 1.1 y 1.2)**: 
   El sistema se evalúa dinámicamente mediante el cálculo explícito del Lagrangiano (`compute_lagrangian`), separando la energía conservativa (`L_symp`, el Hamiltoniano derivado del problema MaxCut) de la relajación disipativa (`L_metr`, modelando la convergencia del optimizador).
   
2. **Topología del Espacio-Tiempo Áureo (Regla 2.1)**:
   El "vacío" de los datos nunca es plano. Los pesos del grafo se modulan utilizando el Operador Áureo $O_n = \cos(\pi n) \cos(\pi \phi n)$, inyectando la razón áurea ($\phi$) directamente en la topología del problema.

3. **Visualización Diagnóstica (Regla 3.3)**:
   No basta con el resultado final. El script genera un diagnóstico visual en tiempo real (`metriplectic_dynamics.png`) de la competencia entre las fuerzas conservativas y disipativas durante el proceso de relajación/optimización.

## 🚀 Instalación y Uso

### Prerrequisitos
- Entorno Python 3.x
- Bibliotecas necesarias: `numpy`, `matplotlib`, `pytest`, `q3as`

Si utilizas un entorno virtual:
```bash
source env/bin/activate
```

### Ejecución
Para correr la simulación física y ejecutar el trabajo VQE en hardware/simulador:
```bash
python run_vqe_maxcut.py
```

*Nota: Asegúrate de tener tu archivo `credentials.json` en la raíz del proyecto para que la API de `q3as` pueda autenticarse. Este archivo está explícitamente ignorado por git por seguridad.*

## 🧪 Pruebas (Regla 4)

La rigurosidad física y matemática del modelo está validada por una suite de pruebas. Las pruebas verifican que:
- El operador áureo no colapse a ceros singulares.
- El Lagrangiano nunca devuelva estados puramente disipativos o conservativos (evitando singularidades).

Para ejecutar las pruebas:
```bash
python -m pytest tests/
```
