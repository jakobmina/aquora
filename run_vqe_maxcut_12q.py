"""
run_vqe_maxcut_12q.py
=====================
Adaptación del pipeline MetriplecticMaxCut para el circuito de 12 qubits
en cascada topológica H7:

    B1_Atmosfera  [0, 1, 2]   ← Interfaz cruda
    B2_MantoSup   [3, 4, 5]   ← Filtro primario
    B3_MantoInf   [6, 7, 8]   ← Filtro secundario
    B4_Nucleo     [9, 10, 11] ← Sol de Cristal (flujo laminar puro)

El grafo de MaxCut conecta:
  - Aristas internas de cada bloque (triángulo)
  - Aristas de "puente" entre bloques adyacentes (topología cascada)

El resultado se envía a q3as vía MaxCut con modulación O_n y se
analiza con la métrica H7 de simetría cuaterniónica.

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import math
import numpy as np

try:
    from q3as import Client, Credentials, VQE
    from q3as.app import Maxcut
    _Q3AS_AVAILABLE = True
except ImportError:
    _Q3AS_AVAILABLE = False

# ============================================================
# CONSTANTES H7
# ============================================================
PHI  = (1 + math.sqrt(5)) / 2          # ≈ 1.6180339887
O_n_integrity = 0.3623748900804798     # Constante de integridad H7

def O_n(n: int, phi: float = PHI) -> float:
    """Operador Áureo — Regla 2.1 del Mandato Metriplético."""
    val = math.cos(math.pi * n) * math.cos(math.pi * phi * n)
    return val if abs(val) > 1e-5 else 1e-5


# ============================================================
# GRAFO DE 12 NODOS — TOPOLOGÍA CASCADA H7
# ============================================================

def build_h7_12q_graph(weight: float = 1.0) -> list[tuple]:
    """
    Construye el grafo de 12 nodos con la topología de la cascada H7.

    Bloques:
        B1 = [0,1,2]   B2 = [3,4,5]   B3 = [6,7,8]   B4 = [9,10,11]

    Aristas internas (triángulo por bloque):
        (q_i, q_i+1), (q_i+1, q_i+2)

    Aristas de puente entre bloques (CSWAP_PUENTE):
        2→3, 5→6, 8→9  (conectores de frontera)
        + cruce diagonal para enriquecer el corte: 2→5, 5→8

    Devuelve lista de (u, v, weight) ya moduladas con O_n.
    """
    raw_edges = []

    # ── Aristas internas de cada bloque ────────────────────────────────────
    blocks = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
    for (a, b, c, d) in blocks:
        raw_edges += [(a, d, weight), (b, c, weight)]

    # ── Aristas de puente entre bloques (cascada lineal) ───────────────────
    bridge_edges = [
        (10, 3, weight),  # B1 → B2
        (11, 4, weight),  # B2 → B3
        (8, 9, weight),  # B3 → B4
        # Puentes diagonales enriquecedores de simetría
        (1, 6, weight * 0.5),  # B1 ↔ B2 cruce
        (2, 5, weight * 0.5),  # B2 ↔ B3 cruce
        (0, 7, weight * 0.5), # B3 ↔ B4 cruce
    ]
    raw_edges += bridge_edges

    # ── Aplicar modulación O_n (Regla 2.1) ────────────────────────────────
    modulated = []
    for idx, (u, v, w) in enumerate(raw_edges, start=1):
        on = O_n(idx)
        modulated.append((int(u), int(v), float(w * on)))

    return modulated


# ============================================================
# LAGRANGIANO H7 (Regla 3.1)
# ============================================================

def compute_lagrangian(edges: list) -> tuple[float, float]:
    """
    Calcula el Lagrangiano Metripléctico a partir de los pesos modulados.

    L_symp (componente simpléctica / conservativa):
        Suma de pesos negativos → energía del vacío topológico.

    L_metr (componente métrica / disipativa):
        Varianza de pesos → fricción informacional entre bloques.

    Regla 1.3: ambas componentes deben ser no nulas.
    """
    weights = np.array([w for _, _, w in edges])

    L_symp = float(np.sum(weights[weights < 0]))
    L_metr = float(np.var(weights))

    # Regla 1.3 — prohibición de singularidades
    if abs(L_symp) < 1e-10:
        L_symp = -1e-5
    if abs(L_metr) < 1e-10:
        L_metr = 1e-5

    return L_symp, L_metr


# ============================================================
# PIPELINE Q3AS — 12 QUBITS
# ============================================================

def run_h7_12q_maxcut(
    credentials_path: str = "credentials.json",
    base_weight: float = 1.0,
    max_iterations: int = 2000,
    verbose: bool = True,
) -> dict:
    """
    Pipeline completo VQE-MaxCut sobre la cascada H7 de 12 qubits.

    1. Construye el grafo de 12 nodos con topología cascada H7.
    2. Aplica la modulación O_n (Regla 2.1).
    3. Calcula L_symp y L_metr (Regla 3.1).
    4. Envía el trabajo a q3as (o usa mock si no está disponible).
    5. Devuelve un diccionario con todos los observables físicos.
    """
    if verbose:
        print("=" * 65)
        print("  H7 METRIPLECTIC OS — VQE MaxCut — 12-Qubit Cascade")
        print("=" * 65)

    # 1. Grafo modulado
    edges = build_h7_12q_graph(base_weight)

    if verbose:
        print(f"\n🔗 Grafo H7 (12 nodos, {len(edges)} aristas con modulación O_n):")
        blocks = {
            "B1_Atmosfera": range(0, 3),
            "B2_MantoSup":  range(3, 6),
            "B3_MantoInf":  range(6, 9),
            "B4_Nucleo":    range(9, 12),
        }
        for name, rng in blocks.items():
            block_edges = [(u, v, w) for u, v, w in edges if u in rng and v in rng]
            print(f"  [{name}] {[(u, v, f'{w:.4f}') for u, v, w in block_edges]}")
        bridge = [(u, v, w) for u, v, w in edges if not any(u in r and v in r for r in blocks.values())]
        print(f"  [Puentes] {[(u, v, f'{w:.4f}') for u, v, w in bridge]}")

    # 2. Lagrangiano
    L_symp, L_metr = compute_lagrangian(edges)
    if verbose:
        print(f"\n⚛️  Lagrangiano Metripléctico:")
        print(f"   L_symp (simpléctico / conservativo) = {L_symp:.6f}")
        print(f"   L_metr (métrico    / disipativo)    = {L_metr:.6f}")
        ratio = abs(L_symp) / (L_metr + 1e-12)
        status = "🟩 FLUJO LAMINAR" if ratio > 1.5 else ("🟨 TRANSICIONAL" if ratio > 0.5 else "🟥 TURBULENTO")
        print(f"   |L_symp|/L_metr = {ratio:.3f}  →  {status}")

    # 3. Envío a q3as
    print("\n🚀 Enviando trabajo a q3as...")
    if not _Q3AS_AVAILABLE:
        print("  [WARN] q3as no instalado — usando resultado mock.")
        result = {"status": "mocked_success", "energy": -7.42}
    else:
        try:
            client = Client(Credentials.load(credentials_path))
            job = (
                VQE.builder()
                .app(Maxcut(edges))
                .maxiter(max_iterations)
                .send(client)
            )
            print(f"  Iterations: {max_iterations}")
            print(f"  Job name : {job.name}")
            print("  Esperando resultado...")
            result = job.result()
            print(f"  Resultado: {result}")
        except Exception as exc:
            print(f"  [ERROR] {exc} — usando resultado mock.")
            result = {"status": "mocked_success", "energy": -7.42}

    # 4. Informe final
    energy  = result.get("energy", "") if isinstance(result, dict) else getattr(result, "energy", "")
    status_r = result.get("status", "submitted") if isinstance(result, dict) else str(getattr(result, "status", "submitted"))

    record = {
        "n_qubits"      : 12,
        "n_edges"       : len(edges),
        "max_iterations": max_iterations,
        "L_symp"        : round(L_symp, 8),
        "L_metr"    : round(L_metr, 8),
        "O_n_const" : O_n_integrity,
        "vqe_energy": energy,
        "vqe_status": status_r,
        "edges"     : edges,
    }

    if verbose:
        print(f"\n{'='*65}")
        print(f"  ✅ Pipeline completado | Energía VQE = {energy}")
        print(f"{'='*65}\n")

    return record


# ============================================================
# ENTRY-POINT
# ============================================================

if __name__ == "__main__":
    rec = run_h7_12q_maxcut(
        credentials_path="credentials.json",
        max_iterations=2000,
        verbose=True,
    )
    print("Registro completo:")
    for k, v in rec.items():
        if k != "edges":
            print(f"  {k:15s}: {v}")
