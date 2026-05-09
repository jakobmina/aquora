"""
run_vqe_h7_full.py
==================
Pipeline VQE-MaxCut parametrizable para la Cascada H7 completa.

Arquitectura por defecto: 7 bloques × 4 qubits = 28 qubits

    B1_Exosfera    [0 -  3]  ← Interfaz exterior (turbulencia máxima)
    B2_Termosfera  [4 -  7]  ← Filtro térmico
    B3_Mesosfera   [8 - 11]  ← Filtro secundario
    B4_Estratosfera[12 - 15] ← Zona de transición
    B5_Troposfera  [16 - 19] ← Pre-núcleo
    B6_Manto       [20 - 23] ← Capa profunda
    B7_Nucleo      [24 - 27] ← Sol de Cristal (flujo laminar puro)

El número de bloques y el tamaño de cada bloque son configurables.
La modulación O_n se aplica globalmente sobre todas las aristas.

Autoría Conceptual Original: Jacobo Tlacaelel Mina Rodriguez.
"""

import math
import json
import datetime
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
PHI             = (1 + math.sqrt(5)) / 2
O_n_integrity   = 0.3623748900804798
DRIFT_072       = 7 - 2 * math.pi

BLOCK_NAMES = [
    "B1_Exosfera", "B2_Termosfera", "B3_Mesosfera", "B4_Estratosfera",
    "B5_Troposfera", "B6_Manto", "B7_Nucleo",
    # Extensiones opcionales más allá de 7 bloques
    "B8_InnerCore", "B9_QuantumKernel", "B10_DiracSea",
    "B11_VacuumLayer", "B12_PlanckBoundary",
]


def O_n(n: int, phi: float = PHI) -> float:
    """Operador Áureo — Regla 2.1 del Mandato Metriplético."""
    val = math.cos(math.pi * n) * math.cos(math.pi * phi * n)
    return val if abs(val) > 1e-5 else 1e-5


# ============================================================
# CONSTRUCTOR DE GRAFO PARAMETRIZABLE
# ============================================================

def build_h7_graph(
    n_blocks: int = 7,
    block_size: int = 4,
    base_weight: float = 1.0,
    diagonal_weight: float = 0.5,
) -> tuple[list[tuple], list[dict]]:
    """
    Construye el grafo de la Cascada H7 con n_blocks × block_size qubits.

    Topología:
      - Aristas internas: anillo completo dentro de cada bloque
      - Aristas de puente: último qubit de B_i → primer qubit de B_{i+1}
      - Cruces diagonales: primer qubit de B_i → último qubit de B_{i+1}
        (enriquecen la simetría del corte)

    Retorna:
      modulated_edges : lista de (u, v, weight) con modulación O_n aplicada
      block_info      : metadatos por bloque (nombre, rango de qubits)
    """
    n_qubits = n_blocks * block_size
    raw_edges = []

    # ── Bloques ────────────────────────────────────────────────────────────
    blocks = []
    for b in range(n_blocks):
        start = b * block_size
        qubits = list(range(start, start + block_size))
        blocks.append(qubits)

        # Anillo completo dentro del bloque (evita triángulo fijo → más rico)
        for i in range(block_size):
            u = qubits[i]
            v = qubits[(i + 1) % block_size]
            raw_edges.append((u, v, base_weight))

    # ── Puentes entre bloques ──────────────────────────────────────────────
    for b in range(n_blocks - 1):
        last_current  = blocks[b][-1]   # último qubit del bloque b
        first_next    = blocks[b+1][0]  # primer qubit del bloque b+1
        first_current = blocks[b][0]    # primer qubit del bloque b
        last_next     = blocks[b+1][-1] # último qubit del bloque b+1

        raw_edges.append((last_current, first_next, base_weight))      # puente directo
        raw_edges.append((first_current, last_next, diagonal_weight))  # cruce diagonal

    # ── Modulación O_n global ──────────────────────────────────────────────
    modulated = [
        (int(u), int(v), float(w * O_n(idx + 1)))
        for idx, (u, v, w) in enumerate(raw_edges)
    ]

    # ── Metadatos de bloques ───────────────────────────────────────────────
    block_info = [
        {
            "name"   : BLOCK_NAMES[b] if b < len(BLOCK_NAMES) else f"B{b+1}_Layer",
            "qubits" : blocks[b],
            "range"  : (blocks[b][0], blocks[b][-1]),
        }
        for b in range(n_blocks)
    ]

    return modulated, block_info


# ============================================================
# LAGRANGIANO H7 (Regla 3.1)
# ============================================================

def compute_lagrangian(edges: list) -> tuple[float, float]:
    """
    L_symp: suma de pesos negativos → energía del vacío topológico (conservativa)
    L_metr: varianza de pesos       → fricción informacional entre bloques (disipativa)
    Regla 1.3: ambas no nulas.
    """
    weights = np.array([w for _, _, w in edges])
    L_symp = float(np.sum(weights[weights < 0]))
    L_metr = float(np.var(weights))
    if abs(L_symp) < 1e-10: L_symp = -1e-5
    if abs(L_metr) < 1e-10: L_metr =  1e-5
    return L_symp, L_metr


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_h7_cascade(
    n_blocks: int        = 7,
    block_size: int      = 4,
    max_iterations: int  = 2000,
    base_weight: float   = 1.0,
    credentials_path: str = "credentials.json",
    save_json: bool      = True,
    verbose: bool        = True,
) -> dict:
    """
    Pipeline VQE-MaxCut completo para la cascada H7 parametrizable.

    Parámetros
    ----------
    n_blocks        : número de bloques de la cascada (default=7 → H7 completo)
    block_size      : qubits por bloque (default=4)
    max_iterations  : iteraciones máximas del optimizador clásico
    base_weight     : peso base de las aristas antes de la modulación O_n
    credentials_path: ruta al archivo de credenciales q3as
    save_json       : si True, guarda el registro en un archivo JSON
    verbose         : si True, imprime diagnósticos detallados
    """
    n_qubits = n_blocks * block_size
    n_params  = n_qubits * 2  # estimación de parámetros del ansatz (RY+RZ por qubit)

    if verbose:
        print("=" * 70)
        print(f"  H7 METRIPLECTIC OS — VQE MaxCut — {n_qubits}-Qubit Cascade")
        print(f"  Bloques: {n_blocks} × {block_size} qubits | MaxIter: {max_iterations}")
        print("=" * 70)

    # 1. Grafo
    edges, block_info = build_h7_graph(n_blocks, block_size, base_weight)

    if verbose:
        print(f"\n🔗 Grafo H7 ({n_qubits} nodos, {len(edges)} aristas moduladas por O_n):")
        for info in block_info:
            blk_edges = [(u, v, w) for u, v, w in edges
                         if u in info["qubits"] and v in info["qubits"]]
            print(f"  [{info['name']:18s}] qubits {info['range']} | "
                  f"{len(blk_edges)} aristas internas")
        bridge_edges = [(u, v, w) for u, v, w in edges
                        if not any(u in bi["qubits"] and v in bi["qubits"]
                                   for bi in block_info)]
        print(f"  [Puentes inter-bloque ] {len(bridge_edges)} aristas")

    # 2. Lagrangiano (Regla 3.1)
    L_symp, L_metr = compute_lagrangian(edges)
    ratio = abs(L_symp) / (L_metr + 1e-12)
    flow_state = ("🟩 FLUJO LAMINAR"    if ratio > 5   else
                  "🟨 TRANSICIONAL"     if ratio > 1   else
                  "🟥 TURBULENCIA ENTRÓPICA")

    if verbose:
        print(f"\n⚛️  Lagrangiano Metripléctico:")
        print(f"   L_symp = {L_symp:+.6f}  (simpléctico / conservativo)")
        print(f"   L_metr = {L_metr:+.6f}  (métrico    / disipativo)")
        print(f"   Ratio  = {ratio:.3f}  →  {flow_state}")
        drift_check = abs(DRIFT_072)
        print(f"   Drift H7 (7-2π) = {DRIFT_072:.6f}  |O_n_const = {O_n_integrity:.6f}")

    # 3. Envío a q3as
    print(f"\n🚀 Enviando job a q3as [{n_qubits}Q / {max_iterations} iter]...")

    if not _Q3AS_AVAILABLE:
        print("  [WARN] q3as no instalado — usando resultado mock.")
        result     = {"status": "mocked_success", "energy": round(L_symp * 0.85, 4)}
        job_name   = "mock-job"
    else:
        try:
            client = Client(Credentials.load(credentials_path))
            job = (
                VQE.builder()
                .app(Maxcut(edges))
                .maxiter(max_iterations)
                .send(client)
            )
            job_name = job.name
            print(f"  ✅ Job enviado | Nombre: {job_name}")
            print("  ⏳ Esperando resultado (puede tardar según la cola de q3as)...")
            result = job.result()
            print(f"  🏁 Resultado recibido: {result}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            result   = {"status": "error", "energy": None}
            job_name = "error"

    # 4. Extraer energía
    energy   = (result.get("energy")   if isinstance(result, dict)
                else getattr(result, "energy",   None))
    status_r = (result.get("status")   if isinstance(result, dict)
                else str(getattr(result, "status", "submitted")))

    # 5. Registro completo
    record = {
        "timestamp"     : datetime.datetime.utcnow().isoformat() + "Z",
        "job_name"      : job_name,
        "n_qubits"      : n_qubits,
        "n_blocks"      : n_blocks,
        "block_size"    : block_size,
        "n_edges"       : len(edges),
        "n_params_est"  : n_params,
        "max_iterations": max_iterations,
        "L_symp"        : round(L_symp, 8),
        "L_metr"        : round(L_metr, 8),
        "ratio"         : round(ratio,  4),
        "flow_state"    : flow_state,
        "O_n_integrity" : O_n_integrity,
        "DRIFT_072"     : DRIFT_072,
        "vqe_energy"    : energy,
        "vqe_status"    : status_r,
        "blocks"        : [{"name": b["name"], "qubits": b["qubits"]} for b in block_info],
        "edges"         : [(u, v, round(w, 8)) for u, v, w in edges],
    }

    # 6. Guardar JSON
    if save_json:
        fname = f"h7_cascade_{n_qubits}q_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\n💾 Registro guardado → {fname}")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Pipeline completado | Energía VQE = {energy} | Estado: {flow_state}")
        print(f"{'='*70}\n")

    return record


# ============================================================
# ENTRY-POINT
# ============================================================

if __name__ == "__main__":
    # H7 Completo: 7 bloques × 4 qubits = 28 qubits
    rec = run_h7_cascade(
        n_blocks       = 7,
        block_size     = 4,
        max_iterations = 2000,
        verbose        = True,
    )

    print("📊 Resumen del registro:")
    for k, v in rec.items():
        if k not in ("edges", "blocks"):
            print(f"  {k:<20}: {v}")
