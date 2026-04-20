"""
generate_submission.py
======================
Batch submission generator for the Metriplectic VQE pipeline.

Runs the MaxCut pipeline across a grid of (n_param, phi_param) values
and exports a single submission.csv accumulating all physics records.

Usage
-----
    python generate_submission.py                   # default grid
    python generate_submission.py --out my_run.csv  # custom output path

Each row in the CSV captures:
  run_id, timestamp, n_param, phi_param, phi_golden,
  particle_type, golden, quasiperiod,
  L_symp, L_metr,                    ← Metriplectic brackets
  symmetry_ratio, h7_state,          ← Virtual-particle H7 classification
  vqe_energy, vqe_status,
  n_edges, on_weights                ← Golden Operator O_n edge modulation
"""

import argparse
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_vqe_maxcut import MetriplecticMaxCut

# ── Default parameter grid ────────────────────────────────────────────────────
# n_param   : integer O_n parity index
# phi_param : quasiperiod modulation (use values near 1/φ ≈ 0.618 or 0.3624)

DEFAULT_GRID = [
    # (n_param, phi_param)
    (1, 0.3624),
    (2, 0.3624),
    (3, 0.3624),   # original VirtualParticleQ3AS params
    (4, 0.3624),
    (1, 0.6180),
    (2, 0.6180),
    (3, 0.6180),
    (4, 0.6180),
]

# ── H7 Hamiltonian edges (default MaxCut problem) ─────────────────────────────
H7_EDGES = [
    (0, 1,  0.695864585574),
    (0, 2,  0.159413943867),
    (1, 2, -0.362374889934),
]


def run_grid(
    grid: list = DEFAULT_GRID,
    edges: list = H7_EDGES,
    output_path: str = "submission.csv",
    credentials_path: str = "credentials.json",
    verbose: bool = True,
) -> list:
    """
    Execute the pipeline for every (n_param, phi_param) in grid and
    accumulate records into a submission CSV.

    Returns
    -------
    List of record dicts (one per run).
    """
    records = []

    for idx, (n, phi) in enumerate(grid, start=1):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Run {idx}/{len(grid)}  |  n={n}  phi_param={phi}")
            print(f"{'─'*60}")

        system = MetriplecticMaxCut(
            edges            = edges,
            n_param          = n,
            phi_param        = phi,
            credentials_path = credentials_path,
        )
        record = system.run()
        records.append(record)

    # Write all records at once
    dummy = MetriplecticMaxCut(edges=edges)
    dummy.export_submission_csv(records, output_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Batch complete — {len(records)} runs exported to {output_path}")
        print(f"{'='*60}\n")

    return records


def print_summary(records: list) -> None:
    """Print a compact summary table to stdout."""
    print(f"\n{'n':>4}  {'phi':>6}  {'type':>10}  {'L_symp':>10}  "
          f"{'L_metr':>10}  {'sym_ratio':>9}  {'h7_state':>13}")
    print("-" * 70)
    for r in records:
        print(
            f"{r['n_param']:>4}  {r['phi_param']:>6}  {r['particle_type']:>10}  "
            f"{r['L_symp']:>10.5f}  {r['L_metr']:>10.5f}  "
            f"{r['symmetry_ratio']:>9.2f}  {r['h7_state']:>13}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch Metriplectic VQE submission generator"
    )
    parser.add_argument(
        "--out", default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--credentials", default="credentials.json",
        help="q3as credentials JSON path"
    )
    args = parser.parse_args()

    records = run_grid(
        output_path      = args.out,
        credentials_path = args.credentials,
    )
    print_summary(records)
