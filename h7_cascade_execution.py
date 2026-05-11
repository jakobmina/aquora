"""
h7_cascade_execution.py
=======================
Gestor de Niveles de Gobernanza H7 (Tiers).

Soporta 3 modos de ejecución para optimizar cuotas y energía:
- LITE: 12 Qubits (Rápido, económico)
- STANDARD: 20 Qubits (Balanceado)
- SOVEREIGN: 80 Qubits (Máxima resolución)
"""

import argparse
import os
import json
from run_vqe_h7_full import run_h7_cascade
from h7_bayesian_oracle import run_extraction_pipeline

TIERS = {
    "lite": {"blocks": 2, "size": 6, "desc": "Gobernanza 12Q (Lite)"},
    "standard": {"blocks": 4, "size": 5, "desc": "Gobernanza 20Q (Standard)"},
    "sovereign": {"blocks": 10, "size": 8, "desc": "Gobernanza 80Q (Sovereign)"}
}

def deploy_cascade(tier_name="standard"):
    tier = TIERS.get(tier_name, TIERS["standard"])
    print(f"🚀 INICIANDO DESPLIEGUE H7 - MODO: {tier_name.upper()}")
    print(f"📋 {tier['desc']}")
    print("="*70)
    
    os.makedirs("h7_outputs", exist_ok=True)
    
    # Ejecución de la Cascada según el Tier
    results = run_h7_cascade(
        n_blocks=tier["blocks"],
        block_size=tier["size"],
        max_iterations=1000, 
        base_weight=1.0,
        save_json=True,
        verbose=True
    )
    
    job_id = results.get("job_name", "pending")
    
    # Extracción y Firma Hexadecimal
    print(f"\n📡 Extrayendo entropía del Job: {job_id}...")
    extraction = run_extraction_pipeline(job_id)
    
    if extraction:
        hex_sig = extraction.get("hex_signature", "N/A")
        print(f"🧬 FIRMA TOPOLÓGICA (uint128): {hex_sig}")
        
        # Guardar como el estado más reciente
        with open("h7_outputs/h7_cascade_80q_latest.json", "w") as f:
            json.dump(extraction, f)
    
    with open("h7_outputs/active_80q_job.txt", "w") as f:
        f.write(job_id)
        
    print(f"\n✅ Ciclo H7-{tier_name.upper()} Completado.")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H7 Cascade Governance Tier Manager")
    parser.add_argument("--tier", choices=["lite", "standard", "sovereign"], default="standard")
    args = parser.parse_args()
    
    deploy_cascade(args.tier)
