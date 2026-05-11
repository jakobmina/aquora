#!/usr/bin/env python3
"""
main.py
=======
Orchestrator for the H7 Metriplectic VQE Pipeline.

This script provides a single entry point to run the training grid,
start the dashboard API, or execute the validation test suite.

Usage:
  python main.py --train    # Generate submission.csv
  python main.py --serve    # Start the Dashboard API
  python main.py --test     # Run all pytest suites
"""

import argparse
import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n{'-'*60}")
    print(f"  ACTION: {description}")
    print(f"  CMD:    {' '.join(command)}")
    print(f"{'-'*60}\n")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="H7 Metriplectic Pipeline Orchestrator")
    parser.add_argument("--train", action="store_true", help="Run VQE grid and generate submission.csv")
    parser.add_argument("--serve", action="store_true", help="Start the Dashboard API (FastAPI)")
    parser.add_argument("--test", action="store_true", help="Run the full test suite (Physics + Bridge)")
    parser.add_argument("--governor", action="store_true", help="Run the Phase Governor (Vertex AI Bridge)")
    parser.add_argument("--os", action="store_true", help="Launch the H7 Metriplectic OS (Radar + Auto-Governor)")
    parser.add_argument("--daemon", action="store_true", help="Run OS in infinite loop for systemd service")
    parser.add_argument("--kernel", action="store_true", help="Train the H7 OS Kernel (Quantum Entropy Training)")
    parser.add_argument("--radar", action="store_true", help="Run the H7 Quantum Radar scan")
    parser.add_argument("--cycles", type=int, default=-1, help="Limit number of cycles for the governor/daemon")
    parser.add_argument("--all", action="store_true", help="Run everything in sequence")

    args = parser.parse_args()

    # Default to help if no args
    valid_args = [args.train, args.serve, args.test, args.governor, args.os, args.daemon, args.kernel, args.radar, args.all]
    if not any(valid_args):
        parser.print_help()
        return

    # 0. Compile C Kernel (Auto-build)
    try:
        run_command(
            ["make", "-C", "core_physics/"],
            "Compiling High-Performance C Kernel (Metriplex Core)"
        )
    except Exception as e:
        print(f"[WARN] Failed to compile C kernel: {e}")

    # 1. Run Tests
    if args.test or args.all:
        run_command(
            [sys.executable, "-m", "pytest", "tests/"],
            "Executing full validation suite (H7 Physics + VQE + Bridge)"
        )

    # 2. Run Phase Governor (Vertex AI)
    if args.governor or args.all:
        run_command(
            [sys.executable, "vertex_h7_bridge.py"],
            "Activating H7 Phase Governor (Vertex AI Cognitive Mapping)"
        )

    # 3. Run Training
    if args.train or args.all:
        creds = "credentials.json"
        if not os.path.exists(creds):
            print(f"[WARN] {creds} not found. Some hardware paths might fallback to mock.")
        
        run_command(
            [sys.executable, "generate_submission.py", "--out", "submission.csv", "--credentials", creds],
            "Generating H7 Metriplectic Submission Grid"
        )

    # 4. H7 Metriplectic OS (Real-time Regulation)
    if args.os or args.daemon or args.all:
        cmd = [sys.executable, "h7_auto_governor.py"]
        if args.daemon: cmd.append("--daemon")
        if args.cycles > 0:
            cmd.extend(["--cycles", str(args.cycles)])
        
        run_command(
            cmd,
            "Activating H7 Metriplectic OS (Quantum Radar + Auto-Governor)"
        )

    # 5. OS Kernel Training
    if args.kernel:
        run_command(
            [sys.executable, "h7_os_kernel_training.py"],
            "Training H7 OS Kernel with Quantum Entropy"
        )

    # 6. Quantum Radar Scan
    if args.radar:
        run_command(
            [sys.executable, "h7_quantum_radar.py"],
            "Executing H7 Quantum Radar Environment Scan"
        )

    # 7. Serve Dashboard
    if args.serve or args.all:
        run_command(
            [sys.executable, "api.py"],
            "Launching H7 Metriplectic Dashboard API (Ctrl+C to stop)"
        )

if __name__ == "__main__":
    main()
