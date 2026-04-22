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
    parser.add_argument("--test", action="store_true", help="Run the full test suite")
    parser.add_argument("--all", action="store_true", help="Run tests, then train, then serve")

    args = parser.parse_args()

    # Default to help if no args
    if not any([args.train, args.serve, args.test, args.all]):
        parser.print_help()
        return

    # 1. Run Tests
    if args.test or args.all:
        run_command(
            [sys.executable, "-m", "pytest", "tests/"],
            "Executing full validation suite (H7 Physics + VQE Logic)"
        )

    # 2. Run Training
    if args.train or args.all:
        creds = "credentials.json"
        if not os.path.exists(creds):
            print(f"[WARN] {creds} not found. Some hardware paths might fallback to mock.")
        
        run_command(
            [sys.executable, "generate_submission.py", "--out", "submission.csv", "--credentials", creds],
            "Generating H7 Metriplectic Submission Grid"
        )

    # 3. Serve Dashboard
    if args.serve or args.all:
        run_command(
            [sys.executable, "api.py"],
            "Launching H7 Metriplectic Dashboard API (Ctrl+C to stop)"
        )

if __name__ == "__main__":
    main()
