"""
Master runner: executes all stages in order and prints a final summary table.
Run this after generating data: python run_all.py
"""
import subprocess, sys, time

stages = [
    ("Stage 1+2: Generate graphs + compute exact BC", "generate_data.py"),
    ("Stage 3+4a: Train Model A (Learned Pivot)",     "model_a_pivot.py"),
    ("Stage 3+4b: Train Model B (Pairwise XGBoost)",  "model_b_pairwise.py"),
    ("Stage 3+4c: Train Model C (GraphSAGE GNN)",     "model_c_gnn.py"),
    ("Stage 5+6:  BOLT Baseline evaluation",          "bolt_baseline.py"),
    ("Stage 7:    Plots + final report",               "evaluate_and_plot.py"),
]

print("=" * 60)
print("  BETWEENNESS ORDERING — ML IMPROVEMENT PIPELINE")
print("=" * 60)

for label, script in stages:
    print(f"\n>>> {label}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"    [{status}] completed in {elapsed:.1f}s")
    if result.returncode != 0:
        print("    ERROR: stopping pipeline.")
        sys.exit(1)

print("\n" + "=" * 60)
print("  ALL STAGES COMPLETE")
print("  Check outputs/ folder for plots and report")
print("=" * 60)
