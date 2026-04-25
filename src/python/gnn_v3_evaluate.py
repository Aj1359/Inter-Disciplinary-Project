#!/usr/bin/env python3
"""
Final evaluation: GNN v3 vs BOLT on all real-world datasets.
Generates comprehensive comparison with timing benchmarks.
"""
import argparse
import glob
import math
import os
import random
import subprocess
import time
import csv

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau

from gnn_v3_model import (
    BCModel, NODE_FEAT_DIM, 
    build_sparse_adj, compute_node_features, load_graph
)


def get_exact_bc(graph_file, bc_cache_dir="bc_cache"):
    """Get exact BC using cached values or C++ compute_bc."""
    os.makedirs(bc_cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(graph_file))[0]
    csv_path = os.path.join(bc_cache_dir, f"{base}_bc.csv")

    compute_bin = os.path.join(os.path.dirname(os.path.abspath(graph_file)), "compute_bc")

    if not os.path.exists(csv_path):
        if not os.path.exists(compute_bin):
            return None
        print(f"    Computing exact BC via C++...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [compute_bin, graph_file, csv_path],
                capture_output=True, text=True, timeout=600
            )
            print(f"done")
        except subprocess.TimeoutExpired:
            print("timeout!")
            return None
        except Exception as e:
            print(f"error: {e}")
            return None

    bc = {}
    with open(csv_path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                bc[int(parts[0])] = float(parts[1])
    return bc


def eval_gnn(model, adj, deg, n, row, col, bc_dict, rng, device, num_pairs=1000):
    """Evaluate GNN ordering efficiency."""
    x = compute_node_features(adj, deg, n).to(device)
    adj_norm = build_sparse_adj(row, col, n).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        scores = model(x, adj_norm).cpu().numpy()
    infer_time = time.time() - t0

    nonzero = [i for i in range(n) if bc_dict.get(i, 0) > 0]
    if len(nonzero) < 10:
        return 0, 0, 0, infer_time

    # Pairwise ordering efficiency
    correct = total = 0
    for _ in range(num_pairs):
        u, v = rng.choice(nonzero), rng.choice(nonzero)
        if u == v or bc_dict[u] == bc_dict[v]:
            continue
        if (scores[u] > scores[v]) == (bc_dict[u] > bc_dict[v]):
            correct += 1
        total += 1
    eff = correct / max(1, total)

    # Spearman rank correlation
    bc_nz = [bc_dict[i] for i in nonzero]
    sc_nz = [scores[i] for i in nonzero]
    rho, _ = spearmanr(sc_nz, bc_nz)
    rho = 0.0 if math.isnan(rho) else rho
    
    # Kendall tau
    tau, _ = kendalltau(sc_nz, bc_nz)
    tau = 0.0 if math.isnan(tau) else tau

    return eff, rho, tau, infer_time


def eval_bolt(graph_file, T=25, trials=1000):
    """Run BOLT and parse efficiency."""
    bolt_bin = os.path.join(os.path.dirname(os.path.abspath(graph_file)), "bolt")
    if not os.path.exists(bolt_bin):
        return None, None, None

    try:
        result = subprocess.run(
            [bolt_bin, graph_file, str(T), str(trials), "1"],
            capture_output=True, text=True, timeout=600
        )
        mode = None
        eff_at_T = err_at_T = bolt_time = None
        for line in result.stdout.splitlines():
            if "Efficiency vs T" in line:
                mode = "eff"; continue
            if "Average Error vs T" in line:
                mode = "err"; continue
            if "Time per estimateBOLT" in line:
                try:
                    bolt_time = float(line.split(":")[-1].strip().replace("ms", "").strip())
                except:
                    pass
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t_val, val = int(parts[0]), float(parts[1])
                    if mode == "eff":
                        eff_at_T = val
                    elif mode == "err":
                        err_at_T = val
                except:
                    pass

        return eff_at_T, err_at_T, bolt_time
    except:
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--gnn-model", default="gnn_model_v3.pt")
    parser.add_argument("--max-nodes", type=int, default=40000)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="final_comparison_v3.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cpu")

    # Load GNN
    print("Loading GNN v3 model...")
    ckpt = torch.load(args.gnn_model, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = BCModel(
        cfg["in_dim"], cfg["hidden_dim"], cfg["num_layers"],
        cfg["num_heads"], cfg["dropout"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val_acc={ckpt.get('val_acc', 0)*100:.2f}%, "
          f"val_rho={ckpt.get('val_rho', 0):.4f})")

    txt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))
    results = []

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "n", "m", "model", "efficiency_pct", 
                         "spearman_rho", "kendall_tau", "time_ms"])

        for path in txt_files:
            bn = os.path.basename(path)
            if "extracted_text" in bn or "synthetic" in bn.lower():
                continue

            adj, deg, n, m, row, col = load_graph(path)
            if n > args.max_nodes or n < 50:
                continue

            print(f"\n{'='*65}")
            print(f"  {bn} — n={n:,}, m={m:,}")
            print(f"{'='*65}")

            # Get exact BC
            bc_dict = get_exact_bc(path)
            if bc_dict is None:
                print("  Skipped (could not compute BC)")
                continue

            # GNN v3
            print(f"  GNN v3...", end=" ", flush=True)
            gnn_eff, gnn_rho, gnn_tau, gnn_t = eval_gnn(
                model, adj, deg, n, row, col, bc_dict, rng, device, args.trials)
            print(f"eff={gnn_eff*100:.2f}% ρ={gnn_rho:.4f} τ={gnn_tau:.4f} "
                  f"time={gnn_t*1000:.1f}ms")
            writer.writerow([bn, n, m, "GNN-v3", f"{gnn_eff*100:.2f}",
                           f"{gnn_rho:.4f}", f"{gnn_tau:.4f}", f"{gnn_t*1000:.1f}"])
            results.append(("GNN-v3", bn, n, gnn_eff, gnn_rho, gnn_tau, gnn_t*1000))

            # BOLT
            if n <= 20000:
                print(f"  BOLT (T=25)...", end=" ", flush=True)
                bolt_eff, bolt_err, bolt_t = eval_bolt(path, 25, args.trials)
                if bolt_eff is not None:
                    print(f"eff={bolt_eff:.2f}% err={bolt_err:.2f}% "
                          f"time={bolt_t:.1f}ms")
                    writer.writerow([bn, n, m, "BOLT-T25", f"{bolt_eff:.2f}",
                                   "", "", f"{bolt_t:.1f}"])
                    results.append(("BOLT-T25", bn, n, bolt_eff/100, 0, 0, bolt_t))
                else:
                    print("skipped")
            f.flush()

    # ── Summary ──
    print(f"\n\n{'='*75}")
    print(f"{'FINAL COMPARISON — GNN v3 vs BOLT':^75}")
    print(f"{'='*75}")
    print(f"{'Dataset':<24} {'Model':<12} {'Eff%':>8} {'Spearman':>9} "
          f"{'Kendall':>8} {'Time(ms)':>10}")
    print(f"{'-'*73}")
    for m_name, ds, n, eff, rho, tau, t in sorted(results, key=lambda x: (x[1], x[0])):
        rho_s = f"{rho:.4f}" if rho else ""
        tau_s = f"{tau:.4f}" if tau else ""
        print(f"{ds:<24} {m_name:<12} {eff*100:>7.2f} {rho_s:>9} "
              f"{tau_s:>8} {t:>9.1f}")

    # Per-model averages
    print(f"\n{'Model':<12} {'Avg Eff%':>10} {'Avg ρ':>8} {'Avg Time':>12} {'Count':>6}")
    print(f"{'-'*50}")
    for m_name in sorted(set(r[0] for r in results)):
        mr = [r for r in results if r[0] == m_name]
        avg_eff = np.mean([r[3] for r in mr]) * 100
        avg_rho = np.mean([r[4] for r in mr]) if mr[0][4] else 0
        avg_t = np.mean([r[6] for r in mr])
        print(f"{m_name:<12} {avg_eff:>9.2f} {avg_rho:>8.4f} {avg_t:>11.1f} {len(mr):>6}")

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
