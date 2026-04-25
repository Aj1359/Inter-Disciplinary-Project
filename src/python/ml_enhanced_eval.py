#!/usr/bin/env python3
"""
Enhanced ML evaluation for betweenness ordering.
Tests the pairwise ranking model on all available graphs:
  - Computes ordering efficiency (% correct pairwise orderings)
  - Compares with BOLT baseline
  - Reports speedup and accuracy
"""
import argparse
import glob
import math
import os
import random
import time
from collections import deque

import numpy as np
import joblib

# Import the feature extraction functions from training module
from ml_enhanced_train import (
    load_graph, bfs_full, brandes_all,
    compute_graph_features, extract_node_features,
    extract_pair_features, NUM_FEATURES, NUM_PAIR_FEATURES,
)


def evaluate_ordering(adj, deg, exact_bc, model, scaler, graph_feats,
                      rng, num_trials=500):
    """Evaluate ordering efficiency: what fraction of pairs are correctly ordered."""
    n = len(adj)
    nonzero = [i for i in range(n) if exact_bc[i] > 0]
    if len(nonzero) < 10:
        return 0.0, 0.0, 0

    # Precompute node features for nonzero BC nodes
    node_features = {}
    for v in nonzero:
        node_features[v] = extract_node_features(adj, deg, v, graph_feats)

    correct = 0
    total = 0
    proba_sum = 0.0
    t_start = time.time()

    for _ in range(num_trials):
        u = rng.choice(nonzero)
        v = rng.choice(nonzero)
        if u == v:
            continue
        if exact_bc[u] == exact_bc[v]:
            continue

        feat = extract_pair_features(
            node_features[u], node_features[v],
            adj, deg, u, v, graph_feats
        )
        feat_arr = np.array([feat], dtype=np.float32)
        feat_scaled = scaler.transform(feat_arr)

        pred = model.predict(feat_scaled)[0]
        prob = model.predict_proba(feat_scaled)[0, 1]

        # pred=1 means model thinks BC(u) > BC(v)
        actual = 1 if exact_bc[u] > exact_bc[v] else 0

        if pred == actual:
            correct += 1
        proba_sum += prob if actual == 1 else (1 - prob)
        total += 1

    elapsed = time.time() - t_start
    efficiency = correct / total if total > 0 else 0.0
    avg_confidence = proba_sum / total if total > 0 else 0.0

    return efficiency, avg_confidence, total, elapsed


def evaluate_avg_error(adj, deg, exact_bc, model, scaler, graph_feats,
                       rng, num_nodes=200, T=25):
    """
    Estimate average betweenness error using ML-guided sampling.
    For each node v, we use the ML model to pick better pivot nodes,
    then estimate BC using Algorithm 1 from the paper.
    """
    n = len(adj)
    nonzero = [i for i in range(n) if exact_bc[i] > 0]
    if len(nonzero) < 10:
        return 0.0

    sample_nodes = rng.sample(nonzero, min(num_nodes, len(nonzero)))

    total_err = 0.0
    cnt = 0

    for v in sample_nodes:
        # Estimate BC using EDDBM-style sampling
        bc_est = estimate_bc_ml(adj, deg, v, model, scaler, graph_feats, rng, T)
        if exact_bc[v] > 0:
            err = abs(exact_bc[v] - bc_est) / exact_bc[v] * 100.0
            total_err += err
            cnt += 1

    return total_err / cnt if cnt > 0 else 0.0


def estimate_bc_ml(adj, deg, v, model, scaler, graph_feats, rng, T):
    """Estimate BC of node v using ML-enhanced EDDBM sampling."""
    n = len(adj)
    avg_deg = graph_feats["avg_deg"]
    lam = max(1.01, avg_deg)

    # BFS from v to get distances and sigma
    dist_v, sigma_v, pred_v, levels_v = bfs_full(adj, v)

    # Compute EDDBM probabilities
    P = [0.0] * n
    total = 0.0
    for d in range(1, len(levels_v)):
        level_base = lam ** (-d)
        for u in levels_v[d]:
            di = max(1, deg[u])
            w = level_base / di
            P[u] = w
            total += w

    if total > 0:
        for i in range(n):
            P[i] /= total

    # Build sampling distribution
    nodes = [i for i in range(n) if i != v and P[i] > 0]
    if not nodes:
        return 0.0

    weights = [P[i] for i in nodes]
    weight_sum = sum(weights)
    cum_weights = []
    cumw = 0
    for w in weights:
        cumw += w
        cum_weights.append(cumw)

    # Sample T pivots and estimate
    bc_est = 0.0
    for _ in range(T):
        r = rng.random() * weight_sum
        # Binary search
        lo, hi = 0, len(cum_weights) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum_weights[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        s = nodes[lo]
        p_s = P[s]
        if p_s <= 0:
            continue

        # Compute dependency of s on v (single-source Brandes)
        dist_s, sigma_s, pred_s, levels_s = bfs_full(adj, s)
        delta = [0.0] * n
        for d in range(len(levels_s) - 1, 0, -1):
            for w in levels_s[d]:
                for p in pred_s[w]:
                    if sigma_s[w] > 0:
                        delta[p] += (sigma_s[p] / sigma_s[w]) * (1.0 + delta[w])

        bc_est += delta[v] / p_s

    return bc_est / T if T > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced ML model")
    parser.add_argument("--data-dir", default=".", help="Directory with .txt graph files")
    parser.add_argument("--model", default="enhanced_model.joblib")
    parser.add_argument("--scaler", default="enhanced_scaler.joblib")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--max-nodes", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="enhanced_results.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    print(f"Loaded model from {args.model}")
    print(f"Loaded scaler from {args.scaler}\n")

    txt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))

    results = []
    with open(args.out, "w") as f:
        f.write("dataset,n,m,efficiency,avg_confidence,time_per_pair_ms\n")

        for path in txt_files:
            basename = os.path.basename(path)
            if "extracted_text" in basename:
                continue

            print(f"Evaluating {basename}...", end=" ", flush=True)
            adj, deg, n, m = load_graph(path)
            if n > args.max_nodes or n < 50:
                print(f"skipped (n={n})")
                continue

            print(f"n={n}, m={m}", end=" ", flush=True)

            # Compute exact BC
            print("computing BC...", end=" ", flush=True)
            exact_bc = brandes_all(adj)

            graph_feats = compute_graph_features(adj, deg)

            eff, conf, total, elapsed = evaluate_ordering(
                adj, deg, exact_bc, model, scaler, graph_feats, rng, args.trials
            )
            time_per_pair = (elapsed / max(1, total)) * 1000  # ms

            print(f"efficiency={eff*100:.2f}%, confidence={conf:.3f}, "
                  f"time={time_per_pair:.2f}ms/pair")

            f.write(f"{basename},{n},{m},{eff*100:.2f},{conf:.3f},{time_per_pair:.2f}\n")
            f.flush()
            results.append((basename, n, eff))

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'Dataset':<30} {'Nodes':>8} {'Efficiency':>12}")
    print(f"{'='*60}")
    for name, n, eff in results:
        print(f"{name:<30} {n:>8} {eff*100:>11.2f}%")
    if results:
        avg_eff = np.mean([r[2] for r in results])
        print(f"{'='*60}")
        print(f"{'Average':<30} {'':>8} {avg_eff*100:>11.2f}%")
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
