"""
rank_k_nodes_presentation.py
============================
Comprehensive K-Node ranking script for presentation.
 
ALGORITHM SUMMARY:
------------------
Given K nodes from a complex graph G(V, E):

1. OFFLINE PHASE - Feature Extraction [O(m) once]:
   - For the entire graph, compute 10 structural features per node:
     degree, PageRank, k-core, clustering, closeness, etc.
   - Save all features as a K x 10 matrix F.
   - Cost: O(m) or O(m log n) for the whole graph. Done once.
   - Store trained ML model (.pkl) - K=any, constant lookup.

2. ONLINE PHASE - K-Node Ranking [O(K^2) inference]:
   - Form K*(K-1)/2 difference vectors: X_ij = F[i] - F[j]
   - Batch-predict all K*(K-1)/2 pairs in ONE model.predict() call
   - Tally wins per node (Round-Robin Tournament)
   - Sort by wins -> Final ranking
   - Cost: O(K^2) -- for K << n, this is sub-O(m)!

COMPLEXITY vs EDDBM:
   EDDBM    : O(K * m)   -- runs O(m) estimate for each of K nodes
   ML Model : O(m) + O(K^2)  -- O(m) feature extraction ONCE per graph
   If K << sqrt(m), then K^2 << m => ML is FASTER than EDDBM!

Example: Wiki-Vote (m=103,000 edges, K=20 nodes)
   EDDBM cost: 20 * 103,000 = 2,060,000 ops
   ML cost: 103,000 + 20^2 = 103,400 ops (barely louder than EDDBM on 1 node)
"""

import os
import sys
import pickle
import argparse
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score

from node_comparison_model import load_graph, extract_node_features, eddbm_estimate_bc

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

def run_eddbm_ranking(G, target_nodes, avg_deg, T=30):
    """Rank K nodes using EDDBM O(K*m) approach."""
    rng = np.random.RandomState(42)
    eddbm_scores = {}
    for node in target_nodes:
        eddbm_scores[node] = eddbm_estimate_bc(G, node, T, avg_deg, rng)
    ranked = sorted(target_nodes, key=lambda n: eddbm_scores[n], reverse=True)
    return ranked, eddbm_scores

def run_ml_ranking(F, node_to_idx, model, target_nodes):
    """Rank K nodes using ML classifier O(K^2) approach."""
    k = len(target_nodes)
    scores = {node: 0 for node in target_nodes}
    pairs = [(target_nodes[i], target_nodes[j]) for i in range(k) for j in range(i+1, k)]
    
    X_batch = np.array([F[node_to_idx[u]] - F[node_to_idx[v]] for u, v in pairs])
    preds = model.predict(X_batch)
    
    for idx, (u, v) in enumerate(pairs):
        if preds[idx] == 1:
            scores[u] += 1
        else:
            scores[v] += 1
    
    ranked = sorted(target_nodes, key=lambda n: scores[n], reverse=True)
    return ranked, scores

def spearman_correlation(ranked_pred, ranked_true):
    """Compute Spearman rank correlation coefficient."""
    k = len(ranked_pred)
    ml_ranks = {n: i for i, n in enumerate(ranked_pred)}
    true_ranks = {n: i for i, n in enumerate(ranked_true)}
    d_sq_sum = sum((ml_ranks[n] - true_ranks[n])**2 for n in ranked_pred)
    if k > 1:
        return 1.0 - (6 * d_sq_sum) / (k * (k**2 - 1))
    return 1.0

def plot_ranking_comparison(target_nodes, ranked_ml, ranked_eddbm, ranked_true, bc_dict, ds_name, out_dir):
    """Bar chart: True BC values of nodes ordered by each method."""
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    methods = [("Optimal (Exact Brandes)", ranked_true, "#2ecc71"),
               ("ML Model (K^2)", ranked_ml, "#3498db"),
               ("EDDBM (K*m)", ranked_eddbm, "#e74c3c")]
    
    for ax, (label, ranking, color) in zip(axes, methods):
        bc_vals = [bc_dict[n] for n in ranking]
        bars = ax.barh([f"Node {n}" for n in ranking], bc_vals, color=color, alpha=0.8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Betweenness Centrality")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"K-Node Ranking Comparison ({ds_name})", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_ranking_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  => Saved: {path}")
    return path

def plot_complexity_timing(k_values, ml_times, eddbm_times, ds_name, m_edges, out_dir):
    """Plot wall-clock time vs K for ML vs EDDBM."""
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, [t * 1000 for t in ml_times], "o-", color="#3498db", label="ML Model O(K²)", linewidth=2)
    ax.plot(k_values, [t * 1000 for t in eddbm_times], "s--", color="#e74c3c", label="EDDBM O(K·m)", linewidth=2)
    
    ax.set_xlabel("Number of Nodes K", fontsize=12)
    ax.set_ylabel("Wall-Clock Time (ms)", fontsize=12)
    ax.set_title(f"Ranking Time: ML vs EDDBM ({ds_name}, m={m_edges:,} edges)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    path = os.path.join(out_dir, f"{ds_name}_timing_vs_k.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  => Saved: {path}")
    return path

def plot_spearman_vs_k(k_values, ml_spearman, eddbm_spearman, ds_name, out_dir):
    """Plot Spearman correlation accuracy vs K for both methods."""
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, ml_spearman, "o-", color="#3498db", label="ML Model", linewidth=2)
    ax.plot(k_values, eddbm_spearman, "s--", color="#e74c3c", label="EDDBM", linewidth=2)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="Perfect (Exact Brandes)")
    
    ax.set_xlabel("Number of Nodes K", fontsize=12)
    ax.set_ylabel("Spearman Rank Correlation", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Ranking Accuracy vs K ({ds_name})", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    path = os.path.join(out_dir, f"{ds_name}_spearman_vs_k.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  => Saved: {path}")
    return path

def plot_node_score_scatter(target_nodes, scores_ml, scores_eddbm, bc_dict, ds_name, out_dir):
    """Scatter: ML win score and EDDBM score vs true BC, both normalized."""
    os.makedirs(out_dir, exist_ok=True)
    
    bc_vals = np.array([bc_dict[n] for n in target_nodes])
    ml_vals = np.array([scores_ml[n] for n in target_nodes], dtype=float)
    eddbm_vals = np.array([scores_eddbm[n] for n in target_nodes], dtype=float)
    
    # Normalize scores to [0, 1]
    def norm(x):
        r = x.max() - x.min()
        return x / r if r > 0 else x
    
    ml_norm = norm(ml_vals)
    eddbm_norm = norm(eddbm_vals)
    bc_norm = norm(bc_vals)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(bc_norm, eddbm_norm, color="#e74c3c", alpha=0.7, label="EDDBM Score", s=60)
    ax.scatter(bc_norm, ml_norm, color="#3498db", alpha=0.7, marker="^", label="ML Win Score", s=60)
    
    # Ideal line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect Correlation")
    
    ax.set_xlabel("Normalized True Brandes BC", fontsize=12)
    ax.set_ylabel("Normalized Score", fontsize=12)
    ax.set_title(f"Node Score vs. True BC: ML vs EDDBM ({ds_name})", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    path = os.path.join(out_dir, f"{ds_name}_score_vs_bc_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  => Saved: {path}")
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="node_comparison_model.pkl")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=1500)
    args = parser.parse_args()

    ds_name = os.path.splitext(os.path.basename(args.dataset))[0]
    out_dir = "plots"
    
    print(f"\n{'='*65}")
    print(f" ML K-Node Comparison Presentation Script")
    print(f" Dataset: {ds_name}  |  K={args.k}")
    print(f"{'='*65}")

    print("\n[Phase 1] Loading Graph & Exact Brandes...")
    G = load_graph(args.dataset, args.max_nodes)
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    m = G.number_of_edges()
    avg_deg = 2.0 * m / G.number_of_nodes()
    print(f"  Graph: {G.number_of_nodes()} nodes, {m:,} edges")

    print("\n[Phase 2] Extracting ML Features (O(m))...")
    t_feat_start = time.time()
    F, F_nodes = extract_node_features(G)
    t_feat = time.time() - t_feat_start
    node_to_idx = {v: i for i, v in enumerate(F_nodes)}
    print(f"  Feature extraction took: {t_feat:.3f}s")

    print("\n[Phase 3] Loading Trained ML Model...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Select K nodes
    all_nodes = list(G.nodes())
    rng = np.random.RandomState(99)
    k = min(args.k, len(all_nodes))
    target_nodes = rng.choice(all_nodes, size=k, replace=False).tolist()

    # --- Single K comparison (full output) ---
    print(f"\n[Phase 4] Running ML Tournament for K={k} nodes...")
    t0 = time.time()
    ranked_ml, scores_ml = run_ml_ranking(F, node_to_idx, model, target_nodes)
    ml_time = time.time() - t0

    print(f"[Phase 5] Running EDDBM Ranking for K={k} nodes...")
    t0 = time.time()
    ranked_eddbm, scores_eddbm = run_eddbm_ranking(G, target_nodes, avg_deg)
    eddbm_time = time.time() - t0

    ranked_true = sorted(target_nodes, key=lambda n: bc_dict[n], reverse=True)
    sp_ml = spearman_correlation(ranked_ml, ranked_true)
    sp_eddbm = spearman_correlation(ranked_eddbm, ranked_true)

    # --- Print results table ---
    print(f"\n{'='*70}")
    print(f"{'Rank':<6} | {'Node':<8} | {'ML Wins':<9} | {'True BC':<14} | {'Optimal Rank'}")
    print(f"{'='*70}")
    for i, node in enumerate(ranked_ml):
        true_rank = ranked_true.index(node) + 1
        marker = " *" if (i + 1) == true_rank else "  "
        print(f"{i+1:<6} | {node:<8} | {scores_ml[node]:<9} | {bc_dict[node]:<14.4e} | {true_rank}{marker}")
    print(f"{'='*70}")

    print(f"\nTimings:")
    print(f"  ML  Tournament (K={k}): {ml_time*1000:.2f} ms  [{m*k:,} ops (EDDBM) vs {k*k:,} ops (ML)]")
    print(f"  EDDBM Ranking  (K={k}): {eddbm_time*1000:.2f} ms")
    print(f"\nAccuracy (Spearman Correlation vs Exact Brandes):")
    print(f"  ML   : {sp_ml:.3f}")
    print(f"  EDDBM: {sp_eddbm:.3f}")

    # --- Sweep K values for timing and spearman plots ---
    print("\n[Phase 6] Sweeping K to build timing & accuracy plots...")
    k_values = [5, 10, 15, 20, 30, 40, 50]
    ml_times_sweep, eddbm_times_sweep = [], []
    ml_sp_sweep, eddbm_sp_sweep = [], []
    
    for kk in k_values:
        if kk > len(all_nodes):
            continue
        nodes_kk = rng.choice(all_nodes, size=kk, replace=False).tolist()
        
        t0 = time.time()
        rkk_ml, _ = run_ml_ranking(F, node_to_idx, model, nodes_kk)
        ml_times_sweep.append(time.time() - t0)
        
        t0 = time.time()
        rkk_eddbm, _ = run_eddbm_ranking(G, nodes_kk, avg_deg)
        eddbm_times_sweep.append(time.time() - t0)
        
        rkk_true = sorted(nodes_kk, key=lambda n: bc_dict[n], reverse=True)
        ml_sp_sweep.append(spearman_correlation(rkk_ml, rkk_true))
        eddbm_sp_sweep.append(spearman_correlation(rkk_eddbm, rkk_true))
        print(f"  K={kk}: ML {ml_times_sweep[-1]*1000:.1f}ms, EDDBM {eddbm_times_sweep[-1]*1000:.1f}ms, ML-Spearman {ml_sp_sweep[-1]:.2f}")

    print("\n[Phase 7] Saving Presentation Plots...")
    plot_ranking_comparison(target_nodes, ranked_ml, ranked_eddbm, ranked_true, bc_dict, ds_name, out_dir)
    plot_complexity_timing(k_values[:len(ml_times_sweep)], ml_times_sweep, eddbm_times_sweep, ds_name, m, out_dir)
    plot_spearman_vs_k(k_values[:len(ml_sp_sweep)], ml_sp_sweep, eddbm_sp_sweep, ds_name, out_dir)
    plot_node_score_scatter(target_nodes, scores_ml, scores_eddbm, bc_dict, ds_name, out_dir)

    print(f"\nAll plots saved to: {os.path.abspath(out_dir)}")
    print("Done!")

if __name__ == "__main__":
    main()
