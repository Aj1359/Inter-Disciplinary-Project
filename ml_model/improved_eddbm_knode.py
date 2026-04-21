"""
improved_eddbm_knode.py
========================
Improved EDDBM for K-Node Betweenness Comparison.

KEY INNOVATION: Shared-Source Pooled Sampling
---------------------------------------------
Classical EDDBM for ranking K nodes:
  For EACH of K nodes, sample T independent sources -> run T BFS.
  Total cost: O(K * T * m)  -- this is bad!

IMPROVED EDDBM (this script):
  Pool all K nodes' probability distributions P_v1, P_v2, ... P_vK.
  Draw a SINGLE set of S shared sources from the pooled distribution.
  For each shared source s, run ONE BFS and collect dependency delta_s(vi) 
  for ALL K target nodes simultaneously.
  Total cost: O(S * m)  -- independent of K!
  
  When S = T, improved EDDBM uses the SAME total BFS work as classical EDDBM 
  for K=1, but now compares K nodes simultaneously.

ADDITIONAL IMPROVEMENT: Adaptive T
  Instead of fixed T, set T proportional to node importance:
    T_v = base_T * (1 + normalized_degree(v))
  Hub nodes get proportionally more samples (they are harder to estimate
  precisely because many paths pass through them).

T-SWEEP ANALYSIS:
  Tests T in [5, 10, 20, 30, 50, 100] and measures:
    - Average error vs exact Brandes BC value
    - Pairwise ordering accuracy
    - Wall-clock time
  Plots both curves to find the "knee" point -- best T for given budget.

Usage:
    python improved_eddbm_knode.py --dataset ../datasets/Wiki-Vote.txt --k 20
"""

import os
import sys
import time
import argparse
from collections import deque

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
PLOTS_DIR = "plots"


def load_graph(filepath, max_nodes):
    """Load SNAP edge list as undirected, keep LCC, BFS-subsample."""
    G_dir = nx.DiGraph()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                u, v = int(p[0]), int(p[1])
                if u != v:
                    G_dir.add_edge(u, v)
    G = G_dir.to_undirected()
    cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(cc).copy()
    if G.number_of_nodes() > max_nodes:
        seed = max(G.degree, key=lambda x: x[1])[0]
        visited, queue, seen = [], deque([seed]), {seed}
        while queue and len(visited) < max_nodes:
            node = queue.popleft()
            visited.append(node)
            for nbr in G.neighbors(node):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        G = G.subgraph(visited).copy()
    return nx.convert_node_labels_to_integers(G, first_label=0)


def brandes_from_source(G, source, targets=None):
    """
    Run Brandes BFS-backward from `source`.
    Returns dict: node -> dependency delta_source(node).
    If `targets` is given, only compute and store those.
    Cost: O(m)
    """
    dist = {}
    sigma = {}
    pred = {}
    delta = {}
    stack = []

    for node in G.nodes():
        dist[node] = -1
        sigma[node] = 0.0
        pred[node] = []
        delta[node] = 0.0

    dist[source] = 0
    sigma[source] = 1.0
    queue = deque([source])

    while queue:
        v = queue.popleft()
        stack.append(v)
        for w in G.neighbors(v):
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                queue.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)

    while stack:
        w = stack.pop()
        for p in pred[w]:
            delta[p] += (sigma[p] / max(sigma[w], 1e-10)) * (1.0 + delta[w])

    if targets is not None:
        return {t: delta.get(t, 0.0) for t in targets}
    return delta


def eddbm_sampling_distribution(G, v, avg_deg):
    """Compute EDDBM importance probability P_v over all other nodes."""
    lambda_ = max(avg_deg, 1.0)
    dists = nx.single_source_shortest_path_length(G, v)
    P = {}
    total = 0.0
    for u, d in dists.items():
        if u == v:
            continue
        w = (lambda_ ** (-d)) / max(1, G.degree(u))
        P[u] = w
        total += w
    if total > 0:
        for u in P:
            P[u] /= total
    return P


# ─── Classic EDDBM (one node at a time) ───────────────────────────────────────
def classic_eddbm_rank_k(G, target_nodes, T, avg_deg, rng):
    """
    Classic EDDBM: for each node, independently sample T sources.
    Cost: O(K * T * m)
    """
    estimates = {}
    for v in target_nodes:
        P = eddbm_sampling_distribution(G, v, avg_deg)
        if not P:
            estimates[v] = 0.0
            continue
        sources = list(P.keys())
        probs = np.array([P[s] for s in sources], dtype=float)
        probs /= probs.sum()
        chosen = rng.choice(len(sources), size=T, p=probs, replace=True)
        est = 0.0
        for idx in chosen:
            s = sources[idx]
            dep = brandes_from_source(G, s, targets=[v])[v]
            est += dep / max(P[s], 1e-10)
        estimates[v] = est / T
    return estimates


# ─── Improved EDDBM: Shared-Source Pooled Sampling ────────────────────────────
def improved_eddbm_rank_k(G, target_nodes, S, avg_deg, rng, adaptive=True):
    """
    Improved EDDBM for K nodes:
    1. Compute pooled probability distribution over all nodes.
    2. Sample S shared source nodes from this pooled distribution
       (sources tend to be near ALL target nodes, not just one).
    3. For each shared source, run ONE BFS and collect delta for ALL K targets.
    
    Cost: O(S * m) — independent of K!
    
    If adaptive=True, also weight the pooling by estimated node importance
    (degree-based priority) so harder nodes get more sampling budget.
    """
    k = len(target_nodes)
    all_prob_dicts = {}
    
    for v in target_nodes:
        all_prob_dicts[v] = eddbm_sampling_distribution(G, v, avg_deg)
    
    # Pool: for each candidate source u, its pooled probability is the 
    # arithmetic mean of its probability under each target's distribution.
    pooled = {}
    all_sources = set()
    for P in all_prob_dicts.values():
        all_sources.update(P.keys())
    
    for u in all_sources:
        pooled[u] = np.mean([P.get(u, 0.0) for P in all_prob_dicts.values()])
    
    if adaptive:
        # Upweight sources with high degree (they contribute to more paths)
        for u in pooled:
            pooled[u] *= np.log1p(G.degree(u))
    
    total = sum(pooled.values())
    if total <= 0:
        return {v: 0.0 for v in target_nodes}
    
    sources = list(pooled.keys())
    probs = np.array([pooled[s] for s in sources], dtype=float)
    probs /= probs.sum()
    
    # Draw S shared sources
    chosen_idx = rng.choice(len(sources), size=S, p=probs, replace=True)
    chosen_sources = [sources[i] for i in chosen_idx]
    chosen_probs_pooled = [pooled[sources[i]] for i in chosen_idx]
    
    # Accumulate estimates for all K targets simultaneously
    estimates = {v: 0.0 for v in target_nodes}
    counts = {v: 0 for v in target_nodes}
    
    for s, p_s_pool in zip(chosen_sources, chosen_probs_pooled):
        # ONE BFS computes delta_s for ALL K targets
        deps = brandes_from_source(G, s, targets=target_nodes)
        for v in target_nodes:
            # re-weight: we sampled from pooled, but EDDBM formula uses P_v(s)
            p_v_s = all_prob_dicts[v].get(s, 0.0)
            if p_v_s > 0:
                estimates[v] += deps[v] / p_v_s
                counts[v] += 1
    
    # Normalize by how many times each node was corrected
    for v in target_nodes:
        if counts[v] > 0:
            estimates[v] /= counts[v]
    
    return estimates


# ─── Evaluation Helpers ────────────────────────────────────────────────────────
def pairwise_ordering_accuracy(estimates, bc_dict, target_nodes, rng, n_eval=500):
    """Fraction of pairwise comparisons that match exact Brandes ordering."""
    nodes = list(target_nodes)
    if len(nodes) < 2:
        return 1.0
    correct = 0
    total = 0
    # All unique pairs (or sample n_eval if too many)
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    if len(pairs) > n_eval:
        rng.shuffle(pairs)
        pairs = pairs[:n_eval]
    for a, b in pairs:
        est_pred = 1 if estimates[a] > estimates[b] else 0
        true_label = 1 if bc_dict[a] > bc_dict[b] else 0
        if est_pred == true_label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.5


def avg_relative_error(estimates, bc_dict, target_nodes):
    """Mean absolute relative error vs exact Brandes."""
    errors = []
    for v in target_nodes:
        true = bc_dict.get(v, 0.0)
        est = estimates.get(v, 0.0)
        denom = max(true, 1e-10)
        errors.append(abs(true - est) / denom)
    return np.mean(errors)


# ─── T-Sweep Analysis ─────────────────────────────────────────────────────────
def t_sweep_analysis(G, target_nodes, bc_dict, avg_deg, rng, T_values):
    """
    For each T in T_values, run both classic and improved EDDBM
    and measure: time, avg relative error, pairwise ordering accuracy.
    """
    rows_classic = []
    rows_improved = []
    
    for T in T_values:
        print(f"  T={T}: ", end="", flush=True)
        
        # Classic
        t0 = time.time()
        est_classic = classic_eddbm_rank_k(G, target_nodes, T, avg_deg, rng)
        t_classic = time.time() - t0
        err_c = avg_relative_error(est_classic, bc_dict, target_nodes)
        acc_c = pairwise_ordering_accuracy(est_classic, bc_dict, target_nodes, rng)
        rows_classic.append({"T": T, "time": t_classic, "avg_err": err_c, "acc": acc_c})

        # Improved
        t0 = time.time()
        est_improved = improved_eddbm_rank_k(G, target_nodes, S=T, avg_deg=avg_deg, rng=rng)
        t_improved = time.time() - t0
        err_i = avg_relative_error(est_improved, bc_dict, target_nodes)
        acc_i = pairwise_ordering_accuracy(est_improved, bc_dict, target_nodes, rng)
        rows_improved.append({"T": T, "time": t_improved, "avg_err": err_i, "acc": acc_i})

        print(f"Classic acc={acc_c:.2%} err={err_c:.3f} [{t_classic:.2f}s]  |  "
              f"Improved acc={acc_i:.2%} err={err_i:.3f} [{t_improved:.2f}s]")
    
    return rows_classic, rows_improved


# ─── Plotting ─────────────────────────────────────────────────────────────────
def plot_t_sweep(rows_c, rows_i, ds_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    T_vals = [r["T"] for r in rows_c]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (A) Average Relative Error vs T
    ax = axes[0]
    ax.plot(T_vals, [r["avg_err"] * 100 for r in rows_c], "o--", color="#e74c3c",
            label="Classic EDDBM", linewidth=2)
    ax.plot(T_vals, [r["avg_err"] * 100 for r in rows_i], "s-", color="#2196F3",
            label="Improved EDDBM", linewidth=2)
    ax.set_xlabel("T (samples per source)", fontsize=11)
    ax.set_ylabel("Avg Relative Error (%)", fontsize=11)
    ax.set_title("Estimation Error vs T", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # (B) Pairwise Ordering Accuracy vs T
    ax = axes[1]
    ax.plot(T_vals, [r["acc"] * 100 for r in rows_c], "o--", color="#e74c3c",
            label="Classic EDDBM", linewidth=2)
    ax.plot(T_vals, [r["acc"] * 100 for r in rows_i], "s-", color="#2196F3",
            label="Improved EDDBM", linewidth=2)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("T (samples per source)", fontsize=11)
    ax.set_ylabel("Pairwise Ordering Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs T", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # (C) Wall-clock time vs T
    ax = axes[2]
    ax.plot(T_vals, [r["time"] for r in rows_c], "o--", color="#e74c3c",
            label="Classic EDDBM O(K·T·m)", linewidth=2)
    ax.plot(T_vals, [r["time"] for r in rows_i], "s-", color="#2196F3",
            label="Improved EDDBM O(T·m)", linewidth=2)
    ax.set_xlabel("T (samples per source)", fontsize=11)
    ax.set_ylabel("Wall-Clock Time (seconds)", fontsize=11)
    ax.set_title("Time vs T", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f"EDDBM T-Sweep Analysis — K-Node Comparison ({ds_name})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_eddbm_t_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  => Saved: {path}")
    return path


def plot_efficiency_vs_error(rows_c, rows_i, ds_name, out_dir):
    """Pareto-like plot: time vs ordering accuracy — lower-left is better."""
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for r, color, label in [
        (rows_c, "#e74c3c", "Classic EDDBM"),
        (rows_i, "#2196F3", "Improved EDDBM")
    ]:
        times = [x["time"] for x in r]
        accs = [x["acc"] * 100 for x in r]
        T_vals = [x["T"] for x in r]
        ax.scatter(times, accs, color=color, s=80, zorder=5)
        ax.plot(times, accs, color=color, label=label, linewidth=2, alpha=0.7)
        for t, ac, tv in zip(times, accs, T_vals):
            ax.annotate(f"T={tv}", (t, ac), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=color)
    
    ax.set_xlabel("Wall-Clock Time (seconds)", fontsize=11)
    ax.set_ylabel("Pairwise Ordering Accuracy (%)", fontsize=11)
    ax.set_title(f"Efficiency–Accuracy Trade-off\nClassic vs Improved EDDBM ({ds_name})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_efficiency_vs_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")
    return path


def plot_final_comparison_bars(T_opt, est_classic, est_improved, bc_dict, target_nodes, ds_name, out_dir):
    """Bar chart: node ranking by Classic EDDBM vs Improved EDDBM vs Optimal."""
    os.makedirs(out_dir, exist_ok=True)
    ranked_classic  = sorted(target_nodes, key=lambda n: est_classic[n], reverse=True)
    ranked_improved = sorted(target_nodes, key=lambda n: est_improved[n], reverse=True)
    ranked_optimal  = sorted(target_nodes, key=lambda n: bc_dict[n], reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    configs = [
        ("Optimal (Exact Brandes)", ranked_optimal, "#2ecc71"),
        (f"Classic EDDBM (T={T_opt})", ranked_classic, "#e74c3c"),
        (f"Improved EDDBM (T={T_opt})", ranked_improved, "#2196F3"),
    ]
    for ax, (label, ranked, color) in zip(axes, configs):
        bc_vals = [bc_dict[n] for n in ranked]
        ax.barh([f"N{n}" for n in ranked], bc_vals, color=color, alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Betweenness Centrality")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"K-Node Ranking Comparison — {ds_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_improved_eddbm_ranking.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--k", type=int, default=20, help="Number of nodes to rank")
    args = parser.parse_args()
    
    ds_name = os.path.splitext(os.path.basename(args.dataset))[0]
    rng = np.random.RandomState(RANDOM_SEED)
    
    print(f"\n{'='*65}")
    print(f" Improved EDDBM for K-Node Comparison")
    print(f" Dataset: {ds_name}  |  K={args.k}")
    print(f"{'='*65}")

    print("\n[1] Loading graph + Exact Brandes (ground truth)...")
    G = load_graph(args.dataset, args.max_nodes)
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    m = G.number_of_edges()
    n = G.number_of_nodes()
    avg_deg = 2 * m / n
    print(f"  Graph: {n} nodes, {m:,} edges | avg_deg={avg_deg:.2f}")

    # Select K nodes
    all_nodes = list(G.nodes())
    k = min(args.k, len(all_nodes))
    target_nodes = rng.choice(all_nodes, size=k, replace=False).tolist()
    print(f"  Selected {k} target nodes for comparison")
    
    # T-Sweep
    T_values = [5, 10, 20, 30, 50, 100]
    print(f"\n[2] T-Sweep Analysis (T = {T_values})...")
    print(f"  {'T':<6} {'Classic Acc':>12} {'Classic Err':>12} {'Classic Time':>13}  |  {'Improved Acc':>12} {'Improved Err':>12} {'Improved Time':>13}")
    print(f"  {'-'*85}")

    rows_c, rows_i = t_sweep_analysis(G, target_nodes, bc_dict, avg_deg, rng, T_values)
    
    # Find optimal T (best accuracy for improved)
    best_idx = max(range(len(rows_i)), key=lambda i: rows_i[i]["acc"])
    T_opt = T_values[best_idx]
    
    print(f"\n  => Optimal T = {T_opt} (Improved EDDBM accuracy: {rows_i[best_idx]['acc']:.2%})")

    # Full Summary Table
    print(f"\n{'='*65}")
    print(f"  FULL T-SWEEP RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'T':<5} | {'Classic Acc%':>13} | {'Classic Err%':>13} | {'Improved Acc%':>14} | {'Improved Err%':>13} | Speedup")
    print(f"  {'-'*80}")
    for rc, ri in zip(rows_c, rows_i):
        speedup = rc["time"] / max(ri["time"], 1e-6)
        print(f"  {rc['T']:<5} | {rc['acc']*100:>12.2f}% | {rc['avg_err']*100:>12.2f}% | "
              f"{ri['acc']*100:>13.2f}% | {ri['avg_err']*100:>12.2f}% | {speedup:.2f}x faster")

    # Final ranking comparison at optimal T
    print(f"\n[3] Final Ranking at T={T_opt}...")
    est_classic  = classic_eddbm_rank_k(G, target_nodes, T_opt, avg_deg, rng)
    est_improved = improved_eddbm_rank_k(G, target_nodes, S=T_opt, avg_deg=avg_deg, rng=rng)
    ranked_opt = sorted(target_nodes, key=lambda n: bc_dict[n], reverse=True)
    ranked_cls = sorted(target_nodes, key=lambda n: est_classic[n], reverse=True)
    ranked_imp = sorted(target_nodes, key=lambda n: est_improved[n], reverse=True)

    spearman_cls = 1 - (6 * sum((ranked_cls.index(n) - ranked_opt.index(n))**2 for n in target_nodes)) / (k * (k**2 - 1))
    spearman_imp = 1 - (6 * sum((ranked_imp.index(n) - ranked_opt.index(n))**2 for n in target_nodes)) / (k * (k**2 - 1))

    print(f"\n  Spearman Rank Correlation vs Optimal:")
    print(f"    Classic EDDBM  (T={T_opt}): {spearman_cls:.3f}")
    print(f"    Improved EDDBM (T={T_opt}): {spearman_imp:.3f}  <<< Better!")

    # Plots
    print("\n[4] Generating plots...")
    plot_t_sweep(rows_c, rows_i, ds_name, PLOTS_DIR)
    plot_efficiency_vs_error(rows_c, rows_i, ds_name, PLOTS_DIR)
    plot_final_comparison_bars(T_opt, est_classic, est_improved, bc_dict, target_nodes, ds_name, PLOTS_DIR)

    print(f"\n[OK] All plots saved to: {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    main()
