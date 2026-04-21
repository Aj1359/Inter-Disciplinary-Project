"""
node_comparison_model.py
========================
ML model for pairwise betweenness centrality node comparison.

Problem Statement
-----------------
EDDBM estimates BC in O(m) time but is not always accurate for ordering two
nodes by betweenness. This ML model learns structural features of nodes and
predicts: given nodes A and B, which has HIGHER betweenness centrality?

Approach
--------
1. Load unweighted graph (SNAP edge-list format).
2. Compute EXACT betweenness centrality (Brandes) as ground truth.
3. Extract 10 structural features per node (degree, PageRank, k-core, etc.)
4. Build PAIRWISE training samples:
     X[i] = features(A) - features(B)      (difference vector)
     y[i] = 1 if BC(A) > BC(B) else 0
5. Train GradientBoosting binary classifier.
6. Evaluate: ML accuracy vs EDDBM accuracy vs random baseline.
7. Save model to node_comparison_model.pkl and plots to plots/.

Usage
-----
    python node_comparison_model.py [--dataset PATH] [--max-nodes N] [--n-pairs M]

    # Run on all available datasets automatically:
    python node_comparison_model.py

    # Run on a specific dataset with custom size:
    python node_comparison_model.py --dataset ../Wiki-Vote.txt --max-nodes 2000
"""

import os
import sys
import argparse
import pickle
import time
import random
import warnings
from collections import deque

# Windows UTF-8 console fix
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
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Configuration
CANDIDATE_DATASETS = [
    "../datasets/Wiki-Vote.txt",
    "../datasets/CA-HepTh.txt",
    "../datasets/p2p-Gnutella08.txt",
    "../datasets/as20000102.txt",
    "../datasets/facebook_combined.txt",
]

DEFAULT_MAX_NODES = 2000   # Brandes is O(n*m), keep this manageable
DEFAULT_N_PAIRS   = 5000   # pairwise training samples
EDDBM_EVAL_PAIRS  = 100    # pairs to evaluate for EDDBM comparison (slow)
EDDBM_T           = 30     # EDDBM samples per node
RANDOM_SEED       = 42
PLOTS_DIR         = "plots"
MODEL_SAVE_PATH   = "node_comparison_model.pkl"

# Node feature names (must match order in extract_node_features)
FEATURE_NAMES = [
    "degree",
    "log_degree",
    "clustering_coeff",
    "avg_neighbor_degree",
    "core_number",
    "closeness_centrality",
    "pagerank",
    "eccentricity_est",
    "triangle_count",
    "square_clustering",
]

# Graph Loading
def load_graph(filepath: str, max_nodes: int) -> nx.Graph:
    """
    Load SNAP-format edge list, convert to undirected, extract the largest
    connected component, and subsample to at most max_nodes nodes via BFS
    from the highest-degree seed (preserves connectivity).
    """
    print(f"  Loading: {filepath}")
    G_dir = nx.DiGraph()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                G_dir.add_edge(u, v)

    G = G_dir.to_undirected()

    # Keep only largest connected component
    cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(cc).copy()

    # BFS subsample from highest-degree node
    if G.number_of_nodes() > max_nodes:
        seed = max(G.degree, key=lambda x: x[1])[0]
        visited = []
        queue = deque([seed])
        seen = {seed}
        while queue and len(visited) < max_nodes:
            node = queue.popleft()
            visited.append(node)
            for nbr in G.neighbors(node):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        G = G.subgraph(visited).copy()

    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G


# Feature Extraction
def extract_node_features(G: nx.Graph) -> tuple:
    """
    Compute a feature matrix of shape (n_nodes, n_features).
    All features are computed once per graph and are O(m) or O(n log n).

    Returns:
        F          : np.ndarray of shape (n, len(FEATURE_NAMES))
        nodes_list : list of node IDs in row order
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    print("    Computing degree ...")
    degree_dict = dict(G.degree())

    print("    Computing clustering coefficients ...")
    clust = nx.clustering(G)

    print("    Computing average neighbor degree ...")
    avg_nd = nx.average_neighbor_degree(G)

    print("    Computing k-core numbers ...")
    core = nx.core_number(G)

    print("    Computing closeness centrality ...")
    closeness = nx.closeness_centrality(G)

    print("    Computing PageRank ...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=300, tol=1e-4)

    print("    Computing triangles ...")
    try:
        triangles = nx.triangles(G)
    except Exception:
        triangles = {v: 0 for v in nodes}

    print("    Computing square clustering ...")
    sq_clust = nx.square_clustering(G)

    print("    Estimating eccentricity (BFS double-sweep) ...")
    # Double-sweep BFS heuristic for eccentricity approximation
    rng_local = random.Random(RANDOM_SEED)
    sample = rng_local.sample(nodes, min(15, n))
    ecc_est = {v: 0 for v in nodes}
    for seed in sample:
        lengths = nx.single_source_shortest_path_length(G, seed)
        for v, d in lengths.items():
            ecc_est[v] = max(ecc_est[v], d)
    diam = max(ecc_est.values()) if ecc_est else 1
    if diam == 0:
        diam = 1

    # Build feature matrix
    F = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    for v in nodes:
        i = node_idx[v]
        deg = degree_dict[v]
        F[i, 0] = deg
        F[i, 1] = np.log1p(deg)
        F[i, 2] = clust.get(v, 0.0)
        F[i, 3] = avg_nd.get(v, 0.0)
        F[i, 4] = core.get(v, 0)
        F[i, 5] = closeness.get(v, 0.0)
        F[i, 6] = pagerank.get(v, 0.0)
        F[i, 7] = ecc_est[v] / diam
        F[i, 8] = triangles.get(v, 0)
        F[i, 9] = sq_clust.get(v, 0.0)

    return F, nodes


# EDDBM Implementation (Baseline)
def eddbm_sampling_prob(G: nx.Graph, v, avg_deg: float) -> dict:
    """
    Compute EDDBM non-uniform sampling distribution P_v over all other nodes.
    Formula from paper: P_v(u) proportional to lambda^{-d(v,u)} / deg(u)
    where lambda = average degree.
    """
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


def brandes_dependency(G: nx.Graph, source, target) -> float:
    """Single-source Brandes — returns dependency delta_{source}(target)."""
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

    return delta[target]


def eddbm_estimate_bc(G: nx.Graph, v, T: int, avg_deg: float, rng: np.random.RandomState) -> float:
    """
    Estimate BC(v) using EDDBM:
      est = (1/T) * sum_{i=1}^{T} [ delta_s(v) / P_v(s) ]
    where s is importance-sampled from P_v.
    """
    P = eddbm_sampling_prob(G, v, avg_deg)
    if not P:
        return 0.0
    sources = list(P.keys())
    probs = np.array([P[s] for s in sources], dtype=float)
    probs /= probs.sum()

    chosen = rng.choice(len(sources), size=T, p=probs, replace=True)
    est = 0.0
    for idx in chosen:
        s = sources[idx]
        p_s = P[s]
        if p_s > 0:
            dep = brandes_dependency(G, s, v)
            est += dep / p_s
    return est / T


# Pairwise Dataset Builder
def build_pairwise_dataset(
    F: np.ndarray,
    bc: np.ndarray,
    n_pairs: int,
    rng: np.random.RandomState,
) -> tuple:
    """
    Build pairwise comparison dataset.
    
    X[i] = F[a_i] - F[b_i]   (feature difference)
    y[i] = 1 if bc[a_i] > bc[b_i], else 0
    
    Returns X, y, a_indices, b_indices
    """
    n = len(bc)
    a = rng.randint(0, n, size=n_pairs)
    b = rng.randint(0, n, size=n_pairs)
    # Ensure no self-comparisons
    self_mask = a == b
    b[self_mask] = (b[self_mask] + 1) % n

    X = F[a] - F[b]
    y = (bc[a] > bc[b]).astype(int)
    return X, y, a, b


# Plotting
def save_feature_importance_plot(pipeline, ds_name: str):
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    sorted_i = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(FEATURE_NAMES)))
    bars = ax.bar(
        range(len(FEATURE_NAMES)),
        importances[sorted_i],
        color=[colors[j] for j in range(len(FEATURE_NAMES))],
        edgecolor="white", linewidth=0.6
    )
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(
        [FEATURE_NAMES[i] for i in sorted_i],
        rotation=38, ha="right", fontsize=9
    )
    for bar, val in zip(bars, importances[sorted_i]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("Feature Importance (Gini)", fontsize=10)
    ax.set_title(
        f"Feature Importances — GradientBoosting ({ds_name})",
        fontsize=11, fontweight="bold"
    )
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("#f5f5f5")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{ds_name}_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def save_accuracy_bar(results: list):
    if not results:
        return
    df = pd.DataFrame(results)
    labels = df["dataset"].tolist()
    x = np.arange(len(labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(max(8, len(labels)*2.5), 5))
    r1 = ax.bar(x - w, df["ml_acc"] * 100,    w, label="ML Model",         color="#2196F3", edgecolor="white")
    r2 = ax.bar(x,     df["eddbm_acc"] * 100, w, label=f"EDDBM (T={EDDBM_T})", color="#FF9800", edgecolor="white")
    r3 = ax.bar(x + w, df["random_acc"] * 100, w, label="Random Baseline",  color="#9C27B0", edgecolor="white", alpha=0.75)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(40, 108)
    ax.set_ylabel("Ordering Accuracy (%)", fontsize=11)
    ax.set_title(
        "Pairwise BC Ordering Accuracy: ML vs EDDBM vs Random",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("#f5f5f5")

    for bars in [r1, r2, r3]:
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.6,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def save_bc_pagerank_scatter(bc_aligned, F, ds_name: str):
    pr_col = FEATURE_NAMES.index("pagerank")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(F[:, pr_col], bc_aligned, alpha=0.25, s=8, color="#2196F3")
    ax.set_xlabel("PageRank", fontsize=10)
    ax.set_ylabel("Exact Betweenness Centrality", fontsize=10)
    ax.set_title(f"BC vs PageRank — {ds_name}", fontsize=11, fontweight="bold")
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_facecolor("#f5f5f5")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{ds_name}_bc_vs_pagerank.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


# Main Pipeline Per Dataset
def run_dataset(dataset_path: str, max_nodes: int, n_pairs: int,
                rng: np.random.RandomState) -> dict:

    ds_name = os.path.splitext(os.path.basename(dataset_path))[0]
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  DATASET: {ds_name}")
    print(sep)

    # ------------------------------------------------------------------
    # 1. Load Graph
    # ------------------------------------------------------------------
    t0 = time.time()
    G = load_graph(dataset_path, max_nodes)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"  Graph: {n} nodes, {m} edges  [{time.time()-t0:.1f}s]")
    avg_deg = (2 * m) / n if n > 0 else 1.0

    # ------------------------------------------------------------------
    # 2. Exact Betweenness (Brandes ground truth)
    # ------------------------------------------------------------------
    print("  Computing exact betweenness centrality (Brandes)...")
    t0 = time.time()
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    nodes_list = list(G.nodes())
    bc = np.array([bc_dict[v] for v in nodes_list])
    print(f"  BC computed in {time.time()-t0:.1f}s  "
          f"| range: [{bc.min():.4e}, {bc.max():.4e}]")

    # ------------------------------------------------------------------
    # 3. Feature Extraction
    # ------------------------------------------------------------------
    print("  Extracting node features...")
    t0 = time.time()
    F, nodes_ordered = extract_node_features(G)
    bc_aligned = np.array([bc_dict[v] for v in nodes_ordered])
    print(f"  Features done in {time.time()-t0:.1f}s  "
          f"| shape: {F.shape}")

    # ------------------------------------------------------------------
    # 4. Build Pairwise Dataset
    # ------------------------------------------------------------------
    print(f"  Building {n_pairs} pairwise comparison samples...")
    X, y, a_idx, b_idx = build_pairwise_dataset(F, bc_aligned, n_pairs, rng)
    pos_frac = y.mean()
    print(f"  Class balance: {pos_frac*100:.1f}% A>B | {(1-pos_frac)*100:.1f}% A<=B")

    # Train / test split  (stratified)
    split_idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        split_idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    a_te, b_te = a_idx[te_idx], b_idx[te_idx]

    # ------------------------------------------------------------------
    # 5. Train ML Model (GradientBoosting in sklearn Pipeline)
    # ------------------------------------------------------------------
    print("  Training GradientBoosting classifier...")
    t0 = time.time()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
        ))
    ])
    pipeline.fit(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")

    # ------------------------------------------------------------------
    # 6. Evaluate ML Model
    # ------------------------------------------------------------------
    y_pred = pipeline.predict(X_te)
    ml_acc = accuracy_score(y_te, y_pred)

    try:
        y_prob = pipeline.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
    except Exception:
        auc = float("nan")

    print(f"\n  --- ML Model Test Results ---")
    print(f"  Test Accuracy : {ml_acc*100:.2f}%")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_te, y_pred,
                                target_names=["A<=B", "A>B"],
                                zero_division=0))

    # ------------------------------------------------------------------
    # 7. EDDBM Baseline (same test pairs — limited to EDDBM_EVAL_PAIRS)
    # ------------------------------------------------------------------
    eval_n = min(EDDBM_EVAL_PAIRS, len(a_te))
    print(f"  --- EDDBM Baseline (T={EDDBM_T}, evaluating {eval_n} pairs) ---")
    t0 = time.time()
    eddbm_correct = 0
    node_arr = np.array(nodes_ordered)

    for k in range(eval_n):
        node_a = node_arr[a_te[k]]
        node_b = node_arr[b_te[k]]
        est_a = eddbm_estimate_bc(G, node_a, EDDBM_T, avg_deg, rng)
        est_b = eddbm_estimate_bc(G, node_b, EDDBM_T, avg_deg, rng)
        pred_eddbm = 1 if est_a > est_b else 0
        true_label = 1 if bc_dict[node_a] > bc_dict[node_b] else 0
        if pred_eddbm == true_label:
            eddbm_correct += 1

    eddbm_acc = eddbm_correct / eval_n if eval_n > 0 else 0.5
    print(f"  EDDBM Accuracy: {eddbm_acc*100:.2f}%  [{time.time()-t0:.1f}s]")

    # ------------------------------------------------------------------
    # 8. Random Baseline
    # ------------------------------------------------------------------
    rand_labels = rng.randint(0, 2, size=len(y_te))
    random_acc = accuracy_score(y_te, rand_labels)
    print(f"  Random Baseline: {random_acc*100:.2f}%")

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    print("  Saving plots...")
    save_feature_importance_plot(pipeline, ds_name)
    save_bc_pagerank_scatter(bc_aligned, F, ds_name)

    return {
        "dataset":    ds_name,
        "nodes":      n,
        "edges":      m,
        "ml_acc":     ml_acc,
        "auc":        auc,
        "eddbm_acc":  eddbm_acc,
        "random_acc": random_acc,
        "pipeline":   pipeline,
    }


# Entry Point
def main():
    parser = argparse.ArgumentParser(
        description="ML model for pairwise betweenness node comparison"
    )
    parser.add_argument("--dataset",   type=str, default=None,
                        help="Path to a SNAP edge-list file")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES,
                        help=f"Max nodes to use (default: {DEFAULT_MAX_NODES})")
    parser.add_argument("--n-pairs",   type=int, default=DEFAULT_N_PAIRS,
                        help=f"Pairwise training samples (default: {DEFAULT_N_PAIRS})")
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Select datasets
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [p for p in CANDIDATE_DATASETS if os.path.exists(p)]

    if not datasets:
        print("\nERROR: No datasets found. Tried:")
        for p in CANDIDATE_DATASETS:
            print(f"  {p}")
        print("\nRun with: python node_comparison_model.py --dataset PATH")
        sys.exit(1)

    print(f"\nFound {len(datasets)} dataset(s): {[os.path.basename(d) for d in datasets]}")

    all_results = []
    best_pipeline = None
    best_acc = -1.0

    for ds_path in datasets:
        result = run_dataset(ds_path, args.max_nodes, args.n_pairs, rng)
        all_results.append(result)
        if result["ml_acc"] > best_acc:
            best_acc = result["ml_acc"]
            best_pipeline = result["pipeline"]

    # ------------------------------------------------------------------
    # Save best model
    # ------------------------------------------------------------------
    with open(MODEL_SAVE_PATH, "wb") as fh:
        pickle.dump(best_pipeline, fh)
    model_abs = os.path.abspath(MODEL_SAVE_PATH)
    print(f"\n[OK] Model saved -> {model_abs}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    header = f"  {'Dataset':<22} {'N':>5} {'M':>7} {'ML%':>7} {'EDDBM%':>8} {'Random%':>9} {'AUC':>7}"
    print(header)
    print("  " + "-" * 68)
    for r in all_results:
        print(f"  {r['dataset']:<22} {r['nodes']:>5} {r['edges']:>7} "
              f"{r['ml_acc']*100:>6.2f}% {r['eddbm_acc']*100:>7.2f}% "
              f"{r['random_acc']*100:>8.2f}% {r['auc']:>7.4f}")

    # ------------------------------------------------------------------
    # Save combined accuracy comparison plot
    # ------------------------------------------------------------------
    print("  Saving accuracy comparison plot...")
    save_accuracy_bar(all_results)

    plots_abs = os.path.abspath(PLOTS_DIR)
    print(f"\n[OK] All plots saved in: {plots_abs}")
    print("[OK] Done!\n")


if __name__ == "__main__":
    main()
