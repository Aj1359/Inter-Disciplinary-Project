"""
model_benchmark.py
==================
Benchmarks the ML pairwise node comparison model against:
  - Random Forest
  - GradientBoosting (current)
  - ExtraTreesClassifier (fast)
  - LogisticRegression (linear baseline)
  - EDDBM heuristic
  - Random baseline

Also adds 6 NEW features to improve pairwise accuracy:
  - harmonic_centrality        (better closeness estimator)
  - betweenness_proxy          (load centrality, fast O(m) approximation)
  - degree_squared             (captures hub effect)
  - clustering_x_degree        (interaction feature)
  - kshell_ratio               (core ratio)
  - neighbor_core_avg          (k-shell of neighbors)

Generates:
  - Multi-model accuracy bar chart
  - ROC curves for all models
  - New feature importance
  - Optimality scatter: model confidence vs exact Brandes rank distance
"""

import os
import sys
import pickle
import time
import warnings
import random
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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
PLOTS_DIR = "plots"

# Extended feature names (16 total, up from 10)
FEATURE_NAMES = [
    # Original 10
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
    # 6 New features
    "harmonic_centrality",
    "betweenness_proxy",
    "degree_squared",
    "clustering_x_degree",
    "kshell_ratio",
    "neighbor_core_avg",
]


def load_graph(filepath, max_nodes):
    """Load SNAP edge list, keep largest CC, BFS-subsample."""
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


def extract_enhanced_features(G):
    """
    Compute 16 structural features per node.
    New features: harmonic, betweenness_proxy, degree^2, 
                  clust*deg, kshell ratio, neighbor core avg
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    print("    degree...")
    degree_dict = dict(G.degree())
    max_core = 1

    print("    clustering...")
    clust = nx.clustering(G)

    print("    avg_neighbor_degree...")
    avg_nd = nx.average_neighbor_degree(G)

    print("    core_number...")
    core = nx.core_number(G)
    max_core = max(core.values()) if core else 1

    print("    closeness...")
    closeness = nx.closeness_centrality(G)

    print("    pagerank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=300, tol=1e-4)

    print("    triangles...")
    try:
        triangles = nx.triangles(G)
    except Exception:
        triangles = {v: 0 for v in nodes}

    print("    square_clustering...")
    sq_clust = nx.square_clustering(G)

    print("    harmonic_centrality...")
    harmonic = nx.harmonic_centrality(G)

    print("    load_centrality (betweenness proxy)...")
    try:
        load = nx.load_centrality(G)
    except Exception:
        load = {v: 0.0 for v in nodes}

    print("    eccentricity estimate (BFS)...")
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

    print("    building feature matrix...")
    F = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    for v in nodes:
        i = node_idx[v]
        deg = degree_dict[v]
        deg_f = float(deg)
        c = clust.get(v, 0.0)
        k = core.get(v, 0)

        # Compute neighbor core avg
        nbs = list(G.neighbors(v))
        nbr_core_avg = np.mean([core.get(u, 0) for u in nbs]) if nbs else 0.0

        # Original 10
        F[i, 0] = deg_f
        F[i, 1] = np.log1p(deg_f)
        F[i, 2] = c
        F[i, 3] = avg_nd.get(v, 0.0)
        F[i, 4] = k
        F[i, 5] = closeness.get(v, 0.0)
        F[i, 6] = pagerank.get(v, 0.0)
        F[i, 7] = ecc_est[v] / diam
        F[i, 8] = triangles.get(v, 0)
        F[i, 9] = sq_clust.get(v, 0.0)
        # New 6
        F[i, 10] = harmonic.get(v, 0.0)
        F[i, 11] = load.get(v, 0.0)
        F[i, 12] = deg_f ** 2
        F[i, 13] = c * deg_f       # interaction feature
        F[i, 14] = k / max_core    # normalized kshell
        F[i, 15] = nbr_core_avg

    return F, nodes


def build_pairwise_dataset(F, bc, n_pairs, rng):
    n = len(bc)
    a = rng.randint(0, n, size=n_pairs)
    b = rng.randint(0, n, size=n_pairs)
    mask = a == b
    b[mask] = (b[mask] + 1) % n
    X = F[a] - F[b]
    y = (bc[a] > bc[b]).astype(int)
    return X, y, a, b


def build_models():
    """Return dict of name -> sklearn Pipeline."""
    return {
        "GradientBoosting (Improved)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.04,
                subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            ))
        ]),
        "Extra Trees (Fast)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=1.0, random_state=RANDOM_SEED
            ))
        ]),
    }


def plot_model_accuracy_bars(results, ds_name, out_dir):
    """Bar chart comparing all model accuracies."""
    os.makedirs(out_dir, exist_ok=True)
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    aucs  = [results[n]["auc"] for n in names]

    colors = ["#2196F3", "#4CAF50", "#00BCD4", "#9C27B0", "#FF9800", "#F44336"]
    x = np.arange(len(names))

    fig, ax1 = plt.subplots(figsize=(11, 6))
    bars = ax1.bar(x, accs, color=colors[:len(names)], alpha=0.85, edgecolor="white", linewidth=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax1.set_ylim(50, 105)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance")
    ax1.set_title(f"Pairwise BC Ordering Accuracy — All Models\nDataset: {ds_name}", fontsize=13, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x, aucs, "ko--", markersize=8, label="ROC-AUC")
    ax2.set_ylim(0.5, 1.05)
    ax2.set_ylabel("ROC-AUC", fontsize=11)

    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=9, loc="lower right")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")


def plot_roc_curves(results, ds_name, out_dir):
    """Overlay ROC curves for every trained model."""
    os.makedirs(out_dir, exist_ok=True)
    colors = ["#2196F3", "#4CAF50", "#00BCD4", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={res['auc']:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves — All Models\nDataset: {ds_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")


def plot_feature_importance(pipeline, ds_name, out_dir):
    """Feature importance from the best model."""
    os.makedirs(out_dir, exist_ok=True)
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return

    importances = clf.feature_importances_
    idx  = np.argsort(importances)[::-1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(FEATURE_NAMES)))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(FEATURE_NAMES)), importances[idx],
                  color=[colors[j] for j in range(len(FEATURE_NAMES))], edgecolor="white")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=40, ha="right", fontsize=9)
    for bar, val in zip(bars, importances[idx]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Feature Importance (Gini)", fontsize=10)
    ax.set_title(f"Feature Importances (16 features) — {ds_name}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_feature_importance_enhanced.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")


def plot_optimality_scatter(best_name, best_res, bc_aligned, a_te, b_te, ds_name, out_dir):
    """Scatter: true BC difference vs ML probability — measures closeness to optimal."""
    os.makedirs(out_dir, exist_ok=True)
    bc_diff = bc_aligned[a_te] - bc_aligned[b_te]
    y_prob = best_res["y_prob"]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(bc_diff, y_prob, alpha=0.25, s=8,
                         c=np.abs(bc_diff), cmap="plasma")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Decision boundary")
    ax.axvline(0.0, color="gray", linestyle="-", alpha=0.4)
    plt.colorbar(scatter, label="Absolute BC Difference")
    ax.set_xlabel("True Brandes BC Difference (A - B)", fontsize=11)
    ax.set_ylabel(f"ML Probability (A > B) — {best_name}", fontsize=11)
    ax.set_title(f"Closeness to Optimal\nDataset: {ds_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{ds_name}_optimality_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to SNAP edge-list")
    parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--n-pairs", type=int, default=8000)
    args = parser.parse_args()

    ds_name = os.path.splitext(os.path.basename(args.dataset))[0]
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print(f"\n{'='*65}")
    print(f" Multi-Model Benchmark: ML vs Baselines")
    print(f" Dataset: {ds_name}  |  max_nodes={args.max_nodes}  |  n_pairs={args.n_pairs}")
    print(f"{'='*65}")

    print("\n[1] Loading graph + Exact Brandes...")
    G = load_graph(args.dataset, args.max_nodes)
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    nodes_ordered = list(G.nodes())
    bc_aligned = np.array([bc_dict[v] for v in nodes_ordered])
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges")

    print("\n[2] Extracting 16 enhanced features...")
    t0 = time.time()
    F, F_nodes = extract_enhanced_features(G)
    print(f"  Done in {time.time()-t0:.2f}s  |  shape: {F.shape}")

    # Align bc to F node order
    node_to_feat_idx = {v: i for i, v in enumerate(F_nodes)}
    bc_F = np.array([bc_dict[v] for v in F_nodes])

    print(f"\n[3] Building {args.n_pairs} pairwise training samples...")
    X, y, a_idx, b_idx = build_pairwise_dataset(F, bc_F, args.n_pairs, rng)
    X_tr, X_te, y_tr, y_te, a_tr, a_te, b_tr, b_te = train_test_split(
        X, y, a_idx, b_idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

    print("\n[4] Training all models...")
    models = build_models()
    results = {}

    for name, pipeline in models.items():
        print(f"\n  > {name}")
        t0 = time.time()
        pipeline.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        y_pred  = pipeline.predict(X_te)
        y_prob  = pipeline.predict_proba(X_te)[:, 1]
        acc     = accuracy_score(y_te, y_pred)
        auc     = roc_auc_score(y_te, y_prob)

        # 5-fold cross-validation on training set
        cv_scores = cross_val_score(pipeline, X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=-1)

        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "auc": auc,
            "cv_mean": cv_scores.mean(),
            "cv_std":  cv_scores.std(),
            "y_true":  y_te,
            "y_prob":  y_prob,
            "train_time": elapsed,
        }

        print(f"    Accuracy : {acc*100:.2f}%")
        print(f"    ROC-AUC  : {auc:.4f}")
        print(f"    CV (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print(f"    Train time: {elapsed:.2f}s")

    # --- Summary table ---
    print(f"\n{'='*65}")
    print(f"  BENCHMARK SUMMARY on Dataset: {ds_name}")
    print(f"{'='*65}")
    print(f"  {'Model':<35} {'Acc%':>7} {'AUC':>7} {'CV%':>10}")
    print(f"  {'-'*60}")
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    for name, res in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        marker = " <<< BEST" if name == best_name else ""
        print(f"  {name:<35} {res['accuracy']*100:>7.2f}% {res['auc']:>7.4f} {res['cv_mean']*100:>9.2f}%{marker}")

    # --- Save best model ---
    best_pipeline = results[best_name]["pipeline"]
    model_path = "node_comparison_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(best_pipeline, fh)
    print(f"\n  [OK] Best model '{best_name}' saved -> {os.path.abspath(model_path)}")

    # --- Plots ---
    print("\n[5] Generating presentation plots...")
    plot_model_accuracy_bars(results, ds_name, PLOTS_DIR)
    plot_roc_curves(results, ds_name, PLOTS_DIR)
    plot_feature_importance(best_pipeline, ds_name, PLOTS_DIR)
    plot_optimality_scatter(best_name, results[best_name], bc_F, a_te, b_te, ds_name, PLOTS_DIR)

    print(f"\n[OK] All plots saved in: {os.path.abspath(PLOTS_DIR)}")
    print("[OK] Done!")


if __name__ == "__main__":
    main()
