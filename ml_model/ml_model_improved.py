"""
ml_model_improved.py
====================
Improved ML model for pairwise betweenness centrality node comparison.

Improvements over node_comparison_model.py:
  - 20 rich structural features (vs 10)
  - Stacking ensemble: GradientBoosting + RandomForest + ExtraTrees + Ridge meta
  - CalibratedClassifierCV for reliable probability output
  - Stratified train/test split + 5-fold CV
  - Two summary plots across ALL datasets:
      1. Efficiency plot  : pairwise accuracy (ML vs EDDBM vs Random) per dataset
      2. Avg Error plot   : mean absolute BC-rank error per dataset
  - Per-dataset plots saved in plots/

Usage
-----
    # Run on all available datasets (auto-detected):
    python ml_model_improved.py

    # Single dataset:
    python ml_model_improved.py --dataset ../datasets/Wiki-Vote.txt
"""

import os
import sys
import argparse
import pickle
import time
import random
import warnings
from collections import deque

# Windows UTF-8 fix
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

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
CANDIDATE_DATASETS = [
    "../datasets/Wiki-Vote.txt",
    "../datasets/CA-HepTh.txt",
    "../datasets/p2p-Gnutella08.txt",
    "../datasets/as20000102.txt",
    "../datasets/facebook_combined.txt",
    "../datasets/oregon1_010331.txt",
]

DEFAULT_MAX_NODES = 2000
DEFAULT_N_PAIRS   = 6000
EDDBM_EVAL_PAIRS  = 120
EDDBM_T           = 30
RANDOM_SEED       = 42
PLOTS_DIR         = "plots"
MODEL_SAVE_PATH   = "ml_model_improved.pkl"

# 20 features
FEATURE_NAMES = [
    "degree",                 # 0
    "log_degree",             # 1
    "clustering_coeff",       # 2
    "avg_neighbor_degree",    # 3
    "core_number",            # 4
    "closeness_centrality",   # 5
    "pagerank",               # 6
    "eccentricity_est",       # 7
    "triangle_count",         # 8
    "square_clustering",      # 9
    "harmonic_centrality",    # 10
    "betweenness_proxy",      # 11  (load centrality)
    "degree_squared",         # 12
    "clustering_x_degree",    # 13
    "kshell_ratio",           # 14
    "neighbor_core_avg",      # 15
    "neighbor_pagerank_avg",  # 16  NEW
    "degree_centrality",      # 17  NEW
    "log_triangles",          # 18  NEW
    "core_x_degree",          # 19  NEW
]
N_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────────────────────
# Graph Loading
# ─────────────────────────────────────────
def load_graph(filepath: str, max_nodes: int) -> nx.Graph:
    """Load SNAP-format edge list, largest CC, BFS-subsample."""
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


# ─────────────────────────────────────────
# Feature Extraction (20 features)
# ─────────────────────────────────────────
def extract_features(G: nx.Graph):
    """
    Returns F (n x 20) and nodes_list.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    print("    degree ...")
    degree_dict = dict(G.degree())

    print("    clustering ...")
    clust = nx.clustering(G)

    print("    avg_neighbor_degree ...")
    avg_nd = nx.average_neighbor_degree(G)

    print("    core_number ...")
    core = nx.core_number(G)
    max_core = max(core.values()) if core else 1

    print("    closeness ...")
    closeness = nx.closeness_centrality(G)

    print("    pagerank ...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=300, tol=1e-4)

    print("    triangles ...")
    try:
        triangles = nx.triangles(G)
    except Exception:
        triangles = {v: 0 for v in nodes}

    print("    square_clustering ...")
    sq_clust = nx.square_clustering(G)

    print("    harmonic_centrality ...")
    harmonic = nx.harmonic_centrality(G)

    print("    load_centrality (BC proxy) ...")
    try:
        load = nx.load_centrality(G)
    except Exception:
        load = {v: 0.0 for v in nodes}

    print("    degree_centrality ...")
    deg_cen = nx.degree_centrality(G)

    print("    eccentricity estimate (BFS double-sweep) ...")
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

    print("    building feature matrix ...")
    F = np.zeros((n, N_FEATURES), dtype=np.float64)
    for v in nodes:
        i = node_idx[v]
        deg = float(degree_dict[v])
        c   = clust.get(v, 0.0)
        k   = float(core.get(v, 0))

        nbs = list(G.neighbors(v))
        nbr_core_avg = float(np.mean([core.get(u, 0) for u in nbs])) if nbs else 0.0
        nbr_pr_avg   = float(np.mean([pagerank.get(u, 0.0) for u in nbs])) if nbs else 0.0

        F[i, 0]  = deg
        F[i, 1]  = np.log1p(deg)
        F[i, 2]  = c
        F[i, 3]  = avg_nd.get(v, 0.0)
        F[i, 4]  = k
        F[i, 5]  = closeness.get(v, 0.0)
        F[i, 6]  = pagerank.get(v, 0.0)
        F[i, 7]  = ecc_est[v] / diam
        F[i, 8]  = float(triangles.get(v, 0))
        F[i, 9]  = sq_clust.get(v, 0.0)
        F[i, 10] = harmonic.get(v, 0.0)
        F[i, 11] = load.get(v, 0.0)
        F[i, 12] = deg ** 2
        F[i, 13] = c * deg
        F[i, 14] = k / max_core
        F[i, 15] = nbr_core_avg
        F[i, 16] = nbr_pr_avg
        F[i, 17] = deg_cen.get(v, 0.0)
        F[i, 18] = np.log1p(float(triangles.get(v, 0)))
        F[i, 19] = k * deg

    return F, nodes


# ─────────────────────────────────────────
# Pairwise Dataset Builder
# ─────────────────────────────────────────
def build_pairwise_dataset(F, bc, n_pairs, rng):
    """
    X[i] = F[a] - F[b]  (feature difference)
    y[i] = 1 if bc[a] > bc[b]
    """
    n = len(bc)
    a = rng.randint(0, n, size=n_pairs)
    b = rng.randint(0, n, size=n_pairs)
    same = a == b
    b[same] = (b[same] + 1) % n
    X = F[a] - F[b]
    y = (bc[a] > bc[b]).astype(int)
    return X, y, a, b


# ─────────────────────────────────────────
# EDDBM Baseline
# ─────────────────────────────────────────
def _brandes_dependency(G, source, target):
    dist  = {n: -1  for n in G.nodes()}
    sigma = {n: 0.0 for n in G.nodes()}
    pred  = {n: []  for n in G.nodes()}
    delta = {n: 0.0 for n in G.nodes()}
    stack = []
    dist[source]  = 0
    sigma[source] = 1.0
    q = deque([source])
    while q:
        v = q.popleft()
        stack.append(v)
        for w in G.neighbors(v):
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                q.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)
    while stack:
        w = stack.pop()
        for p in pred[w]:
            delta[p] += (sigma[p] / max(sigma[w], 1e-10)) * (1.0 + delta[w])
    return delta[target]


def _eddbm_prob(G, v, avg_deg):
    lam = max(avg_deg, 1.0)
    dists = nx.single_source_shortest_path_length(G, v)
    P = {}
    total = 0.0
    for u, d in dists.items():
        if u == v:
            continue
        w = (lam ** (-d)) / max(1, G.degree(u))
        P[u] = w
        total += w
    if total > 0:
        for u in P:
            P[u] /= total
    return P


def eddbm_estimate(G, v, T, avg_deg, rng):
    P = _eddbm_prob(G, v, avg_deg)
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
            est += _brandes_dependency(G, s, v) / p_s
    return est / T


# ─────────────────────────────────────────
# Model Builder — Stacking Ensemble
# ─────────────────────────────────────────
def build_stacking_pipeline():
    """
    Stacking: GBM + RF + ExtraTrees as base; Logistic as meta.
    Wrapped in CalibratedClassifierCV for calibrated probabilities.
    """
    base_estimators = [
        ("gb", GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=3,
            n_jobs=-1, random_state=RANDOM_SEED
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=3,
            n_jobs=-1, random_state=RANDOM_SEED
        )),
    ]
    meta = LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_SEED)
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=5,
        passthrough=False,
        n_jobs=1,
    )
    # Calibrate for reliable probabilities
    calibrated = CalibratedClassifierCV(stacker, cv="prefit", method="isotonic")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("stack",  stacker),   # replaced with calibrated below after first fit
    ])
    return pipeline, stacker, calibrated


def build_simple_pipelines():
    """Individual model pipelines for comparison."""
    return {
        "GradBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=14, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            ))
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=200, max_depth=14, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            ))
        ]),
    }


# ─────────────────────────────────────────
# BC-Rank Error (Avg Error metric)
# ─────────────────────────────────────────
def compute_rank_error(bc_true, y_prob, a_idx, b_idx):
    """
    For each test pair (a, b):
      - Predicted rank diff: prob(a>b) - 0.5  (positive = model says a is higher)
      - True rank diff: rank[a] - rank[b]  (from Brandes BC)
    Returns average absolute rank distance error (normalised by n).
    """
    n = len(bc_true)
    ranks = np.argsort(np.argsort(bc_true))   # rank 0 = lowest BC
    true_rank_diff  = (ranks[a_idx] - ranks[b_idx]).astype(float)
    # Normalize to [-1, 1]
    true_rank_diff /= max(n - 1, 1)
    # Model "signed confidence": prob - 0.5 in [-0.5, 0.5]
    pred_signed = y_prob - 0.5
    mae = np.mean(np.abs(true_rank_diff - pred_signed))
    return float(mae)


# ─────────────────────────────────────────
# Per-dataset Main Pipeline
# ─────────────────────────────────────────
def run_dataset(ds_path, max_nodes, n_pairs, rng):
    ds_name = os.path.splitext(os.path.basename(ds_path))[0]
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  DATASET: {ds_name}")
    print(sep)

    # 1. Load graph
    t0 = time.time()
    G = load_graph(ds_path, max_nodes)
    n, m = G.number_of_nodes(), G.number_of_edges()
    avg_deg = 2 * m / n if n > 0 else 1.0
    print(f"  Graph: {n} nodes, {m} edges  [{time.time()-t0:.1f}s]")

    # 2. Exact BC (Brandes)
    print("  Computing exact betweenness centrality...")
    t0 = time.time()
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    nodes_list = list(G.nodes())
    bc = np.array([bc_dict[v] for v in nodes_list])
    print(f"  BC done [{time.time()-t0:.1f}s] | range [{bc.min():.3e}, {bc.max():.3e}]")

    # 3. Features
    print("  Extracting 20 structural features...")
    t0 = time.time()
    F, feat_nodes = extract_features(G)
    bc_aligned = np.array([bc_dict[v] for v in feat_nodes])
    print(f"  Features done [{time.time()-t0:.1f}s] | shape {F.shape}")

    # 4. Pairwise dataset
    print(f"  Building {n_pairs} pairwise samples...")
    X, y, a_idx, b_idx = build_pairwise_dataset(F, bc_aligned, n_pairs, rng)
    print(f"  Class balance: {y.mean()*100:.1f}% A>B")

    X_tr, X_te, y_tr, y_te, a_tr, a_te, b_tr, b_te = train_test_split(
        X, y, a_idx, b_idx,
        test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

    # 5. Train individual models
    print("\n  Training models...")
    models = build_simple_pipelines()
    model_results = {}

    for mname, pipe in models.items():
        print(f"    > {mname} ...", end=" ", flush=True)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        acc  = accuracy_score(y_te, y_pred)
        auc  = roc_auc_score(y_te, y_prob)
        brier = brier_score_loss(y_te, y_prob)
        rank_err = compute_rank_error(bc_aligned, y_prob, a_te, b_te)
        model_results[mname] = {
            "acc": acc, "auc": auc, "brier": brier,
            "rank_err": rank_err, "y_prob": y_prob, "y_pred": y_pred,
            "pipe": pipe, "train_time": elapsed
        }
        print(f"acc={acc*100:.2f}%  AUC={auc:.4f}  [{elapsed:.1f}s]")

    # 6. Stacking Ensemble with Calibration
    print("    > Stacking (GB+RF+ET -> LR) ...", end=" ", flush=True)
    t0 = time.time()

    # Scale once, then fit stacker
    scaler_stack = StandardScaler()
    X_tr_sc = scaler_stack.fit_transform(X_tr)
    X_te_sc  = scaler_stack.transform(X_te)

    stacker = StackingClassifier(
        estimators=[
            ("gb", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.06,
                subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED
            )),
            ("rf", RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            )),
        ],
        final_estimator=LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_SEED),
        cv=5, passthrough=False, n_jobs=1,
    )
    stacker.fit(X_tr_sc, y_tr)

    # Calibrate on a held-out split from train (20%)
    X_cal_tr, X_cal_val, y_cal_tr, y_cal_val = train_test_split(
        X_tr_sc, y_tr, test_size=0.2, random_state=RANDOM_SEED + 1, stratify=y_tr
    )
    stacker_sub = StackingClassifier(
        estimators=[
            ("gb", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.06,
                subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED
            )),
            ("rf", RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_SEED
            )),
        ],
        final_estimator=LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_SEED),
        cv=3, passthrough=False, n_jobs=1,
    )
    stacker_sub.fit(X_cal_tr, y_cal_tr)
    calibrated = CalibratedClassifierCV(stacker_sub, cv="prefit", method="isotonic")
    calibrated.fit(X_cal_val, y_cal_val)

    y_pred_st = stacker.predict(X_te_sc)
    y_prob_st = calibrated.predict_proba(X_te_sc)[:, 1]
    acc_st  = accuracy_score(y_te, y_pred_st)
    auc_st  = roc_auc_score(y_te, y_prob_st)
    brier_st = brier_score_loss(y_te, y_prob_st)
    rank_err_st = compute_rank_error(bc_aligned, y_prob_st, a_te, b_te)
    elapsed_st = time.time() - t0
    print(f"acc={acc_st*100:.2f}%  AUC={auc_st:.4f}  [{elapsed_st:.1f}s]")

    model_results["Stacking+Calib"] = {
        "acc": acc_st, "auc": auc_st, "brier": brier_st,
        "rank_err": rank_err_st, "y_prob": y_prob_st, "y_pred": y_pred_st,
        "pipe": (scaler_stack, stacker, calibrated), "train_time": elapsed_st
    }

    # 7. EDDBM Baseline
    eval_n = min(EDDBM_EVAL_PAIRS, len(a_te))
    print(f"\n  EDDBM baseline (T={EDDBM_T}, {eval_n} pairs)...", end=" ", flush=True)
    t0 = time.time()
    node_arr = np.array(feat_nodes)
    eddbm_correct = 0
    eddbm_probs = []
    for k in range(eval_n):
        na = node_arr[a_te[k]]; nb = node_arr[b_te[k]]
        est_a = eddbm_estimate(G, na, EDDBM_T, avg_deg, rng)
        est_b = eddbm_estimate(G, nb, EDDBM_T, avg_deg, rng)
        pred = 1 if est_a > est_b else 0
        true = 1 if bc_dict[na] > bc_dict[nb] else 0
        if pred == true:
            eddbm_correct += 1
        # Simple soft-prob for EDDBM: clamp ratio
        tot = abs(est_a) + abs(est_b) + 1e-12
        eddbm_probs.append(est_a / tot if tot > 0 else 0.5)
    eddbm_acc = eddbm_correct / eval_n if eval_n > 0 else 0.5
    print(f"acc={eddbm_acc*100:.2f}%  [{time.time()-t0:.1f}s]")

    # Rank error for EDDBM on eval_n pairs
    eddbm_prob_arr = np.array(eddbm_probs)
    eddbm_rank_err = compute_rank_error(bc_aligned, eddbm_prob_arr, a_te[:eval_n], b_te[:eval_n])

    # Random baseline
    rand_labels = rng.randint(0, 2, size=len(y_te))
    random_acc = accuracy_score(y_te, rand_labels)
    random_rank_err = compute_rank_error(bc_aligned, rng.rand(len(y_te)), a_te, b_te)
    print(f"  Random baseline: {random_acc*100:.2f}%")

    # Best ML model by accuracy
    best_ml = max(model_results, key=lambda k: model_results[k]["acc"])
    best_ml_acc = model_results[best_ml]["acc"]
    best_ml_rank_err = model_results[best_ml]["rank_err"]

    print(f"\n  Best ML: {best_ml}  acc={best_ml_acc*100:.2f}%  rank_err={best_ml_rank_err:.4f}")
    print(f"  EDDBM acc={eddbm_acc*100:.2f}%  rank_err={eddbm_rank_err:.4f}")

    # Save feature importance for best model that supports it
    _save_feature_importance(model_results, ds_name)

    return {
        "dataset":    ds_name,
        "nodes":      n,
        "edges":      m,
        "model_results": model_results,
        "best_ml":    best_ml,
        "best_ml_acc": best_ml_acc,
        "best_ml_rank_err": best_ml_rank_err,
        "eddbm_acc":  eddbm_acc,
        "eddbm_rank_err": eddbm_rank_err,
        "random_acc": random_acc,
        "random_rank_err": random_rank_err,
        "best_pipe":  model_results[best_ml]["pipe"],
    }


# ─────────────────────────────────────────
# Per-dataset Feature Importance Plot
# ─────────────────────────────────────────
def _save_feature_importance(model_results, ds_name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for mname in ["GradBoost", "RandomForest", "ExtraTrees"]:
        if mname not in model_results:
            continue
        pipe = model_results[mname]["pipe"]
        clf  = pipe.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            continue
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1]
        colors = plt.cm.plasma(np.linspace(0.15, 0.9, N_FEATURES))

        fig, ax = plt.subplots(figsize=(13, 5))
        bars = ax.bar(range(N_FEATURES), importances[idx],
                      color=[colors[j] for j in range(N_FEATURES)],
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(N_FEATURES))
        ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=42, ha="right", fontsize=8)
        for bar, val in zip(bars, importances[idx]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_ylabel("Gini Importance", fontsize=10)
        ax.set_title(f"Feature Importances ({mname}) — {ds_name}",
                     fontsize=11, fontweight="bold")
        ax.set_facecolor("#f7f7f7")
        fig.patch.set_facecolor("#f7f7f7")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"{ds_name}_{mname}_feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  => Saved: {path}")
        break  # Only for the first available model


# ─────────────────────────────────────────
# SUMMARY PLOT 1 — Efficiency (Accuracy per Dataset)
# ─────────────────────────────────────────
def plot_efficiency(all_results):
    """
    Grouped bar: for each dataset, show accuracy of each ML model + EDDBM + Random.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    datasets = [r["dataset"] for r in all_results]
    nd = len(datasets)
    if nd == 0:
        return

    # Collect model names across all datasets
    all_model_names = []
    for r in all_results:
        for mname in r["model_results"]:
            if mname not in all_model_names:
                all_model_names.append(mname)
    all_model_names = all_model_names + ["EDDBM", "Random"]

    palette = [
        "#2196F3", "#4CAF50", "#00BCD4", "#E91E63",
        "#FF9800", "#9C27B0", "#607D8B",
    ]
    n_groups = len(all_model_names)
    w = 0.8 / n_groups
    x = np.arange(nd)

    fig, ax = plt.subplots(figsize=(max(10, nd * 3.2), 6))
    for gi, mname in enumerate(all_model_names):
        accs = []
        for r in all_results:
            if mname == "EDDBM":
                accs.append(r["eddbm_acc"] * 100)
            elif mname == "Random":
                accs.append(r["random_acc"] * 100)
            else:
                accs.append(r["model_results"].get(mname, {}).get("acc", 0.0) * 100)
        color = palette[gi % len(palette)]
        alpha = 0.65 if mname == "Random" else 0.88
        bars = ax.bar(x + gi * w - w * (n_groups - 1) / 2, accs,
                      w * 0.92, label=mname, color=color, alpha=alpha,
                      edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, accs):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold", rotation=75)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=18, ha="right", fontsize=10)
    ax.set_ylim(40, 115)
    ax.set_ylabel("Pairwise Ordering Accuracy (%)", fontsize=12)
    ax.set_title(
        "Model Efficiency: Pairwise BC Ordering Accuracy Across All Datasets",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.set_facecolor("#f0f2f5")
    fig.patch.set_facecolor("#f0f2f5")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "efficiency_all_datasets.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\n  => Efficiency plot saved: {path}")


# ─────────────────────────────────────────
# SUMMARY PLOT 2 — Avg Error per Dataset
# ─────────────────────────────────────────
def plot_avg_error(all_results):
    """
    Line + scatter plot: mean absolute rank error per dataset for ML, EDDBM, Random.
    Lower is better.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    datasets = [r["dataset"] for r in all_results]
    nd = len(datasets)
    if nd == 0:
        return

    all_model_names = []
    for r in all_results:
        for mname in r["model_results"]:
            if mname not in all_model_names:
                all_model_names.append(mname)

    palette = ["#2196F3", "#4CAF50", "#00BCD4", "#E91E63",
               "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(max(9, nd * 2.5), 6))
    x = np.arange(nd)

    for gi, mname in enumerate(all_model_names):
        errs = [r["model_results"].get(mname, {}).get("rank_err", np.nan)
                for r in all_results]
        color = palette[gi % len(palette)]
        ax.plot(x, errs, "o-", color=color, label=mname, linewidth=2.2,
                markersize=8, markeredgecolor="white", markeredgewidth=0.8)
        for xi, (v, ds) in enumerate(zip(errs, datasets)):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7.5, color=color)

    # EDDBM
    eddbm_errs = [r["eddbm_rank_err"] for r in all_results]
    ax.plot(x, eddbm_errs, "s--", color="#FF9800", label="EDDBM",
            linewidth=2, markersize=8, markeredgecolor="white")
    for xi, v in enumerate(eddbm_errs):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7.5, color="#FF9800")

    # Random
    rand_errs = [r["random_rank_err"] for r in all_results]
    ax.plot(x, rand_errs, "^:", color="#9C27B0", label="Random",
            linewidth=1.5, markersize=7, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Mean Absolute Rank Error (normalised)", fontsize=12)
    ax.set_title(
        "Average BC-Rank Error Across All Datasets\n(Lower = Better)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.set_facecolor("#f0f2f5")
    fig.patch.set_facecolor("#f0f2f5")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "avg_error_all_datasets.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  => Avg error plot saved: {path}")


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Improved ML model for pairwise BC node comparison"
    )
    parser.add_argument("--dataset",   type=str, default=None)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--n-pairs",   type=int, default=DEFAULT_N_PAIRS)
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [p for p in CANDIDATE_DATASETS if os.path.exists(p)]

    if not datasets:
        print("\nERROR: No datasets found. Tried:")
        for p in CANDIDATE_DATASETS:
            print(f"  {p}")
        print("Run with: python ml_model_improved.py --dataset PATH")
        sys.exit(1)

    print(f"\nFound {len(datasets)} dataset(s): {[os.path.basename(d) for d in datasets]}")
    print(f"Config: max_nodes={args.max_nodes}, n_pairs={args.n_pairs}, EDDBM_T={EDDBM_T}\n")

    all_results = []
    best_pipe = None
    best_acc  = -1.0

    for ds_path in datasets:
        result = run_dataset(ds_path, args.max_nodes, args.n_pairs, rng)
        all_results.append(result)
        if result["best_ml_acc"] > best_acc:
            best_acc  = result["best_ml_acc"]
            best_pipe = result["best_pipe"]

    # ── Save best model ──────────────────
    with open(MODEL_SAVE_PATH, "wb") as fh:
        pickle.dump(best_pipe, fh)
    print(f"\n[OK] Best model saved -> {os.path.abspath(MODEL_SAVE_PATH)}")

    # ── Summary table ────────────────────
    print("\n" + "=" * 75)
    print("  FINAL SUMMARY")
    print("=" * 75)
    hdr = f"  {'Dataset':<22} {'N':>5} {'M':>7}  {'BestML%':>8} {'EDDBM%':>8} {'Rand%':>7} {'RankErr':>9}"
    print(hdr)
    print("  " + "-" * 73)
    for r in all_results:
        print(f"  {r['dataset']:<22} {r['nodes']:>5} {r['edges']:>7}  "
              f"{r['best_ml_acc']*100:>7.2f}% {r['eddbm_acc']*100:>7.2f}% "
              f"{r['random_acc']*100:>6.2f}% {r['best_ml_rank_err']:>9.4f}")

    # ── Combined cross-dataset plots ─────
    print("\nGenerating cross-dataset summary plots...")
    plot_efficiency(all_results)
    plot_avg_error(all_results)

    print(f"\n[OK] All plots in: {os.path.abspath(PLOTS_DIR)}")
    print("[OK] Done!\n")


if __name__ == "__main__":
    main()
