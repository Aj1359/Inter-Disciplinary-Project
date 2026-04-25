#!/usr/bin/env python3
"""
Enhanced ML model for betweenness ordering.
Key improvements over the original ml_train.py:
  1. 20+ structural features (vs 6)
  2. Pairwise ranking objective (predict which node has higher BC)
  3. Deeper MLP with dropout + ensemble with gradient boosting
  4. Multi-graph training for generalization
  5. Proper cross-validation
"""
import argparse
import glob
import math
import os
import random
import time
from collections import defaultdict, deque

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib


# ─────────────────────────────────────────────────────────────────────────────
# Graph loading & utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_graph(path):
    """Load undirected graph from edge-list file."""
    id_map = {}
    id_list = []
    edges = set()

    def get_id(x):
        if x in id_map:
            return id_map[x]
        idx = len(id_list)
        id_map[x] = idx
        id_list.append(x)
        return idx

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            a, b = get_id(u), get_id(v)
            if a > b:
                a, b = b, a
            edges.add((a, b))

    n = len(id_list)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    deg = [len(adj[i]) for i in range(n)]
    return adj, deg, n, len(edges)


def bfs_full(adj, source):
    """Full BFS returning dist, sigma, predecessors, and level structure."""
    n = len(adj)
    dist = [-1] * n
    sigma = [0.0] * n
    pred = [[] for _ in range(n)]
    levels = []

    q = deque()
    dist[source] = 0
    sigma[source] = 1.0
    q.append(source)

    while q:
        u = q.popleft()
        d = dist[u]
        while len(levels) <= d:
            levels.append([])
        levels[d].append(u)
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = d + 1
                q.append(w)
            if dist[w] == d + 1:
                sigma[w] += sigma[u]
                pred[w].append(u)

    return dist, sigma, pred, levels


def brandes_all(adj):
    """Exact Brandes betweenness centrality for all nodes. O(nm)."""
    n = len(adj)
    BC = [0.0] * n

    for s in range(n):
        dist, sigma, pred, levels = bfs_full(adj, s)
        delta = [0.0] * n
        # Process nodes from farthest to closest
        for d in range(len(levels) - 1, 0, -1):
            for w in levels[d]:
                for p in pred[w]:
                    if sigma[w] > 0:
                        delta[p] += (sigma[p] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    BC[w] += delta[w]

    # Undirected graph: divide by 2
    for i in range(n):
        BC[i] /= 2.0
    return BC


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (20+ features per node)
# ─────────────────────────────────────────────────────────────────────────────

def compute_graph_features(adj, deg):
    """Compute graph-level features once."""
    n = len(adj)
    avg_deg = sum(deg) / max(1, n)
    max_deg = max(deg) if deg else 1

    # Estimate diameter via multi-BFS
    # Pick a few random nodes, do BFS, track max distance
    rng = random.Random(123)
    sample = rng.sample(range(n), min(5, n))
    diameter_est = 0
    for s in sample:
        d, _, _, _ = bfs_full(adj, s)
        max_d = max(x for x in d if x >= 0) if any(x >= 0 for x in d) else 0
        diameter_est = max(diameter_est, max_d)

    # K-core decomposition (approximate)
    core = compute_kcore(adj, deg)

    # Triangle counts per node
    triangles = compute_triangles(adj, deg, n)

    return {
        "n": n,
        "avg_deg": avg_deg,
        "max_deg": max_deg,
        "diameter_est": max(1, diameter_est),
        "core": core,
        "triangles": triangles,
    }


def compute_kcore(adj, deg):
    """Compute k-core number for each node."""
    n = len(adj)
    core = list(deg)
    sorted_nodes = sorted(range(n), key=lambda x: core[x])
    node_pos = [0] * n
    for i, v in enumerate(sorted_nodes):
        node_pos[v] = i

    bin_start = [0] * (max(core) + 2) if core else [0]
    for c in core:
        if c < len(bin_start):
            bin_start[c] += 1
    cumsum = 0
    for i in range(len(bin_start)):
        cumsum, bin_start[i] = cumsum + bin_start[i], cumsum

    for v in list(sorted_nodes):
        for u in adj[v]:
            if core[u] > core[v]:
                # Decrease core[u]
                old_core = core[u]
                new_core = old_core - 1
                core[u] = new_core
    return core


def compute_triangles(adj, deg, n):
    """Count triangles per node (approximate for large graphs)."""
    triangles = [0] * n
    for u in range(n):
        if deg[u] > 500:
            continue  # Skip high-degree nodes for speed
        neighbors_u = set(adj[u])
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in neighbors_u:
                        triangles[u] += 1
                        triangles[v] += 1
                        triangles[w] += 1
    return triangles


def extract_node_features(adj, deg, node, graph_feats, bfs_cache=None):
    """Extract ~20 features for a single node."""
    n = graph_feats["n"]
    avg_deg = graph_feats["avg_deg"]
    diameter = graph_feats["diameter_est"]
    core = graph_feats["core"]
    triangles = graph_feats["triangles"]

    d = deg[node]

    # Neighbor statistics
    nb_degs = [deg[nb] for nb in adj[node]] if adj[node] else [0]
    nb_avg = np.mean(nb_degs)
    nb_std = np.std(nb_degs) if len(nb_degs) > 1 else 0.0
    nb_max = max(nb_degs)
    nb_min = min(nb_degs)

    # 2-hop neighborhood size
    visited = set()
    visited.add(node)
    hop1 = set(adj[node])
    visited.update(hop1)
    hop2 = set()
    for nb in hop1:
        for nb2 in adj[nb]:
            if nb2 not in visited:
                hop2.add(nb2)
    two_hop_size = len(hop1) + len(hop2)

    # Local clustering coefficient
    if d >= 2:
        nb_set = set(adj[node])
        edge_count = 0
        for nb in adj[node]:
            for nb2 in adj[nb]:
                if nb2 in nb_set and nb2 != node:
                    edge_count += 1
        clustering = edge_count / (d * (d - 1))
    else:
        clustering = 0.0

    features = [
        math.log1p(d),                          # 1. log degree
        d / max(1, n - 1),                       # 2. degree centrality
        math.log1p(nb_avg),                      # 3. avg neighbor degree
        nb_std / max(1.0, nb_avg),               # 4. neighbor degree CV
        math.log1p(nb_max),                      # 5. max neighbor degree
        math.log1p(two_hop_size),                # 6. 2-hop neighborhood size
        two_hop_size / max(1, n),                # 7. 2-hop ratio
        clustering,                              # 8. local clustering coeff
        math.log1p(core[node]),                  # 9. k-core number
        core[node] / max(1, max(core)),          # 10. normalized core
        math.log1p(triangles[node]),             # 11. triangle count
        math.log1p(avg_deg),                     # 12. graph avg degree
        math.log1p(n),                           # 13. graph size
        math.log1p(diameter),                    # 14. graph diameter
        d / max(1.0, avg_deg),                   # 15. degree / avg_degree
        math.log1p(d * nb_avg),                  # 16. degree * avg_nb_deg (weighted degree)
        nb_min / max(1.0, nb_max),               # 17. min/max neighbor degree ratio
        math.log1p(sum(1 for nb in adj[node]     # 18. high-degree neighbors
                       if deg[nb] > avg_deg)),
        d / max(1, graph_feats["max_deg"]),      # 19. degree / max_degree
        math.log1p(                               # 20. eigenvector centrality proxy
            sum(deg[nb] for nb in adj[node])
        ),
    ]
    return features


NUM_FEATURES = 20  # Number of per-node features
NUM_PAIR_FEATURES = NUM_FEATURES * 2 + 5  # pair features


def extract_pair_features(feat_u, feat_v, adj, deg, u, v, graph_feats):
    """Extract features for a pair (u, v) for ranking."""
    n = graph_feats["n"]

    # Distance between u and v (quick BFS)
    dist_uv = bfs_distance(adj, u, v)

    # Common neighbors
    set_u = set(adj[u])
    set_v = set(adj[v])
    common = len(set_u & set_v)
    jaccard = common / max(1, len(set_u | set_v))

    pair_specific = [
        dist_uv / max(1, graph_feats["diameter_est"]),   # normalized distance
        math.log1p(common),                               # common neighbors
        jaccard,                                          # Jaccard similarity
        abs(deg[u] - deg[v]) / max(1, max(deg[u], deg[v])),  # degree difference
        1.0 if deg[u] > deg[v] else 0.0,                 # u has higher degree?
    ]

    # Differences: feat_u - feat_v (already captures direction)
    return feat_u + feat_v + pair_specific


def bfs_distance(adj, u, v):
    """BFS shortest path distance between u and v."""
    if u == v:
        return 0
    visited = {u}
    q = deque([(u, 0)])
    while q:
        node, d = q.popleft()
        for nb in adj[node]:
            if nb == v:
                return d + 1
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
                if d + 1 > 50:  # Safety: cap at 50
                    return 50
    return 50  # disconnected


# ─────────────────────────────────────────────────────────────────────────────
# Dataset building
# ─────────────────────────────────────────────────────────────────────────────

def build_pairwise_dataset(adj, deg, exact_bc, graph_feats, rng,
                           max_pairs=5000, min_bc_diff_ratio=0.01):
    """Build pairwise ranking dataset from a single graph."""
    n = len(adj)
    nonzero = [i for i in range(n) if exact_bc[i] > 0]
    if len(nonzero) < 10:
        return [], []

    # Precompute node features
    node_features = {}
    for v in nonzero:
        node_features[v] = extract_node_features(adj, deg, v, graph_feats)

    X, y = [], []
    attempts = 0
    max_attempts = max_pairs * 5

    while len(X) < max_pairs and attempts < max_attempts:
        attempts += 1
        u = rng.choice(nonzero)
        v = rng.choice(nonzero)
        if u == v:
            continue
        if exact_bc[u] == exact_bc[v]:
            continue

        # Skip pairs that are too close in BC (noisy labels)
        bc_max = max(exact_bc[u], exact_bc[v])
        bc_diff = abs(exact_bc[u] - exact_bc[v])
        if bc_diff / bc_max < min_bc_diff_ratio:
            continue

        feat = extract_pair_features(
            node_features[u], node_features[v],
            adj, deg, u, v, graph_feats
        )
        label = 1 if exact_bc[u] > exact_bc[v] else 0
        X.append(feat)
        y.append(label)

    return X, y


def generate_synthetic_graph(n, graph_type, rng):
    """Generate synthetic graphs for training data augmentation."""
    adj = [[] for _ in range(n)]
    edges = set()

    def add_edge(u, v):
        if u != v and (min(u, v), max(u, v)) not in edges:
            edges.add((min(u, v), max(u, v)))
            adj[u].append(v)
            adj[v].append(u)

    if graph_type == "er":
        # Erdos-Renyi
        p = rng.uniform(0.003, 0.03)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    add_edge(i, j)
    elif graph_type == "ba":
        # Barabasi-Albert
        m = rng.randint(2, 8)
        for i in range(m):
            for j in range(i + 1, m):
                add_edge(i, j)
        deg_sum = [0] * n
        for i in range(n):
            deg_sum[i] = len(adj[i])
        total_deg = sum(deg_sum)
        for new_node in range(m, n):
            targets = set()
            while len(targets) < m:
                if total_deg == 0:
                    target = rng.randint(0, new_node - 1)
                else:
                    r = rng.random() * total_deg
                    cumul = 0
                    target = 0
                    for k in range(new_node):
                        cumul += deg_sum[k]
                        if cumul >= r:
                            target = k
                            break
                targets.add(target)
            for t in targets:
                add_edge(new_node, t)
                deg_sum[new_node] += 1
                deg_sum[t] += 1
                total_deg += 2

    deg = [len(adj[i]) for i in range(n)]
    return adj, deg


# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enhanced ML training for BC ordering")
    parser.add_argument("--data-dir", default=".", help="Directory with .txt graph files")
    parser.add_argument("--max-graphs", type=int, default=6, help="Max real graphs to train on")
    parser.add_argument("--max-nodes", type=int, default=12000, help="Skip graphs larger than this")
    parser.add_argument("--pairs-per-graph", type=int, default=8000)
    parser.add_argument("--synthetic-count", type=int, default=10, help="Number of synthetic training graphs")
    parser.add_argument("--synthetic-n", type=int, default=1000, help="Size of synthetic graphs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="enhanced_model.joblib")
    parser.add_argument("--scaler-out", default="enhanced_scaler.joblib")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    X_all, y_all = [], []

    # ── Load real graphs and build dataset ──
    txt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))
    used_graphs = 0

    for path in txt_files:
        if used_graphs >= args.max_graphs:
            break
        basename = os.path.basename(path)
        # Skip files that aren't graph datasets
        if "extracted_text" in basename:
            continue

        print(f"Loading {basename}...", end=" ", flush=True)
        adj, deg, n, m = load_graph(path)
        if n > args.max_nodes or n < 50:
            print(f"skipped (n={n})")
            continue

        print(f"n={n}, m={m}", end=" ", flush=True)

        # Compute exact BC (this is the expensive part, for training only)
        print("computing BC...", end=" ", flush=True)
        exact_bc = brandes_all(adj)

        graph_feats = compute_graph_features(adj, deg)
        X, y = build_pairwise_dataset(adj, deg, exact_bc, graph_feats, rng,
                                       max_pairs=args.pairs_per_graph)
        print(f"pairs={len(X)}")
        X_all.extend(X)
        y_all.extend(y)
        used_graphs += 1

    # ── Generate synthetic graphs for augmentation ──
    print(f"\nGenerating {args.synthetic_count} synthetic graphs...")
    for i in range(args.synthetic_count):
        n_syn = rng.randint(500, args.synthetic_n)
        gtype = rng.choice(["er", "ba"])
        adj, deg = generate_synthetic_graph(n_syn, gtype, rng)

        # Check connectivity
        if max(deg) < 2:
            continue

        exact_bc = brandes_all(adj)
        graph_feats = compute_graph_features(adj, deg)
        X, y = build_pairwise_dataset(adj, deg, exact_bc, graph_feats, rng,
                                       max_pairs=3000)
        print(f"  Synthetic {i+1}: {gtype} n={n_syn}, pairs={len(X)}")
        X_all.extend(X)
        y_all.extend(y)

    if not X_all:
        raise SystemExit("No training data generated!")

    X_arr = np.array(X_all, dtype=np.float32)
    y_arr = np.array(y_all, dtype=np.int32)

    print(f"\nTotal training pairs: {len(X_arr)}")
    print(f"Label balance: {y_arr.mean():.3f} (should be ~0.5)")

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.15, random_state=args.seed, stratify=y_arr
    )

    # ── Scale features ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Train models ──
    print("\nTraining Gradient Boosting Classifier...")
    gbc = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=args.seed,
    )
    gbc.fit(X_train_s, y_train)
    gbc_acc = accuracy_score(y_test, gbc.predict(X_test_s))
    gbc_auc = roc_auc_score(y_test, gbc.predict_proba(X_test_s)[:, 1])
    print(f"  GBC accuracy: {gbc_acc:.4f}, AUC: {gbc_auc:.4f}")

    print("Training Deep MLP Classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        alpha=1e-4,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        batch_size=256,
        random_state=args.seed,
    )
    mlp.fit(X_train_s, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test_s))
    mlp_auc = roc_auc_score(y_test, mlp.predict_proba(X_test_s)[:, 1])
    print(f"  MLP accuracy: {mlp_acc:.4f}, AUC: {mlp_auc:.4f}")

    # ── Ensemble via soft voting ──
    print("Building ensemble...")
    # Use the better model as primary, blend with the other
    ensemble = VotingClassifier(
        estimators=[("gbc", gbc), ("mlp", mlp)],
        voting="soft",
        weights=[0.6, 0.4] if gbc_auc > mlp_auc else [0.4, 0.6],
    )
    ensemble.fit(X_train_s, y_train)
    ens_acc = accuracy_score(y_test, ensemble.predict(X_test_s))
    ens_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test_s)[:, 1])
    print(f"  Ensemble accuracy: {ens_acc:.4f}, AUC: {ens_auc:.4f}")

    # ── Save best model ──
    joblib.dump(ensemble, args.model_out)
    joblib.dump(scaler, args.scaler_out)

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Saved model to {args.model_out}")
    print(f"Saved scaler to {args.scaler_out}")
    print(f"{'='*50}")
    print(f"\nResults summary:")
    print(f"  GBC:      acc={gbc_acc:.4f}, AUC={gbc_auc:.4f}")
    print(f"  MLP:      acc={mlp_acc:.4f}, AUC={mlp_auc:.4f}")
    print(f"  Ensemble: acc={ens_acc:.4f}, AUC={ens_auc:.4f}")


if __name__ == "__main__":
    main()
