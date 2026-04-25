#!/usr/bin/env python3
import argparse
import os
import random
import math
from collections import defaultdict

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def load_graph(path):
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
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            a = get_id(u)
            b = get_id(v)
            if a > b:
                a, b = b, a
            edges.add((a, b))

    n = len(id_list)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    deg = [len(adj[i]) for i in range(n)]
    return adj, deg


def downsample_graph(adj, deg, max_nodes, rng):
    n = len(adj)
    if n <= max_nodes:
        return adj, deg

    nodes = rng.sample(range(n), max_nodes)
    node_set = set(nodes)
    idx_map = {old: new for new, old in enumerate(nodes)}

    new_adj = [[] for _ in range(max_nodes)]
    for old_u in nodes:
        u = idx_map[old_u]
        for old_v in adj[old_u]:
            if old_v in node_set:
                v = idx_map[old_v]
                new_adj[u].append(v)

    new_deg = [len(new_adj[i]) for i in range(max_nodes)]
    return new_adj, new_deg


def bfs_features(adj, deg, target):
    n = len(adj)
    dist = [-1] * n
    parent = [-1] * n
    sigma = [0.0] * n
    levels = []

    from collections import deque
    q = deque()
    dist[target] = 0
    sigma[target] = 1.0
    q.append(target)

    while q:
        u = q.popleft()
        d = dist[u]
        while len(levels) <= d:
            levels.append([])
        levels[d].append(u)
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = d + 1
                parent[w] = u
                q.append(w)
            if dist[w] == d + 1:
                sigma[w] += sigma[u]

    level_avg_deg = [1.01] * len(levels)
    for d, nodes in enumerate(levels):
        if not nodes:
            continue
        level_avg_deg[d] = max(1.01, float(sum(deg[u] for u in nodes)) / len(nodes))

    return dist, parent, sigma, level_avg_deg


def featurize(d, deg_i, sigma_i, c_hat, lvl_avg, deg_parent):
    return [
        float(d),
        math.log1p(float(deg_i)),
        math.log1p(float(sigma_i)),
        float(c_hat),
        math.log1p(float(lvl_avg)),
        math.log1p(float(deg_parent)),
    ]


def brandes_dependency_to_target(adj, source, target):
    n = len(adj)
    pred = [[] for _ in range(n)]
    dist = [-1] * n
    sigma = [0.0] * n

    from collections import deque
    q = deque()
    stack = []

    dist[source] = 0
    sigma[source] = 1.0
    q.append(source)

    while q:
        v = q.popleft()
        stack.append(v)
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                q.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)

    delta = [0.0] * n
    while stack:
        w = stack.pop()
        for p in pred[w]:
            if sigma[w] > 0:
                delta[p] += (sigma[p] / sigma[w]) * (1.0 + delta[w])

    return delta[target]


def build_dataset(adj, deg, rng, max_targets, max_sources):
    n = len(adj)
    targets = list(range(n))
    rng.shuffle(targets)
    targets = targets[: min(max_targets, n)]

    X = []
    y = []
    target_ids = []

    mark = [0] * n
    stamp = 1

    for v in targets:
        dist, parent, sigma, level_avg_deg = bfs_features(adj, deg, v)
        sources = [i for i in range(n) if i != v and dist[i] >= 0]
        rng.shuffle(sources)
        sources = sources[: min(max_sources, len(sources))]
        if not sources:
            continue

        labels = []
        feats = []

        for s in sources:
            p = parent[s]
            c_hat = 0.0
            if p >= 0 and deg[s] > 0:
                stamp += 1
                for nb in adj[p]:
                    mark[nb] = stamp
                common = sum(1 for nb in adj[s] if mark[nb] == stamp)
                c_hat = float(common) / max(1, deg[s])

            d = dist[s]
            lvl_avg = level_avg_deg[d] if d < len(level_avg_deg) else 1.01
            deg_parent = deg[p] if p >= 0 else 0

            feats.append(featurize(d, deg[s], sigma[s], c_hat, lvl_avg, deg_parent))
            labels.append(brandes_dependency_to_target(adj, s, v))

        label_sum = sum(labels)
        if label_sum <= 0.0:
            continue

        for feat, lab in zip(feats, labels):
            X.append(feat)
            y.append(lab / label_sum)
            target_ids.append(v)

    return X, y, target_ids


def build_features_for_target(adj, deg, target):
    n = len(adj)
    dist, parent, sigma, level_avg_deg = bfs_features(adj, deg, target)

    mark = [0] * n
    stamp = 1

    X = []
    nodes = []
    for s in range(n):
        if s == target or dist[s] < 0:
            continue
        p = parent[s]
        c_hat = 0.0
        if p >= 0 and deg[s] > 0:
            stamp += 1
            for nb in adj[p]:
                mark[nb] = stamp
            common = sum(1 for nb in adj[s] if mark[nb] == stamp)
            c_hat = float(common) / max(1, deg[s])

        d = dist[s]
        lvl_avg = level_avg_deg[d] if d < len(level_avg_deg) else 1.01
        deg_parent = deg[p] if p >= 0 else 0

        X.append(featurize(d, deg[s], sigma[s], c_hat, lvl_avg, deg_parent))
        nodes.append(s)

    return np.array(X, dtype=np.float32), nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset file name (e.g. Wiki-Vote.txt)")
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--max-targets", type=int, default=25)
    parser.add_argument("--max-sources", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="ml_model.joblib")
    parser.add_argument("--scaler-out", default="ml_scaler.joblib")
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--infer-dataset", default="", help="Optional dataset for inference")
    parser.add_argument("--infer-out", default="", help="Output CSV for inference")
    parser.add_argument("--infer-max-targets", type=int, default=20)
    parser.add_argument("--infer-targets", default="", help="Comma-separated target node IDs")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    rng = random.Random(args.seed)

    adj, deg = load_graph(args.dataset)
    adj, deg = downsample_graph(adj, deg, args.max_nodes, rng)
    if max(deg) < args.min_degree:
        raise SystemExit("Graph too sparse after downsampling")

    X, y, t_ids = build_dataset(adj, deg, rng, args.max_targets, args.max_sources)
    if not X:
        raise SystemExit("No training samples were generated")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X_arr, y_arr, t_ids, test_size=0.2, random_state=args.seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_log = np.log(y_train + 1e-9)
    y_test_log = np.log(y_test + 1e-9)

    model = MLPRegressor(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        alpha=1e-3,
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=args.seed,
    )
    model.fit(X_train_scaled, y_train_log)

    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.exp(y_pred_log)

    mse = float(np.mean((y_pred - y_test) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_test)))

    target_to_indices = defaultdict(list)
    for idx, t in enumerate(t_test):
        target_to_indices[t].append(idx)

    kls = []
    eps = 1e-12
    for t, idxs in target_to_indices.items():
        p = y_test[idxs]
        q = y_pred[idxs]
        p = p / (p.sum() + eps)
        q = q / (q.sum() + eps)
        kl = float(np.sum(p * np.log((p + eps) / (q + eps))))
        kls.append(kl)

    print(f"Samples: {len(X)}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    if kls:
        print(f"Mean KL: {np.mean(kls):.6f}")

    joblib.dump(model, args.model_out)
    joblib.dump(scaler, args.scaler_out)
    print(f"Saved model to {args.model_out}")
    print(f"Saved scaler to {args.scaler_out}")

    infer_path = args.infer_dataset.strip() if args.infer_dataset else ""
    if infer_path:
        if not os.path.exists(infer_path):
            raise SystemExit(f"Inference dataset not found: {infer_path}")

        out_csv = args.infer_out.strip()
        if not out_csv:
            base = os.path.splitext(os.path.basename(infer_path))[0]
            out_csv = f"{base}.csv"

        adj_i, deg_i = load_graph(infer_path)
        adj_i, deg_i = downsample_graph(adj_i, deg_i, args.max_nodes, rng)

        if args.infer_targets:
            targets = [int(t.strip()) for t in args.infer_targets.split(",") if t.strip()]
        else:
            targets = list(range(len(adj_i)))
            rng.shuffle(targets)
            targets = targets[: min(args.infer_max_targets, len(adj_i))]

        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("target,node,prob\n")
            for v in targets:
                X_inf, nodes = build_features_for_target(adj_i, deg_i, v)
                if X_inf.size == 0:
                    continue
                X_inf_scaled = scaler.transform(X_inf)
                pred_log = model.predict(X_inf_scaled)
                weights = np.exp(pred_log)
                total = float(weights.sum())
                if total <= 0.0:
                    continue
                probs = weights / total
                for node, prob in zip(nodes, probs):
                    f.write(f"{v},{node},{prob}\n")

        print(f"Saved inference CSV to {out_csv}")


if __name__ == "__main__":
    main()
