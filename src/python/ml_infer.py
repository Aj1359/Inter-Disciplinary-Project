#!/usr/bin/env python3
import argparse
import random
import math

import numpy as np
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
    parser.add_argument("graph_file")
    parser.add_argument("--model", default="ml_model.joblib")
    parser.add_argument("--scaler", default="ml_scaler.joblib")
    parser.add_argument("--targets", default="", help="Comma-separated target node IDs")
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="ml_probs.csv")
    args = parser.parse_args()

    adj, deg = load_graph(args.graph_file)
    n = len(adj)

    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    rng = random.Random(args.seed)

    if args.targets:
        targets = [int(t.strip()) for t in args.targets.split(",") if t.strip()]
    else:
        targets = list(range(n))
        rng.shuffle(targets)
        targets = targets[: min(args.max_targets, n)]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("target,node,prob\n")
        for v in targets:
            X, nodes = build_features_for_target(adj, deg, v)
            if X.size == 0:
                continue
            X_scaled = scaler.transform(X)
            pred_log = model.predict(X_scaled)
            weights = np.exp(pred_log)
            total = float(weights.sum())
            if total <= 0.0:
                continue
            probs = weights / total
            for node, prob in zip(nodes, probs):
                f.write(f"{v},{node},{prob}\n")

    print(f"Saved ML probabilities to {args.out}")


if __name__ == "__main__":
    main()
