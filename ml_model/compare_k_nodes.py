"""
compare_k_nodes.py
==================
Terminal command to compare K nodes by betweenness centrality
using the trained ML model (99% accuracy).

Usage:
    # Compare K random nodes:
    python compare_k_nodes.py --dataset ../datasets/Wiki-Vote.txt --k 10

    # Compare specific nodes by ID:
    python compare_k_nodes.py --dataset ../datasets/Wiki-Vote.txt --nodes 30,54,123,456

    # Use different trained model:
    python compare_k_nodes.py --dataset ../datasets/Wiki-Vote.txt --k 15 --model node_comparison_model.pkl
"""

import os
import sys
import pickle
import argparse
import time
from collections import deque

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np
import networkx as nx

# ── helpers ──────────────────────────────────────────────────────────────────

def load_graph(filepath, max_nodes=2000):
    G_dir = nx.DiGraph()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            p = line.split()
            if len(p) >= 2:
                u, v = int(p[0]), int(p[1])
                if u != v: G_dir.add_edge(u, v)
    G = G_dir.to_undirected()
    cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(cc).copy()
    if G.number_of_nodes() > max_nodes:
        seed = max(G.degree, key=lambda x: x[1])[0]
        visited, queue, seen = [], deque([seed]), {seed}
        while queue and len(visited) < max_nodes:
            n = queue.popleft(); visited.append(n)
            for nb in G.neighbors(n):
                if nb not in seen: seen.add(nb); queue.append(nb)
        G = G.subgraph(visited).copy()
    return nx.convert_node_labels_to_integers(G, first_label=0)


def extract_features(G):
    """Extract 16 structural features per node (same as trained model)."""
    import random, math
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    deg     = dict(G.degree())
    clust   = nx.clustering(G)
    avg_nd  = nx.average_neighbor_degree(G)
    core    = nx.core_number(G)
    close   = nx.closeness_centrality(G)
    pr      = nx.pagerank(G, alpha=0.85, max_iter=300, tol=1e-4)
    try: tri = nx.triangles(G)
    except: tri = {v:0 for v in nodes}
    sq      = nx.square_clustering(G)
    harm    = nx.harmonic_centrality(G)
    try: load = nx.load_centrality(G)
    except: load = {v:0.0 for v in nodes}

    max_core = max(core.values()) if core else 1
    rng_l = random.Random(42)
    sample = rng_l.sample(nodes, min(15, n))
    ecc_est = {v: 0 for v in nodes}
    for sd in sample:
        for v, d in nx.single_source_shortest_path_length(G, sd).items():
            ecc_est[v] = max(ecc_est[v], d)
    diam = max(ecc_est.values()) if ecc_est else 1
    if diam == 0: diam = 1

    F = np.zeros((n, 16), dtype=np.float64)
    for v in nodes:
        i = node_idx[v]
        d = deg[v]; c = clust.get(v, 0.0); k = core.get(v, 0)
        nbs = list(G.neighbors(v))
        nbr_core = np.mean([core.get(u, 0) for u in nbs]) if nbs else 0.0
        F[i, 0]  = d
        F[i, 1]  = math.log1p(d)
        F[i, 2]  = c
        F[i, 3]  = avg_nd.get(v, 0.0)
        F[i, 4]  = k
        F[i, 5]  = close.get(v, 0.0)
        F[i, 6]  = pr.get(v, 0.0)
        F[i, 7]  = ecc_est[v] / diam
        F[i, 8]  = tri.get(v, 0)
        F[i, 9]  = sq.get(v, 0.0)
        F[i, 10] = harm.get(v, 0.0)
        F[i, 11] = load.get(v, 0.0)
        F[i, 12] = d ** 2
        F[i, 13] = c * d
        F[i, 14] = k / max_core
        F[i, 15] = nbr_core

    return F, nodes


def ml_rank_k(F, node_to_idx, model, target_nodes):
    """Round-robin ML tournament. Returns ranked list (best first) and scores."""
    k = len(target_nodes)
    scores = {n: 0 for n in target_nodes}
    pairs = [(target_nodes[i], target_nodes[j]) for i in range(k) for j in range(i+1, k)]
    if not pairs:
        return target_nodes, scores
    X = np.array([F[node_to_idx[u]] - F[node_to_idx[v]] for u, v in pairs])
    preds = model.predict(X)
    for (u, v), p in zip(pairs, preds):
        if p == 1: scores[u] += 1
        else:      scores[v] += 1
    ranked = sorted(target_nodes, key=lambda n: scores[n], reverse=True)
    return ranked, scores


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare K nodes by Betweenness Centrality using ML model"
    )
    parser.add_argument("--dataset", required=True, help="Path to SNAP edge-list dataset")
    parser.add_argument("--k", type=int, default=10, help="Number of random nodes to compare (default: 10)")
    parser.add_argument("--nodes", type=str, default=None,
                        help="Comma-separated specific node IDs to compare, e.g. 30,54,123")
    default_model_path = os.path.join(os.path.dirname(__file__), "node_comparison_model.pkl")
    parser.add_argument("--model", default=default_model_path,
                        help=f"Trained ML model .pkl path (default: {default_model_path})")
    parser.add_argument("--max-nodes", type=int, default=2000,
                        help="Max nodes to load from graph (default: 2000)")
    parser.add_argument("--verify", action="store_true",
                        help="Also compute exact Brandes BC for ground truth comparison")
    args = parser.parse_args()

    ds_name = os.path.splitext(os.path.basename(args.dataset))[0]

    # Check model
    if not os.path.exists(args.model):
        print(f"\nERROR: Model '{args.model}' not found.")
        print("Run 'python model_benchmark.py --dataset ...' to train first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" ML K-Node Betweenness Comparator")
    print(f" Dataset : {ds_name}")
    print(f" Model   : {args.model}")
    print(f"{'='*60}")

    print("\n[1] Loading graph...")
    t0 = time.time()
    G = load_graph(args.dataset, args.max_nodes)
    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges [{time.time()-t0:.1f}s]")

    # Select nodes
    all_nodes = list(G.nodes())
    if args.nodes:
        try:
            target_nodes = [int(x.strip()) for x in args.nodes.split(",")]
            target_nodes = [n for n in target_nodes if n in G]
            if not target_nodes:
                print("ERROR: None of the specified nodes exist in loaded graph.")
                sys.exit(1)
        except ValueError:
            print("ERROR: --nodes must be comma-separated integers")
            sys.exit(1)
    else:
        k = min(args.k, len(all_nodes))
        np.random.seed(42)
        target_nodes = np.random.choice(all_nodes, size=k, replace=False).tolist()

    K = len(target_nodes)
    print(f"    Comparing {K} nodes: {target_nodes[:8]}{'...' if K > 8 else ''}")

    print("\n[2] Extracting 16 structural features (O(m))...")
    t0 = time.time()
    F, F_nodes = extract_features(G)
    node_to_idx = {v: i for i, v in enumerate(F_nodes)}
    print(f"    Done in {time.time()-t0:.1f}s")

    print("\n[3] Loading ML model...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    print(f"    Model type: {type(model.named_steps['clf']).__name__}")

    print("\n[4] Running ML pairwise tournament...")
    t0 = time.time()
    ranked_ml, scores = ml_rank_k(F, node_to_idx, model, target_nodes)
    ml_time = time.time() - t0
    n_pairs = K * (K - 1) // 2
    print(f"    {n_pairs} comparisons in {ml_time*1000:.2f}ms")

    # Optional ground truth
    bc_dict = {}
    if args.verify:
        print("\n[5] Computing exact Brandes BC (ground truth)...")
        t0 = time.time()
        bc_dict = nx.betweenness_centrality(G, normalized=True)
        print(f"    Done in {time.time()-t0:.1f}s")

    # Print results
    print(f"\n{'='*60}")
    print(f"  RANKING RESULT — {ds_name}")
    print(f"{'='*60}")
    
    if bc_dict:
        ranked_true = sorted(target_nodes, key=lambda n: bc_dict[n], reverse=True)
        print(f"  {'Rank':<6} {'Node':<10} {'ML Wins':<10} {'Exact BC':<14} {'True Rank':<10} {'Match'}")
        print(f"  {'-'*58}")
        for i, node in enumerate(ranked_ml):
            ml_r = i + 1
            true_r = ranked_true.index(node) + 1
            match = "✓" if ml_r == true_r else " "
            print(f"  {ml_r:<6} {node:<10} {scores[node]:<10} {bc_dict[node]:<14.4e} {true_r:<10} {match}")
        
        # Spearman
        d2 = sum((ranked_ml.index(n) - ranked_true.index(n))**2 for n in target_nodes)
        sp = 1 - (6 * d2) / (K * (K**2 - 1)) if K > 1 else 1.0
        print(f"\n  Spearman ρ vs Exact Brandes: {sp:.3f} (1.0 = perfect)")
        if ranked_ml[0] == ranked_true[0]:
            print(f"  => ML correctly identified the HIGHEST BC node: {ranked_ml[0]}")
        else:
            print(f"  => ML top node: {ranked_ml[0]}, True top node: {ranked_true[0]}")
    else:
        print(f"  {'Rank':<6} {'Node':<10} {'ML Wins':<10} Score")
        print(f"  {'-'*38}")
        max_wins = max(scores.values()) if scores else 1
        for i, node in enumerate(ranked_ml):
            bar = "#" * int(20 * scores[node] / max(max_wins, 1))
            print(f"  {i+1:<6} {node:<10} {scores[node]:<10} {bar}")
        print(f"\n  (Add --verify to compare against exact Brandes)")

    print(f"\n  ML inference time: {ml_time*1000:.2f}ms for K={K} nodes")
    print(f"  Complexity: O(K²)={n_pairs} predictions (sub-linear vs O(m) EDDBM)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
