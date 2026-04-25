"""
BOLT baseline: exact implementation of Algorithm 2 from Singh et al. 2017.
Saves: results/bolt_xis.pkl
"""
import networkx as nx
import numpy as np
import pickle, time, os
from collections import deque
sys_path_fix = __import__('sys'); sys_path_fix.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Core BOLT functions (also imported by model_a_pivot.py) ───────────────────

def bfs_levels(G, source):
    dist = {source: 0}
    q = deque([source])
    while q:
        v = q.popleft()
        for w in G.neighbors(v):
            if w not in dist:
                dist[w] = dist[v] + 1
                q.append(w)
    return dist

def eddbm_probabilities(G, target_v, avg_deg):
    dist   = bfs_levels(G, target_v)
    lam    = max(avg_deg, 1.001)
    level_groups = {}
    for node, d in dist.items():
        if node == target_v: continue
        level_groups.setdefault(d, []).append(node)

    raw_dist = {node: lam**(-d) for node, d in dist.items() if node != target_v}
    total    = sum(raw_dist.values()) + 1e-12
    p_d_level = {}
    for node, val in raw_dist.items():
        d = dist[node]
        p_d_level[d] = p_d_level.get(d, 0) + val / total

    probs = {}
    for d, nodes in level_groups.items():
        inv_degs    = np.array([1.0 / (G.degree(n) + 1e-9) for n in nodes])
        inv_deg_sum = inv_degs.sum() + 1e-12
        p_d         = p_d_level.get(d, 0)
        for i, n in enumerate(nodes):
            probs[n] = p_d * (inv_degs[i] / inv_deg_sum) * len(nodes)

    total2 = sum(probs.values()) + 1e-12
    return {n: v/total2 for n, v in probs.items()}, dist

def single_source_dependency(G, source, target):
    pred  = {v: [] for v in G.nodes()}
    sigma = dict.fromkeys(G.nodes(), 0.0)
    dist  = dict.fromkeys(G.nodes(), -1)
    sigma[source] = 1.0; dist[source] = 0
    queue = deque([source]); stack = []
    while queue:
        v = queue.popleft(); stack.append(v)
        for w in G.neighbors(v):
            if dist[w] < 0:
                queue.append(w); dist[w] = dist[v] + 1
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]; pred[w].append(v)
    delta = dict.fromkeys(G.nodes(), 0.0)
    while stack:
        w = stack.pop()
        for v in pred[w]:
            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
    return delta.get(target, 0.0)

def estimate_bc(G, probs, target_v, T=25, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    nodes  = list(probs.keys())
    p_vals = np.array([probs[n] for n in nodes])
    p_vals = np.clip(p_vals, 1e-12, None); p_vals /= p_vals.sum()
    bc_est = 0.0
    for _ in range(T):
        idx    = rng.choice(len(nodes), p=p_vals)
        pivot  = nodes[idx]
        dep    = single_source_dependency(G, pivot, target_v)
        bc_est += dep / (p_vals[idx] + 1e-12)
    return bc_est / T

def bolt_order(G, u, v, T=25, rng=None):
    avg_deg  = np.mean([d for _, d in G.degree()])
    probs_u, _ = eddbm_probabilities(G, u, avg_deg)
    probs_v, _ = eddbm_probabilities(G, v, avg_deg)
    return estimate_bc(G, probs_u, u, T, rng) > estimate_bc(G, probs_v, v, T, rng)

def ordering_efficiency(G, bc_exact, predict_fn, max_pairs=None, rng=None):
    nodes = [n for n in G.nodes() if bc_exact[n] > 0]
    if len(nodes) < 2: return 1.0
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes))
             for j in range(i+1, len(nodes)) if bc_exact[nodes[i]] != bc_exact[nodes[j]]]
    if max_pairs and len(pairs) > max_pairs:
        idx   = rng.choice(len(pairs), max_pairs, replace=False) if rng else range(max_pairs)
        pairs = [pairs[i] for i in idx]
    if not pairs: return 1.0
    correct = sum(predict_fn(u,v) == (bc_exact[u] > bc_exact[v]) for u,v in pairs)
    return correct / len(pairs)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  BOLT BASELINE (T=25, EDDBM)")
    print("=" * 55)
    rng = np.random.RandomState(42)
    with open('data/test_data.pkl','rb') as f: test_data = pickle.load(f)

    xis = []
    for d in test_data:
        G, bc = d['G'], d['bc']
        def bp(u, v, G=G, rng=rng): return bolt_order(G, u, v, T=25, rng=rng)
        xi = ordering_efficiency(G, bc, bp, max_pairs=250, rng=rng)
        xis.append(xi)
        print(f"  {d['type']} n={G.number_of_nodes():3d} param={d['param']} -> ξ={xi:.4f}")

    print(f"\n  BOLT mean ξ = {np.mean(xis):.4f}  std = {np.std(xis):.4f}")
    with open('results/bolt_xis.pkl','wb') as f: pickle.dump(xis, f)
    print("  Saved -> results/bolt_xis.pkl")
