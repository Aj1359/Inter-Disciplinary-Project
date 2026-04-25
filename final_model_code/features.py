"""
Feature extraction utilities — imported by all three model scripts.
"""
import networkx as nx
import numpy as np
from collections import deque

def bfs_levels(G, source):
    dist = {source: 0}
    q    = deque([source])
    while q:
        v = q.popleft()
        for w in G.neighbors(v):
            if w not in dist:
                dist[w] = dist[v] + 1
                q.append(w)
    return dist

def extract_node_features(G):
    """
    8-dimensional structural feature vector per node.
    Returns dict: node -> np.array(float32, shape=(8,))
    """
    n        = G.number_of_nodes()
    degrees  = dict(G.degree())
    max_deg  = max(degrees.values()) if degrees else 1
    avg_deg  = np.mean(list(degrees.values()))
    clust    = nx.clustering(G)
    kcore    = nx.core_number(G)
    max_kc   = max(kcore.values()) if kcore else 1
    avg_nd   = nx.average_neighbor_degree(G)
    max_and  = max(avg_nd.values()) if avg_nd else 1

    # Approximate eccentricity via BFS from 5 sample nodes
    sample_src = list(G.nodes())[:min(5, n)]
    bfs_cache  = {s: bfs_levels(G, s) for s in sample_src}

    features = {}
    for v in G.nodes():
        dv      = degrees[v]
        ecc_app = max(bfs_cache[s].get(v, n) for s in sample_src)
        features[v] = np.array([
            dv / max_deg,                              # 0: normalized degree
            clust.get(v, 0.0),                         # 1: clustering coeff
            kcore.get(v, 1) / max_kc,                  # 2: normalized k-core
            avg_nd.get(v, 0) / (max_and + 1e-9),       # 3: avg neighbor degree
            np.log1p(dv) / np.log1p(max_deg + 1),      # 4: log-degree
            float(dv == 1),                             # 5: is leaf
            dv / (avg_deg + 1e-9),                     # 6: degree / avg_degree
            ecc_app / (n + 1),                          # 7: approx eccentricity
        ], dtype=np.float32)
    return features

def extract_pivot_features(G, target_v, pivot_i, dist_from_v, level_counts, node_feats, avg_deg):
    """
    13-dimensional feature vector for (target_v, pivot_i) pair.
    Used by Model A to predict pivot quality δ_{i•}(v).
    """
    d     = dist_from_v.get(pivot_i, 999)
    deg_i = G.degree(pivot_i)
    lam   = max(avg_deg, 1.001)
    sibs  = level_counts.get(d, 1)
    return np.array([
        d / (G.number_of_nodes() + 1),        # 0: normalized distance
        lam ** (-min(d, 20)),                  # 1: EDDBM distance term λ^{-d}
        1.0 / (deg_i + 1e-9),                 # 2: inverse degree of pivot
        1.0 / (G.degree(target_v) + 1e-9),   # 3: inverse degree of target
        deg_i / (avg_deg + 1e-9),             # 4: normalized pivot degree
        node_feats[pivot_i][1],               # 5: clustering of pivot
        node_feats[pivot_i][2],               # 6: k-core of pivot
        1.0 / (sibs + 1),                     # 7: inverse sibling count
        node_feats[target_v][1],              # 8: clustering of target
        node_feats[target_v][2],              # 9: k-core of target
        float(d == 1),                         # 10: is direct neighbor
        float(d == 2),                         # 11: is 2-hop
        np.log1p(d),                           # 12: log distance
    ], dtype=np.float32)

def make_pairwise_features(feat_u, feat_v):
    """
    16-dim feature for (u,v) pair used by Model B.
    Concatenates (feat_u - feat_v) and |feat_u - feat_v|.
    """
    diff = feat_u - feat_v
    return np.concatenate([diff, np.abs(diff)])
