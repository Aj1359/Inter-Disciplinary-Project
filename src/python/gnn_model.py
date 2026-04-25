#!/usr/bin/env python3
"""
DrBC-style Graph Neural Network for Betweenness Centrality Ordering.

Architecture:
  1. Node Feature Initialization: degree, clustering coeff, k-core, etc.
  2. GNN Encoder: Multi-layer message-passing (GraphSAGE-style)
  3. MLP Decoder: Per-node score prediction
  4. Pairwise Ranking Loss: BPR/margin-based loss for correct ordering

Key advantages over BOLT/EDDBM:
  - Learns structural patterns from diverse graphs
  - Inductive: works on unseen graphs without retraining
  - Inference is O(n*d*L) where d=avg_degree, L=layers (much faster than O(Tm))
  - Once computed, pairwise comparisons are O(1)
"""
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Graph utilities
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
    edge_index = [[], []]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_index[0].extend([a, b])
        edge_index[1].extend([b, a])

    deg = [len(adj[i]) for i in range(n)]
    return adj, deg, n, len(edges), edge_index


def brandes_all(adj):
    """Exact Brandes betweenness centrality. O(nm)."""
    n = len(adj)
    BC = [0.0] * n
    for s in range(n):
        pred = [[] for _ in range(n)]
        dist = [-1] * n
        sigma = [0.0] * n
        stack = []
        q = deque()
        dist[s] = 0
        sigma[s] = 1.0
        q.append(s)
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
            if w != s:
                BC[w] += delta[w]
    for i in range(n):
        BC[i] /= 2.0
    return BC


def generate_ba_graph(n, m_attach, rng):
    """Generate Barabasi-Albert graph."""
    adj = [[] for _ in range(n)]
    edge_index = [[], []]
    edges = set()

    def add_edge(u, v):
        if u != v and (min(u, v), max(u, v)) not in edges:
            edges.add((min(u, v), max(u, v)))
            adj[u].append(v)
            adj[v].append(u)
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])

    # Start with a complete graph of size m_attach
    for i in range(min(m_attach, n)):
        for j in range(i + 1, min(m_attach, n)):
            add_edge(i, j)

    deg = [len(adj[i]) for i in range(n)]
    total_deg = sum(deg)

    for new in range(m_attach, n):
        targets = set()
        attempts = 0
        while len(targets) < m_attach and attempts < m_attach * 10:
            attempts += 1
            if total_deg <= 0:
                t = rng.randint(0, new - 1)
            else:
                r = rng.random() * total_deg
                cumul = 0
                t = 0
                for k in range(new):
                    cumul += deg[k]
                    if cumul >= r:
                        t = k
                        break
            targets.add(t)
        for t in targets:
            add_edge(new, t)
            deg[new] += 1
            deg[t] += 1
            total_deg += 2

    deg = [len(adj[i]) for i in range(n)]
    return adj, deg, n, len(edges), edge_index


def generate_er_graph(n, p, rng):
    """Generate Erdos-Renyi graph."""
    adj = [[] for _ in range(n)]
    edge_index = [[], []]
    edges = set()

    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.add((i, j))
                adj[i].append(j)
                adj[j].append(i)
                edge_index[0].extend([i, j])
                edge_index[1].extend([j, i])

    deg = [len(adj[i]) for i in range(n)]
    return adj, deg, n, len(edges), edge_index


# ─────────────────────────────────────────────────────────────────────────────
# Node Feature Initialization
# ─────────────────────────────────────────────────────────────────────────────

def compute_node_features(adj, deg, n):
    """Compute initial node features for GNN input. Returns (n, feat_dim) tensor."""
    features = []
    avg_deg = sum(deg) / max(1, n)
    max_deg = max(deg) if deg else 1

    for v in range(n):
        d = deg[v]

        # Neighbor degree stats
        nb_degs = [deg[nb] for nb in adj[v]] if adj[v] else [0]
        nb_avg = sum(nb_degs) / max(1, len(nb_degs))
        nb_max = max(nb_degs) if nb_degs else 0

        # Local clustering coefficient
        if d >= 2:
            nb_set = set(adj[v])
            tri_count = 0
            for nb in adj[v]:
                for nb2 in adj[nb]:
                    if nb2 in nb_set and nb2 != v:
                        tri_count += 1
            clustering = tri_count / (d * (d - 1))
        else:
            clustering = 0.0

        # 2-hop reach
        hop1 = set(adj[v])
        hop2 = set()
        for nb in hop1:
            for nb2 in adj[nb]:
                if nb2 != v and nb2 not in hop1:
                    hop2.add(nb2)
        reach2 = len(hop1) + len(hop2)

        feat = [
            math.log1p(d),
            d / max(1, n - 1),
            math.log1p(nb_avg),
            nb_max / max(1, max_deg),
            clustering,
            math.log1p(reach2),
            reach2 / max(1, n),
            d / max(1.0, avg_deg),
        ]
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)


NODE_FEAT_DIM = 8


# ─────────────────────────────────────────────────────────────────────────────
# GNN Model
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGELayer(nn.Module):
    """GraphSAGE-style message passing layer with mean aggregation."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj_list, deg):
        """
        x: (n, in_dim) node features
        adj_list: list of lists, adj_list[v] = neighbors of v
        deg: (n,) degrees
        """
        n = x.size(0)
        device = x.device

        # Self transform
        h_self = self.W_self(x)

        # Neighbor aggregation (mean)
        h_neigh = torch.zeros_like(h_self)
        for v in range(n):
            if adj_list[v]:
                nb_idx = adj_list[v]
                nb_feats = x[nb_idx]  # (num_neighbors, in_dim)
                h_neigh[v] = self.W_neigh(nb_feats.mean(dim=0))

        h = h_self + h_neigh + self.bias
        h = self.norm(h)
        return h


class DrBCEncoder(nn.Module):
    """Multi-layer GNN encoder for betweenness centrality."""

    def __init__(self, in_dim=NODE_FEAT_DIM, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_list, deg):
        h = F.relu(self.input_proj(x))
        h = self.dropout(h)

        for layer in self.layers:
            h_new = F.relu(layer(h, adj_list, deg))
            h_new = self.dropout(h_new)
            h = h + h_new  # Residual connection

        return h  # (n, hidden_dim)


class DrBCDecoder(nn.Module):
    """MLP decoder: node embedding -> betweenness score."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, h):
        return self.mlp(h).squeeze(-1)  # (n,)


class DrBCModel(nn.Module):
    """Full DrBC model: Encoder + Decoder."""

    def __init__(self, in_dim=NODE_FEAT_DIM, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.encoder = DrBCEncoder(in_dim, hidden_dim, num_layers, dropout)
        self.decoder = DrBCDecoder(hidden_dim)

    def forward(self, x, adj_list, deg):
        h = self.encoder(x, adj_list, deg)
        scores = self.decoder(h)
        return scores  # (n,) predicted betweenness scores

    def predict_ordering(self, x, adj_list, deg, u, v):
        """Returns True if model predicts BC(u) > BC(v)."""
        scores = self.forward(x, adj_list, deg)
        return scores[u].item() > scores[v].item()


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_ranking_loss(scores, bc_values, pairs, margin=0.5):
    """
    BPR-style pairwise ranking loss.
    For each pair (i, j) where BC(i) > BC(j), we want score(i) > score(j).
    """
    loss = torch.tensor(0.0, device=scores.device)
    count = 0

    for i, j in pairs:
        if bc_values[i] > bc_values[j]:
            diff = scores[i] - scores[j]
        elif bc_values[j] > bc_values[i]:
            diff = scores[j] - scores[i]
        else:
            continue

        # Margin-based hinge loss
        loss += F.relu(margin - diff)
        count += 1

    return loss / max(1, count)


def listwise_ranking_loss(scores, bc_values, temperature=1.0):
    """
    ListNet-style listwise ranking loss.
    Minimizes KL divergence between predicted and true ranking distributions.
    """
    # Normalize BC values to form probability distribution
    bc_tensor = torch.tensor(bc_values, dtype=torch.float32, device=scores.device)
    bc_nonzero_mask = bc_tensor > 0
    if bc_nonzero_mask.sum() < 2:
        return torch.tensor(0.0, device=scores.device)

    bc_subset = bc_tensor[bc_nonzero_mask]
    score_subset = scores[bc_nonzero_mask]

    # Softmax distributions
    true_dist = F.softmax(torch.log(bc_subset + 1e-10) / temperature, dim=0)
    pred_dist = F.softmax(score_subset / temperature, dim=0)

    # KL divergence
    loss = F.kl_div(pred_dist.log(), true_dist, reduction="batchmean")
    return loss


def combined_loss(scores, bc_values, pairs, alpha=0.5):
    """Combined pairwise + listwise loss."""
    L_pair = pairwise_ranking_loss(scores, bc_values, pairs)
    L_list = listwise_ranking_loss(scores, bc_values)
    return alpha * L_pair + (1 - alpha) * L_list
