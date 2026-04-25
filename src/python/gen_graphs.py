#!/usr/bin/env python3
"""
Generate synthetic graphs similar to the datasets in the paper.
- as-22july06: ~22963 nodes, avg degree ~4.22 (sparse, router topology)
  We model it as a random sparse graph (Waxman-like / random regular-ish).
"""

import random
import sys

def generate_barabasi_albert(n, m, seed=42):
    """Generate Barabasi-Albert scale-free graph with n nodes, m edges per new node."""
    random.seed(seed)
    edges = set()
    degrees = [0] * n
    # Start with a small complete graph of size m+1
    for i in range(m+1):
        for j in range(i+1, m+1):
            edges.add((min(i,j), max(i,j)))
            degrees[i] += 1
            degrees[j] += 1

    for new_node in range(m+1, n):
        targets = set()
        total_deg = sum(degrees)
        if total_deg == 0:
            targets = set(range(m))
        else:
            while len(targets) < m:
                r = random.random() * total_deg
                cumulative = 0
                for i in range(new_node):
                    cumulative += degrees[i]
                    if cumulative >= r:
                        targets.add(i)
                        break
        for t in targets:
            edges.add((min(new_node, t), max(new_node, t)))
            degrees[new_node] = degrees[new_node] + 1
            degrees[t] += 1

    return edges

def write_graph(filename, n_actual, edges, description):
    with open(filename, 'w') as f:
        f.write(f"# Synthetic graph: {description}\n")
        f.write(f"# Nodes: {n_actual}  Edges: {len(edges)}\n")
        f.write("# FromNodeId\tToNodeId\n")
        for u, v in sorted(edges):
            f.write(f"{u}\t{v}\n")
    print(f"Written {filename}: {n_actual} nodes, {len(edges)} edges")

if __name__ == "__main__":
    # Generate as-22july06 synthetic equivalent
    # ~22963 nodes, ~48436 edges (undirected), avg_deg ~4.22
    # Using BA model with m=2 gives avg_deg ~4 for large n
    print("Generating synthetic as-22july06 (BA graph, n=22963, m=2)...")
    edges_as = generate_barabasi_albert(22963, 2, seed=2022)
    write_graph("as-22july06-synthetic.txt", 22963, edges_as,
                "as-22july06 synthetic equivalent (BA model n=22963 m=2)")
