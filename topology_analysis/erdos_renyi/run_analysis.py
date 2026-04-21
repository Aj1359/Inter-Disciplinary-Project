#!/usr/bin/env python3
"""
Erdos-Renyi Topology Analysis
Generates random graphs with specified edge probability
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_erdos_renyi(seed, n=40, p=0.12):
    """Generate Erdos-Renyi random graph."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Ensure connected
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Erdos-Renyi (n=40, p=0.12)",
        graph_generator=generate_erdos_renyi,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
