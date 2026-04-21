#!/usr/bin/env python3
"""
Star Graph Topology Analysis
Central hub with peripheral nodes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_star(seed, n=20):
    """Generate star graph (one central node, n-1 leaves)."""
    return nx.star_graph(n - 1)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Star Graph (n=20)",
        graph_generator=generate_star,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
