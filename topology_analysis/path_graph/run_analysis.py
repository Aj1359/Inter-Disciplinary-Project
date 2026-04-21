#!/usr/bin/env python3
"""
Path Graph Topology Analysis
Linear chain of nodes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_path_graph(seed, n=20):
    """Generate path graph (linear chain)."""
    return nx.path_graph(n)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Path Graph (n=20)",
        graph_generator=generate_path_graph,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
