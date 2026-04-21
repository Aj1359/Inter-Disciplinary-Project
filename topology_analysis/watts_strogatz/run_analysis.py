#!/usr/bin/env python3
"""
Watts-Strogatz Topology Analysis
Small-world network model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_watts_strogatz(seed, n=40, k=4, p=0.3):
    """Generate Watts-Strogatz small-world graph."""
    return nx.watts_strogatz_graph(n, k, p, seed=seed)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Watts-Strogatz (n=40, k=4, p=0.3)",
        graph_generator=generate_watts_strogatz,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
