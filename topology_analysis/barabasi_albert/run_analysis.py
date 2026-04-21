#!/usr/bin/env python3
"""
Barabasi-Albert Topology Analysis
Preferential attachment model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_barabasi_albert(seed, n=40, m=2):
    """Generate Barabasi-Albert preferential attachment graph."""
    return nx.barabasi_albert_graph(n, m, seed=seed)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Barabasi-Albert (n=40, m=2)",
        graph_generator=generate_barabasi_albert,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
