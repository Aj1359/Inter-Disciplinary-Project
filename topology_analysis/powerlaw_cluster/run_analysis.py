#!/usr/bin/env python3
"""
Powerlaw Cluster Topology Analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_powerlaw_cluster(seed, n=40, m=2, p=0.3):
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Powerlaw Cluster (n=40)",
        graph_generator=generate_powerlaw_cluster,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
