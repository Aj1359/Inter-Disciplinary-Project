#!/usr/bin/env python3
"""
Random Tree Topology Analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_random_tree(seed, n=30):
    return nx.random_labeled_tree(n, seed=seed)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Random Tree (n=30)",
        graph_generator=generate_random_tree,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
