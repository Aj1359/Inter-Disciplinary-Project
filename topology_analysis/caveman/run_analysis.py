#!/usr/bin/env python3
"""
Caveman Topology Analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_caveman(seed, cl=5, sz=5):
    return nx.connected_caveman_graph(cl, sz)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Caveman (5 subsets of 5)",
        graph_generator=generate_caveman,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
