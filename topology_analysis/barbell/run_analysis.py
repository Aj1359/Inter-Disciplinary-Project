#!/usr/bin/env python3
"""
Barbell Graph Topology Analysis
Two cliques connected by a bridge
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_barbell(seed, m=7, p=2):
    """Generate barbell graph (two cliques with bridge)."""
    return nx.barbell_graph(m, p)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Barbell Graph (m=7, p=2)",
        graph_generator=generate_barbell,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
