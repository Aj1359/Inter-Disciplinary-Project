#!/usr/bin/env python3
"""
Grid2D Topology Analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from analyze_topology import analyze_topology

def generate_grid_2d(seed, side=5):
    # Seed doesn't change grid structure but defined for API compliance
    return nx.grid_2d_graph(side, side)

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results = analyze_topology(
        topology_name="Grid 2D (5x5)",
        graph_generator=generate_grid_2d,
        output_dir=output_dir,
        num_trials=12,
        tau=0.20,
        topk=50
    )
