import os
import subprocess
import csv
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as plt_sns
import numpy as np
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_connected(G):
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G

def get_9_topologies():
    gs = []
    gs.append(('Erdos-Renyi', ensure_connected(nx.erdos_renyi_graph(40, 0.12, seed=1))))
    gs.append(('Barabasi-Albert', nx.barabasi_albert_graph(40, 2, seed=1)))
    gs.append(('Watts-Strogatz', nx.watts_strogatz_graph(40, 4, 0.3, seed=1)))
    gs.append(('Path', nx.path_graph(20)))
    gs.append(('Barbell', nx.barbell_graph(10, 3)))
    gs.append(('Star', nx.star_graph(19)))
    gs.append(('Tree', nx.random_labeled_tree(30, seed=1)))
    gs.append(('Powerlaw', ensure_connected(nx.powerlaw_cluster_graph(30, 2, 0.3, seed=1))))
    gs.append(('Caveman', nx.connected_caveman_graph(5, 5)))
    return gs

def write_snap_graph(G, filepath):
    G = nx.convert_node_labels_to_integers(G)
    with open(filepath, 'w') as f:
        f.write("# Undirected graph\n")
        f.write(f"# Nodes: {G.number_of_nodes()} Edges: {G.number_of_edges()}\n")
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

if __name__ == "__main__":
    cpp_file = os.path.join(SCRIPT_DIR, "bc_k_analysis.cpp")
    exe_file = os.path.join(SCRIPT_DIR, "bc_k_analysis.exe")
    
    print("Compiling C++ engine...")
    r = subprocess.run(["g++", "-O2", "-std=c++11", "-o", exe_file, cpp_file])
    if r.returncode != 0:
        print("Compilation Failed!")
        exit(1)
        
    temp_dir = os.path.join(SCRIPT_DIR, 'temp_graphs')
    os.makedirs(temp_dir, exist_ok=True)
    graphs = get_9_topologies()
    
    results = []
    
    print(f"Testing Candidate Depths on {len(graphs)} topologies...")
    sys.stdout.flush()
    
    for topo_name, G in graphs:
        # Save temporary
        filepath = os.path.join(temp_dir, f"{topo_name}.txt")
        write_snap_graph(G, filepath)
        
        # Run
        cmd = [exe_file, filepath]
        out = subprocess.run(cmd, capture_output=True, text=True)
        
        bf_time = 0
        
        # Parse standard output looking for BF_TIME and CSV_OUT
        for line in out.stdout.split('\n'):
            line = line.strip()
            if line.startswith("BF_TIME:"):
                bf_time = float(line.split(':')[1])
            if line.startswith("CSV_OUT:"):
                parts = line.split(':')[1].split(',')
                # parts: k, reductPct, opt, timeMs, candsEvaluated
                k = int(parts[0])
                reductPct = float(parts[1])
                opt = float(parts[2])
                timeMs = float(parts[3])
                candsE = int(parts[4])
                
                results.append({
                    'Topology': topo_name,
                    'K': k,
                    'ReductPct': reductPct,
                    'Optimality': opt,
                    'TimeMs': timeMs,
                    'BF_Time': bf_time,
                    'Candidates': candsE
                })
        print(f" - {topo_name} completed (BF Time: {bf_time:.2f}ms).")
        sys.stdout.flush()

    # Dump CSV
    df = pd.DataFrame(results)
    results_csv = os.path.join(SCRIPT_DIR, "k_analysis_results.csv")
    df.to_csv(results_csv, index=False)
    print(f"Results saved to: {results_csv}")
    
    # Run Plotter
    print("Generating updated 3x3 visualization...")
    subprocess.run(["python", os.path.join(SCRIPT_DIR, "plot_k_bars.py")])
