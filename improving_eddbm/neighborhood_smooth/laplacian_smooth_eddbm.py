"""
laplacian_smooth_eddbm.py
=========================
Improves EDDBM via Laplacian Neighborhood Smoothing. Betweenness centrality
is locally correlated. EDDBM suffers from random sampling variance. 
By pulling each node's EDDBM estimate towards the average of its neighbors'
EDDBM estimates in a linear O(m) pass, we smooth out random noise and 
move the estimates much closer to the exact Brandes values.
"""

import os
import sys
import argparse
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from node_comparison_model import load_graph, eddbm_estimate_bc
except ImportError:
    print("Error: Must be run with node_comparison_model.py in parent dir.")
    sys.exit(1)

# Windows UTF-8 console fix
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError: pass

def laplacian_smoothing(G, raw_estimates, alpha=0.5):
    """
    Applies 1-hop Laplacian smoothing in O(m) time.
    smooth(v) = (1-alpha) * raw(v) + alpha * Mean_{u in N(v)} raw(u)
    """
    smooth_bc = np.zeros_like(raw_estimates)
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    for i, v in enumerate(nodes):
        neighbors = list(G.neighbors(v))
        if len(neighbors) == 0:
            smooth_bc[i] = raw_estimates[i]
            continue
            
        nbr_sum = sum(raw_estimates[node_to_idx[u]] for u in neighbors)
        nbr_avg = nbr_sum / len(neighbors)
        
        smooth_bc[i] = (1 - alpha) * raw_estimates[i] + alpha * nbr_avg
        
    return smooth_bc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to SNAP edge-list")
    parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=0.5, help="Smoothing strength")
    args = parser.parse_args()

    ds_name = os.path.basename(args.dataset)
    print(f"\n--- [Option 2] Laplacian Smoothing ({ds_name}) ---")

    print("- Loading Graph & computing Exact Brandes...")
    G = load_graph(args.dataset, args.max_nodes)
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    nodes = list(G.nodes())
    bc_optimal = np.array([bc_dict[n] for n in nodes])
    
    print("- Computing raw baseline EDDBM (T=30)...")
    avg_deg = 2.0 * G.number_of_edges() / G.number_of_nodes()
    rng = np.random.RandomState(42)
    t0 = time.time()
    bc_eddbm = np.array([eddbm_estimate_bc(G, n, 30, avg_deg, rng) for n in nodes])
    print(f"  Took {time.time()-t0:.2f}s")

    print(f"- Applying Laplacian Smoothing (alpha={args.alpha})...")
    t0 = time.time()
    bc_smooth = laplacian_smoothing(G, bc_eddbm, args.alpha)
    print(f"  Took {time.time()-t0:.4f}s  (True O(m) time!)")
    
    # Evaluate
    mse_eddbm = mean_squared_error(bc_optimal, bc_eddbm)
    mse_smooth = mean_squared_error(bc_optimal, bc_smooth)
    
    print("\nResults on Full Graph:")
    print(f"  MSE (Original EDDBM) : {mse_eddbm:.6e}")
    print(f"  MSE (LAPLACIAN smooth): {mse_smooth:.6e}")
    print(f"  Improvement          : {mse_eddbm / mse_smooth:.2f}x less error!")

    # Plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.scatter(bc_optimal, bc_eddbm, alpha=0.5, label="Original EDDBM (Noisy)", color="orange")
    plt.scatter(bc_optimal, bc_smooth, alpha=0.5, label="Smoothed EDDBM", color="green", marker="+")
    
    plt.plot([0, max(bc_optimal)], [0, max(bc_optimal)], 'k--', label="Optimal (Exact)")
    
    plt.xlabel("Exact Brandes Betweenness")
    plt.ylabel("Predicted/Estimated Betweenness")
    plt.title(f"Laplacian Smoothing vs EDDBM ({ds_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = f"plots/{ds_name}_laplacian_smooth_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n=> Saved scatter plot to {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    main()
