"""
deterministic_eddbm.py
======================
Improves EDDBM by abandoning high-variance random sampling. Instead of pulling
T samples based on distance probability, we deterministically select the top T
most strongly correlated source nodes (e.g. highest degree nodes closest to target)
and calculate restricted Brandes for just those T nodes in O(T*m) overall time.
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
    from node_comparison_model import load_graph, brandes_dependency, eddbm_sampling_prob
    from node_comparison_model import eddbm_estimate_bc as original_eddbm
except ImportError:
    print("Error: Must be run with node_comparison_model.py in parent dir.")
    sys.exit(1)

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError: pass

def top_t_deterministic_eddbm(G, v, T, avg_deg):
    """
    Instead of sampling dynamically, pick the exact Top T best nodes based on
    EDDBM probability weights, and average their Brandes dependency.
    """
    P = eddbm_sampling_prob(G, v, avg_deg)
    if not P:
        return 0.0
        
    # Sort nodes by probability descending
    sorted_sources = sorted(P.items(), key=lambda item: item[1], reverse=True)
    
    # Grab the Top T highest probability nodes
    top_t = sorted_sources[:T]
    
    est = 0.0
    for s, prob in top_t:
        dep = brandes_dependency(G, s, v)
        # We re-weight by their known theoretical probability in the EDDBM equation
        est += dep / prob
        
    return est / T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to SNAP edge-list")
    parser.add_argument("--max-nodes", type=int, default=1500)
    args = parser.parse_args()

    ds_name = os.path.basename(args.dataset)
    print(f"\n--- [Option 3] Deterministic Top-T EDDBM ({ds_name}) ---")

    print("- Loading Graph & computing Exact Brandes...")
    G = load_graph(args.dataset, args.max_nodes)
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    nodes = list(G.nodes())
    bc_optimal = np.array([bc_dict[n] for n in nodes])
    
    avg_deg = 2.0 * G.number_of_edges() / G.number_of_nodes()
    rng = np.random.RandomState(42)
    T = 30
    
    print("- Computing original random EDDBM (T=30)...")
    t0 = time.time()
    bc_random = np.array([original_eddbm(G, n, T, avg_deg, rng) for n in nodes])
    print(f"  Took {time.time()-t0:.2f}s")

    print(f"- Computing Deterministic TOP-{T} EDDBM...")
    t0 = time.time()
    bc_deter = np.array([top_t_deterministic_eddbm(G, n, T, avg_deg) for n in nodes])
    print(f"  Took {time.time()-t0:.2f}s")
    
    # Evaluate
    mse_rand = mean_squared_error(bc_optimal, bc_random)
    mse_deter = mean_squared_error(bc_optimal, bc_deter)
    
    print("\nResults on Full Graph:")
    print(f"  MSE (Random EDDBM)       : {mse_rand:.6e}")
    print(f"  MSE (Deterministic Top-T): {mse_deter:.6e}")
    print(f"  Improvement              : {mse_rand / mse_deter:.2f}x less error!")

    # Plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.scatter(bc_optimal, bc_random, alpha=0.5, label="Original Random EDDBM", color="orange")
    plt.scatter(bc_optimal, bc_deter, alpha=0.5, label="Deterministic Top-T", color="purple", marker="v")
    
    plt.plot([0, max(bc_optimal)], [0, max(bc_optimal)], 'k--', label="Optimal (Exact)")
    
    plt.xlabel("Exact Brandes Betweenness")
    plt.ylabel("Computed Betweenness")
    plt.title(f"Random vs Deterministic Top-T EDDBM ({ds_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = f"plots/{ds_name}_deterministic_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n=> Saved scatter plot to {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    main()
