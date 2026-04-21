"""
hybrid_ml_eddbm.py
==================
Improves the purely random EDDBM scalar estimation by feeding it 
along with O(m) structural features into an ML Regressor. The regressor
learns to predict the exact Brandes value, smoothing out EDDBM's noise.
"""

import os
import sys
import argparse
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Add parent path to import our library functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from node_comparison_model import load_graph, extract_node_features, eddbm_estimate_bc
except ImportError:
    print("Error: Must be run with node_comparison_model.py present in parent directory.")
    sys.exit(1)

# Windows UTF-8 console fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to SNAP edge-list")
    parser.add_argument("--max-nodes", type=int, default=1500)
    args = parser.parse_args()

    ds_name = os.path.basename(args.dataset)
    print(f"\n--- [Option 1] Hybrid ML + EDDBM Regressor ({ds_name}) ---")

    print("- Loading Graph & exact Brandes...")
    G = load_graph(args.dataset, args.max_nodes)
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    nodes = list(G.nodes())
    bc_optimal = np.array([bc_dict[n] for n in nodes])
    
    print("- Computing baseline EDDBM (T=30)...")
    avg_deg = 2.0 * G.number_of_edges() / G.number_of_nodes()
    rng = np.random.RandomState(42)
    t0 = time.time()
    bc_eddbm = np.array([eddbm_estimate_bc(G, n, 30, avg_deg, rng) for n in nodes])
    eddbm_time = time.time() - t0

    print("- Extracting ML features...")
    F, F_nodes = extract_node_features(G)
    node_to_idx = {v: i for i, v in enumerate(F_nodes)}
    F_aligned = np.array([F[node_to_idx[n]] for n in nodes])
    
    # Feature matrix X is EDDBM estimate + all 10 structural features
    X = np.hstack([bc_eddbm.reshape(-1, 1), F_aligned])
    y = bc_optimal

    print("- Training GradientBoostingRegressor...")
    # Train / test split to evaluate fairly
    X_train, X_test, y_train, y_test, eddbm_train, eddbm_test = train_test_split(
        X, y, bc_eddbm, test_size=0.3, random_state=42
    )
    
    regressor = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Predict
    ml_preds = regressor.predict(X_test)
    
    # Evaluate
    mse_eddbm = mean_squared_error(y_test, eddbm_test)
    mse_ml = mean_squared_error(y_test, ml_preds)
    
    print("\nResults on 30% Test Set:")
    print(f"  MSE (Original EDDBM) : {mse_eddbm:.6e}")
    print(f"  MSE (Hybrid ML EDDBM): {mse_ml:.6e}")
    print(f"  Improvement          : {mse_eddbm / mse_ml:.2f}x less error!")

    # Plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, eddbm_test, alpha=0.5, label="Original EDDBM", color="orange")
    plt.scatter(y_test, ml_preds, alpha=0.5, label="Hybrid ML EDDBM", color="blue", marker="x")
    
    # Optimal line
    plt.plot([0, max(y_test)], [0, max(y_test)], 'k--', label="Optimal (Exact)")
    
    plt.xlabel("Exact Brandes Betweenness")
    plt.ylabel("Predicted Betweenness")
    plt.title(f"Hybrid ML Regression vs EDDBM ({ds_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plot_path = f"plots/{ds_name}_ml_regressor_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n=> Saved scatter plot to {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    main()
