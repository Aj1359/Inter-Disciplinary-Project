"""
test_ml_model.py
================
Test the trained ML model on a specific dataset and plot its performance
compared to the "optimal" (exact Brandes) betweenness centrality ordering.

Usage:
    python test_ml_model.py --dataset ../Wiki-Vote.txt --max-nodes 1500
"""

import os
import sys
import pickle
import argparse
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Import functions from the main script we wrote earlier
from node_comparison_model import load_graph, extract_node_features

# Windows UTF-8 console fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Test ML Comparison Model")
    parser.add_argument("--dataset", required=True, help="Path to SNAP dataset (e.g., ../Wiki-Vote.txt)")
    parser.add_argument("--model", default="node_comparison_model.pkl", help="Path to trained model .pkl")
    parser.add_argument("--max-nodes", type=int, default=1500, help="Max nodes to test on")
    parser.add_argument("--test-pairs", type=int, default=5000, help="Number of test pairs to evaluate")
    args = parser.parse_args()

    ds_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return

    print(f"\n--- Testing Dataset '{ds_name}' ---")

    # 1. Load Graph and compute Exact BC
    print("1. Loading graph and computing exact Brandes BC...")
    G = load_graph(args.dataset, args.max_nodes)
    bc_dict = nx.betweenness_centrality(G, normalized=True)
    
    # 2. Extract Features
    print("2. Extracting ML features...")
    F, nodes_ordered = extract_node_features(G)
    bc_aligned = np.array([bc_dict[v] for v in nodes_ordered])
    n = len(nodes_ordered)

    # 3. Load Model
    print(f"3. Loading trained model '{args.model}'...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # 4. Generate Random Test Pairs
    print(f"4. Generating {args.test_pairs} random test pairs...")
    rng = np.random.RandomState(42)
    a_idx = rng.randint(0, n, size=args.test_pairs)
    b_idx = rng.randint(0, n, size=args.test_pairs)
    
    # Remove self loops
    mask = a_idx != b_idx
    a_idx = a_idx[mask]
    b_idx = b_idx[mask]

    X_test = F[a_idx] - F[b_idx]
    
    # True Labels: 1 if BC(A) > BC(B), else 0
    y_true = (bc_aligned[a_idx] > bc_aligned[b_idx]).astype(int)

    # 5. ML Predict
    print("5. Predicting with ML Model...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability A > B
    ml_time = time.time() - t0
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\nResults:")
    print(f"  ML Accuracy: {acc*100:.2f}%")
    print(f"  Prediction Time for {len(X_test)} pairs: {ml_time:.4f}s ({ml_time/len(X_test)*1000:.4f} ms/pair)")

    # 6. Plotting - ML Prediction Confidence vs Actual BC Difference
    print("\n6. Generating plots...")
    os.makedirs("plots", exist_ok=True)
    
    # We want to see how "close to optimal" the model is.
    # The x-axis will be the TRUE absolute difference in BC. 
    # The y-axis will be the ML predicted probability (confidence).
    
    bc_diff_true = bc_aligned[a_idx] - bc_aligned[b_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot: True difference vs ML Confidence
    scatter = ax.scatter(bc_diff_true, y_prob, alpha=0.3, s=8, c=np.abs(bc_diff_true), cmap='viridis')
    
    # Draw decision boundary lines
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
    ax.axvline(0.0, color='gray', linestyle='-')
    
    ax.set_xlabel("True Difference in Exact BC (A - B)")
    ax.set_ylabel("ML Predicted Probability (A > B)")
    ax.set_title(f"ML Confidence vs True BC Difference\nDataset: {ds_name} (Acc: {acc*100:.1f}%)")
    plt.colorbar(scatter, label='Absolute BC Difference')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plot_path = os.path.abspath(f"plots/{ds_name}_ml_optimal_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {plot_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
