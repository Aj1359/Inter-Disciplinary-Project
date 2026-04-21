"""
rank_k_nodes.py
===============
Given a set of K nodes, rank them by Betweenness Centrality using our trained 
pairwise ML model.

Approach: Round-Robin Tournament
Because the ML model is a pairwise classifier (A vs B), it might not perfectly 
satisfy transitivity (A > B, B > C does not strictly guarantee A > C in ML).
To rank K nodes robustly, we play a round-robin tournament:
Every node is compared against every other node. A node gets +1 point for a "win".
We then sort the nodes by their total win count to determine the absolute ranking.

Usage:
    python rank_k_nodes.py --dataset ../Wiki-Vote.txt --k 20
"""

import os
import sys
import pickle
import argparse
import time
import networkx as nx
import numpy as np

# Import functions from our feature extraction script
from node_comparison_model import load_graph, extract_node_features

# Windows UTF-8 console fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Rank K nodes using ML Pairwise Model")
    parser.add_argument("--dataset", required=True, help="Path to SNAP dataset")
    parser.add_argument("--model", default="node_comparison_model.pkl", help="Path to trained model .pkl")
    parser.add_argument("--k", type=int, default=20, help="Number of nodes to randomly select and rank")
    parser.add_argument("--nodes", type=str, default=None, help="Comma separated list of specific node IDs (overrides --k)")
    args = parser.parse_args()

    ds_name = os.path.basename(args.dataset)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)

    print(f"\n--- Ranking K Nodes Using ML Model ({ds_name}) ---")

    # 1. Load Graph and Exact BC for ground truth comparison
    print("1. Loading graph and computing exact Brandes BC for ground truth...")
    G = load_graph(args.dataset, 1500) # Keep max-nodes reasonable
    bc_dict =nx.betweenness_centrality(G, normalized=True)
    
    # 2. Select the K nodes to compare
    nodes_list = list(G.nodes())
    if args.nodes:
        # If user specified exact nodes: "10,24,159"
        try:
            target_nodes = [int(x.strip()) for x in args.nodes.split(",")]
            # Filter to nodes actually in the sampled graph G
            target_nodes = [n for n in target_nodes if n in G]
            if not target_nodes:
                 print("None of the specified nodes are in the graph sub-sample.")
                 sys.exit(1)
        except ValueError:
            print("Error: --nodes must be comma-separated integers.")
            sys.exit(1)
    else:
        k = min(args.k, len(nodes_list))
        np.random.seed(42)
        target_nodes = np.random.choice(nodes_list, size=k, replace=False).tolist()

    k = len(target_nodes)
    print(f"-> Selected {k} nodes to rank.")

    # 3. Extract Features
    print("2. Extracting ML features...")
    F, F_nodes = extract_node_features(G)
    node_to_idx = {v: i for i, v in enumerate(F_nodes)}

    # 4. Load Model
    print("3. Loading ML pairwise model...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # 5. Round Robin Tournament Rankings
    print("\n4. Running ML Round-Robin Tournament...")
    t0 = time.time()
    
    scores = {node: 0 for node in target_nodes}
    comparisons = 0

    # Build batched X array so we only call model.predict ONCE which is much faster
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            u = target_nodes[i]
            v = target_nodes[j]
            pairs.append((u, v))
            comparisons += 1
            
    if comparisons > 0:
        X_batch = np.zeros((comparisons, F.shape[1]))
        for idx, (u, v) in enumerate(pairs):
            idx_u, idx_v = node_to_idx[u], node_to_idx[v]
            X_batch[idx] = F[idx_u] - F[idx_v]
            
        preds = model.predict(X_batch)
        
        # Tally scores
        for idx, (u, v) in enumerate(pairs):
            if preds[idx] == 1:
                # ML predicts BC(u) > BC(v), so u wins
                scores[u] += 1
            else:
                # v wins
                scores[v] += 1

    ml_time = time.time() - t0

    # Rank nodes by ML score descending
    ranked_by_ml = sorted(target_nodes, key=lambda n: scores[n], reverse=True)
    
    # Ground truth ranking
    ranked_by_true = sorted(target_nodes, key=lambda n: bc_dict[n], reverse=True)

    # 6. Display Results
    print(f"\nCompleted {comparisons} pairwise comparisons in {ml_time*1000:.2f} ms.\n")
    
    print("=" * 70)
    print(f"{'Rank':<6} | {'Node':<8} | {'ML Wins':<9} | {'True Brandes BC':<15} | {'Optimal Rank'}")
    print("=" * 70)
    
    # Track top node match
    perfect_top_node = (ranked_by_ml[0] == ranked_by_true[0])

    for i, node in enumerate(ranked_by_ml):
        ml_rank = i + 1
        points = scores[node]
        true_bc = bc_dict[node]
        true_rank = ranked_by_true.index(node) + 1
        
        # Mark if the ML rank matches the optimal Rank exactly
        rank_mark = "*" if ml_rank == true_rank else " "
        
        print(f"{ml_rank:<6} | {node:<8} | {points:<9} | {true_bc:<15.6e} | {true_rank} {rank_mark}")
        
    print("-" * 70)
    print("\nSummary:")
    print(f"-> ML Selected '#1 Node': {ranked_by_ml[0]}")
    print(f"-> Optimal '#1 Node'    : {ranked_by_true[0]}")
    if perfect_top_node:
        print("=> SUCCESS: ML found the absolute highest betweenness node among the candidates!")
    else:
        print("=> ML ranking was slightly off for the top node.")
        
    # Calculate Spearman rank correlation
    # Create rank dictionaries
    ml_ranks = {node: i for i, node in enumerate(ranked_by_ml)}
    true_ranks = {node: i for i, node in enumerate(ranked_by_true)}
    
    d_sq_sum = sum((ml_ranks[n] - true_ranks[n])**2 for n in target_nodes)
    if k > 1:
        spearman = 1 - (6 * d_sq_sum) / (k * (k**2 - 1))
        print(f"\n=> Spearman Rank Correlation: {spearman:.3f} (1.0 is perfect match)")

if __name__ == "__main__":
    main()
