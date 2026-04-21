import os
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle
import matplotlib.pyplot as plt

def extract_features_and_labels(G, num_anchors=10):
    nodes = list(G.nodes())
    anchors = np.random.choice(nodes, size=num_anchors, replace=False)
    
    X = []
    y = []

    # Calculate betweenness centrality to use as probability label conceptually, 
    # but the paper states label is "optimal probability p* = \delta_{i.}(v) / \sum_j \delta_{j.}(v)".
    # Let's approximate optimal p* by tracking dependency accumulations.
    # To keep it simple and aligned to paper specs: Brandes dependency \delta_{i.}(s).
    
    for s in anchors:
        # Run BFS from s
        dist = {n: -1 for n in nodes}
        sigma = {n: 0.0 for n in nodes}
        parent = {n: -1 for n in nodes}
        
        queue = [s]
        dist[s] = 0
        sigma[s] = 1.0
        
        stack = []
        
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in G.neighbors(v):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        
        # Calculate dependencies delta
        delta = {n: 0.0 for n in nodes}
        for w in reversed(stack):
            for v in G.neighbors(w):
                if dist[v] == dist[w] - 1:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])

        # Convert delta into probability label p*
        total_delta = sum(delta.values())
        p_star = {n: (delta[n] / total_delta) if total_delta > 0 else 0 for n in nodes}
        
        # Compute level average degree
        level_deg_sum = {}
        level_deg_count = {}
        for n in nodes:
            d = dist[n]
            if d >= 0:
                level_deg_sum[d] = level_deg_sum.get(d, 0) + G.degree(n)
                level_deg_count[d] = level_deg_count.get(d, 0) + 1
        
        # Build features for each node i
        # features: d(s,i), deg(i), \sigma_{si}, clustering_approx, level_avg_deg, deg(parent_in_BFS)
        for i in nodes:
            if i == s or dist[i] <= 0:
                continue
            
            d_si = dist[i]
            deg_i = G.degree(i)
            sig_i = sigma[i]
            
            # Clustering approx: (common neighbors with parent) / degree
            p = parent[i]
            c_hat = 0
            deg_parent = 0
            if p != -1:
                deg_parent = G.degree(p)
                common = len(set(G.neighbors(i)) & set(G.neighbors(p)))
                c_hat = common / max(1, deg_i)
                
            level_avg_d = level_deg_sum[d_si] / level_deg_count[d_si]
            
            x_feat = [d_si, deg_i, sig_i, c_hat, level_avg_d, deg_parent]
            y_label = p_star[i]
            
            X.append(x_feat)
            y.append(y_label)
            
    return np.array(X), np.array(y)

# Load dataset
def load_graph(filename, max_nodes=5000):
    G = nx.Graph()
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
    G.add_edges_from(edges)
    
    # Subsample if too large
    if G.number_of_nodes() > max_nodes:
        # Take largest connected component and sample
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(list(cc)[:max_nodes]).copy()
        
    return G

d_path = "../Wiki-Vote.txt"
if os.path.exists(d_path):
    print("Loading Graph for ML Training...")
    G = load_graph(d_path)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    print("Extracting features (this may take a few seconds)...")
    X, y = extract_features_and_labels(G, num_anchors=20)
    print(f"Generated {len(X)} training samples.")
    
    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("Training MLP Regressor...")
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained! Test MSE: {mse:.6e}")
    
    with open("ml_sampler.pkl", "wb") as f:
        pickle.dump(model, f)
        
    # Plot True vs Predicted mapping
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=2)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.xlabel('True Optimal Probability $p^*$')
    plt.ylabel('Predicted Probability')
    plt.title('ML Model vs Brandes Optimal')
    plt.grid()
    
    if not os.path.exists("plot1"):
        os.makedirs("plot1")
    plt.savefig("plot1/ml_model_accuracy.png")
    print("Saved accuracy plot in plot1/ml_model_accuracy.png")
    
else:
    print(f"Dataset {d_path} not found.")
