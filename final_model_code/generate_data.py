"""
Stage 1 + 2: Graph generation and exact BC computation.
Saves: data/train_data.pkl, data/test_data.pkl
"""
import networkx as nx
import numpy as np
import pickle, random, time, os

random.seed(42)
np.random.seed(42)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

def generate_graphs(n_each=120):
    graphs = []
    sizes  = [80, 150, 250, 400]
    for n in sizes:
        for avg_d in [3, 5, 10]:
            p = avg_d / n
            for _ in range(n_each // (len(sizes)*3) + 1):
                G = nx.erdos_renyi_graph(n, p, seed=random.randint(0,99999))
                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                if G.number_of_nodes() > 20:
                    graphs.append(('ER', n, avg_d, G))
    for n in sizes:
        for k in [2, 3, 5]:
            for _ in range(n_each // (len(sizes)*3) + 1):
                G = nx.barabasi_albert_graph(n, k, seed=random.randint(0,99999))
                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                if G.number_of_nodes() > 20:
                    graphs.append(('BA', n, k, G))
    return graphs[:n_each]

def compute_exact_bc(G):
    return nx.betweenness_centrality(G, normalized=False)

if __name__ == "__main__":
    print("=" * 55)
    print("  STAGE 1+2: Graph Generation + Exact BC")
    print("=" * 55)
    t0 = time.time()
    graphs = generate_graphs(n_each=120)
    random.shuffle(graphs)
    split       = int(0.8 * len(graphs))
    train_graphs= graphs[:split]
    test_graphs = graphs[split:]
    print(f"  Generated {len(train_graphs)} train + {len(test_graphs)} test graphs")

    train_data, test_data = [], []
    for i,(gtype,n,param,G) in enumerate(train_graphs):
        bc = compute_exact_bc(G)
        train_data.append({'type':gtype,'n':n,'param':param,'G':G,'bc':bc})
        if i % 15 == 0: print(f"  Computing BC... {i}/{len(train_graphs)}")
    for gtype,n,param,G in test_graphs:
        bc = compute_exact_bc(G)
        test_data.append({'type':gtype,'n':n,'param':param,'G':G,'bc':bc})

    with open('data/train_data.pkl','wb') as f: pickle.dump(train_data, f)
    with open('data/test_data.pkl',  'wb') as f: pickle.dump(test_data, f)

    # Save graph sizes for plotting
    ns = [d['G'].number_of_nodes() for d in test_data]
    with open('results/graph_sizes.pkl','wb') as f: pickle.dump(ns, f)

    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Train sizes sample: {[d['G'].number_of_nodes() for d in train_data[:5]]}")
    print(f"  Test  sizes sample: {[d['G'].number_of_nodes() for d in test_data[:5]]}")
    print(f"  Saved -> data/train_data.pkl, data/test_data.pkl")
