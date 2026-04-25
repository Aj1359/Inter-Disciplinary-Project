"""
Model A: Learned Pivot Selector.
Replaces EDDBM formula with XGBoost trained on δ_{i•}(v) ground truth.
Saves: models/model_a.pkl, results/model_a_xis.pkl
"""
import numpy as np, pickle, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xgboost as xgb
from sklearn.metrics import r2_score
from features import extract_node_features, extract_pivot_features, bfs_levels
from bolt_baseline import single_source_dependency, estimate_bc, ordering_efficiency

def build_pivot_dataset(train_data, targets_per_graph=6, seed=42):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for d in train_data:
        G, bc  = d['G'], d['bc']
        avg_deg= np.mean([dg for _,dg in G.degree()])
        nf     = extract_node_features(G)
        valid  = [n for n in G.nodes() if bc[n] > 0]
        if not valid: continue
        targets = rng.choice(valid, min(targets_per_graph, len(valid)), replace=False)
        for v in targets:
            dist_v  = bfs_levels(G, v)
            lc      = {}
            for nd,dl in dist_v.items(): lc[dl] = lc.get(dl,0)+1
            for i in G.nodes():
                if i == v: continue
                feat  = extract_pivot_features(G, v, i, dist_v, lc, nf, avg_deg)
                delta = single_source_dependency(G, i, v)
                X.append(feat); Y.append(np.log1p(delta))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_model_a(train_data):
    print("  Building pivot dataset...")
    X, Y = build_pivot_dataset(train_data)
    print(f"  Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    model = xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X, Y)
    r2 = r2_score(Y, model.predict(X))
    print(f"  Train R² = {r2:.4f}  (1.0 = perfect pivot score prediction)")
    return model

def learned_probs(G, target_v, model, avg_deg, node_feats):
    dist_v  = bfs_levels(G, target_v)
    lc      = {}
    for nd,dl in dist_v.items(): lc[dl] = lc.get(dl,0)+1
    pivots  = [n for n in G.nodes() if n != target_v]
    if not pivots: return {}, dist_v
    feats   = np.array([extract_pivot_features(G, target_v, i, dist_v, lc, node_feats, avg_deg)
                        for i in pivots])
    scores  = np.exp(model.predict(feats))
    scores  = np.clip(scores, 0, None)
    total   = scores.sum() + 1e-12
    return {n: scores[i]/total for i,n in enumerate(pivots)}, dist_v

def model_a_order(G, u, v, model, T=25, rng=None):
    avg_deg = np.mean([d for _,d in G.degree()])
    nf      = extract_node_features(G)
    pu, _   = learned_probs(G, u, model, avg_deg, nf)
    pv, _   = learned_probs(G, v, model, avg_deg, nf)
    return estimate_bc(G, pu, u, T, rng) > estimate_bc(G, pv, v, T, rng)

if __name__ == "__main__":
    print("=" * 55)
    print("  MODEL A: Learned Pivot Selector (XGBoost)")
    print("=" * 55)
    rng = np.random.RandomState(42)
    with open('data/train_data.pkl','rb') as f: train_data = pickle.load(f)
    with open('data/test_data.pkl',  'rb') as f: test_data  = pickle.load(f)

    model = train_model_a(train_data)
    os.makedirs('models', exist_ok=True)
    with open('models/model_a.pkl','wb') as f: pickle.dump(model, f)
    print("  Saved -> models/model_a.pkl")

    print("\n  Evaluating on test graphs...")
    xis = []
    for d in test_data:
        G, bc = d['G'], d['bc']
        def ma_p(u,v,G=G,m=model,r=rng): return model_a_order(G,u,v,m,T=25,rng=r)
        xi = ordering_efficiency(G, bc, ma_p, max_pairs=250, rng=rng)
        xis.append(xi)
        print(f"  {d['type']} n={G.number_of_nodes():3d} -> ξ={xi:.4f}")

    print(f"\n  Model A mean ξ = {np.mean(xis):.4f}  std = {np.std(xis):.4f}")
    with open('results/model_a_xis.pkl','wb') as f: pickle.dump(xis, f)
    print("  Saved -> results/model_a_xis.pkl")
