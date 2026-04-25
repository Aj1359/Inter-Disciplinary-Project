"""
Model B: Pairwise XGBoost Ranker.
Directly classifies BC(u) > BC(v) from structural node features.
No sampling at inference — just feature extraction + model.predict().
Saves: models/model_b.pkl, results/model_b_xis.pkl
"""
import numpy as np, pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import extract_node_features, make_pairwise_features
from bolt_baseline import ordering_efficiency

def build_pairwise_dataset(train_data, pairs_per_graph=600, seed=42):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for d in train_data:
        G, bc  = d['G'], d['bc']
        nf     = extract_node_features(G)
        valid  = [n for n in G.nodes() if bc[n] > 0]
        if len(valid) < 2: continue
        n_pairs = min(pairs_per_graph, len(valid)*(len(valid)-1)//2)
        seen, attempts = set(), 0
        while len(seen) < n_pairs and attempts < n_pairs*5:
            i,j = rng.choice(len(valid), 2, replace=False)
            u,v = valid[i], valid[j]
            if bc[u] == bc[v]: attempts+=1; continue
            key = (min(u,v), max(u,v))
            if key in seen: attempts+=1; continue
            seen.add(key)
            X.append(make_pairwise_features(nf[u], nf[v]))
            Y.append(1 if bc[u] > bc[v] else 0)
            attempts+=1
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

def train_model_b(train_data):
    print("  Building pairwise dataset...")
    X, Y = build_pairwise_dataset(train_data)
    print(f"  Dataset: {X.shape[0]:,} pairs, {X.shape[1]} features")
    print(f"  Class balance: {Y.mean():.3f}  (0.5 = perfectly balanced)")
    X_tr,X_val,y_tr,y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=False)
    print(f"  Validation accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    return model

def model_b_efficiency(G, bc_exact, model, max_pairs=None, rng=None):
    nf = extract_node_features(G)
    def pred_fn(u, v):
        feat = make_pairwise_features(nf[u], nf[v]).reshape(1,-1)
        return bool(model.predict(feat)[0] == 1)
    return ordering_efficiency(G, bc_exact, pred_fn, max_pairs=max_pairs, rng=rng)

if __name__ == "__main__":
    print("=" * 55)
    print("  MODEL B: Pairwise XGBoost Ranker")
    print("=" * 55)
    rng = np.random.RandomState(42)
    with open('data/train_data.pkl','rb') as f: train_data = pickle.load(f)
    with open('data/test_data.pkl',  'rb') as f: test_data  = pickle.load(f)

    model = train_model_b(train_data)
    os.makedirs('models', exist_ok=True)
    with open('models/model_b.pkl','wb') as f: pickle.dump(model, f)
    print("  Saved -> models/model_b.pkl")

    print("\n  Evaluating on test graphs...")
    xis = []
    for d in test_data:
        G, bc = d['G'], d['bc']
        xi = model_b_efficiency(G, bc, model, max_pairs=250, rng=rng)
        xis.append(xi)
        print(f"  {d['type']} n={G.number_of_nodes():3d} -> ξ={xi:.4f}")

    print(f"\n  Model B mean ξ = {np.mean(xis):.4f}  std = {np.std(xis):.4f}")
    with open('results/model_b_xis.pkl','wb') as f: pickle.dump(xis, f)
    print("  Saved -> results/model_b_xis.pkl")
