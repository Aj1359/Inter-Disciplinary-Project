"""
Model C: 3-layer GraphSAGE — implemented from scratch in NumPy.
Learns inductive node embeddings via message passing, then uses a pairwise
ranking head. No PyTorch required.
Saves: models/model_c.pkl, results/model_c_xis.pkl
"""
import numpy as np, pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_node_features
from bolt_baseline import ordering_efficiency

# ── Math helpers ──────────────────────────────────────────────────────────────
relu    = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
glorot  = lambda fi, fo, rng: rng.randn(fi,fo).astype(np.float32) * np.sqrt(2.0/(fi+fo))

# ── Single GraphSAGE layer ────────────────────────────────────────────────────
class SAGELayer:
    def __init__(self, in_dim, out_dim, rng):
        self.Ws = glorot(in_dim, out_dim, rng)   # self weight
        self.Wn = glorot(in_dim, out_dim, rng)   # neighbor weight
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self.params = [self.Ws, self.Wn, self.b]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def forward(self, H, adj):
        n   = H.shape[0]
        agg = np.array([H[adj[i]].mean(axis=0) if adj[i] else H[i] for i in range(n)])
        out = H @ self.Ws + agg @ self.Wn + self.b
        self._H, self._agg, self._pre = H, agg, out
        return relu(out)

    def backward(self, dout, adj):
        n   = dout.shape[0]
        dp  = dout * (self._pre > 0)
        dWs = self._H.T @ dp
        dWn = self._agg.T @ dp
        db  = dp.sum(axis=0)
        dH  = dp @ self.Ws.T
        da  = dp @ self.Wn.T
        for i in range(n):
            if adj[i]:
                for j in adj[i]: dH[j] += da[i] / len(adj[i])
        self._grads = [dWs, dWn, db]
        return dH

# ── Full GraphSAGE model ──────────────────────────────────────────────────────
class GraphSAGE:
    def __init__(self, in_dim, hidden=32, seed=42):
        rng = np.random.RandomState(seed)
        self.l1 = SAGELayer(in_dim,  hidden,    rng)
        self.l2 = SAGELayer(hidden,  hidden,    rng)
        self.l3 = SAGELayer(hidden,  hidden//2, rng)
        hd      = hidden   # ranking head input = 2 * (hidden//2) = hidden
        self.Wh = glorot(hd, 1, rng)
        self.bh = np.zeros(1, dtype=np.float32)
        self.mWh = np.zeros_like(self.Wh); self.vWh = np.zeros_like(self.Wh)
        self.mbh = np.zeros_like(self.bh); self.vbh = np.zeros_like(self.bh)
        self.t  = 0; self.lr = 3e-3
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8

    def _norm(self, H):
        return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    def embed(self, H, adj):
        h = self._norm(self.l1.forward(H, adj))
        h = self._norm(self.l2.forward(h, adj))
        return self.l3.forward(h, adj)          # (n, hidden//2)

    def score(self, eu, ev):
        diff = eu - ev
        comb = np.concatenate([diff, np.abs(diff)])
        return sigmoid((comb @ self.Wh + self.bh)[0])

    def _adam(self, p, g, m, v):
        m[:] = self.b1*m + (1-self.b1)*g
        v[:] = self.b2*v + (1-self.b2)*g**2
        p   -= self.lr * (m/(1-self.b1**self.t)) / (np.sqrt(v/(1-self.b2**self.t)) + self.eps)

    def train_step(self, H, adj, pairs):
        """One graph forward + backward + Adam update."""
        self.t  += 1
        embs     = self.embed(H, adj)
        n, hd2   = embs.shape
        dE       = np.zeros_like(embs)
        dWh      = np.zeros_like(self.Wh)
        dbh      = np.zeros_like(self.bh)
        loss = acc = 0.0

        for ui, vi, lbl in pairs:
            eu, ev  = embs[ui], embs[vi]
            diff    = eu - ev
            comb    = np.concatenate([diff, np.abs(diff)])
            p       = np.clip(sigmoid((comb @ self.Wh + self.bh)[0]), 1e-7, 1-1e-7)
            loss   += -(lbl*np.log(p) + (1-lbl)*np.log(1-p))
            acc    += (p>0.5) == (lbl>0.5)
            dl      = (p - lbl) / max(len(pairs),1)
            dWh    += comb.reshape(-1,1) * dl
            dbh    += dl
            dc      = self.Wh.flatten() * dl
            dd, da  = dc[:hd2], dc[hd2:]
            sgn     = np.sign(diff)
            dE[ui] += dd + da*sgn
            dE[vi] -= dd + da*sgn

        self._adam(self.Wh, dWh, self.mWh, self.vWh)
        self._adam(self.bh, dbh, self.mbh, self.vbh)

        dH3 = self.l3.backward(dE, adj)
        for p,g,m,v in zip(self.l3.params, self.l3._grads, self.l3.m, self.l3.v):
            self._adam(p,g,m,v)
        dH2 = self.l2.backward(dH3, adj)
        for p,g,m,v in zip(self.l2.params, self.l2._grads, self.l2.m, self.l2.v):
            self._adam(p,g,m,v)
        _   = self.l1.backward(dH2, adj)
        for p,g,m,v in zip(self.l1.params, self.l1._grads, self.l1.m, self.l1.v):
            self._adam(p,g,m,v)

        return loss/max(len(pairs),1), acc/max(len(pairs),1)

# ── Data prep helpers ─────────────────────────────────────────────────────────
def prep_graph(G, node_feats):
    nodes   = sorted(G.nodes())
    n2i     = {n:i for i,n in enumerate(nodes)}
    H       = np.array([node_feats[n] for n in nodes], dtype=np.float32)
    adj     = {i: [n2i[w] for w in G.neighbors(nodes[i])] for i in range(len(nodes))}
    return H, adj, nodes, n2i

def make_pairs(nodes, n2i, bc, rng, n_pairs=120):
    valid = [n for n in nodes if bc.get(n,0) > 0]
    if len(valid) < 2: return []
    pairs, seen, attempts = [], set(), 0
    target = min(n_pairs, len(valid)*(len(valid)-1)//2)
    while len(pairs) < target and attempts < target*5:
        i,j = rng.choice(len(valid), 2, replace=False)
        u,v = valid[i], valid[j]
        if bc[u]==bc[v]: attempts+=1; continue
        key = (min(u,v), max(u,v))
        if key in seen: attempts+=1; continue
        seen.add(key)
        pairs.append((n2i[u], n2i[v], float(bc[u]>bc[v])))
        attempts+=1
    return pairs

# ── Training loop ─────────────────────────────────────────────────────────────
def train_gnn(train_data, epochs=40, seed=42):
    rng = np.random.RandomState(seed)
    gds = []
    for d in train_data:
        nf = extract_node_features(d['G'])
        H, adj, nodes, n2i = prep_graph(d['G'], nf)
        pairs = make_pairs(nodes, n2i, d['bc'], rng)
        if pairs: gds.append((H, adj, nodes, n2i, d['bc'], pairs))

    in_dim = gds[0][0].shape[1]
    model  = GraphSAGE(in_dim=in_dim, hidden=32, seed=seed)
    print(f"  Training: {len(gds)} graphs, input_dim={in_dim}, epochs={epochs}")

    for ep in range(epochs):
        rng.shuffle(gds)
        tot_loss = tot_acc = tot_n = 0
        for H, adj, nodes, n2i, bc, pairs in gds:
            l, a    = model.train_step(H, adj, pairs)
            tot_loss += l*len(pairs); tot_acc += a*len(pairs); tot_n += len(pairs)
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}/{epochs}  loss={tot_loss/tot_n:.4f}  "
                  f"train_acc={tot_acc/tot_n:.4f}")
    return model

def gnn_efficiency(G, bc_exact, model, max_pairs=None, rng=None):
    nf           = extract_node_features(G)
    H, adj, nodes, n2i = prep_graph(G, nf)
    embs         = model.embed(H, adj)
    def pred_fn(u, v): return model.score(embs[n2i[u]], embs[n2i[v]]) > 0.5
    return ordering_efficiency(G, bc_exact, pred_fn, max_pairs=max_pairs, rng=rng)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MODEL C: GraphSAGE GNN (NumPy, from scratch)")
    print("=" * 55)
    rng = np.random.RandomState(42)
    with open('data/train_data.pkl','rb') as f: train_data = pickle.load(f)
    with open('data/test_data.pkl',  'rb') as f: test_data  = pickle.load(f)

    model = train_gnn(train_data, epochs=40)
    os.makedirs('models', exist_ok=True)
    with open('models/model_c.pkl','wb') as f: pickle.dump(model, f)
    print("  Saved -> models/model_c.pkl")

    print("\n  Evaluating on test graphs...")
    xis = []
    for d in test_data:
        G, bc = d['G'], d['bc']
        xi = gnn_efficiency(G, bc, model, max_pairs=250, rng=rng)
        xis.append(xi)
        print(f"  {d['type']} n={G.number_of_nodes():3d} -> ξ={xi:.4f}")

    print(f"\n  Model C mean ξ = {np.mean(xis):.4f}  std = {np.std(xis):.4f}")
    with open('results/model_c_xis.pkl','wb') as f: pickle.dump(xis, f)
    print("  Saved -> results/model_c_xis.pkl")
