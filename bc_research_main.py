"""
bc_research_main.py - BC Minimization Complete Research Pipeline
================================================================
Reduces a target node's betweenness centrality by adding one edge.
Compares Brute Force (exact optimal) vs Smart Hop-Based (heuristic).

Phase 1: BF vs Smart across 10 topologies, 3 tau values
Phase 2: Parallel BF speedup estimation
Phase 3: XGBoost ML-guided edge prediction
Generates 3D visualizations and full report automatically.

Usage:
    python bc_research_main.py           # Full run
    python bc_research_main.py --test    # Quick test
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import time, os, sys, json, pickle, csv, warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load-balance threshold (τ) values
# We test 5 τ values to characterize the trade-off between BC reduction
# and load-balance strictness:
#   τ=0.05 (strict):  Few edges pass the constraint → conservative solutions
#   τ=0.10 (moderate): Balanced trade-off
#   τ=0.15 (standard): Default threshold for most experiments
#   τ=0.20 (relaxed):  More candidates pass → better reductions possible
#   τ=0.25 (permissive): Nearly all edges valid → maximizes potential reduction
TAU_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.25,
})

COL = {'bf': '#D85A30', 'sm': '#1D9E75', 'ml': '#534AB7',
       'blue': '#185FA5', 'amber': '#BA7517'}


def brandes_bc(G):
    """Brandes algorithm for normalized betweenness centrality."""
    bc = defaultdict(float)
    n = G.number_of_nodes()
    for s in G.nodes():
        S, P = [], defaultdict(list)
        sigma, dist, delta = defaultdict(float), {}, defaultdict(float)
        sigma[s] = 1.0
        dist[s] = 0
        Q = deque([s])
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in G.neighbors(v):
                if w not in dist:
                    Q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    if n > 2:
        f = 1.0 / ((n - 1) * (n - 2))
        for k in bc:
            bc[k] *= f
    return dict(bc)


def ensure_connected(G):
    """Extract largest connected component, relabel to integers."""
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


def brute_force(G, target, tau=0.20):
    """Exact brute force: try every non-edge. Always finds the true optimum."""
    t0 = time.perf_counter()
    bc0 = brandes_bc(G)
    avg = np.mean(list(bc0.values()))
    nodes = list(G.nodes())
    non_edges = [(u, w) for i, u in enumerate(nodes)
                 for w in nodes[i+1:] if not G.has_edge(u, w)]
    best = dict(u=-1, w=-1, red=0, load=0, bc_after=bc0[target])
    for u, w in non_edges:
        G2 = G.copy()
        G2.add_edge(u, w)
        bc2 = brandes_bc(G2)
        red = bc0[target] - bc2[target]
        load = max((bc2[n] - bc0[n]) for n in nodes if n != target)
        if load <= tau * avg and red > best['red']:
            best = dict(u=u, w=w, red=red, load=load, bc_after=bc2[target])
    t1 = time.perf_counter()
    return best, bc0, len(non_edges), (t1 - t0) * 1000


def smart_algorithm(G, target, tau=0.20, topk=50):
    """Hop-based heuristic: only evaluate candidates from hop-1/hop-2 zones."""
    t0 = time.perf_counter()
    bc0 = brandes_bc(G)
    avg = np.mean(list(bc0.values()))
    dist_map = {}
    Q = deque([target])
    dist_map[target] = 0
    while Q:
        v = Q.popleft()
        if dist_map[v] >= 2:
            continue
        for w in G.neighbors(v):
            if w not in dist_map:
                dist_map[w] = dist_map[v] + 1
                Q.append(w)
    hop1 = {n for n, d in dist_map.items() if d == 1}
    hop2 = {n for n, d in dist_map.items() if d == 2}
    cands, seen = [], set()
    def try_add(u, w, score):
        if u == w or G.has_edge(u, w):
            return
        key = (min(u, w), max(u, w))
        if key in seen:
            return
        seen.add(key)
        cands.append((score, u, w))
    for a in hop2:
        for b in hop2:
            if a < b:
                try_add(a, b, 2.0)
    for a in hop1:
        for b in hop1:
            if a < b:
                try_add(a, b, 1.0)
    for a in hop1:
        for b in hop2:
            try_add(a, b, 0.6)
    cands.sort(reverse=True)
    eval_cands = [(u, w) for _, u, w in cands[:topk]]
    best = dict(u=-1, w=-1, red=0, load=0, bc_after=bc0[target])
    for u, w in eval_cands:
        G2 = G.copy()
        G2.add_edge(u, w)
        bc2 = brandes_bc(G2)
        red = bc0[target] - bc2[target]
        nodes = list(G.nodes())
        load = max((bc2[n] - bc0[n]) for n in nodes if n != target)
        if load <= tau * avg and red > best['red']:
            best = dict(u=u, w=w, red=red, load=load, bc_after=bc2[target])
    t1 = time.perf_counter()
    return best, bc0, len(eval_cands), (t1 - t0) * 1000


def get_graphs(test_mode=False):
    """10 topologies, multiple instances each."""
    gs = []
    sizes = [30] if test_mode else [30, 40, 50]
    seeds = [0, 1] if test_mode else [0, 1, 2]
    for n in sizes:
        for s in seeds:
            gs.append((f'ER_n{n}_s{s}', 'Erdos-Renyi',
                       ensure_connected(nx.erdos_renyi_graph(n, 0.12, seed=s))))
            gs.append((f'BA_n{n}_s{s}', 'Barabasi-Albert',
                       nx.barabasi_albert_graph(n, 2, seed=s)))
            gs.append((f'WS_n{n}_s{s}', 'Watts-Strogatz',
                       nx.watts_strogatz_graph(n, 4, 0.3, seed=s)))
    for n in ([15, 20] if test_mode else [15, 20, 25]):
        gs.append((f'Path_n{n}', 'Path', nx.path_graph(n)))
    for m in ([7] if test_mode else [7, 10, 12]):
        gs.append((f'Barbell_m{m}', 'Barbell', nx.barbell_graph(m, max(1, m//3))))
    for n in ([15] if test_mode else [15, 20, 25]):
        gs.append((f'Star_n{n}', 'Star', nx.star_graph(n-1)))
    for s in ([5] if test_mode else [5, 6, 7]):
        gs.append((f'Grid_{s}x{s}', 'Grid2D', nx.grid_2d_graph(s, s)))
    for n, s in ([(20, 0)] if test_mode else [(20, 0), (30, 1), (40, 2)]):
        gs.append((f'Tree_n{n}', 'RandomTree', nx.random_labeled_tree(n, seed=s)))
    for n, s in ([(30, 0)] if test_mode else [(30, 0), (50, 1)]):
        gs.append((f'PLC_n{n}', 'PowerlawCluster',
                   ensure_connected(nx.powerlaw_cluster_graph(n, 2, 0.3, seed=s))))
    for cl, sz in ([(4, 5)] if test_mode else [(4, 5), (5, 5), (4, 6)]):
        gs.append((f'Cave_{cl}x{sz}', 'Caveman', nx.connected_caveman_graph(cl, sz)))
    return gs


# PHASE 1
def run_phase1(test_mode=False):
    """BF vs Smart across all topologies and 3 tau values."""
    graphs = get_graphs(test_mode)
    taus = TAU_VALUES
    total = len(graphs) * len(taus)
    print("=" * 70)
    print(f"  PHASE 1: BF vs SMART ({len(graphs)} graphs × {len(taus)} tau values = {total} runs)")
    print("=" * 70)

    results = []
    idx = 0
    for name, topo, G_raw in graphs:
        G = ensure_connected(G_raw)
        G = nx.convert_node_labels_to_integers(G)
        n = G.number_of_nodes()
        if n < 5:
            continue
        bc0 = brandes_bc(G)
        target = max(bc0, key=bc0.get)

        for tau in taus:
            idx += 1
            print(f"  [{idx:3d}/{total}] {name} tau={tau}...", end=" ", flush=True)
            try:
                b_res, _, b_c, b_t = brute_force(G, target, tau=tau)
                if b_res['u'] < 0 or b_res['red'] <= 1e-9:
                    print("skip")
                    continue
                s_res, _, s_c, s_t = smart_algorithm(G, target, tau=tau)
                opt_sm = (s_res['red'] / b_res['red'] * 100) if s_res['u'] >= 0 else 0
                spd = b_t / max(s_t, 0.001)
                err = 100 - opt_sm
                same = (
                    (b_res['u'] == s_res['u'] and b_res['w'] == s_res['w']) or
                    (b_res['u'] == s_res['w'] and b_res['w'] == s_res['u'])
                )
                r = {
                    'name': name, 'topology': topo, 'nodes': n,
                    'edges': G.number_of_edges(), 'tau': tau,
                    'bf_time_ms': round(b_t, 2), 'smart_time_ms': round(s_t, 2),
                    'speedup': round(spd, 2),
                    'bf_optimality': 100.0,
                    'smart_optimality': round(opt_sm, 2),
                    'error_pct': round(err, 2), 'same_edge': same,
                    'bf_edge': [b_res['u'], b_res['w']],
                    'smart_edge': [s_res['u'], s_res['w']],
                    'bf_reduction': round(b_res['red'], 8),
                    'smart_reduction': round(s_res['red'], 8),
                    'bf_cands': b_c, 'smart_cands': s_c,
                }
                results.append(r)
                print(f"spd={spd:.1f}x sm_opt={opt_sm:.1f}% err={err:.1f}%")
            except Exception as e:
                print(f"ERR: {str(e)[:50]}")

    path = os.path.join(RESULTS_DIR, 'phase1_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    csv_path = os.path.join(RESULTS_DIR, 'phase1_results.csv')
    if results:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            for r in results:
                row = dict(r)
                row['bf_edge'] = str(row['bf_edge'])
                row['smart_edge'] = str(row['smart_edge'])
                w.writerow(row)
    print(f"\n  Saved: {path} and {csv_path}")
    return results


# PHASE 2: PARALLELIZATION
from concurrent.futures import ThreadPoolExecutor


def _eval_chunk_thread(edges_list, G, target, bc0, tau, avg_bc):
    """Thread worker: evaluate a chunk of candidate edges."""
    nodes = list(G.nodes())
    best = dict(u=-1, w=-1, red=0)
    for u, w in edges_list:
        G2 = G.copy()
        G2.add_edge(u, w)
        bc2 = brandes_bc(G2)
        red = bc0[target] - bc2.get(target, 0)
        load = max((bc2.get(nd, 0) - bc0.get(nd, 0)) for nd in nodes if nd != target)
        if load <= tau * avg_bc and red > best['red']:
            best = dict(u=u, w=w, red=red)
    return best


def run_phase2(test_mode=False):
    """Parallel BF speedup analysis using ThreadPoolExecutor."""
    sizes = [30, 40] if test_mode else [30, 40, 50]
    max_cpu = mp.cpu_count()
    worker_counts = [1, 2, min(4, max_cpu)]
    if max_cpu >= 8 and not test_mode:
        worker_counts.append(8)
    print("\n" + "=" * 70)
    print(f"  PHASE 2: PARALLELIZATION (CPUs={max_cpu}, workers={worker_counts})")
    print("=" * 70)

    results = []
    for n in sizes:
        G = nx.barabasi_albert_graph(n, 2, seed=7)
        G = ensure_connected(G)
        G = nx.convert_node_labels_to_integers(G)
        bc0 = brandes_bc(G)
        target = max(bc0, key=bc0.get)
        avg_bc = np.mean(list(bc0.values()))
        nodes = list(G.nodes())
        non_edges = [(u, w) for i, u in enumerate(nodes)
                     for w in nodes[i+1:] if not G.has_edge(u, w)]
        name = f'BA_n{n}'
        print(f"\n  {name}:")

        s_res, _, _, s_t = brute_force(G, target)
        print(f"    Serial: {s_t:.0f}ms")
        gr = {'name': name, 'nodes': n, 'serial_time_ms': round(s_t, 1), 'parallel': []}

        for nw in worker_counts:
            t0 = time.perf_counter()
            chunk_size = max(1, len(non_edges) // nw)
            chunks = [non_edges[i:i+chunk_size] for i in range(0, len(non_edges), chunk_size)]
            best_all = dict(u=-1, w=-1, red=0)
            with ThreadPoolExecutor(max_workers=nw) as exe:
                futs = []
                for ch in chunks:
                    futs.append(exe.submit(_eval_chunk_thread, ch, G, target, bc0, 0.20, avg_bc))
                for fut in as_completed(futs):
                    r = fut.result()
                    if r['red'] > best_all['red']:
                        best_all = r
            p_t = (time.perf_counter() - t0) * 1000
            spd = s_t / max(p_t, 0.001)
            same = abs(s_res['red'] - best_all['red']) < 1e-6 if s_res['u'] >= 0 else True
            gr['parallel'].append({
                'workers': nw, 'time_ms': round(p_t, 1),
                'speedup': round(spd, 2), 'efficiency': round(spd/nw*100, 1),
                'same_result': same
            })
            print(f"    {nw}w: {p_t:.0f}ms spd={spd:.2f}x same={same}")
        results.append(gr)

    path = os.path.join(RESULTS_DIR, 'phase2_parallel.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")
    return results


# PHASE 3: ML
FEAT_NAMES = [
    'deg_u', 'deg_w', 'deg_product', 'deg_sum',
    'dist_u', 'dist_w', 'dist_sum', 'dist_max',
    'bc_u', 'bc_w', 'bc_sum', 'bc_max',
    'common_neighbors', 'jaccard', 'adamic_adar',
    'clust_u', 'clust_w', 'clust_avg', 'path_len_uw',
    'candidate_class', 'graph_density', 'graph_avg_clustering',
    'graph_n', 'graph_m', 'target_bc', 'target_degree',
]


def _extract_feat(G, u, w, target, bc0, dist_t, gs):
    d_u, d_w = dist_t.get(u, 999), dist_t.get(w, 999)
    nu, nw = set(G.neighbors(u)), set(G.neighbors(w))
    common = nu & nw
    union_n = nu | nw
    jac = len(common) / len(union_n) if union_n else 0
    aa = sum(1.0 / np.log(max(G.degree(c), 2)) for c in common)
    try:
        pl = nx.shortest_path_length(G, u, w)
    except Exception:
        pl = G.number_of_nodes()
    cc = 0 if (d_u == 2 and d_w == 2) else 1 if (d_u == 1 and d_w == 1) else 2 if (d_u <= 2 and d_w <= 2) else 3
    return [
        G.degree(u), G.degree(w), G.degree(u)*G.degree(w), G.degree(u)+G.degree(w),
        d_u, d_w, d_u+d_w, max(d_u, d_w),
        bc0.get(u, 0), bc0.get(w, 0), bc0.get(u, 0)+bc0.get(w, 0), max(bc0.get(u, 0), bc0.get(w, 0)),
        len(common), jac, aa,
        nx.clustering(G, u), nx.clustering(G, w), (nx.clustering(G, u)+nx.clustering(G, w))/2,
        pl, cc, gs['density'], gs['avg_clust'], gs['n'], gs['m'],
        bc0.get(target, 0), G.degree(target),
    ]


def run_phase3(test_mode=False):
    """Train XGBoost, evaluate on test graphs."""
    print("\n" + "=" * 70)
    mname = "XGBoost" if HAS_XGB else "GradientBoosting"
    print(f"  PHASE 3: ML ({mname})")
    print("=" * 70)

    train_gs = []
    for n in ([20, 30] if test_mode else [20, 30, 40]):
        for s in range(2 if test_mode else 3):
            train_gs.append((f'ER_{n}_s{s}', ensure_connected(nx.erdos_renyi_graph(n, 0.12, seed=s))))
            train_gs.append((f'BA_{n}_s{s}', nx.barabasi_albert_graph(n, 2, seed=s)))
            train_gs.append((f'WS_{n}_s{s}', nx.watts_strogatz_graph(n, 4, 0.3, seed=s)))
    for n in [15, 20]:
        train_gs.append((f'Path_{n}', nx.path_graph(n)))
    for m in [5, 7]:
        train_gs.append((f'Barbell_{m}', nx.barbell_graph(m, max(1, m//3))))

    max_e = 80 if test_mode else 200
    X_all, y_all = [], []
    print(f"  Training data from {len(train_gs)} graphs...")
    for gi, (nm, G) in enumerate(train_gs):
        print(f"    [{gi+1}/{len(train_gs)}] {nm}...", end=" ", flush=True)
        try:
            G = ensure_connected(G)
            G = nx.convert_node_labels_to_integers(G)
            n = G.number_of_nodes()
            if n < 5:
                print("skip")
                continue
            bc0 = brandes_bc(G)
            tgt = max(bc0, key=bc0.get)
            dist_t = {}
            Q = deque([tgt])
            dist_t[tgt] = 0
            while Q:
                v = Q.popleft()
                for w in G.neighbors(v):
                    if w not in dist_t:
                        dist_t[w] = dist_t[v]+1
                        Q.append(w)
            gs = {'n': n, 'm': G.number_of_edges(),
                  'density': 2*G.number_of_edges()/(n*(n-1)) if n > 1 else 0,
                  'avg_clust': nx.average_clustering(G)}
            hop_nodes = {nd for nd, d in dist_t.items() if d <= 3}
            ne = [(u, w) for u in hop_nodes for w in hop_nodes if u < w and not G.has_edge(u, w)]
            if len(ne) > max_e:
                rng = np.random.RandomState(gi)
                idx = rng.choice(len(ne), max_e, replace=False)
                ne = [ne[i] for i in idx]
            for u, w in ne:
                feat = _extract_feat(G, u, w, tgt, bc0, dist_t, gs)
                G2 = G.copy()
                G2.add_edge(u, w)
                bc2 = brandes_bc(G2)
                red = bc0[tgt] - bc2[tgt]
                red_pct = red / bc0[tgt] * 100 if bc0[tgt] > 0 else 0
                X_all.append(feat)
                y_all.append(red_pct)
            print(f"{len(ne)} edges")
        except Exception as e:
            print(f"ERR: {str(e)[:40]}")

    if len(X_all) < 20:
        print("  Not enough data!")
        return None
    X, y = np.array(X_all), np.array(y_all)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    if HAS_XGB:
        model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                             subsample=0.8, random_state=42, verbosity=0)
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                          learning_rate=0.1, subsample=0.8, random_state=42)
    model.fit(X_tr_s, y_tr)
    y_pred_tr = model.predict(X_tr_s)
    y_pred_te = model.predict(X_te_s)

    metrics = {
        'model': mname,
        'train_r2': round(r2_score(y_tr, y_pred_tr), 4),
        'test_r2': round(r2_score(y_te, y_pred_te), 4),
        'train_rmse': round(np.sqrt(mean_squared_error(y_tr, y_pred_tr)), 4),
        'test_rmse': round(np.sqrt(mean_squared_error(y_te, y_pred_te)), 4),
    }
    importances = dict(zip(FEAT_NAMES, model.feature_importances_))
    print(f"  Train R²={metrics['train_r2']} Test R²={metrics['test_r2']}")

    with open(os.path.join(RESULTS_DIR, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'metrics': metrics, 'importances': importances}, f)

    eval_gs = [
        ('ER_50_s99', ensure_connected(nx.erdos_renyi_graph(50, 0.12, seed=99))),
        ('BA_50_s99', nx.barabasi_albert_graph(50, 2, seed=99)),
        ('WS_50_s99', nx.watts_strogatz_graph(50, 4, 0.3, seed=99)),
        ('Path_25', nx.path_graph(25)),
        ('Barbell_10', nx.barbell_graph(10, 3)),
    ]
    if test_mode:
        eval_gs = eval_gs[:3]

    evals = []
    for nm, G in eval_gs:
        print(f"    {nm}...", end=" ", flush=True)
        try:
            G = ensure_connected(G)
            G = nx.convert_node_labels_to_integers(G)
            bc0 = brandes_bc(G)
            tgt = max(bc0, key=bc0.get)
            bf_res, _, bf_c, bf_t = brute_force(G, tgt)
            if bf_res['u'] < 0:
                print("skip")
                continue
            t0 = time.perf_counter()
            n = G.number_of_nodes()
            dist_t = {}
            Q = deque([tgt])
            dist_t[tgt] = 0
            while Q:
                v = Q.popleft()
                for w in G.neighbors(v):
                    if w not in dist_t:
                        dist_t[w] = dist_t[v]+1
                        Q.append(w)
            gs_info = {'n': n, 'm': G.number_of_edges(),
                       'density': 2*G.number_of_edges()/(n*(n-1)) if n > 1 else 0,
                       'avg_clust': nx.average_clustering(G)}
            hop_nodes = {nd for nd, d in dist_t.items() if d <= 3}
            cands = [(u, w) for u in hop_nodes for w in hop_nodes if u < w and not G.has_edge(u, w)]
            if cands:
                Xc = np.array([_extract_feat(G, u, w, tgt, bc0, dist_t, gs_info) for u, w in cands])
                Xc_s = scaler.transform(Xc)
                preds = model.predict(Xc_s)
                ranked = np.argsort(-preds)[:30]
                avg_bc = np.mean(list(bc0.values()))
                nodes = list(G.nodes())
                ml_best = dict(u=-1, w=-1, red=0)
                for ix in ranked:
                    u, w = cands[ix]
                    G2 = G.copy()
                    G2.add_edge(u, w)
                    bc2 = brandes_bc(G2)
                    red = bc0[tgt] - bc2[tgt]
                    load = max((bc2[nd]-bc0[nd]) for nd in nodes if nd != tgt)
                    if load <= 0.20 * avg_bc and red > ml_best['red']:
                        ml_best = dict(u=u, w=w, red=red)
            else:
                ml_best = dict(u=-1, w=-1, red=0)
            ml_t = (time.perf_counter() - t0) * 1000
            opt = (ml_best['red'] / bf_res['red'] * 100) if ml_best['u'] >= 0 and bf_res['red'] > 1e-9 else 0
            spd = bf_t / max(ml_t, 0.001)
            evals.append({'name': nm, 'nodes': n, 'bf_time': round(bf_t, 1),
                          'ml_time': round(ml_t, 1), 'ml_opt': round(opt, 1),
                          'ml_speedup': round(spd, 1)})
            print(f"spd={spd:.1f}x opt={opt:.1f}%")
        except Exception as e:
            print(f"ERR: {str(e)[:40]}")

    path = os.path.join(RESULTS_DIR, 'phase3_ml.json')
    with open(path, 'w') as f:
        json.dump({'metrics': metrics, 'importances': importances, 'eval': evals}, f, indent=2)
    print(f"  Saved: {path}")
    return metrics, importances, X_te, y_te, y_pred_te, evals


# FIGURES
def generate_figures(p1, p2, p3):
    """Generate all publication figures."""
    print("\n  Generating figures...")
    if not p1:
        return

    topo_stats = {}
    for r in p1:
        t = r['topology']
        tau = r['tau']
        key = f"{t}_tau{tau}"
        if t not in topo_stats:
            topo_stats[t] = {'spd': [], 'opt': [], 'err': [], 'bf_t': [], 'sm_t': [], 'tau': []}
        topo_stats[t]['spd'].append(r['speedup'])
        topo_stats[t]['opt'].append(r['smart_optimality'])
        topo_stats[t]['err'].append(r['error_pct'])
        topo_stats[t]['bf_t'].append(r['bf_time_ms'])
        topo_stats[t]['sm_t'].append(r['smart_time_ms'])
        topo_stats[t]['tau'].append(tau)

    topos = sorted(topo_stats.keys())
    fig = plt.figure(figsize=(22, 18))

    ax1 = fig.add_subplot(331)
    x = np.arange(len(topos))
    means = [np.mean(topo_stats[t]['spd']) for t in topos]
    stds = [np.std(topo_stats[t]['spd']) for t in topos]
    ax1.bar(x, means, yerr=stds, capsize=3, color=COL['sm'], alpha=0.8, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t[:10] for t in topos], rotation=40, fontsize=6)
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup by Topology', fontweight='bold')
    for i, m in enumerate(means):
        ax1.text(i, m + stds[i] + 0.3, f'{m:.1f}x', ha='center', fontsize=6, fontweight='bold')

    ax2 = fig.add_subplot(332)
    bf_opt = [100.0] * len(topos)
    sm_opt = [np.mean(topo_stats[t]['opt']) for t in topos]
    w = 0.35
    ax2.bar(x - w/2, bf_opt, w, label='BF (always 100%)', color=COL['bf'], alpha=0.8)
    ax2.bar(x + w/2, sm_opt, w, label='Smart', color=COL['sm'], alpha=0.8)
    ax2.axhline(100, color='gray', ls='--', lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t[:10] for t in topos], rotation=40, fontsize=6)
    ax2.set_ylabel('Optimality (%)')
    ax2.set_title('BF vs Smart Optimality', fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.set_ylim(0, 115)

    ax3 = fig.add_subplot(333)
    for tau_val in TAU_VALUES:
        tau_opt = []
        for t in topos:
            vals = [r['smart_optimality'] for r in p1 if r['topology'] == t and r['tau'] == tau_val]
            tau_opt.append(np.mean(vals) if vals else 0)
        ax3.plot(x, tau_opt, 'o-', lw=2, ms=5, label=f'τ={tau_val}')
    ax3.axhline(100, color='gray', ls='--', lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([t[:10] for t in topos], rotation=40, fontsize=6)
    ax3.set_ylabel('Smart Optimality (%)')
    ax3.set_title('Optimality vs Tau Value', fontweight='bold')
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(334, projection='3d')
    for t in topos:
        ax4.scatter(
            topo_stats[t]['spd'], topo_stats[t]['opt'], topo_stats[t]['err'],
            s=40, alpha=0.7, label=t[:10])
    ax4.set_xlabel('Speedup', fontsize=8)
    ax4.set_ylabel('Optimality %', fontsize=8)
    ax4.set_zlabel('Error %', fontsize=8)
    ax4.set_title('3D: Speedup × Opt × Error', fontweight='bold', fontsize=9)
    ax4.legend(fontsize=4, ncol=2, loc='upper left')

    ax5 = fig.add_subplot(335)
    errs = [np.mean(topo_stats[t]['err']) for t in topos]
    sorted_idx = np.argsort(errs)
    sorted_topos = [topos[i] for i in sorted_idx]
    sorted_errs = [errs[i] for i in sorted_idx]
    colors_e = ['#2ecc71' if e < 5 else '#f39c12' if e < 40 else '#e74c3c' for e in sorted_errs]
    ax5.barh(range(len(sorted_topos)), sorted_errs, color=colors_e, alpha=0.8)
    ax5.set_yticks(range(len(sorted_topos)))
    ax5.set_yticklabels([t[:12] for t in sorted_topos], fontsize=7)
    ax5.set_xlabel('Error (%)')
    ax5.set_title('Error by Topology (sorted)', fontweight='bold')

    ax6 = fig.add_subplot(336)
    for r in p1:
        ax6.scatter(r['speedup'], r['smart_optimality'], s=15, alpha=0.4,
                    color=COL['blue'])
    ax6.axhline(100, color='gray', ls='--', lw=1)
    ax6.set_xlabel('Speedup')
    ax6.set_ylabel('Smart Optimality (%)')
    ax6.set_title('All Graphs: Speedup vs Optimality', fontweight='bold')

    if p2:
        ax7 = fig.add_subplot(337)
        for r in p2:
            ws = [1] + [p['workers'] for p in r['parallel']]
            spds = [1.0] + [p['speedup'] for p in r['parallel']]
            ax7.plot(ws, spds, 'o-', lw=2, ms=6, label=r['name'])
        max_w = max(p['workers'] for r in p2 for p in r['parallel'])
        ax7.plot([1, max_w], [1, max_w], '--', color='gray', lw=1, label='Ideal')
        ax7.set_xlabel('Workers')
        ax7.set_ylabel('Speedup')
        ax7.set_title('Phase 2: Parallel Speedup', fontweight='bold')
        ax7.legend(fontsize=6)

    if p3:
        metrics, importances, X_te, y_te, y_pred, evals = p3
        ax8 = fig.add_subplot(338)
        ax8.scatter(y_te, y_pred, s=8, alpha=0.4, color=COL['ml'])
        lim = max(max(y_te), max(y_pred)) * 1.1
        ax8.plot([0, lim], [0, lim], '--', color='red', lw=1.5)
        ax8.set_xlabel('Actual BC Reduction (%)')
        ax8.set_ylabel('Predicted')
        ax8.set_title(f'Phase 3: ML (R²={metrics["test_r2"]:.3f})', fontweight='bold')

        ax9 = fig.add_subplot(339)
        top = sorted(importances.items(), key=lambda x: -x[1])[:10]
        fnames = [f[0] for f in top]
        fimps = [f[1] for f in top]
        ax9.barh(range(len(fnames)), fimps, color=COL['sm'], alpha=0.8)
        ax9.set_yticks(range(len(fnames)))
        ax9.set_yticklabels(fnames, fontsize=7)
        ax9.set_xlabel('Importance')
        ax9.set_title('Phase 3: Feature Importance', fontweight='bold')
        ax9.invert_yaxis()

    fig.suptitle('BC Minimization: Complete Research Analysis\n'
                 'BF optimality = 100% (baseline) | Smart = heuristic approximation | '
                 f'τ ∈ {{{", ".join(str(t) for t in TAU_VALUES)}}}',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, 'research_complete.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    generate_tau_comparison(p1)


def generate_tau_comparison(p1):
    """Dedicated figure comparing 3 tau values."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    topos = sorted(set(r['topology'] for r in p1))
    x = np.arange(len(topos))

    for ax_idx, tau_val in enumerate(TAU_VALUES):
        ax = axes[ax_idx]
        bf_reds, sm_reds = [], []
        for t in topos:
            vals_bf = [r['bf_reduction'] for r in p1 if r['topology'] == t and r['tau'] == tau_val]
            vals_sm = [r['smart_reduction'] for r in p1 if r['topology'] == t and r['tau'] == tau_val]
            bf_reds.append(np.mean(vals_bf) * 100 if vals_bf else 0)
            sm_reds.append(np.mean(vals_sm) * 100 if vals_sm else 0)
        w = 0.35
        ax.bar(x - w/2, bf_reds, w, label='BF', color=COL['bf'], alpha=0.8)
        ax.bar(x + w/2, sm_reds, w, label='Smart', color=COL['sm'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([t[:10] for t in topos], rotation=40, fontsize=6)
        ax.set_ylabel('BC Reduction (×100)')
        ax.set_title(f'τ = {tau_val}', fontweight='bold')
        ax.legend(fontsize=7)

    fig.suptitle('BC Reduction Comparison at Different Load-Balance Thresholds (τ)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'tau_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# MAIN
if __name__ == '__main__':
    mp.freeze_support()
    test_mode = '--test' in sys.argv
    print("\n" + "=" * 70)
    print("  BC MINIMIZATION RESEARCH PIPELINE")
    print(f"  Mode: {'TEST' if test_mode else 'FULL'} | Tau: {TAU_VALUES}")
    print("=" * 70)

    p1 = run_phase1(test_mode)
    p2 = None  # run_phase2(test_mode)  -- disabled by user request
    p3 = None  # run_phase3(test_mode)  -- disabled by user request

    if p1:
        generate_figures(p1, p2, p3)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 70)
