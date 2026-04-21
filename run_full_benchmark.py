"""
run_full_benchmark.py — Phase 1: Large-Scale BC-Minimize Benchmark
==================================================================
Tests 100+ graphs across 10 diverse topologies.
Compares Brute Force vs Smart (Hop-Based) Algorithm.
Outputs JSON results, CSV summary, and publication-quality figures.

Usage:
    python run_full_benchmark.py           # Full run (~100+ graphs)
    python run_full_benchmark.py --quick   # Quick test (~20 graphs)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
from collections import defaultdict, deque
from scipy import stats
import time, os, sys, json, csv, itertools, warnings
warnings.filterwarnings('ignore')


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
})

COLORS = dict(
    brute='#D85A30', smart='#1D9E75', neutral='#888780',
    blue='#185FA5', amber='#BA7517', purple='#534AB7',
    ltblue='#85B7EB', ltgreen='#C0DD97', coral='#F0997B',
    teal='#009688', pink='#E91E63',
)



def brandes_bc(G):
    """Compute normalized betweenness centrality (Brandes algorithm)."""
    bc = defaultdict(float)
    n = G.number_of_nodes()
    for s in G.nodes():
        S, P = [], defaultdict(list)
        sigma, dist, delta = defaultdict(float), {}, defaultdict(float)
        sigma[s] = 1.0; dist[s] = 0
        Q = deque([s])
        while Q:
            v = Q.popleft(); S.append(v)
            for w in G.neighbors(v):
                if w not in dist: Q.append(w); dist[w] = dist[v]+1
                if dist[w] == dist[v]+1: sigma[w] += sigma[v]; P[w].append(v)
        while S:
            w = S.pop()
            for v in P[w]: delta[v] += (sigma[v]/sigma[w])*(1+delta[w])
            if w != s: bc[w] += delta[w]
    if n > 2:
        f = 1.0/((n-1)*(n-2))
        for k in bc: bc[k] *= f
    return dict(bc)


def find_hop_candidates(G, target, topk=50):
    """Return edge candidates from hop-1 and hop-2 zones of target, sorted by heuristic score."""
    dist = {}
    Q = deque([target]); dist[target] = 0
    while Q:
        v = Q.popleft()
        if dist[v] >= 2: continue
        for w in G.neighbors(v):
            if w not in dist: dist[w] = dist[v]+1; Q.append(w)
    hop1 = {n for n,d in dist.items() if d == 1}
    hop2 = {n for n,d in dist.items() if d == 2}

    cands = []
    seen = set()
    def try_add(u, w, score):
        if u == w or G.has_edge(u, w): return
        key = (min(u,w), max(u,w))
        if key in seen: return
        seen.add(key)
        cands.append((score, u, w))

    for a in hop2:
        for b in hop2:
            if a < b: try_add(a, b, 2.0)
    for a in hop1:
        for b in hop1:
            if a < b: try_add(a, b, 1.0)
    for a in hop1:
        for b in hop2: try_add(a, b, 0.6)

    cands.sort(reverse=True)
    return [(u,w) for _,u,w in cands[:topk]], hop1, hop2


def brute_force(G, target, tau=0.15):
    """Brute force: try every non-edge."""
    t0 = time.perf_counter()
    bc0 = brandes_bc(G)
    avg = np.mean(list(bc0.values()))
    nodes = list(G.nodes())
    non_edges = [(u,w) for i,u in enumerate(nodes) for w in nodes[i+1:] if not G.has_edge(u,w)]

    best = dict(u=-1, w=-1, red=0, load=0, bc_after=bc0[target])
    for u, w in non_edges:
        G2 = G.copy(); G2.add_edge(u, w)
        bc2 = brandes_bc(G2)
        red = bc0[target] - bc2[target]
        load = max((bc2[n]-bc0[n]) for n in nodes if n != target)
        if load <= tau*avg and red > best['red']:
            best = dict(u=u, w=w, red=red, load=load, bc_after=bc2[target])
    t1 = time.perf_counter()
    return best, bc0, len(non_edges), (t1-t0)*1000


def smart_algorithm(G, target, tau=0.15, topk=50):
    """Smart hop-based candidate selection algorithm."""
    t0 = time.perf_counter()
    bc0 = brandes_bc(G)
    avg = np.mean(list(bc0.values()))
    cands, hop1, hop2 = find_hop_candidates(G, target, topk)

    best = dict(u=-1, w=-1, red=0, load=0, bc_after=bc0[target])
    for u, w in cands:
        G2 = G.copy(); G2.add_edge(u, w)
        bc2 = brandes_bc(G2)
        red = bc0[target] - bc2[target]
        nodes = list(G.nodes())
        load = max((bc2[n]-bc0[n]) for n in nodes if n != target)
        if load <= tau*avg and red > best['red']:
            best = dict(u=u, w=w, red=red, load=load, bc_after=bc2[target])
    t1 = time.perf_counter()
    return best, bc0, len(cands), (t1-t0)*1000, hop1, hop2


def ensure_connected(G):
    """Extract the largest connected component."""
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


def adaptive_topk(G, base_topk=50):
    """Compute adaptive topk based on graph density."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2*m / (n*(n-1)) if n > 1 else 0
    # Denser graphs need more candidates; sparser graphs fewer
    return max(20, min(int(base_topk * (1 + density * 5)), n*(n-1)//4))


def make_erdos_renyi(n, p, seed):
    return ensure_connected(nx.erdos_renyi_graph(n, p, seed=seed))

def make_barabasi_albert(n, m, seed):
    return nx.barabasi_albert_graph(n, m, seed=seed)

def make_watts_strogatz(n, k, p, seed):
    return nx.watts_strogatz_graph(n, k, p, seed=seed)

def make_path(n, seed=None):
    return nx.path_graph(n)

def make_barbell(m, seed=None):
    return nx.barbell_graph(m, max(1, m//3))

def make_star(n, seed=None):
    return nx.star_graph(n-1)

def make_grid_2d(side, seed=None):
    return nx.grid_2d_graph(side, side)

def make_random_tree(n, seed):
    return nx.random_labeled_tree(n, seed=seed)

def make_powerlaw_cluster(n, seed):
    G = nx.powerlaw_cluster_graph(n, 2, 0.3, seed=seed)
    return ensure_connected(G)

def make_caveman(cliques, size, seed=None):
    G = nx.connected_caveman_graph(cliques, size)
    return G


def get_topology_configs(quick=False):
    """Return list of (name, topology_type, generator_fn) tuples. 100+ graphs for full, ~20 for quick."""
    configs = []

    if quick:
        # Quick mode: ~20 graphs
        for n in [30, 50]:
            for seed in [0]:
                configs.append((f'ER_n{n}_p0.12_s{seed}', 'Erdos-Renyi',
                               lambda s=seed, n=n: make_erdos_renyi(n, 0.12, s)))
        for n in [30, 50]:
            for seed in [0]:
                configs.append((f'BA_n{n}_m2_s{seed}', 'Barabasi-Albert',
                               lambda s=seed, n=n: make_barabasi_albert(n, 2, s)))
        for n in [30, 50]:
            for seed in [0]:
                configs.append((f'WS_n{n}_k4_p0.3_s{seed}', 'Watts-Strogatz',
                               lambda s=seed, n=n: make_watts_strogatz(n, 4, 0.3, s)))
        for n in [15, 20]:
            configs.append((f'Path_n{n}', 'Path', lambda n=n: make_path(n)))
        for m in [7, 10]:
            configs.append((f'Barbell_m{m}', 'Barbell', lambda m=m: make_barbell(m)))
        for n in [15, 20]:
            configs.append((f'Star_n{n}', 'Star', lambda n=n: make_star(n)))
        for side in [5, 6]:
            configs.append((f'Grid_{side}x{side}', 'Grid2D',
                           lambda s=side: make_grid_2d(s)))
        for n in [20, 30]:
            configs.append((f'Tree_n{n}_s0', 'RandomTree',
                           lambda n=n: make_random_tree(n, 0)))
        for n in [30, 50]:
            configs.append((f'PLCluster_n{n}_s0', 'PowerlawCluster',
                           lambda n=n: make_powerlaw_cluster(n, 0)))
        for cl in [4, 5]:
            configs.append((f'Caveman_{cl}x5', 'Caveman',
                           lambda cl=cl: make_caveman(cl, 5)))
    else:
        # Full mode: 100+ graphs
        # Erdos-Renyi: 3 sizes × 3 probs × 2 seeds = 18
        for n in [30, 50, 70]:
            for p in [0.08, 0.12, 0.15]:
                for seed in [0, 42]:
                    configs.append((f'ER_n{n}_p{p}_s{seed}', 'Erdos-Renyi',
                                   lambda s=seed, n=n, p=p: make_erdos_renyi(n, p, s)))

        # Barabasi-Albert: 4 sizes × 2 m × 2 seeds = 16
        for n in [30, 50, 70, 100]:
            for m in [2, 3]:
                for seed in [0, 42]:
                    configs.append((f'BA_n{n}_m{m}_s{seed}', 'Barabasi-Albert',
                                   lambda s=seed, n=n, m=m: make_barabasi_albert(n, m, s)))

        # Watts-Strogatz: 3 sizes × 2 k × 2 p × 2 seeds = 24
        for n in [30, 50, 70]:
            for k in [4, 6]:
                for p in [0.2, 0.4]:
                    for seed in [0, 42]:
                        configs.append((f'WS_n{n}_k{k}_p{p}_s{seed}', 'Watts-Strogatz',
                                       lambda s=seed, n=n, k=k, p=p: make_watts_strogatz(n, k, p, s)))

        # Path graphs: 4 sizes
        for n in [15, 20, 25, 30]:
            configs.append((f'Path_n{n}', 'Path', lambda n=n: make_path(n)))

        # Barbell: 4 sizes
        for m in [5, 7, 10, 12]:
            configs.append((f'Barbell_m{m}', 'Barbell', lambda m=m: make_barbell(m)))

        # Star: 4 sizes
        for n in [15, 20, 25, 30]:
            configs.append((f'Star_n{n}', 'Star', lambda n=n: make_star(n)))

        # Grid 2D: 4 sizes
        for side in [5, 6, 7, 8]:
            configs.append((f'Grid_{side}x{side}', 'Grid2D',
                           lambda s=side: make_grid_2d(s)))

        # Random Trees: 4 sizes × 2 seeds = 8
        for n in [20, 30, 40, 50]:
            for seed in [0, 42]:
                configs.append((f'Tree_n{n}_s{seed}', 'RandomTree',
                               lambda n=n, s=seed: make_random_tree(n, s)))

        # Powerlaw Cluster: 3 sizes × 2 seeds = 6
        for n in [30, 50, 70]:
            for seed in [0, 42]:
                configs.append((f'PLCluster_n{n}_s{seed}', 'PowerlawCluster',
                               lambda n=n, s=seed: make_powerlaw_cluster(n, s)))

        # Connected Caveman: 3 cliques × 3 sizes = 9
        for cliques in [4, 5, 6]:
            for size in [5, 6, 7]:
                configs.append((f'Caveman_{cliques}x{size}', 'Caveman',
                               lambda cl=cliques, sz=size: make_caveman(cl, sz)))

    return configs


# 
# Main Benchmark Runner
# 

def run_single_benchmark(name, topo_type, gen_fn, tau=0.20, base_topk=50):
    """Run brute-force and smart algorithm on a single graph."""
    try:
        G = gen_fn()
        G = ensure_connected(G)
        G = nx.convert_node_labels_to_integers(G)

        n = G.number_of_nodes()
        m = G.number_of_edges()

        if n < 4:
            return None

        bc0 = brandes_bc(G)
        target = max(bc0, key=bc0.get)
        topk = adaptive_topk(G, base_topk)

        # Brute force
        b_res, b_bc0, b_cands, b_time = brute_force(G, target, tau=tau)

        # Smart algorithm
        s_res, s_bc0, s_cands, s_time, h1, h2 = smart_algorithm(G, target, tau=tau, topk=topk)

        if b_res['u'] < 0 or b_res['red'] <= 1e-9:
            return None

        opt = (s_res['red'] / b_res['red'] * 100) if s_res['u'] >= 0 and b_res['red'] > 1e-9 else 0
        spd = b_time / max(s_time, 0.001)
        red_bf = b_res['red'] / bc0[target] * 100 if bc0[target] > 0 else 0
        red_sm = s_res['red'] / bc0[target] * 100 if s_res['u'] >= 0 and bc0[target] > 0 else 0
        cand_ratio = s_cands / max(b_cands, 1) * 100

        same_edge = (
            (b_res['u'] == s_res['u'] and b_res['w'] == s_res['w']) or
            (b_res['u'] == s_res['w'] and b_res['w'] == s_res['u'])
        )

        return {
            'name': name,
            'topology': topo_type,
            'nodes': n,
            'edges': m,
            'density': round(2*m/(n*(n-1)), 4) if n > 1 else 0,
            'target_node': target,
            'target_bc': round(bc0[target], 6),
            'bf_time_ms': round(b_time, 2),
            'smart_time_ms': round(s_time, 2),
            'speedup': round(spd, 2),
            'bf_candidates': b_cands,
            'smart_candidates': s_cands,
            'candidate_ratio_pct': round(cand_ratio, 2),
            'bf_reduction_pct': round(red_bf, 2),
            'smart_reduction_pct': round(red_sm, 2),
            'optimality_pct': round(opt, 2),
            'same_edge': same_edge,
            'bf_edge': [b_res['u'], b_res['w']],
            'smart_edge': [s_res['u'], s_res['w']],
            'topk_used': topk,
        }

    except Exception as e:
        print(f"    ✗ Error on {name}: {str(e)[:60]}")
        return None


def run_benchmark(quick=False):
    """Run the full benchmark."""
    configs = get_topology_configs(quick=quick)
    total = len(configs)

    print("=" * 80)
    print(f"  BC-MINIMIZE BENCHMARK — {'QUICK' if quick else 'FULL'} MODE")
    print(f"  Total graph configurations: {total}")
    print("=" * 80)

    results = []
    start_time = time.perf_counter()

    for i, (name, topo_type, gen_fn) in enumerate(configs):
        pct = (i+1) / total * 100
        print(f"  [{i+1:3d}/{total}] ({pct:5.1f}%) {name}...", end=" ", flush=True)

        result = run_single_benchmark(name, topo_type, gen_fn)
        if result:
            results.append(result)
            print(f"✓ spd={result['speedup']:.1f}x opt={result['optimality_pct']:.1f}%"
                  f" t_bf={result['bf_time_ms']:.0f}ms t_sm={result['smart_time_ms']:.0f}ms")
        else:
            print("⚠ skipped (no valid solution)")

    elapsed = time.perf_counter() - start_time
    print(f"\n{'=' * 80}")
    print(f"  Benchmark complete: {len(results)}/{total} graphs succeeded")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'=' * 80}\n")

    return results


# 
# Save Results
# 

def save_results(results):
    """Save results as JSON and CSV."""
    # JSON
    json_path = os.path.join(RESULTS_DIR, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON results saved to: {json_path}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, 'benchmark_results.csv')
    if results:
        keys = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                row = dict(r)
                row['bf_edge'] = str(row['bf_edge'])
                row['smart_edge'] = str(row['smart_edge'])
                writer.writerow(row)
    print(f"✓ CSV results saved to: {csv_path}")

    # Summary by topology
    topo_stats = {}
    for r in results:
        t = r['topology']
        if t not in topo_stats:
            topo_stats[t] = {'speedups': [], 'opts': [], 'bf_times': [], 'sm_times': [],
                            'bf_reds': [], 'sm_reds': [], 'same_edge': [], 'nodes': []}
        topo_stats[t]['speedups'].append(r['speedup'])
        topo_stats[t]['opts'].append(r['optimality_pct'])
        topo_stats[t]['bf_times'].append(r['bf_time_ms'])
        topo_stats[t]['sm_times'].append(r['smart_time_ms'])
        topo_stats[t]['bf_reds'].append(r['bf_reduction_pct'])
        topo_stats[t]['sm_reds'].append(r['smart_reduction_pct'])
        topo_stats[t]['same_edge'].append(r['same_edge'])
        topo_stats[t]['nodes'].append(r['nodes'])

    summary = {}
    for t, s in topo_stats.items():
        summary[t] = {
            'count': len(s['speedups']),
            'speedup_mean': round(np.mean(s['speedups']), 2),
            'speedup_std': round(np.std(s['speedups']), 2),
            'optimality_mean': round(np.mean(s['opts']), 2),
            'optimality_std': round(np.std(s['opts']), 2),
            'bf_time_mean_ms': round(np.mean(s['bf_times']), 2),
            'smart_time_mean_ms': round(np.mean(s['sm_times']), 2),
            'bf_reduction_mean': round(np.mean(s['bf_reds']), 2),
            'smart_reduction_mean': round(np.mean(s['sm_reds']), 2),
            'same_edge_pct': round(sum(s['same_edge']) / len(s['same_edge']) * 100, 1),
            'avg_nodes': round(np.mean(s['nodes']), 1),
        }

    summary_path = os.path.join(RESULTS_DIR, 'topology_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Topology summary saved to: {summary_path}")

    return summary


# 
# Figure Generation
# 

def generate_figures(results, summary):
    """Generate all publication-quality figures."""
    if not results:
        print("No results to plot.")
        return

    print("\nGenerating figures...")

    # Figure 1: Speedup vs Optimality by Topology (Main Result)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    topos = sorted(summary.keys())
    topo_colors = {}
    cmap = plt.cm.Set3(np.linspace(0, 1, len(topos)))
    for i, t in enumerate(topos):
        topo_colors[t] = cmap[i]

    # Panel 1: Speedup bar chart
    ax = axes[0]
    x = np.arange(len(topos))
    means = [summary[t]['speedup_mean'] for t in topos]
    stds = [summary[t]['speedup_std'] for t in topos]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=[topo_colors[t] for t in topos],
                  edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n') for t in topos], rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Speedup (BF / Smart)')
    ax.set_title('Mean Speedup by Topology', fontweight='bold')
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}×', ha='center', fontsize=7, fontweight='bold')

    # Panel 2: Optimality bar chart
    ax = axes[1]
    means_o = [summary[t]['optimality_mean'] for t in topos]
    stds_o = [summary[t]['optimality_std'] for t in topos]
    bars = ax.bar(x, means_o, yerr=stds_o, capsize=4, color=[topo_colors[t] for t in topos],
                  edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.axhline(100, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n') for t in topos], rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Optimality (% of BF reduction)')
    ax.set_title('Mean Optimality by Topology', fontweight='bold')
    ax.set_ylim(0, 115)
    for bar, val in zip(bars, means_o):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=7, fontweight='bold')

    # Panel 3: Speedup vs Optimality scatter
    ax = axes[2]
    for t in topos:
        t_results = [r for r in results if r['topology'] == t]
        spds = [r['speedup'] for r in t_results]
        opts = [r['optimality_pct'] for r in t_results]
        ax.scatter(spds, opts, s=50, alpha=0.7, label=t, edgecolors='none')
    ax.set_xlabel('Speedup (BF / Smart)')
    ax.set_ylabel('Optimality (%)')
    ax.set_title('Speedup vs Optimality Trade-off', fontweight='bold')
    ax.legend(fontsize=6, loc='lower right', ncol=2)
    ax.axhline(100, color='gray', ls='--', lw=0.8, alpha=0.5)

    plt.suptitle(f'Phase 1: Large-Scale Benchmark ({len(results)} graphs, {len(topos)} topologies)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'fig1_speedup_optimality.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Figure 1 saved: {fig_path}")

    # Figure 2: Scaling Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: BF time vs n
    ax = axes[0]
    for t in topos:
        t_results = sorted([r for r in results if r['topology'] == t], key=lambda r: r['nodes'])
        if len(t_results) > 2:
            ns = [r['nodes'] for r in t_results]
            ts = [r['bf_time_ms'] for r in t_results]
            ax.plot(ns, ts, 'o-', label=t, alpha=0.7, ms=4, lw=1.5)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Brute Force Time (ms)')
    ax.set_title('BF Runtime Scaling', fontweight='bold')
    ax.legend(fontsize=6, ncol=2)

    # Panel 2: Speedup vs n
    ax = axes[1]
    for t in topos:
        t_results = sorted([r for r in results if r['topology'] == t], key=lambda r: r['nodes'])
        if len(t_results) > 2:
            ns = [r['nodes'] for r in t_results]
            spds = [r['speedup'] for r in t_results]
            ax.plot(ns, spds, 'o-', label=t, alpha=0.7, ms=4, lw=1.5)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Speedup (BF / Smart)')
    ax.set_title('Speedup vs Graph Size', fontweight='bold')
    ax.legend(fontsize=6, ncol=2)
    ax.axhline(1, color='gray', ls='--', lw=0.8)

    # Panel 3: Candidate ratio vs n
    ax = axes[2]
    for t in topos:
        t_results = sorted([r for r in results if r['topology'] == t], key=lambda r: r['nodes'])
        if len(t_results) > 2:
            ns = [r['nodes'] for r in t_results]
            crs = [r['candidate_ratio_pct'] for r in t_results]
            ax.plot(ns, crs, 'o-', label=t, alpha=0.7, ms=4, lw=1.5)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Smart / BF Candidates (%)')
    ax.set_title('Candidate Reduction vs Size', fontweight='bold')
    ax.legend(fontsize=6, ncol=2)

    plt.suptitle('Scaling Behavior Across Topologies', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'fig2_scaling.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Figure 2 saved: {fig_path}")

    # Figure 3: Box Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Speedup box plots
    ax = axes[0]
    data_spd = [[r['speedup'] for r in results if r['topology'] == t] for t in topos]
    bp = ax.boxplot(data_spd, labels=[t[:12] for t in topos], patch_artist=True, notch=False)
    for patch, col in zip(bp['boxes'], [topo_colors[t] for t in topos]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for med in bp['medians']: med.set_color('black'); med.set_linewidth(2)
    ax.set_ylabel('Speedup (BF / Smart)')
    ax.set_title('Speedup Distribution', fontweight='bold')
    ax.tick_params(axis='x', rotation=35, labelsize=7)

    # Panel 2: Optimality box plots
    ax = axes[1]
    data_opt = [[r['optimality_pct'] for r in results if r['topology'] == t] for t in topos]
    bp = ax.boxplot(data_opt, labels=[t[:12] for t in topos], patch_artist=True, notch=False)
    for patch, col in zip(bp['boxes'], [topo_colors[t] for t in topos]):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for med in bp['medians']: med.set_color('black'); med.set_linewidth(2)
    ax.axhline(100, color='gray', ls='--', lw=1)
    ax.set_ylabel('Optimality (%)')
    ax.set_title('Optimality Distribution', fontweight='bold')
    ax.tick_params(axis='x', rotation=35, labelsize=7)

    # Panel 3: BC Reduction comparison
    ax = axes[2]
    all_bf = [r['bf_reduction_pct'] for r in results]
    all_sm = [r['smart_reduction_pct'] for r in results]
    ax.scatter(all_bf, all_sm, s=30, alpha=0.5, color=COLORS['blue'], edgecolors='none')
    lim = max(max(all_bf + [1]), max(all_sm + [1])) * 1.05
    ax.plot([0, lim], [0, lim], '--', color='red', lw=1.5, label='y=x (perfect)')
    if len(all_bf) > 2:
        r_val, p_val = stats.pearsonr(all_bf, all_sm)
        ax.set_title(f'BF vs Smart Reduction\nPearson r={r_val:.4f}', fontweight='bold')
    else:
        ax.set_title('BF vs Smart Reduction', fontweight='bold')
    ax.set_xlabel('Brute Force Reduction (%)')
    ax.set_ylabel('Smart Reduction (%)')
    ax.legend(fontsize=9)

    plt.suptitle('Distribution Analysis', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'fig3_distributions.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Figure 3 saved: {fig_path}")

    # Figure 4: Summary Table as Image
    fig, ax = plt.subplots(figsize=(14, max(4, len(topos)*0.6 + 2)))
    ax.axis('off')

    col_labels = ['Topology', 'Count', 'Avg Nodes', 'Speedup', 'Optimality %',
                  'BF Time (ms)', 'Smart Time (ms)', 'Same Edge %']
    table_data = []
    for t in topos:
        s = summary[t]
        table_data.append([
            t, s['count'], f"{s['avg_nodes']:.0f}",
            f"{s['speedup_mean']:.1f}± {s['speedup_std']:.1f}",
            f"{s['optimality_mean']:.1f}± {s['optimality_std']:.1f}",
            f"{s['bf_time_mean_ms']:.0f}", f"{s['smart_time_mean_ms']:.0f}",
            f"{s['same_edge_pct']:.0f}%",
        ])

    # Grand totals
    all_spd = [r['speedup'] for r in results]
    all_opt = [r['optimality_pct'] for r in results]
    all_bf_t = [r['bf_time_ms'] for r in results]
    all_sm_t = [r['smart_time_ms'] for r in results]
    all_same = [r['same_edge'] for r in results]
    table_data.append([
        'OVERALL', len(results), f"{np.mean([r['nodes'] for r in results]):.0f}",
        f"{np.mean(all_spd):.1f}± {np.std(all_spd):.1f}",
        f"{np.mean(all_opt):.1f}± {np.std(all_opt):.1f}",
        f"{np.mean(all_bf_t):.0f}", f"{np.mean(all_sm_t):.0f}",
        f"{sum(all_same)/len(all_same)*100:.0f}%",
    ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Style last row (totals)
    for j in range(len(col_labels)):
        table[len(table_data), j].set_facecolor('#E8F5E9')
        table[len(table_data), j].set_text_props(fontweight='bold')

    ax.set_title(f'Benchmark Summary Table ({len(results)} graphs)', fontsize=13,
                 fontweight='bold', pad=20)
    fig_path = os.path.join(RESULTS_DIR, 'fig4_summary_table.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Figure 4 saved: {fig_path}")

    # Figure 5: Heatmap of Speedup vs Optimality
    fig, ax = plt.subplots(figsize=(10, 7))

    spd_bins = np.arange(0, max([r['speedup'] for r in results]) + 5, 5)
    opt_bins = np.arange(0, 110, 10)
    spds = [r['speedup'] for r in results]
    opts = [r['optimality_pct'] for r in results]

    ax.scatter(spds, opts, c=[list(topo_colors.values())[topos.index(r['topology'])] for r in results],
               s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    for t in topos:
        t_results = [r for r in results if r['topology'] == t]
        if t_results:
            mean_s = np.mean([r['speedup'] for r in t_results])
            mean_o = np.mean([r['optimality_pct'] for r in t_results])
            ax.annotate(t[:10], (mean_s, mean_o), fontsize=6, fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    ax.set_xlabel('Speedup (BF / Smart)', fontsize=12)
    ax.set_ylabel('Optimality (%)', fontsize=12)
    ax.set_title(f'Speedup vs Optimality Landscape ({len(results)} graphs)',
                 fontsize=13, fontweight='bold')
    ax.axhline(100, color='gray', ls='--', lw=0.8, alpha=0.5)

    fig_path = os.path.join(RESULTS_DIR, 'fig5_landscape.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"  ✓ Figure 5 saved: {fig_path}")

    print(f"\nAll figures saved to: {RESULTS_DIR}/")


# 
# Main
# 

if __name__ == '__main__':
    quick = '--quick' in sys.argv
    results = run_benchmark(quick=quick)
    summary = save_results(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("  TOPOLOGY SUMMARY")
    print("=" * 80)
    print(f"{'Topology':<22} {'Count':>5} {'Speedup':>12} {'Optimality':>12} {'Same Edge':>10}")
    print("-" * 80)
    for t, s in sorted(summary.items()):
        print(f"{t:<22} {s['count']:>5} {s['speedup_mean']:>8.1f}× ± {s['speedup_std']:<4.1f}"
              f"  {s['optimality_mean']:>7.1f}% ± {s['optimality_std']:<4.1f}"
              f"  {s['same_edge_pct']:>7.0f}%")

    all_spd = np.mean([r['speedup'] for r in results])
    all_opt = np.mean([r['optimality_pct'] for r in results])
    print("-" * 80)
    print(f"{'OVERALL':<22} {len(results):>5} {all_spd:>8.1f}×          {all_opt:>7.1f}%          "
          f"  {sum(r['same_edge'] for r in results)/len(results)*100:>5.0f}%")
    print("=" * 80)

    generate_figures(results, summary)
    print(f"\nDone! Results in: {RESULTS_DIR}/")
