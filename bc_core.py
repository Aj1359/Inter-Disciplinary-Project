"""
bc_core.py  —  Betweenness Centrality Minimization: Core Algorithms
===================================================================
Shared module imported by all topology analysis scripts.
Contains:
  - brandes_bc()          Exact BC via Brandes O(nm)
  - find_hop_candidates() 1-hop/2-hop complement-graph candidates
  - brute_force()         O(n²·nm) exhaustive search
  - smart_algorithm()     O(K·nm) hop-based search
  - analyze_topology()    Full analysis + report + plots for one topology
"""

import os, json, csv, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict, deque


# 
# 1.  BRANDES BETWEENNESS CENTRALITY  —  O(nm)
# 

def brandes_bc(G):
    """
    Compute normalized betweenness centrality for every node.

    Algorithm (Brandes 2001):
      For each source s:
        1. BFS to build shortest-path DAG, counting sigma(s,v)
        2. Backward pass to accumulate delta(v|s)
      BC(v) = sum_s delta(v|s), normalised by 1/[(n-1)(n-2)]
    """
    bc = defaultdict(float)
    n  = G.number_of_nodes()

    for s in G.nodes():
        S, P           = [], defaultdict(list)
        sigma, dist, delta = defaultdict(float), {}, defaultdict(float)
        sigma[s] = 1.0
        dist[s]  = 0
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


# 
# 2.  HOP-ZONE CANDIDATE GENERATION
# 

def find_hop_candidates(G, target, topk=50):
    """
    Return the top-K non-edges most likely to reduce BC(target).

    Structural Lemma: only edges in the 1-hop or 2-hop complement
    neighborhood of target* can create bypass shortest paths.

    Three classes (scored highest→lowest):
      Class 1  2-hop × 2-hop  score 2.0  (direct cross-cluster bypass)
      Class 2  1-hop × 1-hop  score 1.0  (triangle-closing around target)
      Class 3  1-hop × 2-hop  score 0.6  (mixed, useful in sparse graphs)

    Returns (candidates, hop1_set, hop2_set)
    """
    dist = {}
    Q = deque([target])
    dist[target] = 0
    while Q:
        v = Q.popleft()
        if dist[v] >= 2:
            continue
        for w in G.neighbors(v):
            if w not in dist:
                dist[w] = dist[v] + 1
                Q.append(w)

    hop1 = {n for n, d in dist.items() if d == 1}
    hop2 = {n for n, d in dist.items() if d == 2}

    cands = []
    seen  = set()

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
    return [(u, w) for _, u, w in cands[:topk]], hop1, hop2


# 
# 3.  BRUTE FORCE  —  O(n² · nm)
# 

def brute_force(G, target, tau=0.15):
    """
    Exhaustively evaluate every non-edge in the complement graph.
    Returns (best_result_dict, bc0, n_candidates, time_ms).
    """
    t0    = time.perf_counter()
    bc0   = brandes_bc(G)
    avg   = np.mean(list(bc0.values()))
    nodes = list(G.nodes())

    non_edges = [
        (u, w)
        for i, u in enumerate(nodes)
        for w in nodes[i + 1:]
        if not G.has_edge(u, w)
    ]

    best = dict(u=-1, w=-1, red=0, load=0, bc_after=bc0[target])

    for u, w in non_edges:
        G2  = G.copy()
        G2.add_edge(u, w)
        bc2 = brandes_bc(G2)
        red  = bc0[target] - bc2[target]
        load = max((bc2[n] - bc0[n]) for n in nodes if n != target)

        if load <= tau * avg and red > best['red']:
            best = dict(u=u, w=w, red=red, load=load,
                        bc_after=bc2[target], bc2=bc2)

    t1 = time.perf_counter()
    return best, bc0, len(non_edges), (t1 - t0) * 1000


# 
# 4.  SMART (HOP-BASED) ALGORITHM  —  O(K · nm)
# 

def smart_algorithm(G, target, tau=0.15, topk=50):
    """
    Evaluate only top-K candidates from the hop zones.
    Returns (best_result_dict, bc0, n_candidates, time_ms, hop1, hop2).
    """
    t0    = time.perf_counter()
    bc0   = brandes_bc(G)
    avg   = np.mean(list(bc0.values()))
    cands, hop1, hop2 = find_hop_candidates(G, target, topk)
    nodes = list(G.nodes())

    best = dict(u=-1, w=-1, red=0, load=0, bc_after=bc0[target])

    for u, w in cands:
        G2  = G.copy()
        G2.add_edge(u, w)
        bc2  = brandes_bc(G2)
        red  = bc0[target] - bc2[target]
        load = max((bc2[n] - bc0[n]) for n in nodes if n != target)

        if load <= tau * avg and red > best['red']:
            best = dict(u=u, w=w, red=red, load=load,
                        bc_after=bc2[target], bc2=bc2)

    t1 = time.perf_counter()
    return best, bc0, len(cands), (t1 - t0) * 1000, hop1, hop2


# 
# 5.  SINGLE-TOPOLOGY ANALYSIS RUNNER
# 

def analyze_topology(topology_name, graph_generator,
                     output_dir, num_trials=10, tau=0.20, topk=50):
    """
    Run brute-force vs smart comparison for one topology.

    Saves:
      results.json         full trial data
      results.csv          tabular summary
      analysis_plots.png   6-panel comparison figure
      REPORT.txt           human-readable report
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'topology':   topology_name,
        'num_trials': num_trials,
        'trials':     [],
        'summary':    {},
    }

    times_bf, times_smart   = [], []
    speedups, opts          = [], []
    reductions_bf, reductions_smart = [], []
    cand_ratios             = []

    print(f"\n{'='*70}")
    print(f"  Analyzing: {topology_name}")
    print(f"{'='*70}")

    for seed in range(num_trials):
        try:
            print(f"  Trial {seed+1}/{num_trials}...", end=' ', flush=True)
            G = graph_generator(seed)

            if not nx.is_connected(G):
                G = G.subgraph(
                    max(nx.connected_components(G), key=len)).copy()
                G = nx.convert_node_labels_to_integers(G)

            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            bc0     = brandes_bc(G)
            target  = max(bc0, key=bc0.get)

            b_res, _, b_cands, b_time = brute_force(G, target, tau=tau)
            times_bf.append(b_time)

            s_res, _, s_cands, s_time, h1, h2 = smart_algorithm(
                G, target, tau=tau, topk=topk)
            times_smart.append(s_time)

            if b_res['u'] >= 0 and b_res['red'] > 1e-9:
                opt      = (s_res['red'] / b_res['red'] * 100
                            if s_res['u'] >= 0 else 0)
                spd      = b_time / max(s_time, 0.001)
                red_bf   = (b_res['red'] / bc0[target] * 100
                            if bc0[target] > 0 else 0)
                red_sm   = (s_res['red'] / bc0[target] * 100
                            if s_res['u'] >= 0 else 0)
                c_ratio  = s_cands / max(b_cands, 1) * 100
                same_edge = ({b_res['u'], b_res['w']} ==
                             {s_res['u'], s_res['w']})

                speedups.append(spd)
                opts.append(opt)
                reductions_bf.append(red_bf)
                reductions_smart.append(red_sm)
                cand_ratios.append(c_ratio)

                results['trials'].append({
                    'seed':    seed,
                    'nodes':   n_nodes,
                    'edges':   n_edges,
                    'target':  target,
                    'target_bc': bc0[target],
                    'brute_force': {
                        'time_ms':        b_time,
                        'candidates_tried': b_cands,
                        'edge':           [b_res['u'], b_res['w']],
                        'reduction_pct':  red_bf,
                    },
                    'smart': {
                        'time_ms':        s_time,
                        'candidates_tried': s_cands,
                        'edge':           [s_res['u'], s_res['w']],
                        'reduction_pct':  red_sm,
                    },
                    'metrics': {
                        'speedup':          spd,
                        'optimality_pct':   opt,
                        'candidate_ratio_pct': c_ratio,
                        'same_edge_found':  same_edge,
                    },
                })
                print(f"speedup={spd:.1f}x  opt={opt:.1f}%  same={same_edge}")
            else:
                print("no valid solution")

        except Exception as e:
            print(f"ERROR: {e}")

    # Summary statistics
    if speedups:
        results['summary'] = {
            'speedup': {
                'mean': float(np.mean(speedups)),
                'std':  float(np.std(speedups)),
                'min':  float(np.min(speedups)),
                'max':  float(np.max(speedups)),
            },
            'optimality_pct': {
                'mean': float(np.mean(opts)),
                'std':  float(np.std(opts)),
                'min':  float(np.min(opts)),
                'max':  float(np.max(opts)),
            },
            'time_ms': {
                'brute_force_mean': float(np.mean(times_bf)),
                'brute_force_std':  float(np.std(times_bf)),
                'smart_mean':       float(np.mean(times_smart)),
                'smart_std':        float(np.std(times_smart)),
            },
            'reduction_pct': {
                'brute_force_mean': float(np.mean(reductions_bf)),
                'smart_mean':       float(np.mean(reductions_smart)),
            },
            'candidates': {
                'ratio_pct_mean': float(np.mean(cand_ratios)),
            },
            'solutions_identical':  sum(
                1 for t in results['trials']
                if t['metrics']['same_edge_found']),
            'total_valid_trials': len(results['trials']),
        }

    # Persist
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Trial', 'Nodes', 'Edges',
                    'BF_Time_ms', 'BF_Red%',
                    'Smart_Time_ms', 'Smart_Red%',
                    'Speedup', 'Optimality%', 'Same_Edge'])
        for i, t in enumerate(results['trials']):
            w.writerow([
                i + 1, t['nodes'], t['edges'],
                round(t['brute_force']['time_ms'], 2),
                round(t['brute_force']['reduction_pct'], 2),
                round(t['smart']['time_ms'], 2),
                round(t['smart']['reduction_pct'], 2),
                round(t['metrics']['speedup'], 2),
                round(t['metrics']['optimality_pct'], 2),
                t['metrics']['same_edge_found'],
            ])

    if results['summary']:
        _plot_topology(results, output_dir)
        _report_topology(results, output_dir)
    else:
        print("  No valid trials — skipping plots/report")
    return results


# Private helpers

def _plot_topology(results, output_dir):
    if not results['trials']:
        return
    trials = results['trials']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{results["topology"]} — Detailed Analysis',
                 fontsize=14, fontweight='bold')

    bf_times = [t['brute_force']['time_ms'] for t in trials]
    sm_times = [t['smart']['time_ms'] for t in trials]
    x = np.arange(len(trials)); w = 0.35

    axes[0, 0].bar(x - w/2, bf_times, w, label='Brute Force',
                   alpha=0.85, color='#D85A30', edgecolor='white')
    axes[0, 0].bar(x + w/2, sm_times, w, label='Smart',
                   alpha=0.85, color='#1D9E75', edgecolor='white')
    axes[0, 0].set(xlabel='Trial', ylabel='Time (ms)',
                   title='Runtime Comparison')
    axes[0, 0].legend(); axes[0, 0].grid(axis='y', alpha=0.3)

    speedups = [t['metrics']['speedup'] for t in trials]
    axes[0, 1].plot(speedups, 'o-', color='#185FA5', lw=2, ms=8)
    axes[0, 1].axhline(np.mean(speedups), color='red', ls='--', lw=1.5,
                        label=f'Mean: {np.mean(speedups):.1f}×')
    axes[0, 1].set(xlabel='Trial', ylabel='Speedup',
                   title='Speedup per Trial')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    opts = [t['metrics']['optimality_pct'] for t in trials]
    axes[0, 2].plot(opts, 's-', color='#534AB7', lw=2, ms=8)
    axes[0, 2].axhline(100, color='green', ls='--', lw=1.5,
                        label='Perfect (100%)')
    axes[0, 2].axhline(np.mean(opts), color='red', ls='--', lw=1.5,
                        label=f'Mean: {np.mean(opts):.1f}%')
    axes[0, 2].set(xlabel='Trial', ylabel='Optimality (%)',
                   title='Solution Quality per Trial',
                   ylim=(max(0, min(opts) - 5), 105))
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    bf_reds = [t['brute_force']['reduction_pct'] for t in trials]
    sm_reds = [t['smart']['reduction_pct'] for t in trials]
    axes[1, 0].bar(x - w/2, bf_reds, w, label='Brute Force',
                   alpha=0.85, color='#D85A30', edgecolor='white')
    axes[1, 0].bar(x + w/2, sm_reds, w, label='Smart',
                   alpha=0.85, color='#1D9E75', edgecolor='white')
    axes[1, 0].set(xlabel='Trial', ylabel='BC Reduction (%)',
                   title='BC Reduction Comparison')
    axes[1, 0].legend(); axes[1, 0].grid(axis='y', alpha=0.3)

    bf_cands = [t['brute_force']['candidates_tried'] for t in trials]
    sm_cands = [t['smart']['candidates_tried'] for t in trials]
    ratios   = [100 * s / max(b, 1) for s, b in zip(sm_cands, bf_cands)]
    axes[1, 1].plot(ratios, 'D-', color='#BA7517', lw=2, ms=8)
    axes[1, 1].axhline(np.mean(ratios), color='red', ls='--', lw=1.5,
                        label=f'Mean: {np.mean(ratios):.1f}%')
    axes[1, 1].set(xlabel='Trial',
                   ylabel='Smart / Brute Candidates (%)',
                   title='Candidate Count Reduction')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    same = [1 if t['metrics']['same_edge_found'] else 0 for t in trials]
    bar_c = ['#1D9E75' if v else '#D85A30' for v in same]
    axes[1, 2].bar(range(len(trials)), same, color=bar_c,
                   alpha=0.85, edgecolor='black')
    axes[1, 2].set(xlabel='Trial', ylabel='Same Edge?',
                   title=f'Solution Match ({sum(same)}/{len(trials)})',
                   ylim=(-0.1, 1.2), yticks=[0, 1],
                   yticklabels=['No', 'Yes'])
    axes[1, 2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_plots.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved")


def _report_topology(results, output_dir):
    s   = results['summary']
    rep = []
    rep.append('=' * 70)
    rep.append(f"TOPOLOGY ANALYSIS: {results['topology']}")
    rep.append('=' * 70)
    rep.append(f"Trials: {s['total_valid_trials']}  |  "
               f"Same edge: {s['solutions_identical']}")
    rep.append('')
    rep.append('PERFORMANCE METRICS')
    rep.append('-' * 70)
    rep.append(f"Speedup   mean={s['speedup']['mean']:.2f}×  "
               f"std={s['speedup']['std']:.2f}  "
               f"range=[{s['speedup']['min']:.1f}×,{s['speedup']['max']:.1f}×]")
    rep.append(f"Optimality mean={s['optimality_pct']['mean']:.2f}%  "
               f"std={s['optimality_pct']['std']:.2f}")
    rep.append(f"BF time   {s['time_ms']['brute_force_mean']:.2f} ± "
               f"{s['time_ms']['brute_force_std']:.2f} ms")
    rep.append(f"Smart time {s['time_ms']['smart_mean']:.2f} ± "
               f"{s['time_ms']['smart_std']:.2f} ms")
    rep.append(f"BF reduction   {s['reduction_pct']['brute_force_mean']:.2f}%")
    rep.append(f"Smart reduction {s['reduction_pct']['smart_mean']:.2f}%")
    rep.append(f"Cand ratio     {s['candidates']['ratio_pct_mean']:.2f}%")
    rep.append('')
    rep.append('TRIAL DETAILS')
    rep.append('-' * 70)
    rep.append(f"{'#':<4} {'n':<6} {'Spd':<8} {'Opt%':<8} "
               f"{'BF%':<8} {'SM%':<8} {'Match'}")
    rep.append('-' * 70)
    for i, t in enumerate(results['trials']):
        m = t['metrics']
        rep.append(
            f"{i+1:<4} {t['nodes']:<6} {m['speedup']:<8.2f} "
            f"{m['optimality_pct']:<8.1f} "
            f"{t['brute_force']['reduction_pct']:<8.2f} "
            f"{t['smart']['reduction_pct']:<8.2f} "
            f"{'YES' if m['same_edge_found'] else 'NO'}")
    rep.append('')
    rep.append('=' * 70)

    txt = '\n'.join(rep)
    with open(os.path.join(output_dir, 'REPORT.txt'), 'w') as f:
        f.write(txt)
    print(f"  Report saved")
    print(txt)
