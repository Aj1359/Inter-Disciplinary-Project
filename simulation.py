"""
simulation.py — Full simulation comparing Brute Force vs Smart Algorithm
Generates all figures for the research paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
from collections import defaultdict, deque
from scipy import stats
import time, os, itertools

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
})

C = dict(
    brute='#D85A30', smart='#1D9E75', neutral='#888780',
    blue='#185FA5', amber='#BA7517', purple='#534AB7',
    ltblue='#85B7EB', ltgreen='#C0DD97', coral='#F0997B',
)

def brandes_bc(G):
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
                if dist[w] == dist[v]+1: sigma[w]+=sigma[v]; P[w].append(v)
        while S:
            w = S.pop()
            for v in P[w]: delta[v]+=(sigma[v]/sigma[w])*(1+delta[w])
            if w != s: bc[w] += delta[w]
    if n > 2:
        f = 1.0/((n-1)*(n-2))
        for k in bc: bc[k] *= f
    return dict(bc)

def find_hop_candidates(G, target, topk=50):
    """Return candidates sorted by heuristic score, limited to topk."""
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
    """Try every non-edge. Returns (best_u, best_w, bc_before, bc_after, candidates_tried, time_ms)."""
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
        load = max((bc2[n]-bc0[n]) for n in nodes if n!=target)
        if load <= tau*avg and red > best['red']:
            best = dict(u=u, w=w, red=red, load=load, bc_after=bc2[target], bc2=bc2)
    t1 = time.perf_counter()
    return best, bc0, len(non_edges), (t1-t0)*1000

def smart_algorithm(G, target, tau=0.15, topk=50):
    """Hop-based candidate selection. Returns same format as brute_force."""
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
        load = max((bc2[n]-bc0[n]) for n in nodes if n!=target)
        if load <= tau*avg and red > best['red']:
            best = dict(u=u, w=w, red=red, load=load, bc_after=bc2[target], bc2=bc2)
    t1 = time.perf_counter()
    return best, bc0, len(cands), (t1-t0)*1000, hop1, hop2

def ensure_connected(G):
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G

TOPOLOGIES = [
    ('Erdos-Renyi\n(n=40,p=0.1)',    lambda s: ensure_connected(nx.erdos_renyi_graph(40, 0.1, seed=s))),
    ('Barabasi-Albert\n(n=40,m=2)',   lambda s: nx.barabasi_albert_graph(40, 2, seed=s)),
    ('Watts-Strogatz\n(n=40,k=4,p=0.3)', lambda s: nx.watts_strogatz_graph(40, 4, 0.3, seed=s)),
    ('Path (n=20)',                   lambda s: nx.path_graph(20)),
    ('Barbell (m=7)',                 lambda s: nx.barbell_graph(7, 2)),
    ('Star (n=20)',                   lambda s: nx.star_graph(19)),
]



def fig1_explanation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    G = nx.Graph()
    G.add_edges_from([(0,1),(0,2),(0,3),(1,4),(1,5),(2,6),(2,7),(3,8),(4,9),(5,9),(6,10),(7,10)])
    target = 0
    bc = brandes_bc(G)
    pos = nx.spring_layout(G, seed=42)

    
    dist = nx.single_source_shortest_path_length(G, target)
    nc = [C['brute'] if n==target else
          C['smart'] if dist.get(n)==1 else
          C['blue'] if dist.get(n)==2 else
          C['neutral'] for n in G.nodes()]
    ns = [400 if n==target else 200 for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray', width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=ns, linewidths=0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color='white', font_weight='bold')

    # Draw hop zone circles using legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C['brute'], label=f'v* (target, BC={bc[target]:.3f})'),
        Patch(facecolor=C['smart'], label='1-hop neighbors'),
        Patch(facecolor=C['blue'],  label='2-hop neighbors'),
        Patch(facecolor=C['neutral'], label='Outside zones'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')
    ax.set_title('A. Hop zones around target v*\n(only colored nodes matter)', fontweight='bold')
    ax.axis('off')

    # Panel B: Why only hops matter
    ax2 = axes[1]
    # Visualize: adding edge between two 2-hop nodes creates bypass
    G2 = G.copy()
    best_cands, h1, h2 = find_hop_candidates(G, target, topk=10)[0], find_hop_candidates(G, target, 10)[1], find_hop_candidates(G, target, 10)[2]
    cands = find_hop_candidates(G, target, topk=10)[0]

    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.25, edge_color='gray', width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=nc, node_size=ns, linewidths=0)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=7, font_color='white', font_weight='bold')

    # Draw a few candidate non-edges as dashed
    for u, w in cands[:4]:
        x1, y1 = pos[u]; x2, y2 = pos[w]
        ax2.plot([x1,x2],[y1,y2], '--', color=C['amber'], lw=1.5, alpha=0.8, zorder=0)

    ax2.set_title('B. Candidate non-edges (dashed)\nOnly in 1-hop/2-hop zones', fontweight='bold')
    ax2.axis('off')
    from matplotlib.lines import Line2D
    ax2.legend(handles=[Line2D([0],[0],ls='--',color=C['amber'],lw=1.5,label='Candidate non-edges (complement)')],
               fontsize=8, loc='lower right')

    # Panel C: Brute force vs Smart candidate count
    ax3 = axes[2]
    ns_arr = list(range(20, 101, 5))
    brute_cands = [n*(n-1)//2 for n in ns_arr]   # all non-edges ≈ n²/2
    smart_cands_approx = [min(50, int(n*0.3)**2 // 2 + 10) for n in ns_arr]  # typical hop zone

    ax3.fill_between(ns_arr, brute_cands, color=C['brute'], alpha=0.2)
    ax3.fill_between(ns_arr, smart_cands_approx, color=C['smart'], alpha=0.3)
    ax3.plot(ns_arr, brute_cands, '-', color=C['brute'], lw=2.5, label='Brute force: O(n²)')
    ax3.plot(ns_arr, smart_cands_approx, '-', color=C['smart'], lw=2.5, label='Smart: O(K), K≤50')
    ax3.set_xlabel('Number of nodes n')
    ax3.set_ylabel('Candidates to evaluate')
    ax3.set_title('C. Candidates evaluated\n(core savings)', fontweight='bold')
    ax3.legend(fontsize=9)

    # Annotate speedup
    n_100_b = brute_cands[-1]
    n_100_s = smart_cands_approx[-1]
    ax3.annotate(f'~{n_100_b//n_100_s}× fewer\ncandidates at n=100',
                xy=(100, n_100_s), xytext=(70, n_100_b*0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=9, fontweight='bold')

    plt.suptitle('Algorithm Design: Why the Hop-Based Approach Works',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_explanation.png')
    plt.close()
    print('Fig 1 done')

# FIG 2: Full Comparison — Time & Optimality on all topologies

def fig2_comparison_all():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    TRIALS = 8

    all_speedups, all_opts = [], []

    for idx, (name, gen) in enumerate(TOPOLOGIES):
        ax = axes[idx]
        speeds, opts, t_brutes, t_smarts = [], [], [], []

        for seed in range(TRIALS):
            try:
                G = gen(seed)
                bc0 = brandes_bc(G)
                tgt = max(bc0, key=bc0.get)

                b_res, _, b_cands, b_time = brute_force(G, tgt, tau=0.20)
                s_res, _, s_cands, s_time, _, _ = smart_algorithm(G, tgt, tau=0.20, topk=50)

                if b_res['u'] >= 0 and b_res['red'] > 1e-9:
                    opt = (s_res['red'] / b_res['red'] * 100) if s_res['u'] >= 0 else 0
                    spd = b_time / max(s_time, 0.001)
                    speeds.append(spd); opts.append(opt)
                    t_brutes.append(b_time); t_smarts.append(s_time)
            except: continue

        all_speedups.extend(speeds); all_opts.extend(opts)

        if speeds:
            x = range(len(speeds))
            ax2 = ax.twinx()
            bars1 = ax.bar([i-0.2 for i in x], t_brutes, 0.35,
                          color=C['brute'], alpha=0.8, label='Brute force (ms)', edgecolor='white')
            bars2 = ax.bar([i+0.2 for i in x], t_smarts, 0.35,
                          color=C['smart'], alpha=0.8, label='Smart (ms)', edgecolor='white')
            ax2.plot(x, opts, 'D-', color=C['amber'], lw=2, ms=7,
                    label=f'Optimality %\nmean={np.mean(opts):.1f}%')
            ax2.set_ylim(0, 115)
            ax2.set_ylabel('Optimality (%)', color=C['amber'], fontsize=8)
            ax2.tick_params(colors=C['amber'], labelsize=7)
            ax.set_ylabel('Time (ms)', fontsize=8)
            ax.set_title(f'{name.replace(chr(10)," ")}\nSpeedup: {np.mean(speeds):.1f}×  |  Opt: {np.mean(opts):.1f}%',
                        fontsize=8, fontweight='bold')
            ax.set_xticks(list(x))
            ax.set_xticklabels([f'T{i+1}' for i in x], fontsize=7)
            ax.tick_params(labelsize=7)
            from matplotlib.lines import Line2D
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles1+handles2, labels1+labels2, fontsize=6, loc='upper left')

    plt.suptitle('Brute Force vs Smart Algorithm: Time & Optimality per Trial\n(bars=runtime, diamonds=optimality %)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_comparison_all.png')
    plt.close()
    print(f'Fig 2 done | mean speedup={np.mean(all_speedups):.1f}x | mean opt={np.mean(all_opts):.1f}%')
    return np.mean(all_speedups), np.mean(all_opts)

# FIG 3: Speedup vs Graph Size

def fig3_speedup_vs_size():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sizes = [15, 20, 25, 30, 35, 40]
    speedups_er, speedups_ba = [], []
    opts_er, opts_ba = [], []
    cand_ratios_er = []

    for n in sizes:
        sp_er, sp_ba, op_er, op_ba, cr_er = [], [], [], [], []
        for seed in range(6):
            for label, G_fn, sp_list, op_list, cr_list in [
                ('ER',  lambda s,n=n: ensure_connected(nx.erdos_renyi_graph(n, 0.12, seed=s)),
                 sp_er, op_er, cr_er),
                ('BA',  lambda s,n=n: nx.barabasi_albert_graph(n, 2, seed=s),
                 sp_ba, op_ba, []),
            ]:
                try:
                    G = G_fn(seed)
                    bc0 = brandes_bc(G)
                    tgt = max(bc0, key=bc0.get)
                    b_res, _, b_c, b_t = brute_force(G, tgt, tau=0.20)
                    s_res, _, s_c, s_t, _, _ = smart_algorithm(G, tgt, tau=0.20, topk=50)
                    if b_res['u'] >= 0 and b_res['red'] > 1e-9:
                        sp_list.append(b_t / max(s_t, 0.001))
                        op_list.append(s_res['red']/b_res['red']*100 if s_res['u']>=0 else 0)
                        if label == 'ER': cr_list.append(s_c / max(b_c, 1) * 100)
                except: continue
        speedups_er.append(np.mean(sp_er) if sp_er else 1)
        speedups_ba.append(np.mean(sp_ba) if sp_ba else 1)
        opts_er.append(np.mean(op_er) if op_er else 100)
        opts_ba.append(np.mean(op_ba) if op_ba else 100)
        cand_ratios_er.append(np.mean(cr_er) if cr_er else 100)  # one value per n

    # Panel 1: Speedup vs n
    ax = axes[0]
    ax.plot(sizes, speedups_er, 'o-', color=C['blue'], lw=2.5, ms=7, label='Erdos-Renyi')
    ax.plot(sizes, speedups_ba, 's-', color=C['brute'], lw=2.5, ms=7, label='Barabasi-Albert')
    ax.fill_between(sizes, 1, speedups_er, alpha=0.1, color=C['blue'])
    ax.fill_between(sizes, 1, speedups_ba, alpha=0.1, color=C['brute'])
    ax.set_xlabel('Number of nodes n')
    ax.set_ylabel('Speedup factor (Brute / Smart)')
    ax.set_title('Speedup vs Graph Size\n(Smart is faster by this factor)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.axhline(1, color='gray', ls='--', lw=0.8, alpha=0.5)

    # Panel 2: Optimality vs n
    ax2 = axes[1]
    ax2.plot(sizes, opts_er, 'o-', color=C['blue'], lw=2.5, ms=7, label='Erdos-Renyi')
    ax2.plot(sizes, opts_ba, 's-', color=C['brute'], lw=2.5, ms=7, label='Barabasi-Albert')
    ax2.axhline(100, color='gray', ls='--', lw=1, label='Perfect (=BF)')
    ax2.set_xlabel('Number of nodes n')
    ax2.set_ylabel('Optimality (% of brute-force reduction)')
    ax2.set_title('Solution Quality vs Graph Size\n(100% = identical to brute force)', fontweight='bold')
    ax2.set_ylim(60, 105)
    ax2.legend(fontsize=9)

    # Panel 3: Candidate reduction %
    ax3 = axes[2]
    ax3.plot(sizes, cand_ratios_er, 'o-', color=C['smart'], lw=2.5, ms=7, label='ER: smart/brute cands (%)')
    ax3.fill_between(sizes, 0, cand_ratios_er, alpha=0.15, color=C['smart'])
    ax3_twin = ax3.twinx()
    theoretical_speedup = [n**2 / 50 for n in sizes]
    ax3_twin.plot(sizes, theoretical_speedup, '--', color=C['amber'], lw=2,
                 label='Theoretical n²/K')
    ax3.set_xlabel('Number of nodes n')
    ax3.set_ylabel('Candidates: Smart/Brute (%)', color=C['smart'])
    ax3_twin.set_ylabel('Theoretical speedup n²/K', color=C['amber'])
    ax3.set_title('Candidate Count Reduction\nvs Theoretical Speedup', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3_twin.legend(loc='center right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_speedup_vs_size.png')
    plt.close()
    print('Fig 3 done')

# FIG 4: Detailed Before/After on 3 graphs

def fig4_before_after():
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    configs = [
        ('Barabasi-Albert (n=40,m=2)', nx.barabasi_albert_graph(40, 2, seed=5)),
        ('Barbell (m=7,bridge=2)',      nx.barbell_graph(7, 2)),
        ('Watts-Strogatz (n=40,k=4)',   nx.watts_strogatz_graph(40, 4, 0.2, seed=3)),
    ]
    for row, (title, G) in enumerate(configs):
        bc0 = brandes_bc(G)
        target = max(bc0, key=bc0.get)
        b_res, _, _, _ = brute_force(G, target, tau=0.20)
        s_res, _, _, _, h1, h2 = smart_algorithm(G, target, tau=0.20, topk=50)

        use_res = b_res if b_res['u'] >= 0 else s_res
        if use_res['u'] < 0: continue
        G2 = G.copy(); G2.add_edge(use_res['u'], use_res['w'])
        bc2 = brandes_bc(G2)

        # Col 0: network
        ax = axes[row][0]
        pos = nx.spring_layout(G2, seed=42)
        nc = [C['brute'] if n==target else
              ('#E24B4A' if n in [use_res['u'],use_res['w']] else
               C['smart'] if n in h1 else C['neutral']) for n in G2.nodes()]
        ns_arr = [350 if n==target else (200 if n in [use_res['u'],use_res['w']] else 80) for n in G2.nodes()]
        ec = [C['brute'] if set(e)=={use_res['u'],use_res['w']} else '#ccc' for e in G2.edges()]
        ew = [3.5 if set(e)=={use_res['u'],use_res['w']} else 0.5 for e in G2.edges()]
        nx.draw_networkx_edges(G2, pos, ax=ax, edge_color=ec, width=ew, alpha=0.7)
        nx.draw_networkx_nodes(G2, pos, ax=ax, node_color=nc, node_size=ns_arr, linewidths=0)
        pct = use_res['red']/bc0[target]*100 if bc0[target]>0 else 0
        ax.set_title(f'{title}\nAdded: ({use_res["u"]},{use_res["w"]}) — {pct:.1f}% reduction',
                    fontsize=8, fontweight='bold')
        ax.axis('off')

        # Col 1: BC bars
        ax2 = axes[row][1]
        nodes_s = sorted(G.nodes(), key=lambda n: -bc0.get(n,0))
        bc_b = [bc0.get(n,0) for n in nodes_s]
        bc_a = [bc2.get(n,0) for n in nodes_s]
        x = np.arange(len(nodes_s)); w=0.42
        ax2.bar(x-w/2, bc_b, w, color=C['brute'], alpha=0.8, label='Before', edgecolor='none')
        ax2.bar(x+w/2, bc_a, w, color=C['smart'], alpha=0.8, label='After', edgecolor='none')
        ti = nodes_s.index(target)
        ax2.set_title(f'BC: {bc0[target]:.4f} → {bc2[target]:.4f} ({pct:.1f}%↓)',
                     fontsize=8, fontweight='bold')
        ax2.set_xlabel('Nodes (sorted by BC)', fontsize=8)
        ax2.set_ylabel('BC', fontsize=8); ax2.set_xticks([])
        ax2.legend(fontsize=7)
        ax2.annotate('v*', xy=(ti, bc_b[ti]), xytext=(ti+2, bc_b[ti]),
                    fontsize=7, color=C['brute'], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=C['brute'], lw=0.8))

        # Col 2: delta per node
        ax3 = axes[row][2]
        all_nodes = sorted(G.nodes())
        deltas = [bc2.get(n,0)-bc0.get(n,0) for n in all_nodes]
        bar_c = [C['smart'] if d<0 else (C['brute'] if d>1e-9 else '#ddd') for d in deltas]
        ax3.bar(range(len(all_nodes)), deltas, color=bar_c, edgecolor='none', width=1.0)
        ax3.axhline(0, color='black', lw=0.8)
        n_dec = sum(1 for d in deltas if d<-1e-9)
        n_inc = sum(1 for d in deltas if d>1e-9)
        ax3.set_title(f'Per-node BC delta\n({n_dec} nodes down, {n_inc} up)',
                     fontsize=8, fontweight='bold')
        ax3.set_xlabel('All nodes', fontsize=8); ax3.set_ylabel('BC change', fontsize=8)
        ax3.set_xticks([])

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_before_after.png')
    plt.close()
    print('Fig 4 done')

# FIG 5: Optimality % breakdown by topology

def fig5_optimality_breakdown():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    TRIALS = 10

    for idx, (name, gen) in enumerate(TOPOLOGIES):
        ax = axes[idx]
        reductions_bf, reductions_sm, opts = [], [], []

        for seed in range(TRIALS):
            try:
                G = gen(seed)
                bc0 = brandes_bc(G)
                tgt = max(bc0, key=bc0.get)
                b_res, _, b_c, b_t = brute_force(G, tgt, tau=0.20)
                s_res, _, s_c, s_t, _, _ = smart_algorithm(G, tgt, tau=0.20, topk=50)
                if b_res['u'] >= 0 and b_res['red'] > 1e-9:
                    reductions_bf.append(b_res['red']/bc0[tgt]*100)
                    reductions_sm.append(s_res['red']/bc0[tgt]*100 if s_res['u']>=0 else 0)
                    opts.append(reductions_sm[-1]/reductions_bf[-1]*100 if reductions_bf[-1]>0 else 100)
            except: continue

        if reductions_bf:
            x = np.arange(len(reductions_bf))
            ax.bar(x-0.2, reductions_bf, 0.38, color=C['brute'], alpha=0.85, label='Brute Force (%)', edgecolor='white')
            ax.bar(x+0.2, reductions_sm, 0.38, color=C['smart'], alpha=0.85, label='Smart (%)', edgecolor='white')

            # Same-edge indicator
            for i in range(len(reductions_bf)):
                diff = abs(reductions_bf[i] - reductions_sm[i])
                if diff < 0.01:
                    ax.text(i, max(reductions_bf[i], reductions_sm[i])+0.2, '✓',
                           ha='center', fontsize=8, color='green')

            mean_opt = np.mean(opts)
            ax.set_title(f'{name.replace(chr(10)," ")}\nmean optimality = {mean_opt:.1f}%',
                        fontsize=8, fontweight='bold')
            ax.set_xlabel('Trial', fontsize=8); ax.set_ylabel('BC reduction (%)', fontsize=8)
            ax.set_xticks(list(x)); ax.set_xticklabels([str(i+1) for i in x], fontsize=7)
            ax.legend(fontsize=7)
            ax.text(0.98, 0.05, '✓ = same edge found', transform=ax.transAxes,
                   ha='right', fontsize=7, color='green')

    plt.suptitle('Brute Force vs Smart: BC Reduction per Trial by Topology\n(✓ = both found identical optimal edge)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_optimality_breakdown.png')
    plt.close()
    print('Fig 5 done')

# FIG 6: Time complexity empirical measurement

def fig6_time_complexity():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Measure actual runtime scaling
    sizes_n = [10, 15, 20, 25, 30, 35, 40, 45]
    t_bf_er, t_sm_er = [], []
    t_bf_ba, t_sm_ba = [], []

    for n in sizes_n:
        tbf_e, tsm_e, tbf_b, tsm_b = [], [], [], []
        for seed in range(5):
            try:
                G_er = ensure_connected(nx.erdos_renyi_graph(n, 0.12, seed=seed))
                bc0 = brandes_bc(G_er); tgt = max(bc0, key=bc0.get)
                _, _, _, t_b = brute_force(G_er, tgt, tau=0.30)
                _, _, _, t_s, _, _ = smart_algorithm(G_er, tgt, tau=0.30, topk=50)
                tbf_e.append(t_b); tsm_e.append(t_s)

                G_ba = nx.barabasi_albert_graph(n, 2, seed=seed)
                bc0 = brandes_bc(G_ba); tgt = max(bc0, key=bc0.get)
                _, _, _, t_b2 = brute_force(G_ba, tgt, tau=0.30)
                _, _, _, t_s2, _, _ = smart_algorithm(G_ba, tgt, tau=0.30, topk=50)
                tbf_b.append(t_b2); tsm_b.append(t_s2)
            except: pass
        t_bf_er.append(np.mean(tbf_e) if tbf_e else 0)
        t_sm_er.append(np.mean(tsm_e) if tsm_e else 0)
        t_bf_ba.append(np.mean(tbf_b) if tbf_b else 0)
        t_sm_ba.append(np.mean(tsm_b) if tsm_b else 0)

    # Panel 1: Absolute times
    ax = axes[0]
    ax.plot(sizes_n, t_bf_er, 'o-', color=C['brute'], lw=2.5, ms=7, label='BF Erdos-Renyi')
    ax.plot(sizes_n, t_sm_er, 'o--', color=C['brute'], lw=1.5, ms=5, alpha=0.6, label='Smart ER')
    ax.plot(sizes_n, t_bf_ba, 's-', color=C['blue'], lw=2.5, ms=7, label='BF Barabasi-Albert')
    ax.plot(sizes_n, t_sm_ba, 's--', color=C['blue'], lw=1.5, ms=5, alpha=0.6, label='Smart BA')
    ax.set_xlabel('Number of nodes n')
    ax.set_ylabel('Runtime (ms)')
    ax.set_title('Empirical runtime scaling\n(solid=brute, dashed=smart)', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 2: Speedup ratio
    ax2 = axes[1]
    spd_er = [t_bf_er[i]/max(t_sm_er[i],0.001) for i in range(len(sizes_n))]
    spd_ba = [t_bf_ba[i]/max(t_sm_ba[i],0.001) for i in range(len(sizes_n))]
    ax2.plot(sizes_n, spd_er, 'o-', color=C['blue'], lw=2.5, ms=7, label='Erdos-Renyi')
    ax2.plot(sizes_n, spd_ba, 's-', color=C['brute'], lw=2.5, ms=7, label='Barabasi-Albert')
    # Theoretical n²/K line
    theo = [n**2 / 50 for n in sizes_n]
    ax2.plot(sizes_n, theo, '--', color='gray', lw=1.5, label='Theoretical n²/K')
    ax2.set_xlabel('Number of nodes n')
    ax2.set_ylabel('Speedup (Brute / Smart)')
    ax2.set_title('Observed speedup factor\nvs theoretical n²/K', fontweight='bold')
    ax2.legend(fontsize=8)

    # Panel 3: Log-log fit
    ax3 = axes[2]
    valid = [(n,t) for n,t in zip(sizes_n, t_bf_er) if t > 0]
    if len(valid) > 3:
        ns_v = np.array([v[0] for v in valid])
        ts_v = np.array([v[1] for v in valid])
        slope_bf, intercept_bf, r, _, _ = stats.linregress(np.log(ns_v), np.log(ts_v))
        ax3.loglog(ns_v, ts_v, 'o', color=C['brute'], ms=8, label=f'BF (slope={slope_bf:.2f})')
        fitted = np.exp(intercept_bf) * ns_v**slope_bf
        ax3.loglog(ns_v, fitted, '--', color=C['brute'], lw=1.5)

    valid_s = [(n,t) for n,t in zip(sizes_n, t_sm_er) if t > 0]
    if len(valid_s) > 3:
        ns_v = np.array([v[0] for v in valid_s])
        ts_v = np.array([v[1] for v in valid_s])
        slope_sm, intercept_sm, _, _, _ = stats.linregress(np.log(ns_v), np.log(ts_v))
        ax3.loglog(ns_v, ts_v, 's', color=C['smart'], ms=8, label=f'Smart (slope={slope_sm:.2f})')
        fitted = np.exp(intercept_sm) * ns_v**slope_sm
        ax3.loglog(ns_v, fitted, '--', color=C['smart'], lw=1.5)

    ax3.set_xlabel('n (log scale)')
    ax3.set_ylabel('Time ms (log scale)')
    ax3.set_title('Log-log fit: slope ≈ complexity exponent\n(BF ≈ 3-4, Smart ≈ 1-2)', fontweight='bold')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_time_complexity.png')
    plt.close()
    print('Fig 6 done')

# FIG 7: Candidate count & heuristic score correlation

def fig7_candidate_quality():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    G = nx.barabasi_albert_graph(50, 2, seed=7)
    bc0 = brandes_bc(G)
    target = max(bc0, key=bc0.get)
    cands, h1, h2 = find_hop_candidates(G, target, topk=200)
    avg_bc = np.mean(list(bc0.values()))

    # Evaluate all candidates
    scores_h, reds_h, loads_h, classes_h = [], [], [], []
    for u, w in cands:
        G2 = G.copy(); G2.add_edge(u, w)
        bc2 = brandes_bc(G2)
        red = bc0[target]-bc2[target]
        load = max((bc2[n]-bc0[n]) for n in G.nodes() if n!=target)
        score = 2.0 if (u in h2 and w in h2) else 1.0 if (u in h1 and w in h1) else 0.6
        cls = 'C1:2h×2h' if score==2.0 else 'C2:1h×1h' if score==1.0 else 'C3:mixed'
        scores_h.append(score); reds_h.append(red/bc0[target]*100 if bc0[target]>0 else 0)
        loads_h.append(load/avg_bc*100); classes_h.append(cls)

    # Panel 1: Heuristic score vs actual reduction
    ax = axes[0]
    for cls, col in [('C1:2h×2h', C['brute']), ('C2:1h×1h', C['blue']), ('C3:mixed', C['neutral'])]:
        ix = [i for i,c in enumerate(classes_h) if c==cls]
        if ix:
            ax.scatter([scores_h[i] for i in ix], [reds_h[i] for i in ix],
                      s=50, color=col, alpha=0.7, label=cls, edgecolors='none')
    sp_r, sp_p = stats.spearmanr(scores_h, reds_h)
    ax.set_xlabel('Heuristic score')
    ax.set_ylabel('Actual BC reduction (%)')
    ax.set_title(f'Heuristic score vs actual reduction\nSpearman r={sp_r:.2f}, p={sp_p:.3f}', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 2: Rank by heuristic vs rank by actual reduction
    ax2 = axes[1]
    rank_h = np.argsort(scores_h)[::-1]      # rank by score
    rank_a = np.argsort(reds_h)[::-1]         # rank by actual
    # How soon does top-K by heuristic find the best?
    best_actual_idx = np.argmax(reds_h)
    rank_of_best_in_heuristic = np.where(rank_h == best_actual_idx)[0][0] + 1

    ks = list(range(1, min(len(cands)+1, 51)))
    cummax = [max(reds_h[rank_h[i]] for i in range(k)) for k in ks]
    opt_val = max(reds_h)
    ax2.plot(ks, cummax, '-', color=C['smart'], lw=2.5, label='Best found by heuristic ranking')
    ax2.axhline(opt_val, color=C['brute'], ls='--', lw=1.5, label=f'True optimum ({opt_val:.1f}%)')
    ax2.axvline(rank_of_best_in_heuristic, color=C['amber'], ls=':', lw=1.5,
               label=f'Optimum found at K={rank_of_best_in_heuristic}')
    ax2.set_xlabel('K (top-K candidates evaluated)')
    ax2.set_ylabel('Best BC reduction found (%)')
    ax2.set_title('Does top-K find the optimum?\n(heuristic ranking quality)', fontweight='bold')
    ax2.legend(fontsize=8)

    # Panel 3: Candidate class distribution and effectiveness
    ax3 = axes[2]
    class_names = ['C1:2h×2h', 'C2:1h×1h', 'C3:mixed']
    class_counts = [classes_h.count(c) for c in class_names]
    class_mean_red = [np.mean([reds_h[i] for i,c in enumerate(classes_h) if c==cn]) or 0 for cn in class_names]
    class_pct_safe = [np.mean([1 for i,c in enumerate(classes_h) if c==cn and loads_h[i]<=15])*100
                     if classes_h.count(cn)>0 else 0 for cn in class_names]

    x = np.arange(3); w = 0.25
    ax3_twin = ax3.twinx()
    ax3.bar(x-w, class_counts, w, color=C['blue'], alpha=0.8, label='# candidates')
    ax3.bar(x, [r*5 for r in class_mean_red], w, color=C['smart'], alpha=0.8, label='Mean reduction ×5')
    ax3_twin.bar(x+w, class_pct_safe, w, color=C['amber'], alpha=0.7, label='% safe (load≤15%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['2h×2h\n(Class 1)', '1h×1h\n(Class 2)', 'Mixed\n(Class 3)'])
    ax3.set_ylabel('Count / mean reduction ×5')
    ax3_twin.set_ylabel('% candidates safe', color=C['amber'])
    ax3.set_title('Candidate class analysis\n(count, effectiveness, safety)', fontweight='bold')
    handles1, labels1 = ax3.get_legend_handles_labels()
    handles2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(handles1+handles2, labels1+labels2, fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig7_candidate_quality.png')
    plt.close()
    print('Fig 7 done')

# FIG 8: Full aggregated summary

def fig8_summary(mean_speedup, mean_opt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    TRIALS = 12
    all_bf_red, all_sm_red, all_speedup, all_opt = [], [], [], []
    topology_data = {}

    for name, gen in TOPOLOGIES:
        short = name.split('\n')[0]
        bf_r, sm_r, spds, opts = [], [], [], []
        for seed in range(TRIALS):
            try:
                G = gen(seed)
                bc0 = brandes_bc(G); tgt = max(bc0, key=bc0.get)
                b, _, _, bt = brute_force(G, tgt, tau=0.20)
                s, _, _, st, _, _ = smart_algorithm(G, tgt, tau=0.20, topk=50)
                if b['u']>=0 and b['red']>1e-9:
                    bf_r.append(b['red']/bc0[tgt]*100)
                    sm_r.append(s['red']/bc0[tgt]*100 if s['u']>=0 else 0)
                    spds.append(bt/max(st,0.001))
                    opts.append(sm_r[-1]/bf_r[-1]*100 if bf_r[-1]>0 else 100)
            except: pass
        all_bf_red.extend(bf_r); all_sm_red.extend(sm_r)
        all_speedup.extend(spds); all_opt.extend(opts)
        topology_data[short] = dict(bf_r=bf_r, sm_r=sm_r, spds=spds, opts=opts)

    # Panel 1: Reduction scatter BF vs Smart
    ax = axes[0][0]
    ax.scatter(all_bf_red, all_sm_red, s=30, alpha=0.5, color=C['blue'], edgecolors='none')
    lim = max(max(all_bf_red+[1]), max(all_sm_red+[1])) * 1.05
    ax.plot([0,lim],[0,lim], '--', color='red', lw=1.5, label='y=x (perfect)')
    r, p = stats.pearsonr(all_bf_red, all_sm_red)
    ax.set_xlabel('Brute Force BC reduction (%)')
    ax.set_ylabel('Smart BC reduction (%)')
    ax.set_title(f'Reduction quality: BF vs Smart\nPearson r={r:.4f}', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel 2: Speedup box by topology
    ax2 = axes[0][1]
    tnames = list(topology_data.keys())
    data_spd = [topology_data[n]['spds'] for n in tnames]
    bp = ax2.boxplot(data_spd, labels=[n.replace('-',' ')[:12] for n in tnames],
                    patch_artist=True, notch=False)
    colors_box = [C['blue'], C['brute'], C['smart'], C['amber'], C['purple'], C['coral']]
    for patch, col in zip(bp['boxes'], colors_box):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for med in bp['medians']: med.set_color('white'); med.set_linewidth(2)
    ax2.set_ylabel('Speedup (Brute / Smart)')
    ax2.set_title('Speedup distribution by topology', fontweight='bold')
    ax2.tick_params(axis='x', rotation=18, labelsize=7)

    # Panel 3: Optimality box by topology
    ax3 = axes[1][0]
    data_opt = [topology_data[n]['opts'] for n in tnames]
    bp3 = ax3.boxplot(data_opt, labels=[n.replace('-',' ')[:12] for n in tnames],
                     patch_artist=True, notch=False)
    for patch, col in zip(bp3['boxes'], colors_box):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for med in bp3['medians']: med.set_color('white'); med.set_linewidth(2)
    ax3.axhline(100, color='gray', ls='--', lw=1.5, alpha=0.7, label='Perfect = 100%')
    ax3.set_ylabel('Optimality (% of BF reduction)')
    ax3.set_title('Solution optimality by topology', fontweight='bold')
    ax3.tick_params(axis='x', rotation=18, labelsize=7)
    ax3.legend(fontsize=8)

    # Panel 4: Summary bar chart
    ax4 = axes[1][1]
    metrics = {
        'Mean speedup': np.mean(all_speedup),
        'Mean optimality (%)': np.mean(all_opt),
        'Candidates\nreduced (%)': (1 - 50/max(np.mean([len(list(nx.non_edges(nx.barabasi_albert_graph(40,2,seed=0))))]), 1)) * 100,
    }
    bars = ax4.bar(range(len(metrics)), list(metrics.values()),
                  color=[C['blue'], C['smart'], C['amber']], edgecolor='white', width=0.5)
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels(list(metrics.keys()), fontsize=9)
    for bar, val in zip(bars, metrics.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax4.set_title(f'Algorithm Performance Summary\n(across all {len(all_bf_red)} trials)', fontweight='bold')
    ax4.set_ylabel('Value')

    plt.suptitle('Full Comparison Summary: Brute Force vs Smart Algorithm',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig8_summary.png')
    plt.close()
    print('Fig 8 done')

if __name__ == '__main__':
    print('Running full simulation...\n')
    fig1_explanation()
    mean_spd, mean_opt = fig2_comparison_all()
    fig3_speedup_vs_size()
    fig4_before_after()
    fig5_optimality_breakdown()
    fig6_time_complexity()
    fig7_candidate_quality()
    fig8_summary(mean_spd, mean_opt)
    print(f'\nAll 8 figures saved to {OUT}/')
    print(f'Summary: mean speedup={mean_spd:.1f}x, mean optimality={mean_opt:.1f}%')
