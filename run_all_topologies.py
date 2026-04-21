#!/usr/bin/env python3
"""
run_all_topologies.py  —  Master Analysis Runner
=================================================
Runs brute-force vs smart-hop comparison for all 10 topology families,
collects results, and produces a combined summary figure + JSON.

USAGE:
    python run_all_topologies.py              # full run (10 trials each)
    python run_all_topologies.py --quick      # 3 trials each (~2 min)
    python run_all_topologies.py --topk 30    # limit smart candidates

OUTPUT:
    topology_analysis/<topology>/results.json
    topology_analysis/<topology>/results.csv
    topology_analysis/<topology>/analysis_plots.png
    topology_analysis/<topology>/REPORT.txt
    results/master_summary.json
    results/master_comparison.png
"""

import argparse, os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_core import analyze_topology

BASE = os.path.dirname(os.path.abspath(__file__))


def _connected(G):
    if not nx.is_connected(G):
        G = G.subgraph(
            max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G

TOPOLOGY_CONFIGS = [
    {
        'key':   'erdos_renyi',
        'name':  'Erdos-Renyi (n=40, p=0.12)',
        'gen':   lambda s: _connected(nx.erdos_renyi_graph(40, 0.12, seed=s)),
    },
    {
        'key':   'barabasi_albert',
        'name':  'Barabasi-Albert (n=40, m=2)',
        'gen':   lambda s: nx.barabasi_albert_graph(40, 2, seed=s),
    },
    {
        'key':   'watts_strogatz',
        'name':  'Watts-Strogatz (n=40, k=4, p=0.3)',
        'gen':   lambda s: nx.watts_strogatz_graph(40, 4, 0.3, seed=s),
    },
    {
        'key':   'path_graph',
        'name':  'Path Graph (n=20)',
        'gen':   lambda s: nx.path_graph(20),
    },
    {
        'key':   'barbell',
        'name':  'Barbell (m=7, bridge=2)',
        'gen':   lambda s: nx.barbell_graph(7, 2),
    },
    {
        'key':   'star',
        'name':  'Star (n=20)',
        'gen':   lambda s: nx.star_graph(19),
    },
    {
        'key':   'tree',
        'name':  'Random Tree (n=30)',
        'gen':   lambda s: nx.random_labeled_tree(30, seed=s),
    },
    {
        'key':   'grid_2d',
        'name':  'Grid 2D (5×5)',
        'gen':   lambda s: nx.convert_node_labels_to_integers(
                     nx.grid_2d_graph(5, 5)),
    },
    {
        'key':   'powerlaw_cluster',
        'name':  'Powerlaw Cluster (n=40, m=2, p=0.3)',
        'gen':   lambda s: nx.powerlaw_cluster_graph(40, 2, 0.3, seed=s),
    },
    {
        'key':   'caveman',
        'name':  'Caveman (5×5)',
        'gen':   lambda s: nx.connected_caveman_graph(5, 5),
    },
]



def run_all(num_trials=10, tau=0.20, topk=50):
    results_dir = os.path.join(BASE, 'results')
    topo_dir    = os.path.join(BASE, 'topology_analysis')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(topo_dir, exist_ok=True)

    print('\n' + '=' * 70)
    print('  BC-MINIMIZE: Master Analysis Runner')
    print(f'  10 topologies  ×  {num_trials} trials  ×  tau={tau}  topk={topk}')
    print('=' * 70)

    all_results = {}

    for cfg in TOPOLOGY_CONFIGS:
        out_dir = os.path.join(topo_dir, cfg['key'])
        res = analyze_topology(
            topology_name   = cfg['name'],
            graph_generator = cfg['gen'],
            output_dir      = out_dir,
            num_trials      = num_trials,
            tau             = tau,
            topk            = topk,
        )
        if res['summary']:
            all_results[cfg['key']] = {
                'name':       cfg['name'],
                'speedup':    res['summary']['speedup'],
                'optimality': res['summary']['optimality_pct'],
                'time_ms':    res['summary']['time_ms'],
                'reduction':  res['summary']['reduction_pct'],
                'candidates': res['summary']['candidates'],
                'identical':  res['summary']['solutions_identical'],
                'trials':     res['summary']['total_valid_trials'],
            }

    
    master_json = os.path.join(results_dir, 'master_summary.json')
    with open(master_json, 'w') as f:
        json.dump(all_results, f, indent=2)

   
    print('\n' + '=' * 70)
    print('  MASTER SUMMARY')
    print('=' * 70)
    print(f"{'Topology':<26} {'Speedup':>9} {'Opt%':>8} {'BF Red%':>9} "
          f"{'Cand%':>7} {'Same':>6}")
    print('-' * 70)
    for key, r in all_results.items():
        print(f"{key:<26} {r['speedup']['mean']:>9.2f}× "
              f"{r['optimality']['mean']:>8.2f}% "
              f"{r['reduction']['brute_force_mean']:>9.2f}% "
              f"{r['candidates']['ratio_pct_mean']:>7.2f}% "
              f"{r['identical']:>3}/{r['trials']}")

    if all_results:
        avg_spd = np.mean([r['speedup']['mean'] for r in all_results.values()])
        avg_opt = np.mean([r['optimality']['mean'] for r in all_results.values()])
        print('-' * 70)
        print(f"{'AVERAGE':<26} {avg_spd:>9.2f}× {avg_opt:>8.2f}%")
    print('=' * 70)


    _plot_master(all_results, results_dir)
    print(f'\n  Master summary → {master_json}')
    return all_results



def _plot_master(all_results, results_dir):
    keys   = list(all_results.keys())
    labels = [k.replace('_', '\n') for k in keys]
    spds   = [all_results[k]['speedup']['mean']     for k in keys]
    opts   = [all_results[k]['optimality']['mean']  for k in keys]
    bf_red = [all_results[k]['reduction']['brute_force_mean'] for k in keys]
    sm_red = [all_results[k]['reduction']['smart_mean']       for k in keys]
    crats  = [all_results[k]['candidates']['ratio_pct_mean']  for k in keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('BC-Minimize: All 10 Topologies — Master Comparison',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(keys))

    # Panel 1 — Speedup
    ax = axes[0, 0]
    bars = ax.bar(x, spds, color='#185FA5', alpha=0.85, edgecolor='white')
    ax.axhline(np.mean(spds), color='red', ls='--', lw=1.5,
               label=f'Mean {np.mean(spds):.1f}×')
    for bar, v in zip(bars, spds):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f'{v:.1f}×', ha='center', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Speedup (Brute / Smart)')
    ax.set_title('Speedup by Topology', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # Panel 2 — Optimality
    ax2 = axes[0, 1]
    bar_c = ['#1D9E75' if o >= 95 else '#BA7517' if o >= 70 else '#D85A30'
             for o in opts]
    bars2 = ax2.bar(x, opts, color=bar_c, alpha=0.85, edgecolor='white')
    ax2.axhline(100, color='green', ls='--', lw=1.5, label='Perfect (100%)')
    ax2.axhline(np.mean(opts), color='red', ls='--', lw=1.5,
                label=f'Mean {np.mean(opts):.1f}%')
    for bar, v in zip(bars2, opts):
        ax2.text(bar.get_x() + bar.get_width()/2, min(v + 0.5, 103),
                 f'{v:.0f}%', ha='center', fontsize=7.5, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel('Optimality (Smart / Brute × 100%)')
    ax2.set_title('Solution Quality by Topology', fontweight='bold')
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)

    # Panel 3 — BC Reduction
    ax3 = axes[1, 0]
    w = 0.38
    ax3.bar(x - w/2, bf_red, w, label='Brute Force',
            color='#D85A30', alpha=0.85, edgecolor='white')
    ax3.bar(x + w/2, sm_red, w, label='Smart',
            color='#1D9E75', alpha=0.85, edgecolor='white')
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=7.5)
    ax3.set_ylabel('Average BC Reduction (%)')
    ax3.set_title('BC Reduction by Topology', fontweight='bold')
    ax3.legend(fontsize=8); ax3.grid(axis='y', alpha=0.3)

    # Panel 4 — Candidate Ratio
    ax4 = axes[1, 1]
    ax4.bar(x, crats, color='#BA7517', alpha=0.85, edgecolor='white')
    ax4.axhline(np.mean(crats), color='red', ls='--', lw=1.5,
                label=f'Mean {np.mean(crats):.1f}%')
    for i, (xi, v) in enumerate(zip(x, crats)):
        ax4.text(xi, v + 0.3, f'{v:.1f}%',
                 ha='center', fontsize=7.5, fontweight='bold')
    ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=7.5)
    ax4.set_ylabel('Smart Cands / Brute Cands (%)')
    ax4.set_title('Candidate Count Reduction', fontweight='bold')
    ax4.legend(fontsize=8); ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, 'master_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Master figure  → {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Run BC-minimize comparison on all 10 topologies.')
    ap.add_argument('--quick',  action='store_true',
                    help='3 trials per topology (~2 min)')
    ap.add_argument('--trials', type=int, default=10,
                    help='Trials per topology (default 10)')
    ap.add_argument('--tau',    type=float, default=0.20,
                    help='Load-balance threshold (default 0.20)')
    ap.add_argument('--topk',   type=int, default=50,
                    help='Smart-algorithm candidate cap (default 50)')
    args = ap.parse_args()

    trials = 3 if args.quick else args.trials
    run_all(num_trials=trials, tau=args.tau, topk=args.topk)
