#!/usr/bin/env python3
"""
generate_graphs.py  —  Graph Generator for All 10 Topologies
=============================================================
Generates one representative graph per topology, saves as SNAP-format
edge lists in  datasets/<topology>/  and prints a summary table.

WHY THESE 10 TOPOLOGIES:
  We select 10 families to span the full structural diversity spectrum:

  Random models (3) — cover different degree distributions:
    • Erdos-Renyi:        Uniform random edges (Poisson degree dist.)
    • Barabasi-Albert:    Preferential attachment (power-law/scale-free)
    • Watts-Strogatz:     Small-world (high clustering + short paths)

  Deterministic/structured (4) — known BC patterns, 100% predictable:
    • Path Graph:         Linear chain, center node has max BC
    • Barbell:            Two cliques + bridge, bridge node dominates BC
    • Star:               Central hub controls all paths
    • Random Tree:        Acyclic, unique paths between all node pairs

  Hybrid (3) — combine regularity with randomness:
    • Grid 2D:            Regular lattice, center-dominated BC
    • Powerlaw Cluster:   Scale-free + high clustering (triangles)
    • Connected Caveman:  Community structure with inter-clique bridges

  This selection ensures the smart algorithm is evaluated on graphs
  where it should excel (structured), graphs where it may struggle
  (random), and intermediate cases (hybrid).

USAGE:
    python generate_graphs.py           # generate all 10 topologies
    python generate_graphs.py --show    # also print graph stats table

OUTPUT FILES (in datasets/):
    erdos_renyi/graph_seed0.txt
    barabasi_albert/graph_seed0.txt
    watts_strogatz/graph_seed0.txt
    path_graph/graph.txt
    barbell/graph.txt
    star/graph.txt
    tree/graph_seed0.txt
    grid_2d/graph.txt
    powerlaw_cluster/graph_seed0.txt
    caveman/graph.txt
"""

import argparse, os, sys
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

# 
# Topology Definitions
# 

TOPOLOGIES = {
    # 1. Erdős–Rényi random graph
    'erdos_renyi': {
        'label': 'Erdos-Renyi (n=40, p=0.12)',
        'desc':  'Random graph; each edge exists with prob p. '
                 'Poisson degree distribution.',
        'gen':   lambda seed=0: _ensure_connected(
                     nx.erdos_renyi_graph(40, 0.12, seed=seed)),
        'seed_dependent': True,
        'params': 'n=40, p=0.12',
    },

    # 2. Barabási–Albert scale-free
    'barabasi_albert': {
        'label': 'Barabasi-Albert (n=40, m=2)',
        'desc':  'Preferential-attachment growth model. '
                 'Power-law degree distribution (scale-free).',
        'gen':   lambda seed=0: nx.barabasi_albert_graph(40, 2, seed=seed),
        'seed_dependent': True,
        'params': 'n=40, m=2',
    },

    # 3. Watts–Strogatz small-world
    'watts_strogatz': {
        'label': 'Watts-Strogatz (n=40, k=4, p=0.3)',
        'desc':  'Ring lattice with random rewiring. '
                 'High clustering + short paths (small-world).',
        'gen':   lambda seed=0: nx.watts_strogatz_graph(40, 4, 0.3, seed=seed),
        'seed_dependent': True,
        'params': 'n=40, k=4, p=0.3',
    },

    # 4. Path graph
    'path_graph': {
        'label': 'Path Graph (n=20)',
        'desc':  'Linear chain. Center node has maximum BC. '
                 'Deterministic.',
        'gen':   lambda seed=0: nx.path_graph(20),
        'seed_dependent': False,
        'params': 'n=20',
    },

    # 5. Barbell graph
    'barbell': {
        'label': 'Barbell (m=7, bridge=2)',
        'desc':  'Two complete subgraphs connected by a path of '
                 'length bridge. Bridge nodes have extreme BC.',
        'gen':   lambda seed=0: nx.barbell_graph(7, 2),
        'seed_dependent': False,
        'params': 'm1=m2=7, bridge=2',
    },

    # 6. Star graph
    'star': {
        'label': 'Star (n=20)',
        'desc':  'One central hub, n-1 leaf nodes. '
                 'Hub has BC=1 (all paths go through it).',
        'gen':   lambda seed=0: nx.star_graph(19),
        'seed_dependent': False,
        'params': 'n=20 (1 hub + 19 leaves)',
    },

    # 7. Random tree
    'tree': {
        'label': 'Random Tree (n=30)',
        'desc':  'Random labelled spanning tree. '
                 'No cycles; BC concentrated on trunk nodes.',
        'gen':   lambda seed=0: nx.random_labeled_tree(30, seed=seed),
        'seed_dependent': True,
        'params': 'n=30',
    },

    # 8. 2-D Grid
    'grid_2d': {
        'label': '2D Grid (5×5)',
        'desc':  'Regular grid graph. Interior nodes share '
                 'elevated BC; corner/edge nodes lower.',
        'gen':   lambda seed=0: nx.convert_node_labels_to_integers(
                     nx.grid_2d_graph(5, 5)),
        'seed_dependent': False,
        'params': '5×5 = 25 nodes',
    },

    # 9. Power-law cluster
    'powerlaw_cluster': {
        'label': 'Powerlaw Cluster (n=40, m=2, p=0.3)',
        'desc':  'Holme–Kim model: BA + triangle-closing. '
                 'Scale-free with higher clustering.',
        'gen':   lambda seed=0: nx.powerlaw_cluster_graph(
                     40, 2, 0.3, seed=seed),
        'seed_dependent': True,
        'params': 'n=40, m=2, p=0.3',
    },

    # 10. Connected caveman graph
    'caveman': {
        'label': 'Caveman (5 cliques × 5 nodes)',
        'desc':  'Dense cliques connected in a ring. '
                 'Bridge nodes between cliques have high BC.',
        'gen':   lambda seed=0: nx.connected_caveman_graph(5, 5),
        'seed_dependent': False,
        'params': 'l=5 cliques, k=5 nodes each',
    },
}


def _ensure_connected(G):
    if not nx.is_connected(G):
        G = G.subgraph(
            max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
    return G


# 
# Save / Load helpers
# 

def save_graph(G, filepath, header=''):
    """Save graph as SNAP-format edge list."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        if header:
            for line in header.splitlines():
                f.write(f'# {line}\n')
        for u, v in G.edges():
            f.write(f'{u} {v}\n')


def load_graph(filepath):
    """Load a SNAP-format edge list into a NetworkX graph."""
    G = nx.Graph()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v)
    return G


# 
# Main: generate all 10 topologies
# 

def generate_all(seed=0, show_stats=False):
    print('\n' + '=' * 65)
    print('  GRAPH GENERATOR — All 10 Topologies')
    print('=' * 65)

    rows = []
    graphs = {}

    for key, cfg in TOPOLOGIES.items():
        G    = _ensure_connected(cfg['gen'](seed))
        fname = (f'graph_seed{seed}.txt'
                 if cfg['seed_dependent'] else 'graph.txt')
        path = os.path.join(DATASETS_DIR, key, fname)
        header = (f"Topology: {cfg['label']}\n"
                  f"Params:   {cfg['params']}\n"
                  f"Nodes:    {G.number_of_nodes()}\n"
                  f"Edges:    {G.number_of_edges()}\n"
                  f"Seed:     {seed}")
        save_graph(G, path, header)

        # Compute quick stats
        degrees = [d for _, d in G.degree()]
        bc_vals = list(nx.betweenness_centrality(G, normalized=True).values())

        rows.append({
            'key':       key,
            'label':     cfg['label'],
            'desc':      cfg['desc'],
            'n':         G.number_of_nodes(),
            'm':         G.number_of_edges(),
            'density':   round(nx.density(G), 4),
            'avg_deg':   round(np.mean(degrees), 2),
            'clustering': round(nx.average_clustering(G), 3),
            'max_bc':    round(max(bc_vals), 4),
            'avg_bc':    round(np.mean(bc_vals), 4),
            'file':      path,
        })
        graphs[key] = G
        print(f"  ✓ {cfg['label']:<40} → {path}")

    # Print summary table
    print('\n' + '=' * 65)
    print('  GRAPH STATISTICS SUMMARY')
    print('=' * 65)
    print(f"{'Topology':<26} {'n':>4} {'m':>6} {'Density':>8} "
          f"{'AvgDeg':>7} {'Clust':>7} {'MaxBC':>8}")
    print('-' * 65)
    for r in rows:
        print(f"{r['key']:<26} {r['n']:>4} {r['m']:>6} "
              f"{r['density']:>8.4f} {r['avg_deg']:>7.2f} "
              f"{r['clustering']:>7.3f} {r['max_bc']:>8.4f}")
    print('=' * 65)

    if show_stats:
        _plot_all(graphs, rows)

    return rows, graphs


# 
# Visualization: 10-panel network overview
# 

def _plot_all(graphs, rows):
    keys  = list(graphs.keys())
    ncols = 5
    nrows = 2
    fig   = plt.figure(figsize=(22, 9))
    gs    = gridspec.GridSpec(nrows, ncols, figure=fig,
                              hspace=0.45, wspace=0.3)

    bc_by_key = {}
    for key, G in graphs.items():
        bc = nx.betweenness_centrality(G, normalized=True)
        bc_by_key[key] = bc

    for idx, key in enumerate(keys):
        G   = graphs[key]
        bc  = bc_by_key[key]
        row = idx // ncols
        col = idx %  ncols
        ax  = fig.add_subplot(gs[row, col])

        target = max(bc, key=bc.get)
        try:
            pos = nx.spring_layout(G, seed=42)
        except Exception:
            pos = nx.random_layout(G, seed=42)

        nc = ['#D85A30' if n == target else '#85B7EB'
              for n in G.nodes()]
        ns = [280 if n == target else 60 for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax,
                               alpha=0.25, edge_color='#888', width=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=nc, node_size=ns, linewidths=0)

        r = next(r for r in rows if r['key'] == key)
        ax.set_title(
            f"{key.replace('_', ' ')}\n"
            f"n={r['n']}, m={r['m']}, maxBC={r['max_bc']:.3f}",
            fontsize=8, fontweight='bold', pad=3)
        ax.axis('off')

    fig.text(0.5, 1.01,
             'All 10 Topology Graphs (orange = highest-BC target node)',
             ha='center', fontsize=12, fontweight='bold')

    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#D85A30', markersize=9,
                   label='Target node (max BC)'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#85B7EB', markersize=7,
                   label='Other nodes'),
        ],
        loc='lower center', ncol=2, fontsize=9,
        bbox_to_anchor=(0.5, -0.04))

    out = os.path.join(DATASETS_DIR, 'all_topologies_overview.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Overview figure saved → {out}')


# 
# CLI
# 

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Generate graphs for all 10 topologies.')
    ap.add_argument('--seed', type=int, default=0,
                    help='Random seed (default 0)')
    ap.add_argument('--show', action='store_true',
                    help='Also generate network overview figure')
    args = ap.parse_args()

    generate_all(seed=args.seed, show_stats=args.show)
    print('\nDone. All graphs saved to datasets/<topology>/\n')
