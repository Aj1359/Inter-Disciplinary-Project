import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style to perfectly match paper
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 200, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.titlesize': 11, 'axes.titleweight': 'bold'
})

def plot_figure10(csv_path="results/phase1_results.csv", out_path="results/Figure10_Sensitivity.png"):
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}. Make sure you ran phase 1!")
        return

    df = pd.read_csv(csv_path)
    
    topologies = df['topology'].unique()
    taus = sorted(df['tau'].unique())
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Load Factor (τ) Sensitivity Analysis — Smart Algorithm\nHow τ controls BC reduction vs load safety trade-off across topologies', fontsize=14, fontweight='bold', y=0.98)
    
    color_map = sns.color_palette("tab10", len(topologies))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('BC Reduction vs τ\n(larger τ → less strict load constraint → more reduction possible)', fontsize=10)
    for i, topo in enumerate(topologies):
        subset = df[df['topology'] == topo].groupby('tau')['smart_optimality'].mean()
        ax1.plot(subset.index, subset.values, marker='o', label=topo, color=color_map[i], lw=2)
    ax1.set_xlabel('τ (load factor threshold)')
    ax1.set_ylabel('Smart BC reduction (%)')
    ax1.legend(loc='lower right', ncol=2, fontsize=8)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Solution Availability vs τ\n(stricter τ → fewer valid edges satisfy load constraint)', fontsize=10)
    for i, topo in enumerate(topologies):
        availability_series = []
        for tau in taus:
            sub = df[(df['topology'] == topo) & (df['tau'] == tau)]
            if len(sub) > 0:
                avail = (sub['smart_reduction'] > 0).mean() * 100
                availability_series.append(avail)
            else:
                availability_series.append(0)
        ax2.plot(taus, availability_series, marker='s', label=topo, color=color_map[i], lw=2)
    ax2.axhline(80, ls=':', color='gray', alpha=0.6)
    ax2.set_xlabel('τ')
    ax2.set_ylabel('Trials finding valid solution (%)')
    ax2.legend(loc='lower right', ncol=2, fontsize=8)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('BC Reduction Heatmap: Topology × τ\n(green=high reduction, red=low)', fontsize=10)
    heatmap_data = pd.pivot_table(df, values='smart_optimality', index='topology', columns='tau', aggfunc='mean')
    heatmap_data = heatmap_data.fillna(0)
    sns.heatmap(heatmap_data, ax=ax3, cmap="RdYlGn", annot=True, fmt=".1f", cbar_kws={'label': 'BC reduction (%)'})
    ax3.set_ylabel('')
    ax3.set_xlabel('τ value')
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Minimum Effective τ per Topology\n(lower = algorithm works well even at strict τ)', fontsize=10)
    
    min_taus = []
    for topo in topologies:
        sub = df[df['topology'] == topo].groupby('tau')['smart_optimality'].mean()
        max_red = sub.max()
        if max_red > 0:
            effective_tau = sub[sub >= 0.90 * max_red].index.min()
        else:
            effective_tau = np.nan
        min_taus.append({'Topology': topo, 'MinTau': effective_tau})
        
    df_min_tau = pd.DataFrame(min_taus).dropna().sort_values('MinTau')
    sns.barplot(data=df_min_tau, y='Topology', x='MinTau', ax=ax4, palette="muted", orient='h')
    
    ax4.axvline(0.15, ls='--', color='red', label='Default τ=0.15')
    ax4.set_xlabel('Minimum τ achieving 90% of maximum reduction')
    ax4.set_ylabel('')
    for i, p in enumerate(ax4.patches):
        ax4.annotate(f"{p.get_width():.2f}", (p.get_width() + 0.01, p.get_y() + 0.5), fontsize=8, fontweight='bold')
    ax4.legend(loc='center right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Figure 10 generated: {out_path}")

if __name__ == "__main__":
    plot_figure10()
