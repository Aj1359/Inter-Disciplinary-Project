import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "k_analysis_results.csv")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "k_analysis_grid.png")

def plot_k_grid(csv_path=DEFAULT_CSV, out_path=DEFAULT_OUT):
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    topos = df['Topology'].unique()
    
    n_topos = len(topos)
    rows = 3
    cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, topo in enumerate(topos):
        if i >= len(axes): break
        ax = axes[i]
        sub = df[df['Topology'] == topo].sort_values('K')
        
        bf_time = sub['BF_Time'].iloc[0] if 'BF_Time' in sub.columns else 0
        
        # Calculate speedup
        # Smart Time is TimeMs. Speedup = BF_Time / Smart Time
        # To avoid division by zero or infinity, floor Smart Time at 0.1ms
        sub['Speedup'] = bf_time / np.maximum(sub['TimeMs'], 0.1)
        
        # Left Axis: Optimality
        ln1 = ax.plot(sub['K'], sub['Optimality'], marker='o', color='#2ca02c', label='Optimality %', linewidth=2)
        ax.set_ylabel('Optimality (%)', fontsize=10)
        ax.set_ylim(-5, 110)
        ax.axhline(100, color='gray', linestyle='--', alpha=0.5)
        
        # Grid
        ax.grid(True, alpha=0.2)
        
        # Right Axis: Speedup
        ax2 = ax.twinx()
        ln2 = ax2.plot(sub['K'], sub['Speedup'], marker='s', color='#1f77b4', linestyle='--', label='Speedup', alpha=0.8)
        ax2.set_ylabel('Speedup', color='#1f77b4', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#1f77b4')
        
        # Title
        ax.set_title(f"{topo}\nBF time={bf_time:.0f}ms", fontsize=12, fontweight='bold')
        ax.set_xlabel('K (candidate cap)', fontsize=10)
        
        # Combined Legend
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='lower right', fontsize=8)
        
        # Identify the "elbow" or stable point (visual guide)
        # Just drawing a vertical line at K=30 for consistency with the requested image's typical threshold
        ax.axvline(30, color='orange', linestyle=':', alpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Grand visualization saved: {out_path}")

if __name__ == "__main__":
    plot_k_grid()
