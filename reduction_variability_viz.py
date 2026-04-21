"""
Create visualization showing why brute force reduction varies across topologies
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TOPOLOGY_DIR = r"C:\Users\ranik\Videos\files\topology_analysis"

topologies = [
    'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'path_graph', 'barbell',
    'star', 'tree', 'powerlaw_cluster', 'caveman'
]

def load_reductions(topology_name):
    """Load brute force reductions"""
    results_file = os.path.join(TOPOLOGY_DIR, topology_name, "results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
        reductions = [trial['brute_force']['reduction_pct'] for trial in data['trials']]
        return reductions
    return None

# Collect all data
all_reductions = {}
for topology in topologies:
    reductions = load_reductions(topology)
    if reductions:
        all_reductions[topology.replace('_', ' ').title()] = reductions

# Create figure
fig = plt.figure(figsize=(18, 12))

# 1. Box plots showing variability
ax1 = plt.subplot(2, 3, 1)
df_box = pd.DataFrame({
    'Topology': [k for k, v in all_reductions.items() for _ in v],
    'Reduction %': [val for v in all_reductions.values() for val in v]
})

# Sort by mean
topology_order = sorted(all_reductions.keys(), 
                       key=lambda x: np.mean(all_reductions[x]))

bp = ax1.boxplot([all_reductions[t] for t in topology_order],
                 labels=[t[:12] for t in topology_order],
                 patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax1.set_ylabel('Brute Force Reduction %', fontsize=11, fontweight='bold')
ax1.set_title('Reduction Distribution Across Topologies\n(Shows Variability)', 
              fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Variance comparison
ax2 = plt.subplot(2, 3, 2)
variances = [np.std(all_reductions[t]) for t in topology_order]
colors = ['red' if v > 2 else 'orange' if v > 0.5 else 'green' for v in variances]

bars = ax2.barh(topology_order, variances, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Standard Deviation (%)', fontsize=11, fontweight='bold')
ax2.set_title('Reduction Variability by Topology\n(High Std = High Variance)', 
              fontsize=12, fontweight='bold')
ax2.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='High Variance Threshold')
ax2.legend(fontsize=9)

# 3. Mean vs Variance scatter
ax3 = plt.subplot(2, 3, 3)
means = [np.mean(all_reductions[t]) for t in topology_order]
stds = [np.std(all_reductions[t]) for t in topology_order]

scatter = ax3.scatter(means, stds, s=200, c=stds, cmap='RdYlGn_r', 
                     alpha=0.7, edgecolors='black', linewidth=2)

for i, t in enumerate(topology_order):
    ax3.annotate(t[:10], (means[i], stds[i]), fontsize=9, ha='center', va='bottom')

ax3.set_xlabel('Mean Reduction %', fontsize=11, fontweight='bold')
ax3.set_ylabel('Std Dev (Variance)', fontsize=11, fontweight='bold')
ax3.set_title('Mean vs Variance\n(Top-right = High variance at high reduction)', 
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add zones
ax3.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
ax3.axvline(x=30, color='red', linestyle=':', alpha=0.3)
ax3.text(5, 12, 'HIGH VARIANCE\nHIGH REDUCTION\n(Unpredictable)', 
        fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# 4. Min-Max ranges
ax4 = plt.subplot(2, 3, 4)
mins = [min(all_reductions[t]) for t in topology_order]
maxs = [max(all_reductions[t]) for t in topology_order]
ranges = [maxs[i] - mins[i] for i in range(len(topology_order))]

y_pos = np.arange(len(topology_order))
for i in range(len(topology_order)):
    ax4.barh(i, maxs[i], color='lightblue', alpha=0.7, edgecolor='black')
    ax4.barh(i, mins[i], color='darkblue', alpha=0.7, edgecolor='black')
    ax4.text(maxs[i] + 2, i, f'{ranges[i]:.1f}%', va='center', fontsize=9, fontweight='bold')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(topology_order, fontsize=10)
ax4.set_xlabel('Reduction % (Min to Max)', fontsize=11, fontweight='bold')
ax4.set_title('Min-Max Range for Each Topology\n(Blue=Min, Light=Max)', 
              fontsize=12, fontweight='bold')
ax4.legend(['Min Value', 'Max Value'], fontsize=9, loc='lower right')

# 5. Trial-by-trial line plot for each topology
ax5 = plt.subplot(2, 3, 5)
for topology in topology_order[:4]:  # Show top 4 for clarity
    reductions = all_reductions[topology]
    ax5.plot(range(len(reductions)), reductions, marker='o', label=topology[:12], 
            alpha=0.7, linewidth=2, markersize=6)

ax5.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
ax5.set_ylabel('Reduction %', fontsize=11, fontweight='bold')
ax5.set_title('Reduction Changes Across Trials (Sample Topologies)', 
              fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Explanation text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

explanation = """
WHY REDUCTION VARIES:

1. GRAPH STRUCTURE
   • Deterministic (Path, Star, Tree)
     → Fixed structure, LOW variance
   • Random (Erdos-Renyi, Barabasi)
     → Different seeds, MODERATE variance
   • Rewired (Watts-Strogatz)
     → New graphs each time, EXTREME variance

2. EDGE CRITICALITY
   • Hubs & Bridges (Star, Barbell)
     → Few critical edges, high impact
   • Random Graphs (Erdos-Renyi)
     → All edges similar, lower impact
   • Small-world (Watts-Strogatz)
     → Unpredictable shortcuts

3. BC DISTRIBUTION
   • Concentrated: Few edges matter
   • Distributed: All edges similar
   • Drives variability of results

RANKING BY VARIANCE:
Stable → Variable
Caveman < Barbell < Path < Star <
Tree < Barabasi < Erdos < 
PowerLaw < Watts-Strogatz (Most Unstable)
"""

ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(TOPOLOGY_DIR, 'brute_force_reduction_variability.png'), 
           dpi=300, bbox_inches='tight')
print("✓ Saved: brute_force_reduction_variability.png")
plt.close()

# Create detailed info graphic
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, topology in enumerate(topologies):
    ax = axes[idx]
    reductions = all_reductions[topology.replace('_', ' ').title()]
    
    # Create distribution
    ax.hist(reductions, bins=6, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    mean_val = np.mean(reductions)
    std_val = np.std(reductions)
    
    # Add mean line
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, 
              label=f'±1 Std Dev: ±{std_val:.2f}%')
    
    ax.set_xlabel('Reduction %', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    
    title_text = topology.replace('_', ' ').title()
    variance_label = 'STABLE' if std_val < 0.5 else 'MODERATE' if std_val < 2 else 'UNSTABLE'
    ax.set_title(f'{title_text}\n({variance_label})\nMean: {mean_val:.2f}%, Range: {min(reductions):.2f}%-{max(reductions):.2f}%',
                fontsize=11, fontweight='bold')
    
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Brute Force Reduction Distribution for Each Topology', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(TOPOLOGY_DIR, 'reduction_distributions_detailed.png'), 
           dpi=300, bbox_inches='tight')
print("✓ Saved: reduction_distributions_detailed.png")
plt.close()

print("\n✓ All visualizations created successfully!")
