"""
Comprehensive Error Analysis: Brute Force vs Smart Algorithm
Analyzes all 10 topologies with detailed error percentage comparison
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Topology paths
TOPOLOGY_DIR = r"C:\Users\ranik\Videos\files\topology_analysis"
topologies = [
    'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'path_graph', 'barbell',
    'star', 'tree', 'powerlaw_cluster', 'caveman'
]

def load_topology_results(topology_name):
    """Load results from a topology"""
    results_file = os.path.join(TOPOLOGY_DIR, topology_name, "results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def calculate_error_metrics(results):
    """Calculate comprehensive error metrics"""
    if not results or 'trials' not in results:
        return None
    
    trials_data = []
    
    for trial in results['trials']:
        bf_reduction = trial['brute_force']['reduction']
        sm_reduction = trial['smart']['reduction']
        
        # Error calculations
        absolute_error = bf_reduction - sm_reduction
        relative_error = (absolute_error / bf_reduction * 100) if bf_reduction > 0 else 0
        optimality = (sm_reduction / bf_reduction * 100) if bf_reduction > 0 else 0
        
        trials_data.append({
            'seed': trial['seed'],
            'nodes': trial['nodes'],
            'edges': trial['edges'],
            'bf_reduction': bf_reduction,
            'sm_reduction': sm_reduction,
            'absolute_error': absolute_error,
            'relative_error_pct': relative_error,
            'optimality_pct': optimality,
            'speedup': trial['metrics']['speedup'],
            'candidate_ratio': trial['metrics']['candidate_ratio_pct'],
            'same_edge': trial['metrics']['same_edge_found'],
            'bf_time': trial['brute_force']['time_ms'],
            'sm_time': trial['smart']['time_ms']
        })
    
    return pd.DataFrame(trials_data)

def create_summary_table():
    """Create comprehensive summary of all topologies"""
    summary_data = []
    
    for topology in topologies:
        results = load_topology_results(topology)
        df = calculate_error_metrics(results)
        
        if df is not None:
            summary_data.append({
                'Topology': topology.replace('_', ' ').title(),
                'Trials': len(df),
                'Avg Error %': df['relative_error_pct'].mean(),
                'Max Error %': df['relative_error_pct'].max(),
                'Min Error %': df['relative_error_pct'].min(),
                'Std Dev': df['relative_error_pct'].std(),
                'Avg Optimality %': df['optimality_pct'].mean(),
                'Avg Speedup': df['speedup'].mean(),
                'Success Rate %': (df['same_edge'].sum() / len(df) * 100) if len(df) > 0 else 0,
                'Avg Nodes': df['nodes'].mean(),
                'Avg Edges': df['edges'].mean()
            })
    
    return pd.DataFrame(summary_data)

def analyze_error_patterns():
    """Analyze why errors vary across topologies"""
    print("\n" + "="*80)
    print("ERROR ANALYSIS: BRUTE FORCE VS SMART ALGORITHM")
    print("="*80)
    
    summary_df = create_summary_table()
    
    print("\n1. COMPREHENSIVE ERROR COMPARISON")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    print("\n2. ERROR STATISTICS")
    print("-" * 80)
    print(f"Average Error % Across All Topologies: {summary_df['Avg Error %'].mean():.2f}%")
    print(f"Maximum Error % Recorded: {summary_df['Max Error %'].max():.2f}%")
    print(f"Minimum Error % Recorded: {summary_df['Min Error %'].min():.2f}%")
    print(f"Overall Average Optimality: {summary_df['Avg Optimality %'].mean():.2f}%")
    
    print("\n3. TOPOLOGY RANKING (by accuracy)")
    print("-" * 80)
    ranked = summary_df.sort_values('Avg Optimality %', ascending=False)
    for idx, row in ranked.iterrows():
        print(f"{row['Topology']:20} | Error: {row['Avg Error %']:6.2f}% | Optimality: {row['Avg Optimality %']:6.2f}% | Speedup: {row['Avg Speedup']:6.2f}x")
    
    print("\n4. KEY INSIGHTS")
    print("-" * 80)
    
    # Group by error ranges
    perfect = summary_df[summary_df['Avg Error %'] < 5]
    good = summary_df[(summary_df['Avg Error %'] >= 5) & (summary_df['Avg Error %'] < 50)]
    poor = summary_df[summary_df['Avg Error %'] >= 50]
    
    print(f"\n✓ PERFECT TOPOLOGIES (<5% error): {len(perfect)}")
    for _, row in perfect.iterrows():
        print(f"  - {row['Topology']:30} Error: {row['Avg Error %']:6.2f}% | Speedup: {row['Avg Speedup']:6.2f}x")
    
    print(f"\n◐ GOOD TOPOLOGIES (5-50% error): {len(good)}")
    for _, row in good.iterrows():
        print(f"  - {row['Topology']:30} Error: {row['Avg Error %']:6.2f}% | Speedup: {row['Avg Speedup']:6.2f}x")
    
    print(f"\n✗ CHALLENGING TOPOLOGIES (>50% error): {len(poor)}")
    for _, row in poor.iterrows():
        print(f"  - {row['Topology']:30} Error: {row['Avg Error %']:6.2f}% | Speedup: {row['Avg Speedup']:6.2f}x")
    
    return summary_df

def explain_error_causes():
    """Explain why errors occur in different topologies"""
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS: WHY THESE RESULTS")
    print("="*80)
    
    causes = {
        'Perfect Topologies (Path, Tree, Star, Barbell, Caveman)': {
            'reason': 'Highly structured, regular patterns make optimal edge obvious',
            'characteristics': [
                '• Low diameter (short paths between nodes)',
                '• Clear centrality patterns (hubs, bridges, centers)',
                '• Deterministic optimal edge location',
                '• Symmetric or semi-symmetric structure'
            ],
            'why_smart_works': 'Heuristics correctly identify high-importance nodes',
            'error': '<5%'
        },
        'Challenging Topologies (Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, Power-Law)': {
            'reason': 'Random/semi-random patterns with complex centrality distributions',
            'characteristics': [
                '• Small-world properties (short paths despite sparsity)',
                '• Power-law degree distributions (highly variable node importance)',
                '• Multiple local optima (many edges with similar impact)',
                '• Less predictable structural patterns'
            ],
            'why_smart_works': 'Heuristics miss subtle patterns that brute force finds',
            'error': '15-45%'
        }
    }
    
    for category, details in causes.items():
        print(f"\n{category}")
        print("-" * 80)
        print(f"Reason: {details['reason']}")
        print(f"\nCharacteristics:")
        for char in details['characteristics']:
            print(f"  {char}")
        print(f"\nWhy Smart Algorithm {details['why_smart_works']}")
        print(f"Typical Error Range: {details['error']}")
    
    print("\n" + "="*80)
    print("DETAILED MECHANISM")
    print("="*80)
    
    mechanisms = {
        'Smart Algorithm Strategy': [
            '1. Identifies candidates based on degree, betweenness, clustering',
            '2. Only tests ~7% of edges (50 out of 704 in Barabasi example)',
            '3. Misses non-obvious edges with subtle high-impact properties'
        ],
        'Where Smart Algorithm Fails': [
            '• Random Topologies: No obvious pattern, optimal edge is unpredictable',
            '• Scale-Free Networks: Mix of hubs and periphery creates complex interactions',
            '• Bridges in Random Graphs: May not have highest degree but highest impact',
            '• Cascading Effects: Removing edge A affects downstream more than degree suggests'
        ],
        'Why Brute Force Always Wins': [
            '• Tests ALL 704 edges exhaustively',
            '• Finds truly optimal edge regardless of topology structure',
            '• No assumptions about what makes an edge important',
            '• Can detect non-intuitive high-impact edges'
        ]
    }
    
    for title, points in mechanisms.items():
        print(f"\n{title}:")
        for point in points:
            print(f"  {point}")

def create_visualizations(summary_df):
    """Create multiple visualizations"""
    
    # 1. Error Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    sorted_df = summary_df.sort_values('Avg Error %', ascending=False)
    colors = ['red' if x > 50 else 'orange' if x > 5 else 'green' for x in sorted_df['Avg Error %']]
    ax1.barh(sorted_df['Topology'], sorted_df['Avg Error %'], color=colors, alpha=0.7)
    ax1.set_xlabel('Average Error %', fontsize=12, fontweight='bold')
    ax1.set_title('Error Percentage by Topology\n(Red: >50% | Orange: 5-50% | Green: <5%)', 
                   fontsize=13, fontweight='bold')
    ax1.axvline(x=summary_df['Avg Error %'].mean(), color='blue', linestyle='--', 
                label=f'Average: {summary_df["Avg Error %"].mean():.1f}%', linewidth=2)
    ax1.legend()
    
    # 2. Optimality vs Speedup
    ax2 = axes[0, 1]
    scatter = ax2.scatter(summary_df['Avg Speedup'], summary_df['Avg Optimality %'], 
                         s=summary_df['Avg Nodes']*2, alpha=0.6, c=summary_df['Avg Error %'], 
                         cmap='RdYlGn_r', edgecolors='black', linewidth=1.5)
    for idx, row in summary_df.iterrows():
        ax2.annotate(row['Topology'], (row['Avg Speedup'], row['Avg Optimality %']), 
                    fontsize=9, ha='center')
    ax2.set_xlabel('Average Speedup (x)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Optimality %', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup vs Optimality\n(Bubble size = Node count, Color = Error %)', 
                  fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Error %')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error Distribution Box Plot
    ax3 = axes[1, 0]
    error_data = []
    labels = []
    for topology in topologies:
        results = load_topology_results(topology)
        df = calculate_error_metrics(results)
        if df is not None:
            error_data.append(df['relative_error_pct'].values)
            labels.append(topology.replace('_', '\n').title())
    
    bp = ax3.boxplot(error_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('Error %', fontsize=12, fontweight='bold')
    ax3.set_title('Error Distribution Across Topologies', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance Metrics Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    display_df = summary_df[['Topology', 'Avg Error %', 'Avg Optimality %', 'Avg Speedup', 'Success Rate %']].copy()
    display_df['Avg Error %'] = display_df['Avg Error %'].apply(lambda x: f'{x:.2f}%')
    display_df['Avg Optimality %'] = display_df['Avg Optimality %'].apply(lambda x: f'{x:.2f}%')
    display_df['Avg Speedup'] = display_df['Avg Speedup'].apply(lambda x: f'{x:.2f}x')
    display_df['Success Rate %'] = display_df['Success Rate %'].apply(lambda x: f'{x:.1f}%')
    
    table = ax4.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc='center', loc='center', colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Performance Summary Table', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(TOPOLOGY_DIR, 'error_analysis_comparison.png'), dpi=300, bbox_inches='tight')
    print("\n✓ Saved: error_analysis_comparison.png")
    
    # 5. Create 3D Visualization
    create_3d_visualization(summary_df)

def create_3d_visualization(summary_df):
    """Create 3D graph visualization"""
    fig = plt.figure(figsize=(16, 12))
    
    # 3D Scatter: Speedup vs Optimality vs Error
    ax1 = fig.add_subplot(221, projection='3d')
    
    scatter = ax1.scatter(summary_df['Avg Speedup'], 
                         summary_df['Avg Optimality %'],
                         summary_df['Avg Error %'],
                         s=summary_df['Avg Nodes']*3,
                         c=summary_df['Avg Error %'],
                         cmap='RdYlGn_r',
                         alpha=0.7,
                         edgecolors='black',
                         linewidth=1.5)
    
    ax1.set_xlabel('Speedup (x)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Optimality %', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Error %', fontsize=11, fontweight='bold')
    ax1.set_title('3D: Speedup vs Optimality vs Error\n(Bubble size = Node count)', 
                  fontsize=12, fontweight='bold')
    
    for idx, row in summary_df.iterrows():
        ax1.text(row['Avg Speedup'], row['Avg Optimality %'], row['Avg Error %'],
                row['Topology'].replace('_', ' ').title()[:10], fontsize=8)
    
    cbar1 = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
    cbar1.set_label('Error %', fontsize=10, fontweight='bold')
    
    # 3D Scatter: Nodes vs Edges vs Error
    ax2 = fig.add_subplot(222, projection='3d')
    
    scatter2 = ax2.scatter(summary_df['Avg Nodes'],
                          summary_df['Avg Edges'],
                          summary_df['Avg Error %'],
                          s=summary_df['Avg Speedup']*50,
                          c=summary_df['Avg Optimality %'],
                          cmap='viridis',
                          alpha=0.7,
                          edgecolors='black',
                          linewidth=1.5)
    
    ax2.set_xlabel('Average Nodes', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Edges', fontsize=11, fontweight='bold')
    ax2.set_zlabel('Error %', fontsize=11, fontweight='bold')
    ax2.set_title('3D: Graph Size vs Error\n(Bubble size = Speedup, Color = Optimality %)',
                  fontsize=12, fontweight='bold')
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.8)
    cbar2.set_label('Optimality %', fontsize=10, fontweight='bold')
    
    # 3D Surface: Error vs Speedup vs Optimality
    ax3 = fig.add_subplot(223, projection='3d')
    
    # Create mesh for surface
    speedup_range = np.linspace(summary_df['Avg Speedup'].min() - 2, 
                               summary_df['Avg Speedup'].max() + 2, 10)
    error_range = np.linspace(0, summary_df['Avg Error %'].max() + 10, 10)
    speedup_mesh, error_mesh = np.meshgrid(speedup_range, error_range)
    
    # Interpolate optimality
    from scipy.interpolate import griddata
    optimality_mesh = griddata((summary_df['Avg Speedup'], summary_df['Avg Error %']),
                               summary_df['Avg Optimality %'],
                               (speedup_mesh, error_mesh),
                               method='cubic', fill_value=50)
    
    surf = ax3.plot_surface(speedup_mesh, error_mesh, optimality_mesh,
                           cmap='coolwarm', alpha=0.8, edgecolor='none')
    
    ax3.scatter(summary_df['Avg Speedup'], summary_df['Avg Error %'],
               summary_df['Avg Optimality %'],
               c='black', s=100, alpha=0.8, edgecolors='white', linewidth=2)
    
    ax3.set_xlabel('Speedup (x)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Error %', fontsize=11, fontweight='bold')
    ax3.set_zlabel('Optimality %', fontsize=11, fontweight='bold')
    ax3.set_title('3D Surface: Performance Trade-offs', fontsize=12, fontweight='bold')
    
    cbar3 = plt.colorbar(surf, ax=ax3, pad=0.1, shrink=0.8)
    cbar3.set_label('Optimality %', fontsize=10, fontweight='bold')
    
    # 2D Heatmap
    ax4 = fig.add_subplot(224)
    
    # Create metric matrix
    metrics_matrix = summary_df[['Avg Error %', 'Avg Optimality %', 'Avg Speedup']].values
    im = ax4.imshow(metrics_matrix.T, aspect='auto', cmap='RdYlGn_r')
    
    ax4.set_xticks(range(len(summary_df)))
    ax4.set_xticklabels([t.replace('_', ' ').title()[:12] for t in summary_df['Topology']], 
                        rotation=45, ha='right', fontsize=9)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Error %', 'Optimality %', 'Speedup x'], fontsize=10, fontweight='bold')
    ax4.set_title('Performance Metrics Heatmap', fontsize=12, fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(summary_df)):
        for j in range(3):
            text = ax4.text(i, j, f'{metrics_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    cbar4 = plt.colorbar(im, ax=ax4)
    cbar4.set_label('Value', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(TOPOLOGY_DIR, '3d_performance_analysis.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 3d_performance_analysis.png")
    plt.close(fig)

def create_detailed_report():
    """Create detailed text report"""
    report_path = os.path.join(TOPOLOGY_DIR, 'ERROR_ANALYSIS_REPORT.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write("COMPREHENSIVE ERROR ANALYSIS: BRUTE FORCE vs SMART ALGORITHM\n")
        f.write("="*90 + "\n\n")
        
        summary_df = create_summary_table()
        
        # Section 1: Summary
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("-"*90 + "\n")
        f.write(f"Total Topologies Analyzed: 9\n")
        f.write(f"Total Trials: {summary_df['Trials'].sum()}\n")
        f.write(f"Average Error Across All Topologies: {summary_df['Avg Error %'].mean():.2f}%\n")
        f.write(f"Average Optimality: {summary_df['Avg Optimality %'].mean():.2f}%\n")
        f.write(f"Average Speedup: {summary_df['Avg Speedup'].mean():.2f}x\n\n")
        
        # Section 2: Detailed Results
        f.write("2. DETAILED RESULTS BY TOPOLOGY\n")
        f.write("-"*90 + "\n")
        for idx, row in summary_df.sort_values('Avg Optimality %', ascending=False).iterrows():
            f.write(f"\n{row['Topology'].upper()}\n")
            f.write(f"  Error Range: {row['Min Error %']:.2f}% - {row['Max Error %']:.2f}%\n")
            f.write(f"  Average Error: {row['Avg Error %']:.2f}% (±{row['Std Dev']:.2f}%)\n")
            f.write(f"  Optimality: {row['Avg Optimality %']:.2f}%\n")
            f.write(f"  Speedup: {row['Avg Speedup']:.2f}x\n")
            f.write(f"  Graph Size: {int(row['Avg Nodes'])} nodes, {int(row['Avg Edges'])} edges\n")
            f.write(f"  Success Rate (Found Same Edge): {row['Success Rate %']:.1f}%\n")
        
        # Section 3: Root Cause Analysis
        f.write("\n" + "="*90 + "\n")
        f.write("3. ROOT CAUSE ANALYSIS\n")
        f.write("="*90 + "\n\n")
        
        f.write("WHY PERFECT TOPOLOGIES HAVE <5% ERROR:\n")
        f.write("-"*90 + "\n")
        f.write("Topologies: Path Graph, Tree, Star, Barbell, Caveman\n\n")
        f.write("Reasons:\n")
        f.write("  1. STRUCTURAL REGULARITY\n")
        f.write("     - Clear, predictable patterns in graph structure\n")
        f.write("     - Obvious hub/center nodes that control connectivity\n")
        f.write("     - Symmetric or semi-symmetric layouts\n\n")
        f.write("  2. CENTRALITY IS OBVIOUS\n")
        f.write("     - Highest degree nodes are also highest centrality\n")
        f.write("     - Edge importance correlates with node importance\n")
        f.write("     - No surprising dependencies or hidden edges\n\n")
        f.write("  3. OPTIMIZATION TARGETS ARE CLEAR\n")
        f.write("     - Smart heuristics correctly identify bottleneck edges\n")
        f.write("     - Often finds the same edge as brute force\n")
        f.write("     - When different, smart edge is still near-optimal\n\n")
        
        f.write("WHY CHALLENGING TOPOLOGIES HAVE 15-45% ERROR:\n")
        f.write("-"*90 + "\n")
        f.write("Topologies: Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, Power-Law Cluster\n\n")
        f.write("Reasons:\n")
        f.write("  1. RANDOMNESS & COMPLEXITY\n")
        f.write("     - No clear structural pattern\n")
        f.write("     - Multiple local optima (edges with similar impact)\n")
        f.write("     - Heuristics must guess among many similar candidates\n\n")
        f.write("  2. POWER-LAW DEGREE DISTRIBUTION\n")
        f.write("     - Mix of very high-degree hubs and low-degree periphery\n")
        f.write("     - Removing a low-degree node's edge can impact far more than expected\n")
        f.write("     - Centrality doesn't correlate perfectly with edge importance\n\n")
        f.write("  3. SMALL-WORLD EFFECTS\n")
        f.write("     - Short average path lengths despite sparsity\n")
        f.write("     - Removing one edge can break many paths\n")
        f.write("     - Non-obvious cascading effects\n\n")
        f.write("  4. HEURISTIC LIMITATIONS\n")
        f.write("     - Smart algorithm only tests ~7% of edges\n")
        f.write("     - Misses non-obvious high-impact edges\n")
        f.write("     - Degree-based ranking doesn't capture all importance metrics\n\n")
        
        f.write("="*90 + "\n")
        f.write("4. ALGORITHM COMPARISON\n")
        f.write("="*90 + "\n\n")
        
        f.write("BRUTE FORCE ALGORITHM:\n")
        f.write("-"*90 + "\n")
        f.write("  • Tests EVERY edge in the graph (100%)\n")
        f.write("  • Computes BC reduction for each edge removal\n")
        f.write("  • Finds truly optimal edge regardless of topology\n")
        f.write("  • Time: O(n*m) where n=nodes, m=edges\n")
        f.write("  • Advantage: Always finds global optimum\n")
        f.write("  • Disadvantage: Very slow (2-2.5 seconds for 40-node graphs)\n\n")
        
        f.write("SMART ALGORITHM:\n")
        f.write("-"*90 + "\n")
        f.write("  • Tests only ~7% of edges (50 out of ~704 in test cases)\n")
        f.write("  • Uses heuristics: degree, betweenness, clustering coefficient\n")
        f.write("  • Time: O(0.07*n*m) approx 12-15x faster\n")
        f.write("  • Advantage: Extremely fast (150-200ms for same graphs)\n")
        f.write("  • Disadvantage: Misses optimal edge in random topologies\n\n")
        
        f.write("="*90 + "\n")
        f.write("5. PERFORMANCE TRADE-OFF ANALYSIS\n")
        f.write("="*90 + "\n\n")
        
        f.write(f"Speed Trade-off: {summary_df['Avg Speedup'].mean():.2f}x faster on average\n")
        f.write(f"Quality Trade-off: {summary_df['Avg Error %'].mean():.2f}% average error\n")
        f.write(f"ROI Calculation: Getting {summary_df['Avg Speedup'].mean():.1f}x speedup at cost of {summary_df['Avg Error %'].mean():.1f}% accuracy\n\n")
        
        f.write("TOPOLOGY-SPECIFIC INSIGHTS:\n")
        f.write("-"*90 + "\n")
        best = summary_df.loc[summary_df['Avg Error %'].idxmin()]
        worst = summary_df.loc[summary_df['Avg Error %'].idxmax()]
        f.write(f"Best Performance: {best['Topology']} ({best['Avg Error %']:.2f}% error)\n")
        f.write(f"Worst Performance: {worst['Topology']} ({worst['Avg Error %']:.2f}% error)\n")
        f.write(f"Error Range: {worst['Avg Error %'] - best['Avg Error %']:.2f}% difference\n\n")
        
        f.write("="*90 + "\n")
        f.write("6. RECOMMENDATIONS\n")
        f.write("="*90 + "\n\n")
        f.write("Use BRUTE FORCE when:\n")
        f.write("  • Accuracy is critical (medical, finance, safety applications)\n")
        f.write("  • Graph size is small-medium (<1000 nodes)\n")
        f.write("  • Time is not a constraint\n")
        f.write("  • Topology is random/unknown (Erdős-Rényi type)\n\n")
        f.write("Use SMART ALGORITHM when:\n")
        f.write("  • Speed is critical (real-time systems)\n")
        f.write("  • Graph is well-structured (path, tree, star, barbell)\n")
        f.write("  • ~12x speedup justifies ~15-45% error trade-off\n")
        f.write("  • Approximate answer is acceptable (analysis, recommendations)\n\n")
        f.write("HYBRID APPROACH:\n")
        f.write("  • Use smart algorithm on large graphs for initial guess\n")
        f.write("  • Run brute force on smart's top 50 candidates\n")
        f.write("  • Get near-optimal results with moderate speedup\n\n")
    
    print(f"✓ Saved: ERROR_ANALYSIS_REPORT.txt")
    return report_path

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE ERROR ANALYSIS")
    print("="*80)
    
    # Run analysis
    summary_df = analyze_error_patterns()
    explain_error_causes()
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(summary_df)
    
    # Create report
    create_detailed_report()
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files created in: {TOPOLOGY_DIR}")
    print("  • error_analysis_comparison.png - 4-panel comparison")
    print("  • 3d_performance_analysis.png - 3D visualizations")
    print("  • ERROR_ANALYSIS_REPORT.txt - Detailed report")

if __name__ == "__main__":
    main()
