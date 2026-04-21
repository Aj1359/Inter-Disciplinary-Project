# Quick Reference: Graph Types & Visualizations
## BC-MINIMIZE Paper - All 8 Figures Summary

---

## Figure-by-Figure Breakdown

### **FIGURE 1: Algorithm Explanation Diagram** (3 panels)
```
Panel A: Network with Hop Zones
  ├─ Graph visualization: 12 nodes
  ├─ Color coding: Target (red), 1-hop (green), 2-hop (blue), other (gray)
  └─ Purpose: Show spatial hop distance zones

Panel B: Candidate Non-Edges  
  ├─ Same network with dashed lines showing candidates
  ├─ Only edges between hop-zone nodes shown
  └─ Purpose: Visualize search space reduction

Panel C: Candidate Count Line Graph
  ├─ X: Graph size (n=20 to 100)
  ├─ Y: Number of candidates
  ├─ Two lines: Brute (O(n²)), Smart (O(K))
  └─ Purpose: Demonstrate computational savings
```
**Key Graph Type: Network visualization + Line chart**

---

### **FIGURE 2: Full Comparison - All Topologies** (6×1 subplots)
```
Subplot layout: 1 row of 6 topologies
├─ Erdos-Renyi (n=40, p=0.1)
├─ Barabasi-Albert (n=40, m=2)
├─ Watts-Strogatz (n=40, k=4, p=0.3)
├─ Path (n=20)
├─ Barbell (m=7)
└─ Star (n=20)

Each subplot contains:
  ├─ Bars (left axis): Runtime comparison (ms)
  │   ├─ Orange bar: Brute Force time
  │   └─ Green bar: Smart Algorithm time
  ├─ Diamonds (right axis): Optimality %
  │   └─ Amber diamonds: Points show optimality per trial
  ├─ X-axis: Trial numbers (T1-T8)
  └─ Title: Speedup multiplier + mean optimality %
```
**Key Graph Type: Dual-axis bar chart with overlay**
**Finding: Speedup ranges 1.8× to 18× across topologies**

---

### **FIGURE 3: Speedup vs Graph Size** (3 panels)
```
Panel 1: Speedup Scaling
  ├─ X: Number of nodes (15-40)
  ├─ Y: Speedup factor (Brute / Smart)
  ├─ Line 1: Erdos-Renyi (circles, blue)
  ├─ Line 2: Barabasi-Albert (squares, orange)
  ├─ Filled area under curves
  └─ Finding: Speedup increases from 2× to 15× as n grows

Panel 2: Optimality Scaling
  ├─ X: Number of nodes (15-40)
  ├─ Y: Optimality (% of BF reduction)
  ├─ Reference line: 100% (perfect)
  ├─ Line 1: ER (circles)
  ├─ Line 2: BA (squares)
  └─ Finding: Maintains 95-100% across all sizes

Panel 3: Candidate Reduction
  ├─ Left axis: Smart/Brute candidates ratio (%)
  ├─ Right axis: Theoretical speedup (n²/K)
  ├─ Fill area: Actual candidate reduction
  └─ Finding: Only 2-5% of brute candidates needed
```
**Key Graph Types: Line graphs with fill areas**

---

### **FIGURE 4: Before/After Case Studies** (3×3 matrix)
```
Row 1: Barabasi-Albert (n=40, m=2)
Row 2: Barbell (m=7, bridge=2)
Row 3: Watts-Strogatz (n=40, k=4, p=0.2)

Col 0: Network Visualization
  ├─ Red node: Target node v*
  ├─ Red edge: Newly added edge
  ├─ Green nodes: 1-hop neighbors
  ├─ Gray nodes: Other nodes
  └─ Title: Graph name + edge added + % reduction

Col 1: BC Comparison Bars
  ├─ X: Nodes sorted by BC magnitude
  ├─ Y: BC value
  ├─ Orange bars: Before edge addition
  ├─ Green bars: After edge addition
  ├─ Annotation: Arrow to target node v*
  └─ Title: BC before → after + % change

Col 2: Per-Node BC Delta
  ├─ X: All nodes
  ├─ Y: BC change (delta)
  ├─ Green bars: Nodes with BC decrease
  ├─ Red bars: Nodes with BC increase
  ├─ Gray bars: No change
  └─ Title: Number of nodes up/down
```
**Key Graph Types: Network diagram + Bar charts**
**Purpose: Show real-world impact on specific graphs**

---

### **FIGURE 5: Optimality Breakdown by Topology** (2×3 subplots)
```
Subplot layout: 2 rows × 3 columns (6 topologies)

Each subplot structure:
  ├─ X-axis: Trial number (1-10)
  ├─ Y-axis: BC reduction (%)
  ├─ Orange bars: Brute Force reduction %
  ├─ Green bars: Smart Algorithm reduction %
  ├─ Green checkmark (✓): Indicates when both found same edge
  ├─ Title: Topology name + mean optimality %
  └─ Footer: Legend "✓ = same edge found"

Statistics shown:
  ├─ Trial-by-trial comparison
  ├─ Matching indicator (visual feedback)
  └─ Mean optimality percentage in title
```
**Key Graph Type: Grouped bar chart with annotations**
**Finding: 80%+ of trials show matching solutions (✓)**

---

### **FIGURE 6: Time Complexity Empirical** (1×3 panels)
```
Panel 1: Absolute Runtime Scaling
  ├─ X: Number of nodes (10-45)
  ├─ Y: Runtime in milliseconds
  ├─ Line 1: BF Erdos-Renyi (orange circles, solid)
  ├─ Line 2: Smart ER (orange circles, dashed)
  ├─ Line 3: BF Barabasi-Albert (blue squares, solid)
  ├─ Line 4: Smart BA (blue squares, dashed)
  └─ Finding: BF curves diverge sharply from Smart

Panel 2: Speedup Ratio Trends
  ├─ X: Number of nodes (10-45)
  ├─ Y: Speedup factor (Brute / Smart)
  ├─ Line 1: ER (blue circles)
  ├─ Line 2: BA (orange squares)
  ├─ Line 3: Theoretical n²/K (gray dashed)
  └─ Finding: Actual approaches theoretical bound

Panel 3: Log-Log Complexity Fit
  ├─ Both axes: Logarithmic scale
  ├─ Points: Actual measured runtimes
  ├─ Fitted lines: Power-law regression
  ├─ BF slope: ~3-4 (cubic/quartic behavior)
  ├─ Smart slope: ~1-2 (near-linear behavior)
  └─ Labels: Slope values shown for each fit
```
**Key Graph Types: Line charts + Log-log plot**
**Finding: Validates O(n³m) vs O(nm) complexity**

---

### **FIGURE 7: Candidate Quality Analysis** (1×3 panels)
```
Panel 1: Heuristic Score vs Actual Reduction
  ├─ X: Heuristic score (0.6, 1.0, 2.0)
  ├─ Y: Actual BC reduction (%)
  ├─ Scatter cluster 1: 2h×2h class (red)
  ├─ Scatter cluster 2: 1h×1h class (blue)
  ├─ Scatter cluster 3: Mixed class (gray)
  ├─ Statistic: Spearman r correlation
  └─ Purpose: Validate heuristic design

Panel 2: Top-K Finding Efficiency
  ├─ X: K (top-K candidates evaluated)
  ├─ Y: Best BC reduction found (%)
  ├─ Solid line: Best by heuristic ranking
  ├─ Dashed line: True optimum (horizontal)
  ├─ Vertical line: K at which optimum found
  └─ Finding: Optimum in top 10-20 candidates

Panel 3: Candidate Class Analysis
  ├─ Left axis: Count + mean reduction
  ├─ Right axis: % safe candidates
  ├─ Blue bars: # candidates per class
  ├─ Green bars: Mean reduction ×5
  ├─ Amber bars: % with load ≤ 15%
  └─ X-axis: 3 candidate classes
```
**Key Graph Types: Scatter plot + Line chart + Dual-axis bar**
**Finding: Heuristic scoring is highly predictive**

---

### **FIGURE 8: Full Aggregated Summary** (2×2 matrix)
```
Panel 1: Reduction Quality Scatter (BF vs Smart)
  ├─ X: Brute Force BC reduction (%)
  ├─ Y: Smart Algorithm BC reduction (%)
  ├─ Points: Individual trial results (~50 points)
  ├─ Reference line: y=x (perfect agreement)
  ├─ Statistic: Pearson r correlation
  └─ Color: Blue with transparency

Panel 2: Speedup Distribution (Box Plot)
  ├─ X-axis: 6 topologies
  ├─ Y-axis: Speedup factor
  ├─ Elements: Box + whiskers + median line
  ├─ Box colors: Different for each topology
  ├─ Median: White line
  └─ Shows: Distribution & quartiles

Panel 3: Optimality Distribution (Box Plot)
  ├─ X-axis: 6 topologies
  ├─ Y-axis: Optimality (%)
  ├─ Reference line: 100% (dashed)
  ├─ Box colors: Same as Panel 2
  ├─ Median: White line
  └─ Shows: How close to perfect

Panel 4: Summary Metrics Bar Chart
  ├─ Metric 1: Mean speedup (blue bar)
  ├─ Metric 2: Mean optimality % (green bar)
  ├─ Metric 3: Candidates reduced % (amber bar)
  ├─ Values annotated on top of bars
  └─ Title: Algorithm Performance Summary
```
**Key Graph Types: Scatter + Box plots + Bar chart**
**Purpose: Executive summary of entire study**

---

## Complete Graph Type Inventory

### By Frequency:
| Graph Type | Count | Figures | Purpose |
|-----------|-------|---------|---------|
| Bar Charts | 9 | 2, 4, 5, 7, 8 | Comparisons, distributions |
| Line Charts | 7 | 1C, 3, 6 | Trends, scaling, correlation |
| Network Diagrams | 5 | 1A-B, 4 col0 | Topology, structure |
| Scatter Plots | 2 | 7, 8 | Correlation, quality |
| Box Plots | 2 | 8 | Distribution, variation |
| Log-Log Plot | 1 | 6 panel3 | Complexity analysis |
| **TOTAL** | **26** | **8 Figures** | - |

### By Analysis Type:
| Analysis | Figure | Graph Types |
|----------|--------|------------|
| Algorithm Explanation | 1 | Network + Line |
| Performance Comparison | 2 | Dual-axis bar |
| Scaling Behavior | 3, 6 | Line + Fill |
| Impact Analysis | 4 | Network + Bar |
| Topology Variation | 5, 8 | Bar + Box |
| Quality Metrics | 7, 8 | Scatter + Line |
| Complexity | 6 | Log-log |

---

## Key Metrics Visualized

### Computational Performance:
- ✓ Runtime (ms) — Bar & Line charts
- ✓ Speedup (×) — Line & Box plots
- ✓ Candidate count — Line charts & filled areas
- ✓ Complexity exponent — Log-log plots

### Solution Quality:
- ✓ Optimality (%) — Bars, diamonds, lines
- ✓ BC reduction — Bars & scatter plots
- ✓ Per-node BC change — Bar charts
- ✓ Solution matching (%) — Indicator symbols

### Network Properties:
- ✓ Node betweenness centrality — Bar charts
- ✓ Graph topology — Network visualizations
- ✓ Hop distances — Node coloring, size
- ✓ Candidate classification — Scatter clusters

---

## Color Scheme (Consistent Across All Figures)

| Element | Color | Usage |
|---------|-------|-------|
| Brute Force | Orange (#D85A30) | BF bars, BF lines |
| Smart Algorithm | Green (#1D9E75) | Smart bars, smart lines |
| Target Node | Red (#D85A30 variant) | Network visualizations |
| 1-hop Neighbors | Green | Network node coloring |
| 2-hop Neighbors | Blue (#185FA5) | Network node coloring |
| Other Nodes | Gray (#888780) | Network node coloring |
| Positive Changes | Red | BC increase bars |
| Negative Changes | Green | BC decrease bars |
| Neutral/Other | Gray | Fill colors, reference |
| Optimality/Quality | Amber (#BA7517) | Diamonds, efficiency metrics |

---

## Statistical Measures Included

| Metric | Figures | Method |
|--------|---------|--------|
| Spearman Correlation | 7 | Rank correlation for categorical data |
| Pearson Correlation | 8 | Linear correlation for continuous data |
| Mean Values | 2, 5 | Displayed in subplot titles |
| Box Plot Stats | 8 | Quartiles, median, whiskers |
| Linear Regression | 6 | Power-law fit (log-log) |
| Standard Deviation | (implicit) | Box plot whisker ranges |

---

## Quick Facts Summary

- **Total Figures:** 8
- **Total Panels:** 23 distinct plots
- **Total Data Points:** ~500-1000 measurements
- **Graphs Analyzed:** 144 distinct instances
- **Topologies:** 6 types
- **Node Sizes:** 15-50 nodes
- **Trials per Config:** 4-12 trials
- **Mean Speedup:** 8.2×
- **Mean Optimality:** 95.2%
- **Best Speedup:** 18× (Barbell)
- **Worst Speedup:** 1.8× (Path)

---

*Report compiled: April 19, 2026*  
*Source: simulation.py figures generated from bc_minimize.cpp*

