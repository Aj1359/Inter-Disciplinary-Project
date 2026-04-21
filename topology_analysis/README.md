# Topology Analysis Framework

This directory contains automated analysis tools for comparing Brute Force vs Smart (Hop-Based) algorithms across 6 different graph topologies.

## Directory Structure

```
topology_analysis/
├── analyze_topology.py           # Core analysis functions (shared)
├── run_all_topologies.py         # Master runner (executes all topologies)
├── MASTER_SUMMARY.txt            # Generated master report (after running)
├── summary.json                  # Generated JSON summary (after running)
│
├── erdos_renyi/                  # Erdos-Renyi random graphs
│   ├── run_analysis.py           # Run this topology
│   ├── results.json              # Generated results
│   ├── REPORT.txt                # Generated text report
│   └── analysis_plots.png        # Generated visualization
│
├── barabasi_albert/              # Preferential attachment
│   ├── run_analysis.py
│   ├── results.json
│   ├── REPORT.txt
│   └── analysis_plots.png
│
├── watts_strogatz/               # Small-world networks
│   ├── run_analysis.py
│   ├── results.json
│   ├── REPORT.txt
│   └── analysis_plots.png
│
├── path_graph/                   # Linear chains
│   ├── run_analysis.py
│   ├── results.json
│   ├── REPORT.txt
│   └── analysis_plots.png
│
├── barbell/                      # Two cliques with bridge
│   ├── run_analysis.py
│   ├── results.json
│   ├── REPORT.txt
│   └── analysis_plots.png
│
└── star/                         # Star/wheel networks
    ├── run_analysis.py
    ├── results.json
    ├── REPORT.txt
    └── analysis_plots.png
```

## Quick Start

### Run All Topologies (Recommended)
```bash
cd topology_analysis
python run_all_topologies.py
```

This will execute all 6 topology analyses sequentially and generate:
- Individual results and reports in each topology subdirectory
- `MASTER_SUMMARY.txt` - comparative overview
- `summary.json` - structured results

### Run Individual Topology
```bash
cd topology_analysis/erdos_renyi
python run_analysis.py
```

## Graph Topologies Analyzed

### 1. Erdos-Renyi (n=40, p=0.12)
- **Type**: Random graph model
- **Parameters**: 40 nodes, edge probability 0.12
- **Characteristics**: Relatively sparse, uniformly random connections
- **Expected**: Moderate speedup and good optimality

### 2. Barabasi-Albert (n=40, m=2)
- **Type**: Preferential attachment (scale-free)
- **Parameters**: 40 nodes, 2 edges per new node
- **Characteristics**: Power-law degree distribution, hub nodes
- **Expected**: High speedup (hubs reduce search space)

### 3. Watts-Strogatz (n=40, k=4, p=0.3)
- **Type**: Small-world network
- **Parameters**: 40 nodes, 4 neighbors, 0.3 rewiring probability
- **Characteristics**: Low diameter, high clustering
- **Expected**: Moderate speedup, balanced performance

### 4. Path Graph (n=20)
- **Type**: Linear chain
- **Parameters**: 20 nodes in a line
- **Characteristics**: Extremely sparse, linear structure
- **Expected**: Low speedup (few edges to add)

### 5. Barbell Graph (m=7, p=2)
- **Type**: Two cliques connected by bridge
- **Parameters**: Two cliques of 7 nodes, 2-node bridge
- **Characteristics**: Bipartite-like, extreme centrality on bridge
- **Expected**: Very high speedup (targeted search on bridge area)

### 6. Star Graph (n=20)
- **Type**: Hub-and-spoke
- **Parameters**: 1 central node, 19 peripheral nodes
- **Characteristics**: Center has very high centrality
- **Expected**: High speedup (hub is target, clear 1-hop/2-hop zones)

## Output Files

### For Each Topology Folder:

#### `results.json`
Structured results containing:
- 12 individual trial records with:
  - Graph properties (n_nodes, n_edges, target node BC)
  - Brute Force results (time, candidates, reduction %, edge)
  - Smart Algorithm results (time, candidates, reduction %, edge)
  - Comparison metrics (speedup, optimality %, same_edge_found)
- Summary statistics:
  - Mean/std/min/max for speedup, optimality
  - Runtime comparisons
  - Candidate ratio reduction
  - Number of trials with identical solutions

#### `REPORT.txt`
Human-readable text report containing:
- Execution summary (total trials, success rate)
- Performance metrics (speedup, optimality, runtime, reduction)
- Trial-by-trial table with key metrics
- Conclusions about topology performance

#### `analysis_plots.png`
6-panel visualization showing:
1. Runtime comparison (Brute Force vs Smart bars)
2. Speedup per trial (line plot)
3. Optimality per trial (line plot)
4. BC reduction comparison (Brute Force vs Smart bars)
5. Candidate count reduction (line plot)
6. Same edge found indicator (bar chart)

### Master-Level Files:

#### `MASTER_SUMMARY.txt`
Comparative overview across all 6 topologies with:
- Quick overview table
- Average speedup and optimality
- File locations for each topology
- References to detailed reports

#### `summary.json`
Structured summary with speedup and optimality for each topology

## Key Parameters

All analyses use consistent parameters for fair comparison:

- **tau (load factor)**: 0.20 (20% threshold)
- **topk**: 50 (maximum candidates for smart algorithm)
- **trials per topology**: 12
- **Algorithm**: Brandes' betweenness centrality (O(nm))
- **Constraint**: Load-balance on max BC increase

## Interpretation Guide

### Speedup
- **Formula**: Time_BruteForce / Time_Smart
- **Value >1**: Smart is faster
- **Typical range**: 1.5× to 18×
- **What it means**: Algorithm acceleration factor

### Optimality (%)
- **Formula**: (Smart_reduction / BruteForce_reduction) × 100
- **Perfect**: 100%
- **Typical range**: 94% to 100%
- **What it means**: How close smart solution is to brute force

### Candidate Reduction (%)
- **Formula**: (Smart_candidates / BruteForce_candidates) × 100
- **Lower is better**: Fewer candidates to evaluate
- **Typical range**: 0.5% to 5%
- **What it means**: Hop-based search space savings

## Statistical Measures

- **Mean**: Average value across all trials
- **Std**: Standard deviation (variability)
- **Min/Max**: Range of values
- **Same Edge Found**: % of trials where both algorithms found identical edge

## Requirements

- Python 3.7+
- networkx
- numpy
- scipy (for statistics)
- matplotlib

Install with:
```bash
pip install networkx numpy scipy matplotlib
```

## Troubleshooting

### Import errors
Ensure the script is run from the topology_analysis directory or that parent directory is in PYTHONPATH.

### Graph generation fails
Verify networkx is installed: `pip install networkx`

### Memory issues
Reduce the number of trials or node counts in individual run_analysis.py scripts.

### Timeout
Increase timeout in run_all_topologies.py (currently 300 seconds per topology).

## Performance Notes

- Each topology analysis takes ~1-5 minutes depending on parameters
- Total time for all 6 topologies: ~10-30 minutes
- Results are saved after each topology completes
- Can safely interrupt and resume individual topologies

## Customization

To modify parameters, edit the relevant run_analysis.py file:

```python
results = analyze_topology(
    topology_name="Custom Name",
    graph_generator=your_generator_function,
    output_dir=output_dir,
    num_trials=12,  # Change number of trials
    tau=0.20,       # Change load factor
    topk=50         # Change candidate limit
)
```

---

**For questions or issues, check the generated REPORT.txt files for detailed diagnostics.**

