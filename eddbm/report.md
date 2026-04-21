# EDDBM (Efficient Dynamic Degree-Based Model) Analysis Report

## Overview
EDDBM is an approximation algorithm for Betweenness Centrality (BC) in graphs, using degree-based sampling to estimate centrality values efficiently compared to exact Brandes algorithm.

## Improvements Attempted

### 1. Hybrid ML Regressor (`improving_eddbm/ml_regressor/`)
- **Approach**: Combines raw EDDBM estimates with graph structural features (degree, clustering) into a Gradient Boosting Regressor to predict exact BC.
- **Goal**: Reduce random sampling noise.
- **Implementation**: Python script using scikit-learn.

### 2. Laplacian Neighborhood Smoothing (`improving_eddbm/neighborhood_smooth/`)
- **Approach**: Smooths EDDBM estimates by averaging with neighbors using Laplacian formula: `smooth(v) = (1-α) * raw(v) + α * mean(neighbors)`.
- **Goal**: Leverage local BC correlations to reduce variance.
- **Implementation**: O(m) pass in Python.

### 3. Top-T Deterministic Sampling (`improving_eddbm/top_t_sampling/`)
- **Approach**: Selects top T nodes by EDDBM probability weights instead of random sampling, computes exact dependencies.
- **Goal**: Eliminate sampling variance for better accuracy.
- **Implementation**: Deterministic selection in Python.

### 4. Improved EDDBM v2 (`improved_eddbm_v2/`)
- **Key Fixes**:
  - BFS subgraph loading from high-degree seed for dense, connected graphs.
  - Target selection by true BC values.
  - Strict pooled sampling: Pool P_vi with sqrt(deg) weighting, S = T*sqrt(K) sources shared across K targets.
- **Goal**: Better accuracy and efficiency.
- **Implementation**: Optimized C++ with shared BFS.

### 5. CAEDDBM and PDEDDBM (`improving_eddbm/`)
- Variants tested on datasets like Wiki-Vote, CA-HepTh.
- Results in smoothed CSV files.

## Success Metrics
- **Efficiency**: Percentage of correct pairwise rankings.
- **AvgError**: Average relative error.
- **Runtime**: Wall-clock time, speedup.

## Results Summary

### Successes
- **Improved EDDBM v2**: Achieved 3-4x speedup, higher accuracy (89-99% vs. 89-98%) on Wiki-Vote, reduced error in some T sweeps.
- **PDEDDBM on dense graphs**: Lower error (e.g., 0.019-0.04 vs. 0.03-0.05 on AS20000102).
- Laplacian smoothing and ML reduced noise in correlated graphs.

### Failures
- **Inconsistent across datasets**: On Wiki-Vote, improved methods had higher error (0.52-0.92 vs. 0.49-0.60) and lower efficiency (72-96% vs. 88-96%).
- **Efficiency trade-offs**: Many variants reduced efficiency without proportional error gains.
- **Overhead**: ML and smoothing added complexity without universal improvement.
- **Dataset sensitivity**: Degree-based sampling failed on non-degree-centric graphs (e.g., social networks).

## Conclusion
Improvements partially succeeded in specific scenarios (dense graphs, v2 optimizations), but failed to consistently outperform original EDDBM due to dataset-specific issues and efficiency losses. Future work should focus on adaptive, graph-aware sampling.

## Files
- Original: `eddbm_model/`
- Improvements: `improving_eddbm/`, `improved_eddbm_v2/`
- Plots: Generated with resampling for smoothness.