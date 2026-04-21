# BC-Minimize: Betweenness Centrality Minimization via Single Edge Addition

## Quick Start

```bash
# Install dependencies
pip install networkx numpy matplotlib scipy scikit-learn

# Phase 1: Run benchmark on 100+ graphs (full ~20-30 min, quick ~3 min)
python run_full_benchmark.py --quick     # Quick test (~20 graphs)
python run_full_benchmark.py             # Full run (100+ graphs)

# Phase 2: Probability-enhanced algorithm
python probability_algorithm.py --test   # Quick test
python probability_algorithm.py          # Full run

# Phase 3: ML model (train + evaluate)
python ml_edge_predictor.py --test       # Quick test
python ml_edge_predictor.py              # Full pipeline

# Phase 4: Parallelization benchmark
python parallel_bc_minimize.py --test    # Quick test
python parallel_bc_minimize.py           # Full run

# Phase 5: Generate LaTeX paper
python generate_paper.py
# Then compile: pdflatex BC_Minimize_Paper.tex
```

## File Structure

| File | Description |
|------|-------------|
| `run_full_benchmark.py` | **Phase 1:** 100+ graph benchmark (10 topologies, speedup vs optimality) |
| `probability_algorithm.py` | **Phase 2:** Probability-weighted scoring with early stopping |
| `ml_edge_predictor.py` | **Phase 3:** GBT model for edge prediction (26 features) |
| `parallel_bc_minimize.py` | **Phase 4:** Multiprocessing parallel brute force |
| `generate_paper.py` | **Phase 5:** LaTeX paper generator |
| `simulation.py` | Original simulation script (8 figures) |
| `bc_minimize.cpp` | C++ implementation |
| `results/` | Auto-generated results (JSON, CSV, PNG figures) |

## Algorithms

1. **Brute Force**: Evaluates all non-edges, O(n³m) — ground truth
2. **Smart (Hop-Based)**: Only evaluates edges in 1-hop/2-hop zone of target, O(K·nm)
3. **Probability-Enhanced**: Structural feature scoring + early stopping
4. **ML-Guided**: GBT model predicts best edges from 26 features
5. **Parallel BF**: Multiprocessing for exact solutions with N workers
