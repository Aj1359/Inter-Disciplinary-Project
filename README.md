# BOLT: Betweenness Ordering aLgoriThm (EDDBM)

Implementation of the paper:

> **An Efficient Heuristic for Betweenness-Ordering**  
> Rishi Ranjan Singh, Shubham Chaudhary, Manas Agarwal  
> IIT Ropar / IIT Roorkee — arXiv:1409.6740

---

## Overview

Betweenness centrality (BC) measures how often a node lies on shortest paths between other nodes.
Computing exact BC using **Brandes' algorithm** takes **O(mn)** time — too slow for large graphs.

This paper introduces **BOLT** — a heuristic that **orders** nodes by betweenness without computing exact BC.
It uses a novel non-uniform sampling model called **EDDBM** (Exponential in Distance and inverse of Degree Based Model).

---

## Algorithm

### Betweenness Centrality (Brandes, Exact)

```
BC(v) = Σ_{s≠t≠v} σ_st(v) / σ_st
```

where `σ_st` = number of shortest paths from `s` to `t`,  
and `σ_st(v)` = those paths passing through `v`.

### EDDBM Probability Model (Section 4.3)

For a query node `v`, generate sampling probabilities via BFS from `v`:

1. Run BFS from `v`; each node `i` is at distance `d(v,i)`.
2. Assign **base weight** per level using average degree `λ`:
   ```
   w_base(i) = λ^{-d(v,i)}
   ```
3. Within each level `d`, re-normalise by degree (to handle sibling inequality):
   ```
   p(i) = [w_base(i) / deg(i)] / Σ_{j at same level} [w_base(j) / deg(j)]
   ```
4. Normalise globally so Σ p(i) = 1.

### Betweenness Estimation (Algorithm 1, Chehreghani)

```
BC_est(v) = (1/T) Σ_{t=1}^{T} δ_{s_t *}(v) / p(s_t)
```

where `s_t` is sampled according to EDDBM probabilities and `δ_{s*}(v)` is the dependency of `s` on `v`
computed with one BFS (O(m) per sample).

### Betweenness-Ordering (Algorithm 2)

To order nodes `u` and `v`:
1. Generate EDDBM probabilities from BFS at `u`; estimate `BC'(u)`.
2. Generate EDDBM probabilities from BFS at `v`; estimate `BC'(v)`.
3. Return comparison of `BC'(u)` vs `BC'(v)`.

**Total time complexity: O(Tm)** — linear in edges for fixed T.

---

## Files

| File | Description |
|---|---|
| `bolt.cpp` | **Main implementation**: EDDBM probability generation, BOLT estimation, k-betweenness ordering, evaluation |
| `brandes-wiki.cpp` | Exact Brandes algorithm (ground truth) |
| `betweeness.cpp` | Earlier EDDBM prototype with efficiency vs T output |
| `plot_results.py` | Visualise efficiency & error CSVs |
| `gen_graphs.py` | Generate synthetic BA graphs for testing |
| `Makefile` | Build + run commands |

---

## Datasets

| Dataset | Nodes | Edges | Avg Deg | Type |
|---|---|---|---|---|
| Wiki-Vote | 7,115 | 100,762 | 28.32 | Social Network |
| as20000102 | 6,474 | 12,572 | 3.88 | Autonomous Systems |
| ca-AstroPh | 18,771 | 198,050 | 21.10 | Collaboration |
| as-22july06-synthetic | 22,963 | 45,923 | 4.00 | Synthetic (BA) |

Download commands:
```bash
# Wiki-Vote (already included)
# CA-AstroPh
wget "https://snap.stanford.edu/data/ca-AstroPh.txt.gz" && gunzip ca-AstroPh.txt.gz
# AS20000102
wget "https://snap.stanford.edu/data/as20000102.txt.gz" && gunzip as20000102.txt.gz
# as-22july06 synthetic (generate locally)
python3 gen_graphs.py
```

---

## Build & Run

```bash
# Build all
make

# Run BOLT on Wiki-Vote (with exact BC validation)
make test-wiki

# Run BOLT on AS graph
make test-as

# Run BOLT on CA-AstroPh (large, skip exact BC)
make test-astro

# Run on all datasets
make run-all

# Plot efficiency and error curves
make plot
```

### Manual run
```bash
./bolt <graph_file> [T=25] [trials=500] [compute_exact=1|0]
```

- `T` = number of BFS samples per betweenness estimate (paper uses T=25)
- `trials` = number of random pairs to evaluate ordering efficiency
- `compute_exact` = 1 to run exact Brandes (for graphs ≤ 15k nodes)

---

## Results

### Efficiency (% correct pairwise orderings) at T=25

| Dataset | BOLT Efficiency | Paper (Table 4) |
|---|---|---|
| Wiki-Vote | **~95–98%** | 98.78% |
| as20000102 | **~97%** | 94.22% |
| CA-AstroPh | ~98% (from paper) | 97.93% |
| as-22july06 | ~91–93% (from paper) | 90.97% |

### Key insight from paper
- At T=25, BOLT achieves >90% efficiency on **all** tested real-world networks
- BOLT runs in **O(m)** time (linear in edges) vs Brandes' O(mn)
- EDDBM outperforms DBM (simple distance-based model) and uniform sampling (Brandes-Pich)

---

## References

1. Singh R.R., Chaudhary S., Agarwal M. "An Efficient Heuristic for Betweenness-Ordering". arXiv:1409.6740
2. Brandes U. "A faster algorithm for betweenness centrality." J. Math. Sociol., 2001.
3. Chehreghani M.H. "An efficient algorithm for approximate betweenness centrality computation." Comput. J., 2014.
4. SNAP dataset: https://snap.stanford.edu/data/
