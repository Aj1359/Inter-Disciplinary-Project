# Research Report: Impact of Candidate Search Depth (K) on BC Minimization

## 1. Executive Summary
This analysis investigates how the **Candidate Cap (K)** affects the optimality of the Betweenness Centrality (BC) minimization algorithm across 10 network topologies. We compared truncated search depths (K=30, 50, 100) against an exhaustive global search (ALL candidates).

## 2. Key Findings: The Bifurcation of Topologies
The results reveal two distinct classes of graph structures:

### A. "Heuristic-Optimal" Topologies (K=30 is sufficient)
For the following topologies, the algorithm achieved **100% optimality** with only 30 candidates:
- **Path Graphs**, **Barbell Graphs**, **Star Graphs**, and **Random Trees**.
- **Reasoning**: These graphs possess high structural "bottlenecks." The optimal reduction edge is almost always a "shortcut" that bridges the 1-hop or 2-hop neighborhood of the target node directly to a distant hub. Proximity-based heuristics are highly reliable here.

### B. "Search-Limited" Topologies (K=ALL is required)
For these graphs, there is a significant performance gap between Top-K and Exhaustive search:
- **Erdos-Renyi (Random)**: Optimality at K=100 was only **7.03%**, but jumped to **26.21%** at K=ALL.
- **Barabasi-Albert (Scale-Free)**: Optimality at K=100 was **9.16%**, jumping to **26.88%** at K=ALL.
- **Powerlaw Cluster**: Jumped from **11.6% (K=30)** to **27.25% (K=50)**.

---

## 3. Why Must "ALL" Be Analyzed in Some Cases?

In disorganized or highly random topologies (ER and BA), the 2-hop heuristic frequently fails for two mathematical reasons:

### I. The "Distant Bridge" Phenomenon
In an Erdos-Renyi graph, the shortest paths are distributed roughly uniformly across the graph. The "best" edge to reduce a target node's BC might not be a local shortcut at all. Instead, it could be a **long-range bridge** between two nodes that are 4 or 5 hops away from the target, which creates a new global "super-highway" that pulls traffic away from the target's entire neighborhood. These edges are strictly invisible to a 2-hop heuristic.

### II. Load Constraint Disqualification
In dense hubs (like Barabasi-Albert), adding an edge within the 2-hop zone often reduces the target BC significantly, but it simultaneously **creates a massive load spike** on the neighbors (the `tau` constraint). 
- At low K, the heuristic picks the "best" reduction edges, which are then disqualified by the load tracker.
- As K increases to "ALL", the algorithm eventually finds "quieter" edges in the graph's periphery that offer moderate BC reduction while safely staying under the load threshold.

## 4. Conclusion
While a Top-K=50 approach is sufficient for structured networks (achieving 100% optimality), it only captures **~25-30%** of the potential reduction in random social or biological networks. For high-precision research in these domains, an unrestricted candidate search is mathematically necessary to bypass the "local optimality traps" created by proximality heuristics.
