/**
 * ================================================================
 * BC-MINIMIZE: Betweenness Centrality Minimization
 * via Single Edge Addition
 * ================================================================
 *
 * Implements TWO approaches for comparison:
 *
 *  1. BRUTE FORCE  — tries every non-edge (u,w) in the complement
 *     graph, recomputes Brandes BC for each, picks the best.
 *     Complexity: O(n² × nm) = O(n³m)
 *
 *  2. SMART (Hop-Based) — analyzes only non-edges between
 *     1-hop and 2-hop neighbors of the target node.
 *     Complexity: O(K × nm), K << n²
 *
 * Both enforce a load-balance constraint: the maximum BC increase
 * on any non-target node must not exceed tau × avg_BC.
 *
 * ================================================================
 * COMPILE:
 *   g++ -O2 -std=c++17 -o bc_minimize bc_minimize.cpp
 *
 * USAGE:
 *   ./bc_minimize <graph_file> [target_node] [--tau F] [--topk N]
 *                [--brute] [--compare]
 *
 * OPTIONS:
 *   --brute      Run brute force only
 *   --compare    Run BOTH and print comparison table
 *   --tau F      Load factor threshold (default 0.15)
 *   --topk N     Candidate cap for smart algorithm (default 50)
 *   --target N   Target node (default = auto: highest BC)
 *
 * GRAPH FILE FORMAT (SNAP edge list):
 *   # comment lines ignored
 *   u v          (one edge per line, 0-indexed node IDs)
 * ================================================================
 */

#include <bits/stdc++.h>
using namespace std;

// ─── Graph ────────────────────────────────────────────────────
struct Graph {
  int n;
  vector<vector<int>> adj;
  set<pair<int, int>> edgeSet;

  Graph() : n(0) {}
  explicit Graph(int n) : n(n), adj(n) {}

  void addEdge(int u, int v) {
    if (u == v)
      return;
    auto key = make_pair(min(u, v), max(u, v));
    if (edgeSet.count(key))
      return;
    adj[u].push_back(v);
    adj[v].push_back(u);
    edgeSet.insert(key);
  }
  bool hasEdge(int u, int v) const {
    return edgeSet.count({min(u, v), max(u, v)}) > 0;
  }
  int edgeCount() const { return (int)edgeSet.size(); }
};

// ─── Brandes BC ───────────────────────────────────────────────
// Returns normalized BC for all nodes: O(nm)
vector<double> brandes(const Graph &G) {
  int n = G.n;
  vector<double> bc(n, 0.0);

  for (int s = 0; s < n; s++) {
    vector<int> S;
    vector<vector<int>> P(n);
    vector<double> sigma(n, 0.0), delta(n, 0.0);
    vector<int> dist(n, -1);

    sigma[s] = 1.0;
    dist[s] = 0;
    queue<int> Q;
    Q.push(s);

    while (!Q.empty()) {
      int v = Q.front();
      Q.pop();
      S.push_back(v);
      for (int w : G.adj[v]) {
        if (dist[w] < 0) {
          Q.push(w);
          dist[w] = dist[v] + 1;
        }
        if (dist[w] == dist[v] + 1) {
          sigma[w] += sigma[v];
          P[w].push_back(v);
        }
      }
    }

    while (!S.empty()) {
      int w = S.back();
      S.pop_back();
      for (int v : P[w])
        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
      if (w != s)
        bc[w] += delta[w];
    }
  }

  if (n > 2) {
    double norm = 1.0 / ((double)(n - 1) * (n - 2));
    for (auto &x : bc)
      x *= norm;
  }
  return bc;
}

// ─── Load safety check ────────────────────────────────────────
double maxOtherIncrease(const vector<double> &before,
                        const vector<double> &after, int target) {
  double mx = 0.0;
  for (int i = 0; i < (int)before.size(); i++) {
    if (i == target)
      continue;
    mx = max(mx, after[i] - before[i]);
  }
  return mx;
}

// ─── Result struct ────────────────────────────────────────────
struct Result {
  int u = -1, w = -1;
  double bcBefore = 0.0;
  double bcAfter = 0.0;
  double reduction = 0.0; // absolute
  double reductPct = 0.0; // percentage
  double maxLoadInc = 0.0;
  long long candidatesEvaluated = 0;
  long long candidatesTotal = 0;
  double timeMs = 0.0;
  bool found = false;
};

void printResult(const Result &r, const string &label) {
  printf("  %s:\n", label.c_str());
  if (!r.found) {
    printf("    No safe edge found\n");
  } else {
    printf("    Node to be added: (%d, %d)\n", r.u, r.w);
    printf("    Target betweenness: %.6f\n", r.bcAfter);
    printf("    Reduction: %.2f%%\n", r.reductPct);
  }
}

// ═══════════════════════════════════════════════════════════════
// APPROACH 1: BRUTE FORCE
// ═══════════════════════════════════════════════════════════════
Result bruteForce(const Graph &G, int target) {
  Result res;
  auto t0 = chrono::high_resolution_clock::now();

  vector<double> bc0 = brandes(G);
  res.bcBefore = bc0[target];
  double avgBC = 0;
  for (double x : bc0)
    avgBC += x;
  avgBC /= G.n;

  // Enumerate ALL non-edges
  vector<pair<int, int>> nonEdges;
  for (int u = 0; u < G.n; u++)
    for (int w = u + 1; w < G.n; w++)
      if (!G.hasEdge(u, w))
        nonEdges.push_back({u, w});

  res.candidatesTotal = (long long)nonEdges.size();

  double bestRed = -1e18;
  for (auto p : nonEdges) {
    int u = p.first;
    int w = p.second;
    res.candidatesEvaluated++;
    Graph G2 = G;
    G2.addEdge(u, w);
    vector<double> bc2 = brandes(G2);
    double red = bc0[target] - bc2[target];
    double load = maxOtherIncrease(bc0, bc2, target);
    if (red > bestRed) {
      bestRed = red;
      res.u = u;
      res.w = w;
      res.bcAfter = bc2[target];
      res.maxLoadInc = load;
      res.found = true;
    }
  }

  if (res.found) {
    res.reduction = res.bcBefore - res.bcAfter;
    res.reductPct =
        (res.bcBefore > 0) ? res.reduction / res.bcBefore * 100.0 : 0.0;
  }

  auto t1 = chrono::high_resolution_clock::now();
  res.timeMs = chrono::duration<double, milli>(t1 - t0).count();
  return res;
}

// ═══════════════════════════════════════════════════════════════
// APPROACH 2: SMART (COMPLEMENT GRAPH HOP ANALYSIS)
// ═══════════════════════════════════════════════════════════════
struct Candidate {
  double score;
  int u, w;
};

vector<Candidate> findCandidates(const Graph &G, int target, int topk) {
  // BFS from target — find 1-hop and 2-hop sets
  vector<int> dist(G.n, -1);
  queue<int> Q;
  dist[target] = 0;
  Q.push(target);
  while (!Q.empty()) {
    int v = Q.front();
    Q.pop();
    if (dist[v] >= 2)
      continue;
    for (int w : G.adj[v])
      if (dist[w] < 0) {
        dist[w] = dist[v] + 1;
        Q.push(w);
      }
  }

  set<int> hop1, hop2;
  for (int i = 0; i < G.n; i++) {
    if (dist[i] == 1)
      hop1.insert(i);
    if (dist[i] == 2)
      hop2.insert(i);
  }

  vector<Candidate> cands;

  auto tryAdd = [&](int u, int w) {
    if (u >= w || G.hasEdge(u, w))
      return;
    bool u1 = hop1.count(u), u2 = hop2.count(u);
    bool w1 = hop1.count(w), w2 = hop2.count(w);
    double score = 0.0;
    if (u2 && w2)
      score = 2.0; // 2-hop × 2-hop: strongest bypass
    else if (u1 && w1)
      score = 1.0; // 1-hop × 1-hop: triangle closing
    else
      score = 0.6; // mixed
    cands.push_back({score, u, w});
  };

  for (int a : hop2)
    for (int b : hop2)
      if (a < b)
        tryAdd(a, b);
  for (int a : hop1)
    for (int b : hop1)
      if (a < b)
        tryAdd(a, b);
  for (int a : hop1)
    for (int b : hop2)
      tryAdd(a, b);

  sort(cands.begin(), cands.end(), [](const Candidate &a, const Candidate &b) {
    return a.score > b.score;
  });

  if ((int)cands.size() > topk)
    cands.resize(topk);
  return cands;
}

Result smartAlgorithm(const Graph &G, int target, double tau, int topk) {
  Result res;
  auto t0 = chrono::high_resolution_clock::now();

  vector<double> bc0 = brandes(G);
  res.bcBefore = bc0[target];
  double avgBC = 0;
  for (double x : bc0)
    avgBC += x;
  avgBC /= G.n;

  auto cands = findCandidates(G, target, topk);
  // Total non-edges (for reference)
  long long totalNonEdges = 0;
  for (int u = 0; u < G.n; u++)
    for (int w = u + 1; w < G.n; w++)
      if (!G.hasEdge(u, w))
        totalNonEdges++;
  res.candidatesTotal = totalNonEdges;
  res.candidatesEvaluated = (long long)cands.size();

  double bestRed = -1e18;
  for (auto &c : cands) {
    Graph G2 = G;
    G2.addEdge(c.u, c.w);
    vector<double> bc2 = brandes(G2);
    double red = bc0[target] - bc2[target];
    double load = maxOtherIncrease(bc0, bc2, target);
    if (load <= tau * avgBC && red > bestRed) {
      bestRed = red;
      res.u = c.u;
      res.w = c.w;
      res.bcAfter = bc2[target];
      res.maxLoadInc = load;
      res.found = true;
    }
  }

  if (res.found) {
    res.reduction = res.bcBefore - res.bcAfter;
    res.reductPct =
        (res.bcBefore > 0) ? res.reduction / res.bcBefore * 100.0 : 0.0;
  }

  auto t1 = chrono::high_resolution_clock::now();
  res.timeMs = chrono::duration<double, milli>(t1 - t0).count();
  return res;
}

// ─── Load graph ───────────────────────────────────────────────
Graph loadGraph(const string &fname) {
  ifstream f(fname);
  if (!f) {
    fprintf(stderr, "Cannot open: %s\n", fname.c_str());
    exit(1);
  }
  int maxId = 0;
  vector<pair<int, int>> edges;
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    istringstream ss(line);
    int u, v;
    if (!(ss >> u >> v))
      continue;
    maxId = max(maxId, max(u, v));
    edges.push_back({u, v});
  }
  Graph G(maxId + 1);
  for (auto p : edges)
    G.addEdge(p.first, p.second);
  printf("[INFO] Loaded: %d nodes, %d edges  (file: %s)\n", G.n, G.edgeCount(),
         fname.c_str());
  return G;
}

// ─── Main ─────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  if (argc < 2) return 1;
  string filename = argv[1];
  Graph G = loadGraph(filename);
  
  vector<double> bc = brandes(G);
  int target = max_element(bc.begin(), bc.end()) - bc.begin();

  printf("BF_TIME:%.2f\n", rBrute.timeMs);
  
  vector<int> ks = {2, 5, 10, 20, 30, 50, 75, 100, 150, 200};
  double tau = 0.15;
  
  for (int k : ks) {
    Result rSmart = smartAlgorithm(G, target, tau, k);
    double opt = (rBrute.reduction > 0) ? (rSmart.reduction / rBrute.reduction * 100.0) : (rSmart.reduction > 0 ? 100.0 : 0.0);
    printf("CSV_OUT:%d,%.2f,%.2f,%.2f,%lld\n", k, rSmart.reductPct, opt, rSmart.timeMs, rSmart.candidatesEvaluated);
  }
  return 0;
}
