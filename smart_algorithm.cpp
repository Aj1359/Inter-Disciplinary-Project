/**
 * ================================================================
 * SMART-ONLY BC MINIMIZATION (For massive datasets)
 * ================================================================
 */
#include <bits/stdc++.h>
using namespace std;

struct Graph {
  int n;
  vector<vector<int>> adj;
  set<pair<int, int>> edgeSet;
  explicit Graph(int n) : n(n), adj(n) {}
  void addEdge(int u, int v) {
    if (u == v) return;
    auto key = make_pair(min(u, v), max(u, v));
    if (edgeSet.count(key)) return;
    adj[u].push_back(v); adj[v].push_back(u);
    edgeSet.insert(key);
  }
  bool hasEdge(int u, int v) const {
    return edgeSet.count({min(u, v), max(u, v)}) > 0;
  }
  int edgeCount() const { return (int)edgeSet.size(); }
};

vector<double> brandes(const Graph &G) {
  int n = G.n;
  vector<double> bc(n, 0.0);
  int k = min(n, 30);
  vector<int> sources(n);
  iota(sources.begin(), sources.end(), 0);
  mt19937 g(42);
  shuffle(sources.begin(), sources.end(), g);
  sources.resize(k);

  for (int s : sources) {
    vector<int> S;
    vector<vector<int>> P(n);
    vector<double> sigma(n, 0.0), delta(n, 0.0);
    vector<int> dist(n, -1);
    sigma[s] = 1.0; dist[s] = 0;
    queue<int> Q; Q.push(s);
    while (!Q.empty()) {
      int v = Q.front(); Q.pop(); S.push_back(v);
      for (int w : G.adj[v]) {
        if (dist[w] < 0) { Q.push(w); dist[w] = dist[v] + 1; }
        if (dist[w] == dist[v] + 1) { sigma[w] += sigma[v]; P[w].push_back(v); }
      }
    }
    while (!S.empty()) {
      int w = S.back(); S.pop_back();
      for (int v : P[w]) delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
      if (w != s) bc[w] += delta[w];
    }
  }
  if (n > 2) {
    double norm = ((double)n / k) / ((double)(n - 1) * (n - 2));
    for (auto &x : bc) x *= norm;
  }
  return bc;
}

double maxOtherIncrease(const vector<double> &before, const vector<double> &after, int target) {
  double mx = 0.0;
  for (int i = 0; i < (int)before.size(); i++) {
    if (i == target) continue;
    mx = max(mx, after[i] - before[i]);
  }
  return mx;
}

struct Candidate { double score; int u, w; };

vector<Candidate> findCandidates(const Graph &G, int target, int topk) {
  vector<int> dist(G.n, -1);
  queue<int> Q;
  dist[target] = 0; Q.push(target);
  while (!Q.empty()) {
    int v = Q.front(); Q.pop();
    if (dist[v] >= 2) continue;
    for (int w : G.adj[v]) if (dist[w] < 0) { dist[w] = dist[v] + 1; Q.push(w); }
  }
  set<int> hop1, hop2;
  for (int i = 0; i < G.n; i++) {
    if (dist[i] == 1) hop1.insert(i);
    if (dist[i] == 2) hop2.insert(i);
  }
  vector<Candidate> cands;
  auto tryAdd = [&](int u, int w) {
    if (u >= w || G.hasEdge(u, w)) return;
    bool u1 = hop1.count(u), u2 = hop2.count(u);
    bool w1 = hop1.count(w), w2 = hop2.count(w);
    double score = (u2 && w2) ? 2.0 : (u1 && w1) ? 1.0 : 0.6;
    cands.push_back({score, u, w});
  };
  for (int a : hop2) for (int b : hop2) if (a < b) tryAdd(a, b);
  for (int a : hop1) for (int b : hop1) if (a < b) tryAdd(a, b);
  for (int a : hop1) for (int b : hop2) tryAdd(a, b);
  sort(cands.begin(), cands.end(), [](const Candidate &a, const Candidate &b) { return a.score > b.score; });
  if ((int)cands.size() > topk) cands.resize(topk);
  return cands;
}

void smartAlgorithm(const Graph &G, int target, double tau, int topk) {
  auto t0 = chrono::high_resolution_clock::now();
  vector<double> bc0 = brandes(G);
  double bcBefore = bc0[target];
  double avgBC = 0;
  for (double x : bc0) avgBC += x;
  avgBC /= G.n;

  auto cands = findCandidates(G, target, topk);
  long long candidatesEvaluated = (long long)cands.size();
  
  double bestRed = -1e18;
  int bestU = -1, bestW = -1;
  double bcAfter = 0.0, maxLoadInc = 0.0;
  bool found = false;

  for (auto &c : cands) {
    Graph G2 = G; G2.addEdge(c.u, c.w);
    vector<double> bc2 = brandes(G2);
    double red = bc0[target] - bc2[target];
    double load = maxOtherIncrease(bc0, bc2, target);
    if (load <= tau * avgBC && red > bestRed) {
      bestRed = red; bestU = c.u; bestW = c.w;
      bcAfter = bc2[target]; maxLoadInc = load; found = true;
    }
  }

  auto t1 = chrono::high_resolution_clock::now();
  double timeMs = chrono::duration<double, milli>(t1 - t0).count();

  if (found) {
    double reduction = bcBefore - bcAfter;
    double reductPct = (bcBefore > 0) ? reduction / bcBefore * 100.0 : 0.0;
    
    printf("\n  ┌─ SMART ALGORITHM ────────────────────────────\n");
    printf("  │  Optimal edge    : (%d, %d)\n", bestU, bestW);
    printf("  │  BC before        : %.6f\n", bcBefore);
    printf("  │  BC after         : %.6f\n", bcAfter);
    printf("  │  Reduction        : %.6f  (%.2f%%)\n", reduction, reductPct);
    printf("  │  Max load increase: %.6f\n", maxLoadInc);
    printf("  │  Candidates eval  : %lld\n", candidatesEvaluated);
    printf("  │  Time             : %.3f ms\n", timeMs);
    printf("  └───────────────────────────────────────────────\n");
  } else {
    printf("\n  ┌─ SMART ALGORITHM ────────────────────────────\n");
    printf("  │  No safe edge found within load constraint.\n");
    printf("  └───────────────────────────────────────────────\n");
  }
}

Graph loadGraph(const string &fname) {
  ifstream f(fname);
  int maxId = 0;
  vector<pair<int, int>> edges;
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream ss(line); int u, v;
    if (!(ss >> u >> v)) continue;
    maxId = max(maxId, max(u, v));
    edges.push_back({u, v});
  }
  Graph G(maxId + 1);
  for (auto p : edges) G.addEdge(p.first, p.second);
  return G;
}

int main(int argc, char *argv[]) {
  if (argc < 2) return 1;
  string filename = argv[1];
  Graph G = loadGraph(filename);
  vector<double> bc = brandes(G);
  int target = max_element(bc.begin(), bc.end()) - bc.begin();

  printf("DATASET:%s\n", filename.c_str());
  vector<double> taus = {0.05, 0.10, 0.15, 0.20, 0.25};
  for (double tau : taus) {
    smartAlgorithm(G, target, tau, 50);
  }
  return 0;
}
