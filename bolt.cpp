/*
 * BOLT: BetweennesS Ordering aLgoriThm
 * =====================================
 * Implementation of the EDDBM (Exponential in Distance and inverse of
 * Degree Based Model) for betweenness estimation and ordering.
 *
 * Reference: "An Efficient Heuristic for Betweenness-Ordering"
 *            Rishi Ranjan Singh, Shubham Chaudhary, Manas Agarwal
 *
 * Algorithm (Section 4.3 of the paper):
 *   For node v, BFS from v. Each node i at distance d gets base weight:
 *       w_base(i) = lambda^(-d)
 *   Within each level d, re-normalise by degree:
 *       p(i) = (p_d * |V_d| * deg(i)^-1) / sum_{j in V_d} deg(j)^-1
 *   where p_d is the share of level d = (sum_{i at dist d} lambda^(-d)) /
 *                                        (sum_{all j != v} lambda^(-d(j,v)))
 *
 *   Betweenness of v is then estimated as:
 *       BC_est(v) = (1/T) sum_{t=1}^{T} delta_{s_t *}(v) / p(s_t)
 *   where s_t is drawn from the non-uniform distribution p.
 *
 * Usage:
 *   ./bolt <graph_file> [T=25] [trials=1000] [k=10]
 *
 * Graph file format (edge list, lines starting with '#' are comments):
 *   u v   (0- or 1-based, compressed automatically)
 */

#include <bits/stdc++.h>
using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
// Graph data structure
// ─────────────────────────────────────────────────────────────────────────────
struct Graph {
    int n;
    vector<vector<int>> adj;
    vector<int> deg;

    Graph() : n(0) {}
    Graph(int n) : n(n), adj(n), deg(n, 0) {}

    void addEdge(int u, int v) {
        // undirected, no self-loops or multi-edges checked outside
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }

    double avgDeg() const {
        long long s = 0;
        for (int d : deg) s += d;
        return (double)s / n;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Load graph from edge-list file, compress IDs, remove self-loops & duplicates
// ─────────────────────────────────────────────────────────────────────────────
Graph loadGraph(const string &path, long long &edgeCount) {
    ifstream f(path);
    if (!f) { cerr << "Cannot open file: " << path << "\n"; exit(1); }

    set<pair<int,int>> edgeSet;
    unordered_map<int,int> idMap;
    vector<int> idVec;
    string line;
    int u, v;

    auto getId = [&](int x) -> int {
        auto it = idMap.find(x);
        if (it != idMap.end()) return it->second;
        int id = (int)idVec.size();
        idMap[x] = id;
        idVec.push_back(x);
        return id;
    };

    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream ss(line);
        if (!(ss >> u >> v)) continue;
        if (u == v) continue; // skip self-loops
        int a = getId(u), b = getId(v);
        if (a > b) swap(a, b);
        edgeSet.insert({a, b});
    }
    f.close();

    int n = (int)idVec.size();
    Graph G(n);
    for (auto &e : edgeSet)
        G.addEdge(e.first, e.second);

    edgeCount = (long long)edgeSet.size();
    return G;
}

// ─────────────────────────────────────────────────────────────────────────────
// Exact Brandes betweenness (for validation on small graphs)
// ─────────────────────────────────────────────────────────────────────────────
vector<double> exactBrandes(const Graph &G) {
    int n = G.n;
    vector<double> BC(n, 0.0);

    for (int s = 0; s < n; s++) {
        vector<vector<int>> pred(n);
        vector<int>    dist(n, -1);
        vector<double> sigma(n, 0.0);
        stack<int> S;
        queue<int> Q;

        dist[s] = 0; sigma[s] = 1.0;
        Q.push(s);

        while (!Q.empty()) {
            int v = Q.front(); Q.pop();
            S.push(v);
            for (int w : G.adj[v]) {
                if (dist[w] < 0) { dist[w] = dist[v]+1; Q.push(w); }
                if (dist[w] == dist[v]+1) { sigma[w] += sigma[v]; pred[w].push_back(v); }
            }
        }

        vector<double> delta(n, 0.0);
        while (!S.empty()) {
            int w = S.top(); S.pop();
            for (int p : pred[w])
                delta[p] += (sigma[p]/sigma[w]) * (1.0 + delta[w]);
            if (w != s) BC[w] += delta[w];
        }
    }
    for (int i = 0; i < n; i++) BC[i] /= 2.0;
    return BC;
}

// ─────────────────────────────────────────────────────────────────────────────
// EDDBM probability generation (Section 4.3 of the paper)
// For a query node v:
//   Step 1: BFS from v, record distance d[i] for each node i.
//   Step 2: base weight = lambda^{-d[i]}  (handled in log to avoid underflow)
//   Step 3: Within level d, weight further divided by deg[i].
//   Step 4: Normalise to get proper probability distribution.
// ─────────────────────────────────────────────────────────────────────────────
vector<double> eddbmProbs(const Graph &G, int v) {
    int n = G.n;
    double lambda = G.avgDeg();
    if (lambda < 1.0) lambda = 1.01; // avoid issues on near-isolated graphs

    // BFS from v
    vector<int> dist(n, -1);
    vector<vector<int>> levels; // levels[d] = list of nodes at distance d
    queue<int> q;
    dist[v] = 0;
    q.push(v);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        int d = dist[u];
        while ((int)levels.size() <= d) levels.push_back({});
        levels[d].push_back(u);
        for (int w : G.adj[u]) {
            if (dist[w] == -1) { dist[w] = dist[u]+1; q.push(w); }
        }
    }

    vector<double> P(n, 0.0);

    // For each level d >= 1, compute level share and distribute by 1/deg
    // level_share(d) = |V_d| * lambda^{-d}
    // Then for node i in V_d:
    //   p(i) = level_share(d) * (1/deg[i]) / sum_{j in V_d} (1/deg[j])

    // First, compute raw unnormalized values per level, then normalize globally
    // Raw weight of node i at level d = lambda^{-d} / deg[i]
    double total = 0.0;
    for (int d = 1; d < (int)levels.size(); d++) {
        double invDegSum = 0.0;
        for (int u : levels[d]) {
            int di = max(1, G.deg[u]);
            invDegSum += 1.0 / di;
        }
        if (invDegSum == 0.0) continue;

        double levelBase = pow(lambda, -(double)d);
        // Assign: p_i = levelBase * (1/deg[i]) / invDegSum
        for (int u : levels[d]) {
            int di = max(1, G.deg[u]);
            double w = levelBase * (1.0/di) / invDegSum;
            P[u] = w;
            total += w;
        }
    }

    // Normalise
    if (total > 0.0)
        for (int i = 0; i < n; i++) P[i] /= total;

    return P;
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-source dependency computation (Brandes-style)
// Returns delta[s*](v) for all v in O(m)
// ─────────────────────────────────────────────────────────────────────────────
vector<double> singleSourceDependency(const Graph &G, int s) {
    int n = G.n;
    vector<vector<int>> pred(n);
    vector<int>    dist(n, -1);
    vector<double> sigma(n, 0.0);
    stack<int> S;
    queue<int> Q;

    dist[s] = 0; sigma[s] = 1.0;
    Q.push(s);

    while (!Q.empty()) {
        int v = Q.front(); Q.pop();
        S.push(v);
        for (int w : G.adj[v]) {
            if (dist[w] < 0) { dist[w] = dist[v]+1; Q.push(w); }
            if (dist[w] == dist[v]+1) { sigma[w] += sigma[v]; pred[w].push_back(v); }
        }
    }

    vector<double> delta(n, 0.0);
    while (!S.empty()) {
        int w = S.top(); S.pop();
        for (int p : pred[w])
            delta[p] += (sigma[p]/sigma[w]) * (1.0 + delta[w]);
        // Note: do NOT subtract self (this returns raw dependencies, no BC formula)
    }
    // delta[v] = delta_{s*}(v) for each v != s
    // (for v == s it is 0 which is correct since s doesn't lie on s-t paths for t!=s)
    return delta;
}

// ─────────────────────────────────────────────────────────────────────────────
// BOLT: Estimate betweenness of node v using T samples with EDDBM
// Algorithm 1 + EDDBM probabilities
// ─────────────────────────────────────────────────────────────────────────────
double estimateBOLT(const Graph &G, int v, int T, mt19937 &rng) {
    vector<double> P = eddbmProbs(G, v);

    // Build discrete distribution
    // (avoid nodes with P=0 being selected)
    vector<int>    nodes;
    vector<double> weights;
    for (int i = 0; i < G.n; i++) {
        if (i != v && P[i] > 0.0) {
            nodes.push_back(i);
            weights.push_back(P[i]);
        }
    }

    if (nodes.empty()) return 0.0;

    discrete_distribution<int> dist(weights.begin(), weights.end());

    double bcEst = 0.0;
    for (int t = 0; t < T; t++) {
        int idx = dist(rng);
        int s = nodes[idx];
        double p_s = P[s];

        vector<double> dep = singleSourceDependency(G, s);
        bcEst += dep[v] / p_s;
    }
    return bcEst / T;
}

// ─────────────────────────────────────────────────────────────────────────────
// Betweenness-Ordering: compare two nodes u, v using BOLT
// Returns true if estimated BC(u) > estimated BC(v)
// ─────────────────────────────────────────────────────────────────────────────
bool betweennessOrdering(const Graph &G, int u, int v, int T, mt19937 &rng) {
    double bu = estimateBOLT(G, u, T, rng);
    double bv = estimateBOLT(G, v, T, rng);
    return bu > bv;
}

// ─────────────────────────────────────────────────────────────────────────────
// k-Betweenness-Ordering: rank k nodes using BOLT
// Returns nodes sorted by estimated BC (descending)
// ─────────────────────────────────────────────────────────────────────────────
vector<pair<double,int>> kBetweennessOrdering(const Graph &G,
                                               const vector<int> &nodes,
                                               int T, mt19937 &rng)
{
    vector<pair<double,int>> scores;
    for (int v : nodes) {
        double est = estimateBOLT(G, v, T, rng);
        scores.push_back({est, v});
    }
    // Sort descending by estimated betweenness
    sort(scores.begin(), scores.end(), [](auto &a, auto &b){ return a.first > b.first; });
    return scores;
}

// ─────────────────────────────────────────────────────────────────────────────
// Efficiency measurement: fraction of node pairs correctly ordered
// ─────────────────────────────────────────────────────────────────────────────
double measureEfficiency(const Graph &G,
                         const vector<double> &exactBC,
                         int T,
                         int trials,
                         mt19937 &rng)
{
    int n = G.n;
    uniform_int_distribution<int> unode(0, n-1);
    int correct = 0, total = 0;

    for (int i = 0; i < trials; i++) {
        int a = unode(rng), b = unode(rng);
        if (a == b) continue;
        if (exactBC[a] == exactBC[b]) continue; // skip ties

        double estA = estimateBOLT(G, a, T, rng);
        double estB = estimateBOLT(G, b, T, rng);

        bool exactOrder = exactBC[a] > exactBC[b];
        bool estOrder   = estA > estB;

        if (exactOrder == estOrder) correct++;
        total++;
    }
    return total > 0 ? (double)correct / total : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Average Error measurement
// ─────────────────────────────────────────────────────────────────────────────
double measureAvgError(const Graph &G,
                       const vector<double> &exactBC,
                       int T,
                       int sampleNodes,
                       mt19937 &rng)
{
    int n = G.n;
    uniform_int_distribution<int> unode(0, n-1);
    double totalErr = 0.0;
    int cnt = 0;

    for (int i = 0; i < sampleNodes; i++) {
        int v = unode(rng);
        if (exactBC[v] == 0.0) continue; // paper considers only nonzero BC nodes

        double est = estimateBOLT(G, v, T, rng);
        double err = abs(exactBC[v] - est) / exactBC[v] * 100.0;
        totalErr += err;
        cnt++;
    }
    return cnt > 0 ? totalErr / cnt : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [T=25] [trials=500] [compute_exact=1]\n";
        return 1;
    }

    string graphFile = argv[1];
    int T            = (argc > 2) ? atoi(argv[2]) : 25;
    int trials       = (argc > 3) ? atoi(argv[3]) : 500;
    bool computeExact= (argc > 4) ? (atoi(argv[4]) != 0) : true;

    // ── Load graph ──────────────────────────────────────────────────────────
    long long edgeCount = 0;
    cout << "Loading graph from: " << graphFile << "\n";
    Graph G = loadGraph(graphFile, edgeCount);
    cout << "Nodes : " << G.n << "\n";
    cout << "Edges : " << edgeCount << "\n";
    cout << "Avg Degree: " << fixed << setprecision(4) << G.avgDeg() << "\n\n";

    mt19937 rng(42);

    // ── Exact Brandes (optional for large graphs) ────────────────────────────
    vector<double> exactBC;
    if (computeExact) {
        if (G.n > 30000) {
            cout << "Graph is large (n=" << G.n << "). Skipping exact Brandes.\n";
            cout << "To force, pass compute_exact=1 after other args.\n";
            computeExact = false;
        } else {
            auto t0 = chrono::steady_clock::now();
            cout << "Computing exact Brandes BC...\n";
            exactBC = exactBrandes(G);
            auto t1 = chrono::steady_clock::now();
            double secs = chrono::duration<double>(t1 - t0).count();
            cout << "Brandes done in " << fixed << setprecision(2) << secs << "s\n\n";
        }
    }

    // ── BOLT: Efficiency vs T ─────────────────────────────────────────────
    string base = graphFile;
    // strip path and extension for output file naming
    size_t sl = base.rfind('/');
    if (sl != string::npos) base = base.substr(sl+1);
    size_t dot = base.rfind('.');
    if (dot != string::npos) base = base.substr(0, dot);

    string effCsv = base + "_efficiency.csv";
    ofstream effOut(effCsv);
    effOut << "T,Efficiency\n";

    vector<int> Tvals = {1, 5, 10, 15, 20, 25, 30, 40, 50};

    cout << "=== Efficiency vs T (BOLT) ===\n";
    cout << left << setw(6) << "T" << setw(14) << "Efficiency(%)" << "\n";
    cout << string(20, '-') << "\n";

    if (computeExact) {
        for (int t : Tvals) {
            // Use fewer trials for large T to keep runtime reasonable
            int tr = min(trials, (int)(trials * 25.0 / t + 10));
            tr = max(tr, 100);
            double eff = measureEfficiency(G, exactBC, t, tr, rng);
            cout << left << setw(6) << t << setw(14) << fixed << setprecision(2) << eff*100.0 << "\n";
            effOut << t << "," << fixed << setprecision(4) << eff << "\n";
        }
    } else {
        cout << "(Skipped: exact BC not available)\n";
    }
    effOut.close();

    // ── BOLT: Average Error vs T ──────────────────────────────────────────
    string errCsv = base + "_error.csv";
    ofstream errOut(errCsv);
    errOut << "T,AvgError\n";

    cout << "\n=== Average Error vs T (BOLT) ===\n";
    cout << left << setw(6) << "T" << setw(16) << "AvgError(%)" << "\n";
    cout << string(22, '-') << "\n";

    if (computeExact) {
        int sampleN = min(G.n, 200);
        for (int t : Tvals) {
            double err = measureAvgError(G, exactBC, t, sampleN, rng);
            cout << left << setw(6) << t << setw(16) << fixed << setprecision(2) << err << "\n";
            errOut << t << "," << fixed << setprecision(4) << err << "\n";
        }
    } else {
        cout << "(Skipped: exact BC not available)\n";
    }
    errOut.close();

    // ── BOLT: Time for betweenness estimate at T=25 ────────────────────────
    {
        int v = rng() % G.n;
        auto t0 = chrono::steady_clock::now();
        int reps = 5;
        for (int r = 0; r < reps; r++)
            estimateBOLT(G, v, T, rng);
        auto t1 = chrono::steady_clock::now();
        double ms = chrono::duration<double>(t1-t0).count() / reps * 1000.0;
        cout << "\nTime per estimateBOLT(T=" << T << "): "
             << fixed << setprecision(2) << ms << " ms\n";
    }

    // ── k-Betweenness-Ordering demo ────────────────────────────────────────
    {
        int k = min(10, G.n);
        uniform_int_distribution<int> unode(0, G.n-1);
        vector<int> kNodes;
        {
            set<int> picked;
            while ((int)picked.size() < k) picked.insert(unode(rng));
            kNodes.assign(picked.begin(), picked.end());
        }

        cout << "\n=== k-Betweenness-Ordering (k=" << k << ", T=" << T << ") ===\n";
        auto ranked = kBetweennessOrdering(G, kNodes, T, rng);

        cout << left << setw(8) << "Rank" << setw(10) << "Node"
             << setw(18) << "Est.BC (BOLT)";
        if (computeExact) cout << setw(18) << "Exact BC";
        cout << "\n" << string(50, '-') << "\n";

        for (int i = 0; i < (int)ranked.size(); i++) {
            auto [est, node] = ranked[i];
            cout << left << setw(8) << (i+1) << setw(10) << node
                 << setw(18) << fixed << setprecision(2) << est;
            if (computeExact) cout << setw(18) << fixed << setprecision(2) << exactBC[node];
            cout << "\n";
        }

        if (computeExact) {
            // Check how many are correctly ordered vs exact
            vector<pair<double,int>> exactRanked;
            for (int v : kNodes) exactRanked.push_back({exactBC[v], v});
            sort(exactRanked.begin(), exactRanked.end(), [](auto &a, auto &b){ return a.first > b.first; });

            // Count pairwise correct orderings
            int pairCorrect = 0, pairTotal = 0;
            for (int i = 0; i < k; i++) for (int j = i+1; j < k; j++) {
                int ni = ranked[i].second, nj = ranked[j].second;
                bool boltSaysIGtJ = exactBC[ni] >= exactBC[nj]; // ranked[i] is "better"
                // We said ni is ranked above nj, so BOLT claims BC(ni) >= BC(nj)
                if (exactBC[ni] == exactBC[nj]) continue; // skip ties
                if (boltSaysIGtJ) pairCorrect++;
                pairTotal++;
            }
            if (pairTotal > 0)
                cout << "\nPairwise ordering accuracy: "
                     << fixed << setprecision(1) << 100.0*pairCorrect/pairTotal << "%\n";
        }
    }

    // ── Summary ────────────────────────────────────────────────────────────
    cout << "\n=== Summary ===\n";
    cout << "Graph       : " << graphFile << "\n";
    cout << "Nodes       : " << G.n << "\n";
    cout << "Edges       : " << edgeCount << "\n";
    cout << "T (samples) : " << T << "\n";
    if (computeExact) {
        int nonzeroCnt = 0;
        double maxBC = 0, minNZBC = 1e18;
        for (double b : exactBC) {
            if (b > 0) { nonzeroCnt++; minNZBC = min(minNZBC, b); maxBC = max(maxBC, b); }
        }
        cout << "Nonzero BC nodes: " << nonzeroCnt << "\n";
        cout << "Max BC          : " << fixed << setprecision(4) << maxBC << "\n";
    }
    cout << "Output files: " << effCsv << ", " << errCsv << "\n";

    return 0;
}
