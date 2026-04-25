/*
 * CAEDDBM: Clustering-Aware EDDBM
 * Uses local clustering approximation via common neighbors with BFS parent.
 */

#include <bits/stdc++.h>
using namespace std;

struct Graph {
    int n;
    vector<vector<int>> adj;
    vector<int> deg;

    Graph() : n(0) {}
    Graph(int n) : n(n), adj(n), deg(n, 0) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }

    double avgDeg() const {
        long long s = 0;
        for (int d : deg) s += d;
        return n ? (double)s / n : 0.0;
    }
};

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
        if (u == v) continue;
        int a = getId(u), b = getId(v);
        if (a > b) swap(a, b);
        edgeSet.insert({a, b});
    }
    f.close();

    int n = (int)idVec.size();
    Graph G(n);
    for (auto &e : edgeSet) G.addEdge(e.first, e.second);
    edgeCount = (long long)edgeSet.size();
    return G;
}

vector<double> exactBrandes(const Graph &G) {
    int n = G.n;
    vector<double> BC(n, 0.0);

    for (int s = 0; s < n; s++) {
        vector<vector<int>> pred(n);
        vector<int> dist(n, -1);
        vector<double> sigma(n, 0.0);
        stack<int> S;
        queue<int> Q;

        dist[s] = 0; sigma[s] = 1.0;
        Q.push(s);

        while (!Q.empty()) {
            int v = Q.front(); Q.pop();
            S.push(v);
            for (int w : G.adj[v]) {
                if (dist[w] < 0) { dist[w] = dist[v] + 1; Q.push(w); }
                if (dist[w] == dist[v] + 1) { sigma[w] += sigma[v]; pred[w].push_back(v); }
            }
        }

        vector<double> delta(n, 0.0);
        while (!S.empty()) {
            int w = S.top(); S.pop();
            for (int p : pred[w])
                delta[p] += (sigma[p] / sigma[w]) * (1.0 + delta[w]);
            if (w != s) BC[w] += delta[w];
        }
    }
    for (int i = 0; i < n; i++) BC[i] /= 2.0;
    return BC;
}

static void bfsLevels(const Graph &G, int v,
                      vector<int> &dist,
                      vector<int> &parent,
                      vector<vector<int>> &levels) {
    int n = G.n;
    dist.assign(n, -1);
    parent.assign(n, -1);
    levels.clear();

    queue<int> q;
    dist[v] = 0;
    q.push(v);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        int d = dist[u];
        while ((int)levels.size() <= d) levels.push_back({});
        levels[d].push_back(u);
        for (int w : G.adj[u]) {
            if (dist[w] == -1) {
                dist[w] = d + 1;
                parent[w] = u;
                q.push(w);
            }
        }
    }
}

vector<double> caeddbmProbs(const Graph &G, int v) {
    int n = G.n;
    vector<int> dist, parent;
    vector<vector<int>> levels;
    bfsLevels(G, v, dist, parent, levels);

    double lambda = G.avgDeg();
    if (lambda < 1.0) lambda = 1.01;

    vector<double> P(n, 0.0);
    vector<int> mark(n, 0);
    int stamp = 1;
    double total = 0.0;

    for (int d = 1; d < (int)levels.size(); d++) {
        double levelBase = pow(lambda, -(double)d);
        for (int u : levels[d]) {
            int p = parent[u];
            double c_hat = 0.0;
            if (p >= 0 && G.deg[u] > 0) {
                stamp++;
                for (int nb : G.adj[p]) mark[nb] = stamp;
                int common = 0;
                for (int nb : G.adj[u]) {
                    if (mark[nb] == stamp) common++;
                }
                c_hat = (double)common / max(1, G.deg[u]);
            }
            double denom = max(1, G.deg[u]) * (1.0 + c_hat);
            double w = levelBase / denom;
            P[u] = w;
            total += w;
        }
    }

    if (total > 0.0) {
        for (int i = 0; i < n; i++) P[i] /= total;
    }
    return P;
}

vector<double> singleSourceDependency(const Graph &G, int s) {
    int n = G.n;
    vector<vector<int>> pred(n);
    vector<int> dist(n, -1);
    vector<double> sigma(n, 0.0);
    stack<int> S;
    queue<int> Q;

    dist[s] = 0; sigma[s] = 1.0;
    Q.push(s);

    while (!Q.empty()) {
        int v = Q.front(); Q.pop();
        S.push(v);
        for (int w : G.adj[v]) {
            if (dist[w] < 0) { dist[w] = dist[v] + 1; Q.push(w); }
            if (dist[w] == dist[v] + 1) { sigma[w] += sigma[v]; pred[w].push_back(v); }
        }
    }

    vector<double> delta(n, 0.0);
    while (!S.empty()) {
        int w = S.top(); S.pop();
        for (int p : pred[w])
            delta[p] += (sigma[p] / sigma[w]) * (1.0 + delta[w]);
    }
    return delta;
}

double estimateBOLT(const Graph &G, int v, int T, mt19937 &rng) {
    vector<double> P = caeddbmProbs(G, v);

    vector<int> nodes;
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

double measureEfficiency(const Graph &G,
                         const vector<double> &exactBC,
                         int T,
                         int trials,
                         mt19937 &rng) {
    int n = G.n;
    uniform_int_distribution<int> unode(0, n - 1);
    int correct = 0, total = 0;

    for (int i = 0; i < trials; i++) {
        int a = unode(rng), b = unode(rng);
        if (a == b) continue;
        if (exactBC[a] == exactBC[b]) continue;

        double estA = estimateBOLT(G, a, T, rng);
        double estB = estimateBOLT(G, b, T, rng);

        bool exactOrder = exactBC[a] > exactBC[b];
        bool estOrder = estA > estB;

        if (exactOrder == estOrder) correct++;
        total++;
    }
    return total > 0 ? (double)correct / total : 0.0;
}

double measureAvgError(const Graph &G,
                       const vector<double> &exactBC,
                       int T,
                       int sampleNodes,
                       mt19937 &rng) {
    int n = G.n;
    uniform_int_distribution<int> unode(0, n - 1);
    double totalErr = 0.0;
    int cnt = 0;

    for (int i = 0; i < sampleNodes; i++) {
        int v = unode(rng);
        if (exactBC[v] == 0.0) continue;

        double est = estimateBOLT(G, v, T, rng);
        double err = abs(exactBC[v] - est) / exactBC[v] * 100.0;
        totalErr += err;
        cnt++;
    }
    return cnt > 0 ? totalErr / cnt : 0.0;
}

int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [T=10] [trials=500] [compute_exact=1]\n";
        return 1;
    }

    string graphFile = argv[1];
    int T = (argc > 2) ? atoi(argv[2]) : 10;
    int trials = (argc > 3) ? atoi(argv[3]) : 500;
    bool computeExact = (argc > 4) ? (atoi(argv[4]) != 0) : true;

    long long edgeCount = 0;
    cout << "Loading graph from: " << graphFile << "\n";
    Graph G = loadGraph(graphFile, edgeCount);
    cout << "Nodes : " << G.n << "\n";
    cout << "Edges : " << edgeCount << "\n";
    cout << "Avg Degree: " << fixed << setprecision(4) << G.avgDeg() << "\n\n";

    mt19937 rng(42);

    vector<double> exactBC;
    if (computeExact) {
        if (G.n > 50000) {
            cout << "Graph is large (n=" << G.n << "). Skipping exact Brandes.\n";
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

    string base = graphFile;
    size_t sl = base.rfind('/');
    if (sl != string::npos) base = base.substr(sl + 1);
    size_t dot = base.rfind('.');
    if (dot != string::npos) base = base.substr(0, dot);

    string effCsv = base + "_caeddbm_efficiency.csv";
    ofstream effOut(effCsv);
    effOut << "T,Efficiency\n";

    vector<int> Tvals;
    if (argc > 2) {
        Tvals = {T};
    } else {
        Tvals = {1, 3, 5, 7, 10, 15, 20, 25};
    }

    cout << "=== Efficiency vs T (CAEDDBM) ===\n";
    cout << left << setw(6) << "T" << setw(14) << "Efficiency(%)" << "\n";
    cout << string(20, '-') << "\n";

    if (computeExact) {
        for (int t : Tvals) {
            int tr = min(trials, (int)(trials * 10.0 / max(1, t) + 10));
            tr = max(tr, 100);
            double eff = measureEfficiency(G, exactBC, t, tr, rng);
            cout << left << setw(6) << t << setw(14) << fixed << setprecision(2) << eff * 100.0 << "\n";
            effOut << t << "," << fixed << setprecision(4) << eff << "\n";
        }
    } else {
        cout << "(Skipped: exact BC not available)\n";
    }
    effOut.close();

    string errCsv = base + "_caeddbm_error.csv";
    ofstream errOut(errCsv);
    errOut << "T,AvgError\n";

    cout << "\n=== Average Error vs T (CAEDDBM) ===\n";
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

    return 0;
}
