#include <bits/stdc++.h>
using namespace std;

/*
    EDDBM Efficiency Calculator
    Usage:
    ./edd dataset.txt T numComparisons
*/

class Graph {
public:
    int n;
    vector<vector<int>> adj;
    vector<int> degree;

    Graph(int n) : n(n), adj(n), degree(n, 0) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
        degree[u]++;
        degree[v]++;
    }

    double averageDegree() const {
        long long sum = 0;
        for (int d : degree) sum += d;
        return (double)sum / n;
    }
};

/* ===============================
   Exact Brandes
=================================*/
vector<double> brandes(Graph &G) {

    int n = G.n;
    vector<double> BC(n, 0.0);

    for (int s = 0; s < n; s++) {

        vector<vector<int>> pred(n);
        vector<int> dist(n, -1);
        vector<double> sigma(n, 0.0);
        stack<int> S;
        queue<int> Q;

        dist[s] = 0;
        sigma[s] = 1.0;
        Q.push(s);

        while (!Q.empty()) {
            int v = Q.front(); Q.pop();
            S.push(v);

            for (int w : G.adj[v]) {
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    Q.push(w);
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    pred[w].push_back(v);
                }
            }
        }

        vector<double> delta(n, 0.0);

        while (!S.empty()) {
            int w = S.top(); S.pop();
            for (int v : pred[w]) {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if (w != s)
                BC[w] += delta[w];
        }
    }

    for (int i = 0; i < n; i++)
        BC[i] /= 2.0;

    return BC;
}

/* ===============================
   EDDBM
=================================*/
class EDDBM {
private:
    Graph &G;
    mt19937 rng;

public:
    EDDBM(Graph &graph) : G(graph), rng(42) {}

    vector<double> generateProb(int v) {

        int n = G.n;
        vector<int> dist(n, -1);
        queue<int> q;

        dist[v] = 0;
        q.push(v);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int w : G.adj[u]) {
                if (dist[w] == -1) {
                    dist[w] = dist[u] + 1;
                    q.push(w);
                }
            }
        }

        double lambda = G.averageDegree();
        vector<double> P(n, 0.0);
        double total = 0.0;

        for (int i = 0; i < n; i++) {
            if (i == v || dist[i] == -1) continue;
            double val = pow(lambda, -dist[i]) / max(1, G.degree[i]);
            P[i] = val;
            total += val;
        }

        for (int i = 0; i < n; i++)
            if (total > 0) P[i] /= total;

        return P;
    }

    double dependency(int s, int target) {

        int n = G.n;
        vector<vector<int>> pred(n);
        vector<int> dist(n, -1);
        vector<double> sigma(n, 0.0);
        stack<int> S;
        queue<int> Q;

        dist[s] = 0;
        sigma[s] = 1.0;
        Q.push(s);

        while (!Q.empty()) {
            int v = Q.front(); Q.pop();
            S.push(v);

            for (int w : G.adj[v]) {
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    Q.push(w);
                }
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    pred[w].push_back(v);
                }
            }
        }

        vector<double> delta(n, 0.0);

        while (!S.empty()) {
            int w = S.top(); S.pop();
            for (int v : pred[w]) {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
        }

        return delta[target];
    }

    double estimate(int v, int T) {

        vector<double> P = generateProb(v);
        uniform_real_distribution<double> dist(0.0, 1.0);

        double est = 0.0;

        for (int i = 0; i < T; i++) {

            double r = dist(rng);
            double cum = 0.0;
            int s = -1;

            for (int j = 0; j < P.size(); j++) {
                cum += P[j];
                if (r <= cum) {
                    s = j;
                    break;
                }
            }

            if (s != -1 && P[s] > 0)
                est += dependency(s, v) / P[s];
        }

        return est / T;
    }
};

/* ===============================
   MAIN
=================================*/
int main(int argc, char* argv[]) {

<<<<<<< HEAD:efficiency.cpp
    if (argc != 4) {
        cout << "Usage: ./edd dataset.txt T numComparisons\n";
        return 0;
    }

    string filename = argv[1];
    int T = stoi(argv[2]);
    int trials = stoi(argv[3]);

    ifstream file(filename);
=======
    ifstream file("Wiki-Vote.txt");
>>>>>>> 87ac4bf094e3a35d5e9884f5cfec689af0e0e204:betweeness.cpp
    if (!file) {
        cout << "File not found\n";
        return 0;
    }

    vector<pair<int,int>> rawEdges;
    unordered_set<int> nodeSet;
    string line;
    int u, v;

    while (getline(file, line)) {
        if (line[0] == '#') continue;
        stringstream ss(line);
        ss >> u >> v;
        rawEdges.push_back({u, v});
        nodeSet.insert(u);
        nodeSet.insert(v);
    }

    unordered_map<int,int> idMap;
    int idx = 0;
    for (int node : nodeSet)
        idMap[node] = idx++;

    Graph G(nodeSet.size());
    for (auto &e : rawEdges)
        G.addEdge(idMap[e.first], idMap[e.second]);

    cout << "Computing Exact BC...\n";
    vector<double> exactBC = brandes(G);

    EDDBM solver(G);

    mt19937 rng(100);
    uniform_int_distribution<int> distNode(0, G.n - 1);

    int correct = 0;

    for (int i = 0; i < trials; i++) {

        int a = distNode(rng);
        int b = distNode(rng);

        if (a == b) continue;

        double estA = solver.estimate(a, T);
        double estB = solver.estimate(b, T);

        bool exactOrder = exactBC[a] > exactBC[b];
        bool estOrder = estA > estB;

        if (exactOrder == estOrder)
            correct++;
    }

    double efficiency = (double)correct / trials;

    cout << "\nEfficiency (T=" << T << "): " << efficiency << endl;

    return 0;
}
