#include <bits/stdc++.h>
using namespace std;

/*
    wiki.cpp
    ----------
    EDDBM Betweenness Ordering
    - Proper ID compression
    - Undirected graph
    - Efficiency vs T output
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
        long long total = 0;
        for (int d : degree) total += d;
        return (double)total / n;
    }
};

/* ==========================
   FULL BRANDES (Exact BC)
========================== */
vector<double> computeExactBC(Graph &G) {

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

/* ==========================
   EDDBM Class
========================== */
class BetweennessOrdering {
private:
    Graph &G;
    mt19937 rng;

public:
    BetweennessOrdering(Graph &graph)
        : G(graph), rng(random_device{}()) {}

    vector<double> generateProbabilities(int v) {

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

    double computeDependency(int s, int target) {

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

    int sampleNode(const vector<double> &P) {
        uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);

        double cumulative = 0.0;
        for (int i = 0; i < (int)P.size(); i++) {
            cumulative += P[i];
            if (r <= cumulative)
                return i;
        }
        return P.size() - 1;
    }

    double estimateBetweenness(int v, int T) {

        vector<double> P = generateProbabilities(v);
        double estimate = 0.0;

        for (int i = 0; i < T; i++) {
            int s = sampleNode(P);
            if (P[s] == 0) continue;

            double dep = computeDependency(s, v);
            estimate += dep / P[s];
        }

        return estimate / T;
    }
};

/* ==========================
   MAIN
========================== */
int main() {

    ifstream file("wiki-Vote.txt");
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

    file.close();

    // ID compression
    unordered_map<int,int> idMap;
    int idx = 0;
    for (int node : nodeSet)
        idMap[node] = idx++;

    int n = nodeSet.size();
    Graph G(n);

    for (auto &e : rawEdges)
        G.addEdge(idMap[e.first], idMap[e.second]);

    cout << "Nodes: " << n << "\n";
    cout << "Edges: " << rawEdges.size() << "\n";

    cout << "Computing Exact BC...\n";
    vector<double> exactBC = computeExactBC(G);

    BetweennessOrdering solver(G);

    ofstream out("efficiency.csv");
    out << "T,Efficiency\n";

    vector<int> T_values = {1,5,10,15,20,25,30};
    mt19937 rng(random_device{}());
    uniform_int_distribution<int> distNode(0, n-1);

    int trials = 200;

    for (int T : T_values) {

        int correct = 0;

        for (int i = 0; i < trials; i++) {
            int a = distNode(rng);
            int b = distNode(rng);
            if (a == b) continue;

            double estA = solver.estimateBetweenness(a, T);
            double estB = solver.estimateBetweenness(b, T);

            bool exactOrder = exactBC[a] > exactBC[b];
            bool estOrder = estA > estB;

            if (exactOrder == estOrder)
                correct++;
        }

        double efficiency = (double)correct / trials;
        cout << "T=" << T << " Efficiency=" << efficiency << "\n";
        out << T << "," << efficiency << "\n";
    }

    out.close();
    return 0;
}
