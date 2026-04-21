#include <bits/stdc++.h>
using namespace std;

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

int main(int argc, char* argv[]) {

    if (argc != 5) {
        cout << "Usage: ./edd dataset.txt nodeA nodeB T\n";
        return 0;
    }

    string filename = argv[1];
    int nodeA_original = stoi(argv[2]);
    int nodeB_original = stoi(argv[3]);
    int T = stoi(argv[4]);

    ifstream file(filename);
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

    if (!idMap.count(nodeA_original) || !idMap.count(nodeB_original)) {
        cout << "One of the nodes not found.\n";
        return 0;
    }

    Graph G(nodeSet.size());
    for (auto &e : rawEdges)
        G.addEdge(idMap[e.first], idMap[e.second]);

    EDDBM solver(G);

    int a = idMap[nodeA_original];
    int b = idMap[nodeB_original];

    double estA = solver.estimate(a, T);
    double estB = solver.estimate(b, T);

    cout << "Estimated BC(A): " << estA << endl;
    cout << "Estimated BC(B): " << estB << endl;

    if (estA > estB)
        cout << "Node A more central (EDDBM)\n";
    else if (estB > estA)
        cout << "Node B more central (EDDBM)\n";
    else
        cout << "Approximately equal\n";

    return 0;
}
