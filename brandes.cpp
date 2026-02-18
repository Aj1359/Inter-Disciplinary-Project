#include <bits/stdc++.h>
using namespace std;

class Graph {
public:
    int n;
    vector<vector<int>> adj;

    Graph(int n) : n(n), adj(n) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
};

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

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cout << "Usage: ./brandes dataset.txt nodeA nodeB\n";
        return 0;
    }

    string filename = argv[1];
    int nodeA_original = stoi(argv[2]);
    int nodeB_original = stoi(argv[3]);

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

    cout << "Computing Exact Betweenness...\n";

    vector<double> BC = brandes(G);

    int a = idMap[nodeA_original];
    int b = idMap[nodeB_original];

    cout << "Exact BC(A): " << BC[a] << endl;
    cout << "Exact BC(B): " << BC[b] << endl;

    if (BC[a] > BC[b])
        cout << "Node A more central (Exact)\n";
    else if (BC[b] > BC[a])
        cout << "Node B more central (Exact)\n";
    else
        cout << "Equal centrality\n";

    return 0;
}
