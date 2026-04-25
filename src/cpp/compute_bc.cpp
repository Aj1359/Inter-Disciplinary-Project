/*
 * compute_bc.cpp
 * Compute exact Brandes BC and output as CSV.
 * Usage: ./compute_bc <graph_file> <output_csv>
 */
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <graph_file> <output_csv>\n";
        return 1;
    }

    string graphFile = argv[1];
    string outFile = argv[2];

    // Load graph
    ifstream f(graphFile);
    if (!f) { cerr << "Cannot open: " << graphFile << "\n"; return 1; }

    unordered_map<int,int> idMap;
    vector<int> idVec;
    set<pair<int,int>> edgeSet;
    string line;

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
        int u, v;
        if (!(ss >> u >> v)) continue;
        if (u == v) continue;
        int a = getId(u), b = getId(v);
        if (a > b) swap(a, b);
        edgeSet.insert({a, b});
    }
    f.close();

    int n = (int)idVec.size();
    vector<vector<int>> adj(n);
    vector<int> deg(n, 0);
    for (auto &e : edgeSet) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
        deg[e.first]++;
        deg[e.second]++;
    }

    cerr << "Graph: n=" << n << " m=" << edgeSet.size() << endl;

    // Exact Brandes
    auto t0 = chrono::steady_clock::now();
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
            for (int w : adj[v]) {
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

    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cerr << "Brandes done in " << fixed << setprecision(2) << secs << "s" << endl;

    // Output exact BC
    ofstream out(outFile);
    out << "node,bc\n";
    for (int i = 0; i < n; i++) {
        out << i << "," << fixed << setprecision(10) << BC[i] << "\n";
    }
    out.close();

    cout << n << "," << edgeSet.size() << "," << fixed << setprecision(2) << secs << endl;
    return 0;
}
