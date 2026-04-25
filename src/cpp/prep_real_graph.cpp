/*
 * prep_real_graph.cpp  
 * Compute features + exact BC for an existing graph file.
 * Just like gen_training_data but for a single real graph.
 *
 * Usage: ./prep_real_graph <graph.txt> <output_dir> <name>
 */
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <graph.txt> <output_dir> <name>\n";
        return 1;
    }
    string graphFile = argv[1];
    string outDir = argv[2];
    string name = argv[3];

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
    cerr << name << ": n=" << n << " m=" << edgeSet.size() << endl;

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
    auto t1 = chrono::steady_clock::now();
    cerr << "Brandes: " << chrono::duration<double>(t1-t0).count() << "s" << endl;

    // Features
    double avgDeg = 0;
    int maxDeg = 0;
    for (int i = 0; i < n; i++) { avgDeg += deg[i]; maxDeg = max(maxDeg, deg[i]); }
    avgDeg /= n;

    system(("mkdir -p " + outDir).c_str());
    string csvFile = outDir + "/" + name + ".csv";
    ofstream out(csvFile);
    out << "node,bc,log_degree,degree_centrality,log_nb_avg_deg,"
        << "nb_max_deg_norm,clustering,log_reach2,reach2_norm,deg_over_avg\n";

    for (int v = 0; v < n; v++) {
        int d = deg[v];
        double nbSum = 0; int nbMax = 0;
        for (int nb : adj[v]) { nbSum += deg[nb]; nbMax = max(nbMax, deg[nb]); }
        double nbAvg = d > 0 ? nbSum / d : 0;

        double clust = 0.0;
        if (d >= 2) {
            unordered_set<int> ns(adj[v].begin(), adj[v].end());
            int tri = 0;
            for (int nb : adj[v])
                for (int nb2 : adj[nb])
                    if (nb2 != v && ns.count(nb2)) tri++;
            clust = (double)tri / (d * (d - 1));
        }

        unordered_set<int> h1(adj[v].begin(), adj[v].end());
        int r2 = (int)h1.size();
        for (int nb : adj[v])
            for (int nb2 : adj[nb])
                if (nb2 != v && !h1.count(nb2)) r2++;

        out << v << "," << fixed << setprecision(10) << BC[v]
            << "," << log(1.0 + d)
            << "," << (double)d / max(1, n-1)
            << "," << log(1.0 + nbAvg)
            << "," << (maxDeg > 0 ? (double)nbMax / maxDeg : 0.0)
            << "," << clust
            << "," << log(1.0 + r2)
            << "," << (double)r2 / max(1, n)
            << "," << (avgDeg > 0 ? d / avgDeg : 0.0)
            << "\n";
    }

    // Edge list
    string edgeFile = outDir + "/" + name + "_edges.txt";
    ofstream eout(edgeFile);
    for (int u = 0; u < n; u++)
        for (int w : adj[u])
            if (u < w) eout << u << " " << w << "\n";

    cout << name << " done" << endl;
    return 0;
}
