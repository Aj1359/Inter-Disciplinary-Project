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
        if(n == 0) return 1;
        long long sum = 0;
        for (int d : degree) sum += d;
        return (double)sum / n;
    }
};

void run_experiments(Graph &G, string out_name) {
    int n = G.n;
    
    mt19937 rng(1337);
    uniform_int_distribution<int> nodeDist(0, n - 1);
    
    // Pick random target v
    int v = nodeDist(rng);
    
    // Exact Dependencies (Opt) mapped onto all nodes for v
    vector<vector<int>> pred(n);
    vector<int> dist(n, -1);
    vector<double> sigma(n, 0.0);
    stack<int> S;
    queue<int> Q;

    dist[v] = 0;
    sigma[v] = 1.0;
    Q.push(v);

    while (!Q.empty()) {
        int u = Q.front(); Q.pop();
        S.push(u);

        for (int w : G.adj[u]) {
            if (dist[w] < 0) {
                dist[w] = dist[u] + 1;
                Q.push(w);
            }
            if (dist[w] == dist[u] + 1) {
                sigma[w] += sigma[u];
                pred[w].push_back(u);
            }
        }
    }

    vector<double> delta(n, 0.0);
    while (!S.empty()) {
        int w = S.top(); S.pop();
        for (int u : pred[w]) {
            delta[u] += (sigma[u] / sigma[w]) * (1.0 + delta[w]);
        }
    }
    
    double total_delta = 0.0;
    for (int i = 0; i < n; i++) {
        if(i != v) total_delta += delta[i];
    }
    vector<double> Opt(n, 0.0);
    if(total_delta > 0) {
        for (int i = 0; i < n; i++) Opt[i] = delta[i] / total_delta;
    }
    
    // EDDBM Probs
    double lambda = G.averageDegree();
    vector<double> p_eddbm(n, 0.0);
    double tot_eddbm = 0.0;
    for (int i = 0; i < n; i++) {
        if(i == v || dist[i] == -1) continue;
        double val = pow(lambda, -dist[i]) / max(1, G.degree[i]);
        p_eddbm[i] = val;
        tot_eddbm += val;
    }
    if(tot_eddbm > 0) {
        for (int i = 0; i < n; i++) p_eddbm[i] /= tot_eddbm;
    }
    
    // DBM Probs (Distance-Based Model roughly P ~ 2^{-d})
    vector<double> p_dbm(n, 0.0);
    double tot_dbm = 0.0;
    for(int i = 0; i < n; i++) {
        if(i == v || dist[i] == -1) continue;
        double val = pow(2.0, -dist[i]) / max(1, G.degree[i]);
        p_dbm[i] = val;
        tot_dbm += val;
    }
    if(tot_dbm > 0) {
        for (int i = 0; i < n; i++) p_dbm[i] /= tot_dbm;
    }
    
    // Pick 100 random nodes consistently
    vector<int> samples;
    unordered_set<int> seen;
    while(samples.size() < 100) {
        int pick = nodeDist(rng);
        if(pick != v && dist[pick] != -1 && !seen.count(pick)) {
            seen.insert(pick);
            samples.push_back(pick);
        }
    }
    
    // We want to sort samples such that Opt descending
    sort(samples.begin(), samples.end(), [&](int a, int b){
        return Opt[a] > Opt[b];
    });
    
    ofstream out(out_name + "_prob.csv");
    out << "NodeIndex,Opt,EDDBM,DBM\n";
    for(int i = 0; i < 100; i++) {
        int s = samples[i];
        out << i << "," << Opt[s] << "," << p_eddbm[s] << "," << p_dbm[s] << "\n";
    }
    out.close();
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        cout << "Usage: ./task4_prob graph.txt outpre\n";
        return 0;
    }
    ifstream file(argv[1]);
    string outpre = argv[2];
    
    vector<pair<int,int>> edges;
    unordered_set<int> nodes;
    int u, v;
    while(file >> u >> v) {
        edges.push_back({u, v});
        nodes.insert(u);
        nodes.insert(v);
    }
    
    unordered_map<int,int> mp;
    int idx = 0;
    for(int x : nodes) mp[x] = idx++;
    
    Graph G(nodes.size());
    for(auto p : edges) {
        if(p.first != p.second)
            G.addEdge(mp[p.first], mp[p.second]);
    }
    
    cout << "Processing " << outpre << "...\n";
    run_experiments(G, outpre);
    cout << "Done " << outpre << ".\n";
    
    return 0;
}
