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
        if(n == 0) return 1.0;
        long long sum = 0;
        for (int d : degree) sum += d;
        return (double)sum / n;
    }
};

vector<double> exact_brandes(Graph &G) {
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

// Helper to compute dependency from s to target
double dependency(Graph& G, int s, int target) {
    int n = G.n;
    static vector<vector<int>> pred;
    static vector<int> dist;
    static vector<double> sigma;
    static vector<double> delta;
    static vector<int> S;
    static vector<int> Q;
    
    if (pred.size() < n) {
        pred.resize(n);
        dist.resize(n);
        sigma.resize(n);
        delta.resize(n);
    }
    for (int i=0; i<n; i++) {
        pred[i].clear();
        dist[i] = -1;
        sigma[i] = 0.0;
        delta[i] = 0.0;
    }
    S.clear();
    Q.clear();
    int q_head = 0;

    dist[s] = 0;
    sigma[s] = 1.0;
    Q.push_back(s);

    while (q_head < Q.size()) {
        int v = Q[q_head++];
        S.push_back(v);

        for (int w : G.adj[v]) {
            if (dist[w] < 0) {
                dist[w] = dist[v] + 1;
                Q.push_back(w);
            }
            if (dist[w] == dist[v] + 1) {
                sigma[w] += sigma[v];
                pred[w].push_back(v);
            }
        }
    }

    while (!S.empty()) {
        int w = S.back(); S.pop_back();
        for (int v : pred[w]) {
            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
        }
    }
    return delta[target];
}

class CAEDDBM {
private:
    Graph &G;
    mt19937 rng;

public:
    CAEDDBM(Graph &graph) : G(graph), rng(42) {}

    vector<double> generateProb(int v) {
        int n = G.n;
        vector<int> dist(n, -1);
        vector<int> parent(n, -1);
        queue<int> q;

        dist[v] = 0;
        q.push(v);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int w : G.adj[u]) {
                if (dist[w] == -1) {
                    dist[w] = dist[u] + 1;
                    parent[w] = u;
                    q.push(w);
                }
            }
        }

        double lambda = G.averageDegree();
        vector<double> P(n, 0.0);
        double total = 0.0;

        for (int i = 0; i < n; i++) {
            if (i == v || dist[i] == -1) continue;
            
            // compute c_hat(i) = common neighbors of i with parent / deg(i)
            int p = parent[i];
            int common = 0;
            if(p != -1) {
                unordered_set<int> neighbors_p(G.adj[p].begin(), G.adj[p].end());
                for(int nbr : G.adj[i]) {
                    if(neighbors_p.count(nbr)) common++;
                }
            }
            
            double c_hat = (double)common / max(1, G.degree[i]);
            
            double val = pow(lambda, -dist[i]) / (max(1, G.degree[i]) * (1.0 + c_hat));
            P[i] = val;
            total += val;
        }

        if(total > 0) {
            for (int i = 0; i < n; i++) P[i] /= total;
        }
        return P;
    }

    double estimate(int v, int T) {
        vector<double> P = generateProb(v);
        uniform_real_distribution<double> dist(0.0, 1.0);
        double est = 0.0;

        for (int i = 0; i < T; i++) {
            double r = dist(rng);
            double cum = 0.0;
            int s = -1;
            for (int j = 0; j < (int)P.size(); j++) {
                cum += P[j];
                if (r <= cum) {
                    s = j;
                    break;
                }
            }
            if (s != -1 && P[s] > 0)
                est += dependency(G, s, v) / P[s];
        }

        return est / T;
    }
};

class PDEDDBM {
private:
    Graph &G;
    mt19937 rng;

public:
    PDEDDBM(Graph &graph) : G(graph), rng(123) {}

    vector<double> generateProb(int v) {
        int n = G.n;
        vector<int> dist(n, -1);
        vector<double> sigma(n, 0.0);
        queue<int> q;

        dist[v] = 0;
        sigma[v] = 1.0;
        q.push(v);

        while (!q.empty()) {
            int u = q.front(); q.pop();

            for (int w : G.adj[u]) {
                if (dist[w] == -1) {
                    dist[w] = dist[u] + 1;
                    q.push(w);
                }
                if (dist[w] == dist[u] + 1) {
                    sigma[w] += sigma[u];
                }
            }
        }

        double lambda = G.averageDegree();
        vector<double> P(n, 0.0);
        double total = 0.0;

        for (int i = 0; i < n; i++) {
            if (i == v || dist[i] == -1) continue;
            
            // PDEDDBM incorporates sigma_{vi} directly
            double val = sigma[i] / (max(1, G.degree[i]) * pow(lambda, dist[i]));
            P[i] = val;
            total += val;
        }

        if(total > 0) {
            for (int i = 0; i < n; i++) P[i] /= total;
        }
        return P;
    }

    double estimate(int v, int T) {
        vector<double> P = generateProb(v);
        uniform_real_distribution<double> dist(0.0, 1.0);
        double est = 0.0;

        for (int i = 0; i < T; i++) {
            double r = dist(rng);
            double cum = 0.0;
            int s = -1;
            for (int j = 0; j < (int)P.size(); j++) {
                cum += P[j];
                if (r <= cum) {
                    s = j;
                    break;
                }
            }
            if (s != -1 && P[s] > 0)
                est += dependency(G, s, v) / P[s];
        }

        return est / T;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: ./task3_experiments dataset.txt out_prefix num_trials [max_nodes]\n";
        return 0;
    }

    string filename = argv[1];
    string out_prefix = argv[2];
    int trials = stoi(argv[3]);
    int max_nodes = (argc >= 5) ? stoi(argv[4]) : -1;

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
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        ss >> u >> v;
        if(u == v) continue;
        rawEdges.push_back({u, v});
        nodeSet.insert(u);
        nodeSet.insert(v);
    }

    unordered_map<int,int> idMap;
    int idx = 0;
    for (int node : nodeSet) idMap[node] = idx++;

    Graph G(nodeSet.size());
    for (auto &e : rawEdges) G.addEdge(idMap[e.first], idMap[e.second]);

    cout << "Loaded graph: " << G.n << " nodes.\n";

    if(max_nodes > 0 && G.n > max_nodes) {
        cout << "Subsampling graph to " << max_nodes << " nodes...\n";
        int startNode = 0;
        vector<int> dist(G.n, -1);
        queue<int> q;
        dist[startNode] = 0;
        q.push(startNode);
        
        unordered_set<int> subNodes;
        subNodes.insert(startNode);

        while(!q.empty() && (int)subNodes.size() < max_nodes) {
            int curr = q.front(); q.pop();
            for(int neighbor : G.adj[curr]) {
                if(dist[neighbor] == -1) {
                    dist[neighbor] = dist[curr] + 1;
                    subNodes.insert(neighbor);
                    q.push(neighbor);
                    if((int)subNodes.size() >= max_nodes) break;
                }
            }
        }
        
        Graph G_sub(subNodes.size());
        unordered_map<int, int> newMap;
        int newIdx = 0;
        for(int n : subNodes) newMap[n] = newIdx++;
        
        for(auto &e : rawEdges) {
            int u_old = idMap[e.first];
            int v_old = idMap[e.second];
            if(subNodes.count(u_old) && subNodes.count(v_old)) {
                G_sub.addEdge(newMap[u_old], newMap[v_old]);
            }
        }
        G = G_sub;
    }

    CAEDDBM ca_solver(G);
    PDEDDBM pd_solver(G);

    cout << "Computing Exact BC...\n";
    vector<double> exactBC = exact_brandes(G);
    double maxBC = 0;
    for(double x : exactBC) maxBC = max(maxBC, x);
    if(maxBC == 0) maxBC = 1;

    vector<int> T_values = {5, 10, 15, 20, 25, 30};
    ofstream res_file(out_prefix + "_improved_results.csv");
    res_file << "Model,T,Efficiency,AvgError\n";

    mt19937 rng(999);
    uniform_int_distribution<int> distNode(0, G.n - 1);

    for(int T : T_values) {
        cout << "Evaluating CAEDDBM & PDEDDBM T=" << T << "...\n";
        int ca_corr = 0, pd_corr = 0;
        double ca_err = 0.0, pd_err = 0.0;
        
        for (int i = 0; i < trials; i++) {
            int a = distNode(rng);
            int b = distNode(rng);
            while(a == b) b = distNode(rng);

            bool exactOrder = exactBC[a] > exactBC[b];
            
            // CAEDDBM
            double estA_ca = ca_solver.estimate(a, T);
            double estB_ca = ca_solver.estimate(b, T);
            if ((estA_ca > estB_ca) == exactOrder) ca_corr++;
            ca_err += (abs((exactBC[a] - estA_ca)/maxBC) + abs((exactBC[b] - estB_ca)/maxBC)) / 2.0;
            
            // PDEDDBM
            double estA_pd = pd_solver.estimate(a, T);
            double estB_pd = pd_solver.estimate(b, T);
            if ((estA_pd > estB_pd) == exactOrder) pd_corr++;
            pd_err += (abs((exactBC[a] - estA_pd)/maxBC) + abs((exactBC[b] - estB_pd)/maxBC)) / 2.0;
        }

        res_file << "CAEDDBM," << T << "," << (double)ca_corr/trials << "," << ca_err/trials << "\n";
        res_file << "PDEDDBM," << T << "," << (double)pd_corr/trials << "," << pd_err/trials << "\n";
    }
    res_file.close();
    
    cout << "Done Improved EDDBM for " << filename << ".\n";
    return 0;
}
