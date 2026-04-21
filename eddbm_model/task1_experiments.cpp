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
            for (int j = 0; j < (int)P.size(); j++) {
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
    if (argc < 4) {
        cout << "Usage: ./task1_experiments dataset.txt out_prefix num_trials [max_nodes]\n";
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
    for (int node : nodeSet)
        idMap[node] = idx++;

    Graph G(nodeSet.size());
    for (auto &e : rawEdges)
        G.addEdge(idMap[e.first], idMap[e.second]);

    cout << "Loaded graph: " << G.n << " nodes, " << rawEdges.size() << " edges.\n";

    // Subsample graph if it is too massive because exact brandes takes O(NM) 
    if(max_nodes > 0 && G.n > max_nodes) {
        cout << "Subsampling graph to " << max_nodes << " nodes via BFS..." << endl;
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
        
        // Rebuild G
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
        cout << "Subsampled G to " << G.n << " nodes.\n";
    }

    EDDBM solver(G);

    // 1. Probabilities: We pick a random node and dump its generated generic probabilities across distances.
    cout << "Generating Probability mappings...\n";
    mt19937 rng(100);
    uniform_int_distribution<int> distNode(0, G.n - 1);
    int target_v = distNode(rng);
    vector<double> P = solver.generateProb(target_v);

    // Get distance array
    vector<int> dists(G.n, -1);
    queue<int> q;
    dists[target_v] = 0;
    q.push(target_v);
    while(!q.empty()) {
        int curr = q.front(); q.pop();
        for(int nbr : G.adj[curr]) {
            if(dists[nbr] == -1) {
                dists[nbr] = dists[curr] + 1;
                q.push(nbr);
            }
        }
    }

    ofstream p_file(out_prefix + "_prob.csv");
    p_file << "Distance,Probability\n";
    for(int i=0; i<G.n; i++) {
        if(i != target_v && dists[i] != -1) {
            p_file << dists[i] << "," << P[i] << "\n";
        }
    }
    p_file.close();

    // 2. Compute Exact BC
    cout << "Computing Exact BC...\n";
    vector<double> exactBC = exact_brandes(G);

    // Normalize Exact BC to [0,1] for accurate error metric scaling if needed
    double maxBC = 0;
    for(double x : exactBC) maxBC = max(maxBC, x);
    if(maxBC == 0) maxBC = 1;

    // 3. Evaluate T
    vector<int> T_values = {10, 50, 100, 200, 500};
    ofstream res_file(out_prefix + "_results.csv");
    res_file << "T,Efficiency,AvgError\n";

    for(int T : T_values) {
        cout << "Evaluating T=" << T << "...\n";
        int correct = 0;
        double totalError = 0.0;
        
        for (int i = 0; i < trials; i++) {
            int a = distNode(rng);
            int b = distNode(rng);
            while(a == b) b = distNode(rng);

            double estA = solver.estimate(a, T);
            double estB = solver.estimate(b, T);

            bool exactOrder = exactBC[a] > exactBC[b];
            bool estOrder = estA > estB;

            if (exactOrder == estOrder) correct++;
            
            // Normalized error
            double errA = abs((exactBC[a] - estA)/maxBC);
            double errB = abs((exactBC[b] - estB)/maxBC);
            totalError += (errA + errB) / 2.0;
        }

        double efficiency = (double)correct / trials;
        double avgError = totalError / trials;
        res_file << T << "," << efficiency << "," << avgError << "\n";
    }
    res_file.close();
    
    cout << "Done for " << filename << ".\n";
    return 0;
}
