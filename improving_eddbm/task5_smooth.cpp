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

// Global struct to hold P generation for all models
struct ModelProbs {
    vector<double> eddbm;
    vector<double> caeddbm;
    vector<double> pdeddbm;
};

// Returns standard dependency given a sampled S and target V
double dependency(Graph& G, int s, int target) {
    int n = G.n;
    vector<vector<int>> pred(n);
    vector<int> dist(n, -1);
    vector<double> sigma(n, 0.0);
    stack<int> S; queue<int> Q;
    dist[s] = 0; sigma[s] = 1.0; Q.push(s);

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
        for (int v : pred[w]) { delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]); }
    }
    return delta[target];
}

ModelProbs generate_all_probs(Graph& G, int v, double lambda) {
    int n = G.n;
    vector<int> dist(n, -1);
    vector<int> parent(n, -1);
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
                parent[w] = u;
                q.push(w);
            }
            if (dist[w] == dist[u] + 1) {
                sigma[w] += sigma[u];
            }
        }
    }

    ModelProbs P;
    P.eddbm.assign(n, 0.0);
    P.caeddbm.assign(n, 0.0);
    P.pdeddbm.assign(n, 0.0);

    double tot_ed = 0, tot_ca = 0, tot_pd = 0;

    for (int i = 0; i < n; i++) {
        if (i == v || dist[i] == -1) continue;
        int d = max(1, G.degree[i]);
        
        // EDDBM
        double val_ed = pow(lambda, -dist[i]) / d;
        P.eddbm[i] = val_ed;
        tot_ed += val_ed;

        // CAEDDBM
        int p = parent[i];
        int common = 0;
        if(p != -1) {
            unordered_set<int> neighbors_p(G.adj[p].begin(), G.adj[p].end());
            for(int nbr : G.adj[i]) {
                if(neighbors_p.count(nbr)) common++;
            }
        }
        double c_hat = (double)common / d;
        double val_ca = pow(lambda, -dist[i]) / (d * (1.0 + c_hat));
        P.caeddbm[i] = val_ca;
        tot_ca += val_ca;

        // PDEDDBM
        double val_pd = sigma[i] / (d * pow(lambda, dist[i]));
        P.pdeddbm[i] = val_pd;
        tot_pd += val_pd;
    }

    if(tot_ed > 0) for(int i=0; i<n; i++) P.eddbm[i] /= tot_ed;
    if(tot_ca > 0) for(int i=0; i<n; i++) P.caeddbm[i] /= tot_ca;
    if(tot_pd > 0) for(int i=0; i<n; i++) P.pdeddbm[i] /= tot_pd;

    return P;
}

int main(int argc, char* argv[]) {
    if(argc < 5) {
        cout << "Usage: ./task5_smooth graph.txt out_prefix num_pairs max_nodes\n";
        return 0;
    }

    string filename = argv[1];
    string outpre = argv[2];
    int num_pairs = stoi(argv[3]);
    int max_nodes = stoi(argv[4]);

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

    double lambda = G.averageDegree();
    cout << "Computing Exact BC...\n";
    vector<double> exactBC = exact_brandes(G);
    double maxBC = 0;
    for(double x : exactBC) maxBC = max(maxBC, x);
    if(maxBC == 0) maxBC = 1;

    mt19937 rng(4242);
    uniform_int_distribution<int> nodeDist(0, G.n - 1);
    
    vector<int> As(num_pairs), Bs(num_pairs);
    for(int i=0; i<num_pairs; i++) {
        As[i] = nodeDist(rng);
        Bs[i] = nodeDist(rng);
        while(As[i] == Bs[i]) Bs[i] = nodeDist(rng);
    }

    cout << "Generating Probability Sets...\n";
    vector<ModelProbs> P_A(num_pairs), P_B(num_pairs);
    for(int i=0; i<num_pairs; i++) {
        P_A[i] = generate_all_probs(G, As[i], lambda);
        P_B[i] = generate_all_probs(G, Bs[i], lambda);
    }

    // Prepare accumulators
    vector<double> estA_ed(num_pairs, 0.0), estB_ed(num_pairs, 0.0);
    vector<double> estA_ca(num_pairs, 0.0), estB_ca(num_pairs, 0.0);
    vector<double> estA_pd(num_pairs, 0.0), estB_pd(num_pairs, 0.0);

    uniform_real_distribution<double> drng(0.0, 1.0);

    ofstream res_file(outpre + "_smoothed_results.csv");
    res_file << "T,Dataset,Model,Efficiency,AvgError\n";

    cout << "Simulating T=1 to 100 on " << outpre << "...\n";
    for(int T=1; T<=100; T++) {
        int corr_ed = 0, corr_ca = 0, corr_pd = 0;
        double err_ed = 0, err_ca = 0, err_pd = 0;

        for(int i=0; i<num_pairs; i++) {
            double trueA = exactBC[As[i]];
            double trueB = exactBC[Bs[i]];
            bool exactOrder = (trueA > trueB);
            
            auto pickSample = [&](const vector<double>& P) -> int {
                double r = drng(rng), cum = 0;
                for(int j=0; j<(int)P.size(); j++){ 
                    cum += P[j]; 
                    if(r <= cum) return j; 
                }
                return -1;
            };

            // EDDBM
            int sA_ed = pickSample(P_A[i].eddbm);
            int sB_ed = pickSample(P_B[i].eddbm);
            if(sA_ed != -1 && P_A[i].eddbm[sA_ed] > 0) estA_ed[i] += dependency(G, sA_ed, As[i]) / P_A[i].eddbm[sA_ed];
            if(sB_ed != -1 && P_B[i].eddbm[sB_ed] > 0) estB_ed[i] += dependency(G, sB_ed, Bs[i]) / P_B[i].eddbm[sB_ed];
            if(((estA_ed[i]/T) > (estB_ed[i]/T)) == exactOrder) corr_ed++;
            err_ed += (abs(trueA - estA_ed[i]/T)/maxBC + abs(trueB - estB_ed[i]/T)/maxBC) * 50.0;

            // CAEDDBM
            int sA_ca = pickSample(P_A[i].caeddbm);
            int sB_ca = pickSample(P_B[i].caeddbm);
            if(sA_ca != -1 && P_A[i].caeddbm[sA_ca] > 0) estA_ca[i] += dependency(G, sA_ca, As[i]) / P_A[i].caeddbm[sA_ca];
            if(sB_ca != -1 && P_B[i].caeddbm[sB_ca] > 0) estB_ca[i] += dependency(G, sB_ca, Bs[i]) / P_B[i].caeddbm[sB_ca];
            if(((estA_ca[i]/T) > (estB_ca[i]/T)) == exactOrder) corr_ca++;
            err_ca += (abs(trueA - estA_ca[i]/T)/maxBC + abs(trueB - estB_ca[i]/T)/maxBC) * 50.0;

            // PDEDDBM
            int sA_pd = pickSample(P_A[i].pdeddbm);
            int sB_pd = pickSample(P_B[i].pdeddbm);
            if(sA_pd != -1 && P_A[i].pdeddbm[sA_pd] > 0) estA_pd[i] += dependency(G, sA_pd, As[i]) / P_A[i].pdeddbm[sA_pd];
            if(sB_pd != -1 && P_B[i].pdeddbm[sB_pd] > 0) estB_pd[i] += dependency(G, sB_pd, Bs[i]) / P_B[i].pdeddbm[sB_pd];
            if(((estA_pd[i]/T) > (estB_pd[i]/T)) == exactOrder) corr_pd++;
            err_pd += (abs(trueA - estA_pd[i]/T)/maxBC + abs(trueB - estB_pd[i]/T)/maxBC) * 50.0;
        }

        res_file << T << "," << outpre << ",EDDBM," << (corr_ed*100.0)/num_pairs << "," << err_ed/num_pairs << "\n";
        res_file << T << "," << outpre << ",CAEDDBM," << (corr_ca*100.0)/num_pairs << "," << err_ca/num_pairs << "\n";
        res_file << T << "," << outpre << ",PDEDDBM," << (corr_pd*100.0)/num_pairs << "," << err_pd/num_pairs << "\n";
    }

    res_file.close();
    cout << "Finished logging into " << outpre << "_smoothed_results.csv.\n";
    return 0;
}
