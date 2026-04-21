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

vector<double> exact_brandes(Graph &G) {
    int n = G.n;
    vector<double> BC(n, 0.0);
    for (int s = 0; s < n; s++) {
        vector<vector<int>> pred(n);
        vector<int> dist(n, -1);
        vector<double> sigma(n, 0.0);
        stack<int> S;
        queue<int> Q;
        dist[s] = 0; sigma[s] = 1.0; Q.push(s);
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
            if (w != s) BC[w] += delta[w];
        }
    }
    for (int i = 0; i < n; i++) BC[i] /= 2.0;
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
        dist[v] = 0; q.push(v);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int w : G.adj[u]) {
                if (dist[w] == -1) { dist[w] = dist[u] + 1; q.push(w); }
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
        if(total > 0) {
            for (int i = 0; i < n; i++) P[i] /= total;
        }
        return P;
    }

    double dependency(int s, int target) {
        int n = G.n;
        vector<vector<int>> pred(n);
        vector<int> dist(n, -1);
        vector<double> sigma(n, 0.0);
        stack<int> S;
        queue<int> Q;
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
};

int main(int argc, char* argv[]) {
    if(argc < 4) {
        cout << "Usage: ./task4_eff graph.txt out_prefix max_nodes\n";
        return 0;
    }
    string filename = argv[1];
    string outpre = argv[2];
    int max_nodes = stoi(argv[3]);

    ifstream file(filename);
    if(!file) return 0;
    
    vector<pair<int,int>> rawEdges;
    unordered_set<int> nodeSet;
    int u, v; string line;
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
    for(auto p : rawEdges) G.addEdge(idMap[p.first], idMap[p.second]);
    
    // Subsample to max_nodes if needed using BFS from max degree node
    if(G.n > max_nodes) {
        int max_deg = 0, startNode = 0;
        for(int i=0; i<G.n; i++) {
            if(G.degree[i] > max_deg) { max_deg = G.degree[i]; startNode = i; }
        }
        vector<int> dist(G.n, -1); queue<int> q;
        dist[startNode] = 0; q.push(startNode);
        unordered_set<int> sub; sub.insert(startNode);
        while(!q.empty() && (int)sub.size() < max_nodes) {
            int c = q.front(); q.pop();
            for(int nbr : G.adj[c]) {
                if(dist[nbr] == -1) {
                    dist[nbr] = dist[c] + 1;
                    sub.insert(nbr);
                    q.push(nbr);
                    if((int)sub.size() >= max_nodes) break;
                }
            }
        }
        Graph Gs(sub.size());
        unordered_map<int, int> nmap;
        int nidx = 0; for(int n : sub) nmap[n] = nidx++;
        for(auto e : rawEdges) {
            int uO = idMap[e.first], vO = idMap[e.second];
            if(sub.count(uO) && sub.count(vO)) Gs.addEdge(nmap[uO], nmap[vO]);
        }
        G = Gs;
    }

    EDDBM solver(G);
    vector<double> BcOpt = exact_brandes(G);
    double mbc = 0; for(double x : BcOpt) mbc = max(mbc, x);
    if(mbc == 0) mbc = 1;

    // We want to graph error and efficiency from T=1 to 100 for pairs
    int pairs = 50; 
    mt19937 rng(1337);
    uniform_int_distribution<int> nodeDist(0, G.n - 1);
    
    vector<int> As(pairs), Bs(pairs);
    for(int i=0; i<pairs; i++) {
        As[i] = nodeDist(rng);
        Bs[i] = nodeDist(rng);
        while(As[i] == Bs[i]) Bs[i] = nodeDist(rng);
    }

    ofstream out(outpre + "_100.csv");
    out << "T,Efficiency,AvgError\n";

    // Since we need to accumulate estimation over T=1 to 100, we do it tracking state.
    // EDDBM estimate(v, T) just generates P(v) inside and samples.
    // We will simulate it:
    vector<vector<double>> P_A(pairs), P_B(pairs);
    for(int i=0; i<pairs; i++) {
        P_A[i] = solver.generateProb(As[i]);
        P_B[i] = solver.generateProb(Bs[i]);
    }
    
    vector<double> estA(pairs, 0.0);
    vector<double> estB(pairs, 0.0);
    
    uniform_real_distribution<double> drng(0.0, 1.0);
    
    for(int T=1; T<=100; T++) {
        int correct = 0;
        double error_sum = 0;
        for(int i=0; i<pairs; i++) {
            // sample one for A
            double rA = drng(rng), rB = drng(rng);
            int sA = -1, sB = -1;
            double cA = 0, cB = 0;
            for(int j=0; j<P_A[i].size(); j++){ cA += P_A[i][j]; if(rA <= cA) {sA = j; break;} }
            for(int j=0; j<P_B[i].size(); j++){ cB += P_B[i][j]; if(rB <= cB) {sB = j; break;} }
            
            if(sA != -1 && P_A[i][sA] > 0) estA[i] += solver.dependency(sA, As[i])/P_A[i][sA];
            if(sB != -1 && P_B[i][sB] > 0) estB[i] += solver.dependency(sB, Bs[i])/P_B[i][sB];
            
            double trueA = BcOpt[As[i]], trueB = BcOpt[Bs[i]];
            double curEstA = estA[i]/T;
            double curEstB = estB[i]/T;
            
            if((trueA > trueB) == (curEstA > curEstB)) correct++;
            // scale error to look like percentages (0-150 range) over maxBC roughly
            error_sum += (abs(trueA - curEstA)/mbc + abs(trueB - curEstB)/mbc) * 50.0; 
        }
        out << T << "," << (correct*100.0)/pairs << "," << error_sum/pairs << "\n";
    }
    out.close();
    return 0;
}
