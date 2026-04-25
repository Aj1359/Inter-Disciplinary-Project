/*
 * gen_training_data.cpp
 * Generate synthetic graphs + exact BC for GNN training.
 * Outputs CSV: one file per graph with node features and BC values.
 *
 * Usage: ./gen_training_data <output_dir> <num_graphs> [min_n] [max_n] [seed]
 */
#include <bits/stdc++.h>
using namespace std;

struct Graph {
    int n;
    vector<vector<int>> adj;
    vector<int> deg;

    Graph(int n) : n(n), adj(n), deg(n, 0) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }

    double avgDeg() const {
        long long s = 0;
        for (int d : deg) s += d;
        return n ? (double)s / n : 0.0;
    }
};

// Exact Brandes
vector<double> exactBrandes(const Graph &G) {
    int n = G.n;
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
            for (int w : G.adj[v]) {
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
    return BC;
}

// Generate BA graph
Graph generateBA(int n, int m_attach, mt19937 &rng) {
    Graph G(n);
    // Start with complete graph of m_attach nodes
    for (int i = 0; i < min(m_attach, n); i++)
        for (int j = i + 1; j < min(m_attach, n); j++)
            G.addEdge(i, j);

    for (int v = m_attach; v < n; v++) {
        long long totalDeg = 0;
        for (int i = 0; i < v; i++) totalDeg += G.deg[i];

        set<int> targets;
        int attempts = 0;
        while ((int)targets.size() < m_attach && attempts < m_attach * 20) {
            attempts++;
            if (totalDeg <= 0) {
                targets.insert(rng() % v);
            } else {
                long long r = rng() % totalDeg;
                long long cum = 0;
                for (int i = 0; i < v; i++) {
                    cum += G.deg[i];
                    if (cum > r) { targets.insert(i); break; }
                }
            }
        }
        for (int t : targets) G.addEdge(v, t);
    }
    return G;
}

// Generate ER graph
Graph generateER(int n, double p, mt19937 &rng) {
    Graph G(n);
    uniform_real_distribution<double> udist(0.0, 1.0);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (udist(rng) < p)
                G.addEdge(i, j);
    return G;
}

// Compute node features
struct NodeFeatures {
    double log_degree;
    double degree_centrality;
    double log_nb_avg_deg;
    double nb_max_deg_norm;
    double clustering;
    double log_reach2;
    double reach2_norm;
    double deg_over_avg;
};

vector<NodeFeatures> computeFeatures(const Graph &G) {
    int n = G.n;
    double avgDeg = G.avgDeg();
    int maxDeg = *max_element(G.deg.begin(), G.deg.end());

    vector<NodeFeatures> feats(n);
    for (int v = 0; v < n; v++) {
        int d = G.deg[v];

        // Neighbor degree stats
        double nbSum = 0;
        int nbMax = 0;
        for (int nb : G.adj[v]) {
            nbSum += G.deg[nb];
            nbMax = max(nbMax, G.deg[nb]);
        }
        double nbAvg = d > 0 ? nbSum / d : 0;

        // Clustering coefficient
        double clustering = 0.0;
        if (d >= 2) {
            unordered_set<int> nbSet(G.adj[v].begin(), G.adj[v].end());
            int triCount = 0;
            for (int nb : G.adj[v])
                for (int nb2 : G.adj[nb])
                    if (nb2 != v && nbSet.count(nb2))
                        triCount++;
            clustering = (double)triCount / (d * (d - 1));
        }

        // 2-hop reach
        unordered_set<int> hop1(G.adj[v].begin(), G.adj[v].end());
        int reach2 = (int)hop1.size();
        for (int nb : G.adj[v])
            for (int nb2 : G.adj[nb])
                if (nb2 != v && !hop1.count(nb2))
                    reach2++;

        feats[v] = {
            log(1.0 + d),
            (double)d / max(1, n - 1),
            log(1.0 + nbAvg),
            maxDeg > 0 ? (double)nbMax / maxDeg : 0.0,
            clustering,
            log(1.0 + reach2),
            (double)reach2 / max(1, n),
            avgDeg > 0 ? d / avgDeg : 0.0,
        };
    }
    return feats;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <output_dir> <num_graphs> [min_n=100] [max_n=800] [seed=42]\n";
        return 1;
    }

    string outDir = argv[1];
    int numGraphs = atoi(argv[2]);
    int minN = argc > 3 ? atoi(argv[3]) : 100;
    int maxN = argc > 4 ? atoi(argv[4]) : 800;
    unsigned seed = argc > 5 ? atoi(argv[5]) : 42;

    mt19937 rng(seed);

    // Create output directory
    system(("mkdir -p " + outDir).c_str());

    for (int g = 0; g < numGraphs; g++) {
        int n = minN + rng() % (maxN - minN + 1);
        bool isBA = rng() % 2;

        Graph G(0);
        string gtype;

        if (isBA) {
            int m = 2 + rng() % 8;  // m_attach in [2, 9]
            G = generateBA(n, m, rng);
            gtype = "BA";
        } else {
            double avgDegTarget = 3.0 + (rng() % 150) / 10.0;  // [3, 18]
            double p = avgDegTarget / (n - 1);
            G = generateER(n, p, rng);
            gtype = "ER";
        }

        // Skip if too sparse
        if (*max_element(G.deg.begin(), G.deg.end()) < 2) continue;

        // Compute exact BC
        auto bc = exactBrandes(G);
        auto feats = computeFeatures(G);

        // Write to CSV
        string filename = outDir + "/graph_" + to_string(g) + ".csv";
        ofstream out(filename);
        out << "node,bc,log_degree,degree_centrality,log_nb_avg_deg,"
            << "nb_max_deg_norm,clustering,log_reach2,reach2_norm,deg_over_avg\n";

        for (int v = 0; v < G.n; v++) {
            out << v << "," << fixed << setprecision(10) << bc[v]
                << "," << feats[v].log_degree
                << "," << feats[v].degree_centrality
                << "," << feats[v].log_nb_avg_deg
                << "," << feats[v].nb_max_deg_norm
                << "," << feats[v].clustering
                << "," << feats[v].log_reach2
                << "," << feats[v].reach2_norm
                << "," << feats[v].deg_over_avg
                << "\n";
        }

        // Write edge list
        string edgefile = outDir + "/graph_" + to_string(g) + "_edges.txt";
        ofstream eout(edgefile);
        for (int u = 0; u < G.n; u++)
            for (int w : G.adj[u])
                if (u < w)
                    eout << u << " " << w << "\n";

        cout << "Graph " << g << ": " << gtype << " n=" << G.n
             << " m=" << G.avgDeg() * G.n / 2 << " avgDeg=" << fixed
             << setprecision(2) << G.avgDeg() << endl;
    }

    cout << "Done! Generated " << numGraphs << " graphs in " << outDir << endl;
    return 0;
}
