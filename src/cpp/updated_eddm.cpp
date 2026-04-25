#include "graph.h"
#include "models.h"
#include "estimator.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <set>

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Graph loader — edge list format (u v per line, 0 or 1 indexed)
// ============================================================
Graph load_graph(const string& path, bool one_indexed = false) {
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "Cannot open: " << path << endl;
        exit(1);
    }
    string line;
    int max_node = -1;
    vector<pair<int,int>> edges;

    while (getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;
        if (one_indexed) { u--; v--; }
        if (u == v) continue;
        max_node = max(max_node, max(u, v));
        edges.push_back({u, v});
    }

    Graph G(max_node + 1);
    // Deduplicate
    set<pair<int,int>> seen;
    for (auto [u,v] : edges) {
        if (u > v) swap(u, v);
        if (!seen.count({u,v})) {
            seen.insert({u,v});
            G.add_edge(u, v);
        }
    }
    G.finalize();
    return G;
}

// ============================================================
// Generate synthetic ER graph
// ============================================================
Graph gen_ER(int n, double p, mt19937& rng) {
    Graph G(n);
    bernoulli_distribution bern(p);
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            if (bern(rng)) G.add_edge(i, j);
    G.finalize();
    return G;
}

// ============================================================
// Generate synthetic BA graph
// ============================================================
Graph gen_BA(int n, int k, mt19937& rng) {
    Graph G(n);
    // Start with complete graph of size k
    for (int i = 0; i < min(k, n); i++)
        for (int j = i+1; j < min(k, n); j++)
            G.add_edge(i, j);

    vector<int> stubs;
    for (int i = 0; i < min(k, n); i++)
        for (int j = 0; j < G.deg[i]; j++)
            stubs.push_back(i);

    for (int i = k; i < n; i++) {
        set<int> chosen;
        while ((int)chosen.size() < min(k, i)) {
            uniform_int_distribution<int> pick(0, (int)stubs.size()-1);
            int c = stubs[pick(rng)];
            if (c != i) chosen.insert(c);
        }
        for (int c : chosen) {
            G.add_edge(i, c);
            stubs.push_back(i);
            stubs.push_back(c);
        }
    }
    G.finalize();
    return G;
}

// ============================================================
// Run full benchmark on one graph
// ============================================================
void run_benchmark(const string& name, Graph& G,
                   int T, int num_pairs, mt19937& rng,
                   ofstream& out_csv) {
    if (G.n < 10) { cerr << "  Graph too small, skip\n"; return; }

    cout << "  Computing exact BC (n=" << G.n << ", m=" << G.m << ")...\n";
    auto t0 = chrono::high_resolution_clock::now();
    auto exact_BC = G.exact_betweenness();
    auto t1 = chrono::high_resolution_clock::now();
    double exact_time = chrono::duration<double>(t1 - t0).count();
    cout << "  Exact BC done in " << fixed << setprecision(2) << exact_time << "s\n";

    vector<ModelType> models = {UNIFORM, EDDBM, WEDDBM, PDEDDBM, CAEDDBM, FUSION};

    for (auto model : models) {
        string mname = model_name(model);
        cout << "    Model: " << mname << " T=" << T << "...";
        auto tm0 = chrono::high_resolution_clock::now();
        auto result = evaluate_model(G, exact_BC, model, T, num_pairs, rng);
        auto tm1 = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(tm1 - tm0).count();
        cout << " Eff=" << fixed << setprecision(2) << result.efficiency
             << "% Err=" << result.avg_error << "% time=" << elapsed << "s\n";

        out_csv << name << "," << G.n << "," << G.m << ","
                << mname << "," << T << ","
                << fixed << setprecision(4)
                << result.efficiency << ","
                << result.avg_error << ","
                << elapsed << "\n";
    }

    // Also run FUSION with Adaptive-T
    cout << "    Model: FUSION-Adaptive...";
    auto tm0 = chrono::high_resolution_clock::now();
    auto result_adp = evaluate_model_adaptive(G, exact_BC, FUSION,
                                               T*3, 0.15, 5,
                                               num_pairs, rng);
    auto tm1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(tm1 - tm0).count();
    cout << " Eff=" << fixed << setprecision(2) << result_adp.efficiency
         << "% Err=" << result_adp.avg_error
         << "% avg_T=" << fixed << setprecision(1) << result_adp.avg_T_used
         << " time=" << elapsed << "s\n";

    out_csv << name << "," << G.n << "," << G.m << ","
            << "FUSION_Adaptive," << (int)result_adp.avg_T_used << ","
            << fixed << setprecision(4)
            << result_adp.efficiency << ","
            << result_adp.avg_error << ","
            << elapsed << "\n";

    out_csv.flush();
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    mt19937 rng(42);
    int T = 25;
    int num_pairs = 500;

    string data_dir = "../data/";
    string results_dir = "../results/";
    fs::create_directories(results_dir);

    ofstream out_csv(results_dir + "benchmark_results.csv");
    out_csv << "graph,n,m,model,T,efficiency,avg_error,time_sec\n";

    cout << "=== BOLT Improved: Benchmark ===\n\n";

    // ---- SYNTHETIC GRAPHS ----
    cout << "--- Synthetic Graphs ---\n";

    vector<tuple<string,int,double>> er_configs = {
        {"ER_1k_3", 1000, 0.01},
        {"ER_1k_4", 1000, 0.00562},
        {"ER_500_3", 500, 0.014},
    };
    for (auto [name, n, p] : er_configs) {
        cout << "\n[" << name << "]\n";
        auto G = gen_ER(n, p, rng);
        cout << "  Generated: n=" << G.n << " m=" << G.m
             << " avg_deg=" << fixed << setprecision(2) << G.avg_deg << "\n";
        run_benchmark(name, G, T, num_pairs, rng, out_csv);
    }

    vector<tuple<string,int,int>> ba_configs = {
        {"BA_1k_3", 1000, 5},
        {"BA_1k_4", 1000, 3},
        {"BA_500_3", 500, 5},
    };
    for (auto [name, n, k] : ba_configs) {
        cout << "\n[" << name << "]\n";
        auto G = gen_BA(n, k, rng);
        cout << "  Generated: n=" << G.n << " m=" << G.m
             << " avg_deg=" << fixed << setprecision(2) << G.avg_deg << "\n";
        run_benchmark(name, G, T, num_pairs, rng, out_csv);
    }

    // ---- REAL-WORLD GRAPHS (if available) ----
    cout << "\n--- Real-World Graphs ---\n";
    vector<pair<string,string>> rw_graphs = {
        {"facebook", data_dir + "facebook_combined.txt"},
        {"as20000102", data_dir + "as20000102.txt"},
        {"CA-GrQc",  data_dir + "CA-GrQc.txt"},
        {"p2p-Gnutella04", data_dir + "p2p-Gnutella04.txt"},
    };
    for (auto [name, path] : rw_graphs) {
        if (!fs::exists(path)) {
            cout << "  [" << name << "] not found at " << path << ", skipping\n";
            continue;
        }
        cout << "\n[" << name << "]\n";
        auto G = load_graph(path);
        cout << "  Loaded: n=" << G.n << " m=" << G.m
             << " avg_deg=" << fixed << setprecision(2) << G.avg_deg << "\n";
        // For large graphs, limit exact BC to subgraph or limit pairs
        if (G.n > 5000) {
            cout << "  Graph large — using T=25, 200 pairs\n";
            run_benchmark(name, G, T, 200, rng, out_csv);
        } else {
            run_benchmark(name, G, T, num_pairs, rng, out_csv);
        }
    }

    out_csv.close();
    cout << "\n=== Done. Results in " << results_dir << "benchmark_results.csv ===\n";
    return 0;
}