#include <bits/stdc++.h>
using namespace std;

struct Graph {
    int n;
    vector<vector<int>> adj;
    vector<int> deg;
    Graph(int n) : n(n), adj(n), deg(n,0) {}
    void addEdge(int u, int v) { adj[u].push_back(v); adj[v].push_back(u); deg[u]++; deg[v]++; }
    double avgDeg() const {
        double s = 0; for (int d : deg) s += d;
        return n>0 ? s/n : 1.0;
    }
};

Graph loadGraph(const string& path, int maxNodes) {
    ifstream f(path);
    vector<pair<int,int>> rawEdges;
    unordered_map<int,int> nodeIdMap;
    string line;
    int nextId = 0;

    auto getId = [&](int raw) -> int {
        auto it = nodeIdMap.find(raw);
        if (it != nodeIdMap.end()) return it->second;
        int id = nextId++;
        nodeIdMap[raw] = id;
        return id;
    };

    while (getline(f, line)) {
        if (line.empty() || line[0]=='#') continue;
        stringstream ss(line);
        int u, v;
        if (!(ss >> u >> v) || u==v) continue;
        int ui = getId(u), vi = getId(v);
        rawEdges.push_back({ui, vi});
    }

    int totalNodes = nextId;
    vector<vector<int>> fullAdj(totalNodes);
    vector<int> fullDeg(totalNodes, 0);
    set<pair<int,int>> seenE;
    for (auto& e : rawEdges) {
        int a = e.first, b = e.second;
        if (a==b || a>=totalNodes || b>=totalNodes) continue;
        if (seenE.count({min(a,b),max(a,b)})) continue;
        seenE.insert({min(a,b),max(a,b)});
        fullAdj[a].push_back(b);
        fullAdj[b].push_back(a);
        fullDeg[a]++; fullDeg[b]++;
    }

    int seed = (int)(max_element(fullDeg.begin(), fullDeg.end()) - fullDeg.begin());
    vector<int> bfsOrder;
    bfsOrder.reserve(maxNodes);
    vector<bool> visited(totalNodes, false);
    queue<int> q;
    q.push(seed); visited[seed] = true;
    while (!q.empty() && (int)bfsOrder.size() < maxNodes) {
        int u = q.front(); q.pop();
        bfsOrder.push_back(u);
        for (int w : fullAdj[u]) {
            if (!visited[w]) { visited[w] = true; q.push(w); }
        }
    }

    int n = bfsOrder.size();
    unordered_map<int,int> remap;
    for (int i = 0; i < n; i++) remap[bfsOrder[i]] = i;

    Graph G(n);
    set<pair<int,int>> added;
    for (auto& e : rawEdges) {
        if (!remap.count(e.first) || !remap.count(e.second)) continue;
        int a = remap[e.first], b = remap[e.second];
        if (a==b) continue;
        if (added.count({min(a,b),max(a,b)})) continue;
        added.insert({min(a,b),max(a,b)});
        G.addEdge(a, b);
    }
    return G;
}

vector<double> brandesDep(const Graph& G, int s) {
    int n = G.n;
    vector<double> sigma(n,0), delta(n,0);
    vector<int> dist(n,-1);
    vector<vector<int>> pred(n);
    stack<int> S;
    queue<int> Q;
    dist[s]=0; sigma[s]=1; Q.push(s);
    while (!Q.empty()) {
        int v=Q.front(); Q.pop(); S.push(v);
        for (int w : G.adj[v]) {
            if (dist[w]<0) { dist[w]=dist[v]+1; Q.push(w); }
            if (dist[w]==dist[v]+1) { sigma[w]+=sigma[v]; pred[w].push_back(v); }
        }
    }
    while (!S.empty()) {
        int w=S.top(); S.pop();
        for (int p : pred[w])
            delta[p]+=(sigma[p]/max(sigma[w],1e-10))*(1.0+delta[w]);
    }
    return delta;
}

vector<double> exactBC(const Graph& G) {
    int n=G.n;
    vector<double> bc(n,0);
    for (int s=0; s<n; s++) {
        auto d=brandesDep(G,s);
        for (int v=0; v<n; v++) if (v!=s) bc[v]+=d[v];
    }
    double norm=(n>2)?(double)(n-1)*(n-2):1.0;
    for (auto& b:bc) b/=norm;
    return bc;
}

vector<double> genProb(const Graph& G, int v) {
    int n=G.n;
    double lam=max(G.avgDeg(),1.0);
    vector<int> dist(n,-1);
    queue<int> Q;
    dist[v]=0; Q.push(v);
    while (!Q.empty()) {
        int u=Q.front(); Q.pop();
        for (int w:G.adj[u]) if (dist[w]<0){dist[w]=dist[u]+1; Q.push(w);}
    }
    vector<double> P(n,0);
    double tot=0;
    for (int i=0;i<n;i++){
        if (i==v||dist[i]<0) continue;
        double w=pow(lam,-(double)dist[i])/max(1,G.deg[i]);
        P[i]=w; tot+=w;
    }
    if (tot>0) for (auto& p:P) p/=tot;
    return P;
}

int sampleFromProb(const vector<double>& P, mt19937& rng) {
    vector<pair<double,int>> cdf;
    double s=0;
    for (int i=0;i<(int)P.size();i++) if (P[i]>0){s+=P[i]; cdf.push_back({s,i});}
    if (cdf.empty()) return 0;
    uniform_real_distribution<double> ud(0,1);
    double r=ud(rng);
    for (auto& c:cdf) if (r<=c.first) return c.second;
    return cdf.back().second;
}

double classicEst(const Graph& G, int v, int T, mt19937& rng) {
    auto P=genProb(G,v);
    double est=0;
    for (int t=0;t<T;t++){
        int s=sampleFromProb(P,rng);
        auto d=brandesDep(G,s);
        if (P[s]>0) est+=d[v]/P[s];
    }
    return est/T;
}

vector<double> improvedEstK(const Graph& G, const vector<int>& targets, int T, mt19937& rng) {
    int K=targets.size(), n=G.n;
    vector<vector<double>> allP(K);
    for (int i=0;i<K;i++) allP[i]=genProb(G,targets[i]);

    vector<double> pool(n,0);
    for (int i=0;i<K;i++)
        for (int u=0;u<n;u++)
            pool[u]+=allP[i][u]*log(1.0+max(1,G.deg[u]));

    double tot=0; for (double p:pool) tot+=p;
    if (tot<=0) return vector<double>(K,0);
    for (auto& p:pool) p/=tot;

    vector<pair<double,int>> poolCdf;
    double s=0;
    for (int i=0;i<n;i++) if (pool[i]>0){s+=pool[i]; poolCdf.push_back({s,i});}

    uniform_real_distribution<double> ud(0,1);
    vector<double> est(K,0);
    vector<int> cnt(K,0);

    int S = T * (int)ceil(sqrt((double)K));
    for (int t=0;t<S;t++){
        double r=ud(rng);
        int src=poolCdf.back().second;
        for (auto& c:poolCdf) if (r<=c.first){src=c.second; break;}

        auto dep=brandesDep(G,src);
        for (int i=0;i<K;i++){
            double pv=allP[i][src];
            if (pv>0){
                double poolW=pool[src];
                est[i]+=(dep[targets[i]]/pv)*(pv/max(poolW,1e-10));
                cnt[i]++;
            }
        }
    }
    for (int i=0;i<K;i++) if (cnt[i]>0) est[i]/=cnt[i];
    return est;
}

double pairAcc(const vector<int>& tgts, const vector<double>& est, const vector<double>& bc, mt19937& rng) {
    int K=tgts.size();
    vector<pair<int,int>> pairs;
    for (int i=0;i<K;i++) for (int j=i+1;j<K;j++) pairs.push_back({i,j});
    shuffle(pairs.begin(),pairs.end(),rng);
    if ((int)pairs.size()>400) pairs.resize(400);
    int ok=0;
    for (auto& p:pairs){
        bool predA=(est[p.first]>est[p.second]);
        bool trueA=(bc[tgts[p.first]]>bc[tgts[p.second]]);
        if (predA==trueA) ok++;
    }
    return pairs.empty()?0.5:(double)ok/pairs.size();
}

double avgRelErr(const vector<int>& tgts, const vector<double>& est, const vector<double>& bc) {
    double e=0;
    for (int i=0;i<(int)tgts.size();i++) {
        double t=bc[tgts[i]], d=max(t,1e-10);
        e+=abs(t-est[i])/d;
    }
    return e/tgts.size();
}

int main(int argc, char* argv[]) {
    if (argc<5){
        cout<<"Usage: improved_eddbm.exe dataset.txt K T_max output.txt [max_nodes]\n";
        return 1;
    }
    string dsPath=argv[1];
    int K=stoi(argv[2]), T_max=stoi(argv[3]);
    string outFile=argv[4];
    int maxN=(argc>=6)?stoi(argv[5]):1500;

    string dsName=dsPath;
    for (char c:{'/','\\'}){
        auto p=dsName.rfind(c);
        if (p!=string::npos) dsName=dsName.substr(p+1);
    }

    cout<<"\n================================================================\n";
    cout<<" Improved EDDBM (v2 — Fixed)\n";
    cout<<" Dataset: "<<dsName<<"  K="<<K<<"  T_max="<<T_max<<"\n";
    cout<<"================================================================\n";

    cout<<"[1] Loading graph...\n";
    Graph G=loadGraph(dsPath,maxN);
    int n=G.n, m=0;
    for (int v=0;v<n;v++) m+=G.deg[v]; m/=2;
    cout<<"  Graph: "<<n<<" nodes, "<<m<<" edges, avg_deg="<<G.avgDeg()<<"\n";

    if (m<100){
        cout<<"ERROR: Graph too sparse after loading. Check dataset path.\n";
        return 1;
    }

    cout<<"[2] Exact Brandes BC...\n";
    auto t0c=chrono::high_resolution_clock::now();
    vector<double> bc=exactBC(G);
    double bcTime=chrono::duration<double>(chrono::high_resolution_clock::now()-t0c).count();
    cout<<"  Done in "<<fixed<<setprecision(2)<<bcTime<<"s\n";

    vector<pair<double,int>> byBC(n);
    for (int i=0;i<n;i++) byBC[i]={bc[i],i};
    sort(byBC.rbegin(),byBC.rend());
    
    mt19937 rng(42);
    vector<int> targets;
    int nTop=min(K/2+1, n);
    for (int i=0;i<nTop && (int)targets.size()<K;i++) targets.push_back(byBC[i].second);
    int mid=n/4, midEnd=3*n/4;
    vector<int> midPool;
    for (int i=mid;i<midEnd;i++) midPool.push_back(byBC[i].second);
    shuffle(midPool.begin(),midPool.end(),rng);
    for (int v:midPool){ if ((int)targets.size()>=K) break; targets.push_back(v); }

    K=min(K,(int)targets.size());
    cout<<"  Selected "<<K<<" target nodes (top BC + diverse)\n";

    vector<int> T_vals;
    for (int T:{5,10,20,30,50,75,100}) if (T<=T_max) T_vals.push_back(T);
    if (T_vals.empty()) T_vals.push_back(T_max);

    struct Row{ int T; double c_acc,c_err,c_t,i_acc,i_err,i_t; };
    vector<Row> rows;

    int pivot=targets[0];
    vector<pair<int,double>> pvtC, pvtI;

    cout<<"\n[3] T-Sweep (Classic vs Improved EDDBM)...\n";
    cout<<setw(5)<<"T"<<setw(13)<<"C-Acc%"<<setw(13)<<"C-Err%"<<setw(12)<<"C-Time"
        <<setw(13)<<"I-Acc%"<<setw(13)<<"I-Err%"<<setw(12)<<"I-Time"<<setw(9)<<"Speedup\n";
    cout<<string(90,'-')<<"\n";

    for (int T:T_vals){
        Row r; r.T=T;

        auto tc0=chrono::high_resolution_clock::now();
        vector<double> estC(K);
        for (int i=0;i<K;i++) estC[i]=classicEst(G,targets[i],T,rng);
        r.c_t=chrono::duration<double>(chrono::high_resolution_clock::now()-tc0).count();
        r.c_acc=pairAcc(targets,estC,bc,rng)*100;
        r.c_err=avgRelErr(targets,estC,bc)*100;

        auto ti0=chrono::high_resolution_clock::now();
        vector<double> estI=improvedEstK(G,targets,T,rng);
        r.i_t=chrono::duration<double>(chrono::high_resolution_clock::now()-ti0).count();
        r.i_acc=pairAcc(targets,estI,bc,rng)*100;
        r.i_err=avgRelErr(targets,estI,bc)*100;

        double spd=(r.i_t>0)?r.c_t/r.i_t:0;
        cout<<setw(5)<<T
            <<setw(12)<<fixed<<setprecision(2)<<r.c_acc<<"%"
            <<setw(12)<<r.c_err<<"%"
            <<setw(11)<<setprecision(3)<<r.c_t<<"s"
            <<setw(12)<<setprecision(2)<<r.i_acc<<"%"
            <<setw(12)<<r.i_err<<"%"
            <<setw(11)<<setprecision(3)<<r.i_t<<"s"
            <<setw(8)<<setprecision(2)<<spd<<"x\n";

        rows.push_back(r);
        pvtC.push_back({T,estC[0]});
        pvtI.push_back({T,estI[0]});
    }

    int bestT=T_vals[0]; double bestAcc=0;
    for (auto& r:rows) if (r.i_acc>bestAcc){bestAcc=r.i_acc; bestT=r.T;}
    cout<<"\n=> Best T="<<bestT<<" (Improved acc="<<bestAcc<<"%)\n";

    ofstream out(outFile);
    out<<"Dataset="<<dsName<<"\nNodes="<<n<<" Edges="<<m<<" K="<<K<<"\n";
    out<<"ExactBrandesTime="<<bcTime<<"s\n\n";
    out<<"# Exact BC for targets\nNodeID,ExactBC\n";
    for (int i=0;i<K;i++) out<<targets[i]<<","<<fixed<<setprecision(8)<<bc[targets[i]]<<"\n";
    out<<"\n# T-Sweep\nT,ClassicAcc%,ClassicErr%,ClassicTimeSec,ImprovedAcc%,ImprovedErr%,ImprovedTimeSec,Speedup\n";
    for (auto& r:rows){
        double spd=(r.i_t>0)?r.c_t/r.i_t:0;
        out<<r.T<<","<<r.c_acc<<","<<r.c_err<<","<<r.c_t<<","
           <<r.i_acc<<","<<r.i_err<<","<<r.i_t<<","<<spd<<"\n";
    }
    out<<"\n# Prob vs T pivotNode="<<pivot<<" exact="<<bc[pivot]<<"\n";
    out<<"T,ClassicEst,ImprovedEst,Exact\n";
    for (int i=0;i<(int)T_vals.size();i++)
        out<<T_vals[i]<<","<<pvtC[i].second<<","<<pvtI[i].second<<","<<bc[pivot]<<"\n";
    out<<"\nBestT="<<bestT<<" BestImprovedAcc="<<bestAcc<<"%\n";
    out.close();

    cout<<"[OK] Written: "<<outFile<<"\n[OK] Done!\n";
    return 0;

struct Graph {
    int n;
    vector<vector<int>> adj;
    vector<int> deg;
    Graph(int n) : n(n), adj(n), deg(n,0) {}
    void addEdge(int u, int v) { adj[u].push_back(v); adj[v].push_back(u); deg[u]++; deg[v]++; }
    double avgDeg() const {
        double s = 0; for (int d : deg) s += d;
        return n>0 ? s/n : 1.0;
    }
};

// Proper BFS subsampling from highest-degree seed (same as Python version)
Graph loadGraph(const string& path, int maxNodes) {
    ifstream f(path);
    vector<pair<int,int>> rawEdges;
    unordered_map<int,int> nodeIdMap;
    string line;
    int nextId = 0;

    auto getId = [&](int raw) -> int {
        auto it = nodeIdMap.find(raw);
        if (it != nodeIdMap.end()) return it->second;
        int id = nextId++;
        nodeIdMap[raw] = id;
        return id;
    };

    while (getline(f, line)) {
        if (line.empty() || line[0]=='#') continue;
        stringstream ss(line);
        int u, v;
        if (!(ss >> u >> v) || u==v) continue;
        int ui = getId(u), vi = getId(v);
        rawEdges.push_back({ui, vi});
    }

    int totalNodes = nextId;
    // Build full adjacency to find highest degree seed
    vector<vector<int>> fullAdj(totalNodes);
    vector<int> fullDeg(totalNodes, 0);
    set<pair<int,int>> seenE;
    for (auto& e : rawEdges) {
        int a = e.first, b = e.second;
        if (a==b || a>=totalNodes || b>=totalNodes) continue;
        if (seenE.count({min(a,b),max(a,b)})) continue;
        seenE.insert({min(a,b),max(a,b)});
        fullAdj[a].push_back(b);
        fullAdj[b].push_back(a);
        fullDeg[a]++; fullDeg[b]++;
    }

    // BFS from highest-degree node
    int seed = (int)(max_element(fullDeg.begin(), fullDeg.end()) - fullDeg.begin());
    vector<int> bfsOrder;
    bfsOrder.reserve(maxNodes);
    vector<bool> visited(totalNodes, false);
    queue<int> q;
    q.push(seed); visited[seed] = true;
    while (!q.empty() && (int)bfsOrder.size() < maxNodes) {
        int u = q.front(); q.pop();
        bfsOrder.push_back(u);
        for (int w : fullAdj[u]) {
            if (!visited[w]) { visited[w] = true; q.push(w); }
        }
    }

    // Remap BFS order to [0, |bfsOrder|)
    int n = bfsOrder.size();
    unordered_map<int,int> remap;
    for (int i = 0; i < n; i++) remap[bfsOrder[i]] = i;

    Graph G(n);
    set<pair<int,int>> added;
    for (auto& e : rawEdges) {
        if (!remap.count(e.first) || !remap.count(e.second)) continue;
        int a = remap[e.first], b = remap[e.second];
        if (a==b) continue;
        if (added.count({min(a,b),max(a,b)})) continue;
        added.insert({min(a,b),max(a,b)});
        G.addEdge(a, b);
    }
    return G;
}

// Brandes single source — O(m)
vector<double> brandesDep(const Graph& G, int s) {
    int n = G.n;
    vector<double> sigma(n,0), delta(n,0);
    vector<int> dist(n,-1);
    vector<vector<int>> pred(n);
    stack<int> S;
    queue<int> Q;
    dist[s]=0; sigma[s]=1; Q.push(s);
    while (!Q.empty()) {
        int v=Q.front(); Q.pop(); S.push(v);
        for (int w : G.adj[v]) {
            if (dist[w]<0) { dist[w]=dist[v]+1; Q.push(w); }
            if (dist[w]==dist[v]+1) { sigma[w]+=sigma[v]; pred[w].push_back(v); }
        }
    }
    while (!S.empty()) {
        int w=S.top(); S.pop();
        for (int p : pred[w])
            delta[p]+=(sigma[p]/max(sigma[w],1e-10))*(1.0+delta[w]);
    }
    return delta;
}

// Exact Brandes BC — O(n*m)
vector<double> exactBC(const Graph& G) {
    int n=G.n;
    vector<double> bc(n,0);
    for (int s=0; s<n; s++) {
        auto d=brandesDep(G,s);
        for (int v=0; v<n; v++) if (v!=s) bc[v]+=d[v];
    }
    double norm=(n>2)?(double)(n-1)*(n-2):1.0;
    for (auto& b:bc) b/=norm;
    return bc;
}

// EDDBM probability distribution P_v
vector<double> genProb(const Graph& G, int v) {
    int n=G.n;
    double lam=max(G.avgDeg(),1.0);
    vector<int> dist(n,-1);
    queue<int> Q;
    dist[v]=0; Q.push(v);
    while (!Q.empty()) {
        int u=Q.front(); Q.pop();
        for (int w:G.adj[u]) if (dist[w]<0){dist[w]=dist[u]+1; Q.push(w);}
    }
    vector<double> P(n,0);
    double tot=0;
    for (int i=0;i<n;i++){
        if (i==v||dist[i]<0) continue;
        double w=pow(lam,-(double)dist[i])/max(1,G.deg[i]);
        P[i]=w; tot+=w;
    }
    if (tot>0) for (auto& p:P) p/=tot;
    return P;
}

// CDF sampler from probability vector
int sampleFromProb(const vector<double>& P, mt19937& rng) {
    vector<pair<double,int>> cdf;
    double s=0;
    for (int i=0;i<(int)P.size();i++) if (P[i]>0){s+=P[i]; cdf.push_back({s,i});}
    if (cdf.empty()) return 0;
    uniform_real_distribution<double> ud(0,1);
    double r=ud(rng);
    for (auto& c:cdf) if (r<=c.first) return c.second;
    return cdf.back().second;
}

// Classic EDDBM: T BFS per node independently — O(K*T*m)
double classicEst(const Graph& G, int v, int T, mt19937& rng) {
    auto P=genProb(G,v);
    double est=0;
    for (int t=0;t<T;t++){
        int s=sampleFromProb(P,rng);
        auto d=brandesDep(G,s);
        if (P[s]>0) est+=d[v]/P[s];
    }
    return est/T;
}

// Improved EDDBM: S total BFS shared across ALL K nodes — O(S*m)
// S = T * sqrt(K) to give extra budget for pooling accuracy
vector<double> improvedEstK(const Graph& G, const vector<int>& targets, int T, mt19937& rng) {
    int K=targets.size(), n=G.n;
    // Compute P_vi for each target
    vector<vector<double>> allP(K);
    for (int i=0;i<K;i++) allP[i]=genProb(G,targets[i]);

    // Adaptive pooled distribution: mean of P_vi weighted by log(1+deg(u))
    // This concentrates sampling budget near high-degree nodes (better coverage)
    vector<double> pool(n,0);
    for (int i=0;i<K;i++)
        for (int u=0;u<n;u++)
            pool[u]+=allP[i][u]*log(1.0+max(1,G.deg[u]));

    double tot=0; for (double p:pool) tot+=p;
    if (tot<=0) return vector<double>(K,0);
    for (auto& p:pool) p/=tot;

    // Build CDF for pooled distribution
    vector<pair<double,int>> poolCdf;
    double s=0;
    for (int i=0;i<n;i++) if (pool[i]>0){s+=pool[i]; poolCdf.push_back({s,i});}

    uniform_real_distribution<double> ud(0,1);
    vector<double> est(K,0);
    vector<int> cnt(K,0);

    // Use S = T * ceil(sqrt(K)) shared BFS — gives more samples per target 
    // on average than classic EDDBM at same total cost
    int S = T * (int)ceil(sqrt((double)K));
    for (int t=0;t<S;t++){
        double r=ud(rng);
        int src=poolCdf.back().second;
        for (auto& c:poolCdf) if (r<=c.first){src=c.second; break;}

        // One BFS contributes to ALL targets
        auto dep=brandesDep(G,src);
        for (int i=0;i<K;i++){
            double pv=allP[i][src];
            if (pv>0){
                // Importance-weight correct for sampling from pool vs P_vi
                double poolW=pool[src];
                est[i]+=(dep[targets[i]]/pv)*(pv/max(poolW,1e-10));
                cnt[i]++;
            }
        }
    }
    for (int i=0;i<K;i++) if (cnt[i]>0) est[i]/=cnt[i];
    return est;
}

// Pairwise accuracy helper
double pairAcc(const vector<int>& tgts, const vector<double>& est, const vector<double>& bc, mt19937& rng) {
    int K=tgts.size();
    vector<pair<int,int>> pairs;
    for (int i=0;i<K;i++) for (int j=i+1;j<K;j++) pairs.push_back({i,j});
    shuffle(pairs.begin(),pairs.end(),rng);
    if ((int)pairs.size()>400) pairs.resize(400);
    int ok=0;
    for (auto& p:pairs){
        bool predA=(est[p.first]>est[p.second]);
        bool trueA=(bc[tgts[p.first]]>bc[tgts[p.second]]);
        if (predA==trueA) ok++;
    }
    return pairs.empty()?0.5:(double)ok/pairs.size();
}

double avgRelErr(const vector<int>& tgts, const vector<double>& est, const vector<double>& bc) {
    double e=0;
    for (int i=0;i<(int)tgts.size();i++) {
        double t=bc[tgts[i]], d=max(t,1e-10);
        e+=abs(t-est[i])/d;
    }
    return e/tgts.size();
}

int main(int argc, char* argv[]) {
    if (argc<5){
        cout<<"Usage: improved_eddbm.exe dataset.txt K T_max output.txt [max_nodes]\n";
        return 1;
    }
    string dsPath=argv[1];
    int K=stoi(argv[2]), T_max=stoi(argv[3]);
    string outFile=argv[4];
    int maxN=(argc>=6)?stoi(argv[5]):1500;

    string dsName=dsPath;
    for (char c:{'/','\\'}){
        auto p=dsName.rfind(c);
        if (p!=string::npos) dsName=dsName.substr(p+1);
    }

    cout<<"\n================================================================\n";
    cout<<" Improved EDDBM (v2 — Fixed)\n";
    cout<<" Dataset: "<<dsName<<"  K="<<K<<"  T_max="<<T_max<<"\n";
    cout<<"================================================================\n";

    cout<<"[1] Loading graph...\n";
    Graph G=loadGraph(dsPath,maxN);
    int n=G.n, m=0;
    for (int v=0;v<n;v++) m+=G.deg[v]; m/=2;
    cout<<"  Graph: "<<n<<" nodes, "<<m<<" edges, avg_deg="<<G.avgDeg()<<"\n";

    if (m<100){
        cout<<"ERROR: Graph too sparse after loading. Check dataset path.\n";
        return 1;
    }

    cout<<"[2] Exact Brandes BC...\n";
    auto t0c=chrono::high_resolution_clock::now();
    vector<double> bc=exactBC(G);
    double bcTime=chrono::duration<double>(chrono::high_resolution_clock::now()-t0c).count();
    cout<<"  Done in "<<fixed<<setprecision(2)<<bcTime<<"s\n";

    // Select K nodes with DIVERSE BC values — pick top and medium BC nodes
    vector<pair<double,int>> byBC(n);
    for (int i=0;i<n;i++) byBC[i]={bc[i],i};
    sort(byBC.rbegin(),byBC.rend());
    
    mt19937 rng(42);
    vector<int> targets;
    // Take top-K/2 by BC and random from rest for comparison spread
    int nTop=min(K/2+1, n);
    for (int i=0;i<nTop && (int)targets.size()<K;i++) targets.push_back(byBC[i].second);
    // Fill rest randomly from middle BC range
    int mid=n/4, midEnd=3*n/4;
    vector<int> midPool;
    for (int i=mid;i<midEnd;i++) midPool.push_back(byBC[i].second);
    shuffle(midPool.begin(),midPool.end(),rng);
    for (int v:midPool){ if ((int)targets.size()>=K) break; targets.push_back(v); }

    K=min(K,(int)targets.size());
    cout<<"  Selected "<<K<<" target nodes (top BC + diverse)\n";

    // T-Sweep
    vector<int> T_vals;
    for (int T:{5,10,20,30,50,75,100}) if (T<=T_max) T_vals.push_back(T);
    if (T_vals.empty()) T_vals.push_back(T_max);

    struct Row{ int T; double c_acc,c_err,c_t,i_acc,i_err,i_t; };
    vector<Row> rows;

    int pivot=targets[0]; // highest BC node for prob-vs-samples tracking
    vector<pair<int,double>> pvtC, pvtI;

    cout<<"\n[3] T-Sweep (Classic vs Improved EDDBM)...\n";
    cout<<setw(5)<<"T"<<setw(13)<<"C-Acc%"<<setw(13)<<"C-Err%"<<setw(12)<<"C-Time"
        <<setw(13)<<"I-Acc%"<<setw(13)<<"I-Err%"<<setw(12)<<"I-Time"<<setw(9)<<"Speedup\n";
    cout<<string(90,'-')<<"\n";

    for (int T:T_vals){
        Row r; r.T=T;

        // Classic
        auto tc0=chrono::high_resolution_clock::now();
        vector<double> estC(K);
        for (int i=0;i<K;i++) estC[i]=classicEst(G,targets[i],T,rng);
        r.c_t=chrono::duration<double>(chrono::high_resolution_clock::now()-tc0).count();
        r.c_acc=pairAcc(targets,estC,bc,rng)*100;
        r.c_err=avgRelErr(targets,estC,bc)*100;

        // Improved
        auto ti0=chrono::high_resolution_clock::now();
        vector<double> estI=improvedEstK(G,targets,T,rng);
        r.i_t=chrono::duration<double>(chrono::high_resolution_clock::now()-ti0).count();
        r.i_acc=pairAcc(targets,estI,bc,rng)*100;
        r.i_err=avgRelErr(targets,estI,bc)*100;

        double spd=(r.i_t>0)?r.c_t/r.i_t:0;
        cout<<setw(5)<<T
            <<setw(12)<<fixed<<setprecision(2)<<r.c_acc<<"%"
            <<setw(12)<<r.c_err<<"%"
            <<setw(11)<<setprecision(3)<<r.c_t<<"s"
            <<setw(12)<<setprecision(2)<<r.i_acc<<"%"
            <<setw(12)<<r.i_err<<"%"
            <<setw(11)<<setprecision(3)<<r.i_t<<"s"
            <<setw(8)<<setprecision(2)<<spd<<"x\n";

        rows.push_back(r);
        pvtC.push_back({T,estC[0]});
        pvtI.push_back({T,estI[0]});
    }

    // Best T
    int bestT=T_vals[0]; double bestAcc=0;
    for (auto& r:rows) if (r.i_acc>bestAcc){bestAcc=r.i_acc; bestT=r.T;}
    cout<<"\n=> Best T="<<bestT<<" (Improved acc="<<bestAcc<<"%)\n";

    // Write output
    ofstream out(outFile);
    out<<"Dataset="<<dsName<<"\nNodes="<<n<<" Edges="<<m<<" K="<<K<<"\n";
    out<<"ExactBrandesTime="<<bcTime<<"s\n\n";
    out<<"# Exact BC for targets\nNodeID,ExactBC\n";
    for (int i=0;i<K;i++) out<<targets[i]<<","<<fixed<<setprecision(8)<<bc[targets[i]]<<"\n";
    out<<"\n# T-Sweep\nT,ClassicAcc%,ClassicErr%,ClassicTimeSec,ImprovedAcc%,ImprovedErr%,ImprovedTimeSec,Speedup\n";
    for (auto& r:rows){
        double spd=(r.i_t>0)?r.c_t/r.i_t:0;
        out<<r.T<<","<<r.c_acc<<","<<r.c_err<<","<<r.c_t<<","
           <<r.i_acc<<","<<r.i_err<<","<<r.i_t<<","<<spd<<"\n";
    }
    out<<"\n# Prob vs T pivotNode="<<pivot<<" exact="<<bc[pivot]<<"\n";
    out<<"T,ClassicEst,ImprovedEst,Exact\n";
    for (int i=0;i<(int)T_vals.size();i++)
        out<<T_vals[i]<<","<<pvtC[i].second<<","<<pvtI[i].second<<","<<bc[pivot]<<"\n";
    out<<"\nBestT="<<bestT<<" BestImprovedAcc="<<bestAcc<<"%\n";
    out.close();

    cout<<"[OK] Written: "<<outFile<<"\n[OK] Done!\n";
    return 0;
}
