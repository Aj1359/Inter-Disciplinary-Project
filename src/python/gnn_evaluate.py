#!/usr/bin/env python3
"""
Final evaluation: GNN vs BOLT on all real-world datasets.
Uses C++ compute_bc for fast exact BC computation.
"""
import argparse, glob, math, os, random, subprocess, time
import numpy as np
import torch
from gnn_model import DrBCModel, NODE_FEAT_DIM


def load_graph(path):
    id_map, id_list, edges = {}, [], set()
    def get_id(x):
        if x in id_map: return id_map[x]
        idx = len(id_list); id_map[x] = idx; id_list.append(x); return idx
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 2: continue
            try: u, v = int(parts[0]), int(parts[1])
            except: continue
            if u == v: continue
            a, b = get_id(u), get_id(v)
            if a > b: a, b = b, a
            edges.add((a, b))
    n = len(id_list)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    deg = [len(adj[i]) for i in range(n)]
    return adj, deg, n, len(edges)


def compute_features(adj, deg, n):
    avg_deg = sum(deg) / max(1, n)
    max_deg = max(deg) if deg else 1
    features = []
    for v in range(n):
        d = deg[v]
        nb = [deg[u] for u in adj[v]] if adj[v] else [0]
        nb_avg = sum(nb) / max(1, len(nb))
        nb_max = max(nb) if nb else 0
        clust = 0.0
        if d >= 2:
            ns = set(adj[v])
            tri = sum(1 for u in adj[v] for w in adj[u] if w in ns and w != v)
            clust = tri / (d * (d - 1))
        h1 = set(adj[v])
        r2 = len(h1) + sum(1 for u in h1 for w in adj[u] if w != v and w not in h1)
        features.append([
            math.log1p(d), d/max(1,n-1), math.log1p(nb_avg),
            nb_max/max(1,max_deg), clust, math.log1p(r2),
            r2/max(1,n), d/max(1.0,avg_deg),
        ])
    return torch.tensor(features, dtype=torch.float32)


def get_exact_bc(graph_file, bc_cache_dir="bc_cache"):
    """Get exact BC using C++ compute_bc (with caching)."""
    os.makedirs(bc_cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(graph_file))[0]
    csv_path = os.path.join(bc_cache_dir, f"{base}_bc.csv")

    compute_bin = os.path.join(os.path.dirname(os.path.abspath(graph_file)), "compute_bc")

    if not os.path.exists(csv_path):
        if not os.path.exists(compute_bin):
            return None, None
        print(f"    Computing exact BC via C++...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [compute_bin, graph_file, csv_path],
                capture_output=True, text=True, timeout=600
            )
            info = result.stdout.strip()
            print(f"done ({info})")
        except subprocess.TimeoutExpired:
            print("timeout!")
            return None, None
        except Exception as e:
            print(f"error: {e}")
            return None, None

    # Read BC values
    bc = {}
    with open(csv_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                bc[int(parts[0])] = float(parts[1])
    return bc, csv_path


def eval_gnn(model, adj, deg, n, bc_dict, rng, num_pairs=500):
    """Evaluate GNN ordering efficiency."""
    x = compute_features(adj, deg, n)
    t0 = time.time()
    with torch.no_grad():
        scores = model(x, adj, deg).cpu().numpy()
    infer_time = time.time() - t0

    nonzero = [i for i in range(n) if bc_dict.get(i, 0) > 0]
    if len(nonzero) < 10:
        return 0, 0, infer_time

    correct = total = 0
    for _ in range(num_pairs):
        u, v = rng.choice(nonzero), rng.choice(nonzero)
        if u == v or bc_dict[u] == bc_dict[v]: continue
        if (scores[u] > scores[v]) == (bc_dict[u] > bc_dict[v]):
            correct += 1
        total += 1

    eff = correct / max(1, total)

    # Spearman
    from scipy.stats import spearmanr
    bc_nz = [bc_dict[i] for i in nonzero]
    sc_nz = [scores[i] for i in nonzero]
    rho, _ = spearmanr(sc_nz, bc_nz)
    rho = 0.0 if math.isnan(rho) else rho

    return eff, rho, infer_time


def eval_bolt(graph_file, T=25, trials=500):
    """Run BOLT and parse efficiency at T=25."""
    bolt_bin = os.path.join(os.path.dirname(os.path.abspath(graph_file)), "bolt")
    if not os.path.exists(bolt_bin):
        return None, None

    try:
        result = subprocess.run(
            [bolt_bin, graph_file, str(T), str(trials), "1"],
            capture_output=True, text=True, timeout=600
        )
        mode = None
        eff_at_T = err_at_T = bolt_time = None
        for line in result.stdout.splitlines():
            if "Efficiency vs T" in line: mode = "eff"; continue
            if "Average Error vs T" in line: mode = "err"; continue
            if "Time per estimateBOLT" in line:
                try: bolt_time = float(line.split(":")[-1].strip().replace("ms","").strip())
                except: pass
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    t_val, val = int(parts[0]), float(parts[1])
                    if mode == "eff": eff_at_T = val
                    elif mode == "err": err_at_T = val
                except: pass

        return eff_at_T, err_at_T, bolt_time
    except:
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--gnn-model", default="gnn_model.pt")
    parser.add_argument("--max-nodes", type=int, default=40000)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="final_comparison.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load GNN
    ckpt = torch.load(args.gnn_model, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = DrBCModel(cfg["in_dim"], cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"GNN model loaded (val_acc={ckpt.get('val_acc',0)*100:.2f}%)\n")

    txt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))
    results = []

    with open(args.out, "w") as f:
        f.write("dataset,n,m,model,efficiency_pct,rank_corr,time_ms\n")

        for path in txt_files:
            bn = os.path.basename(path)
            if "extracted_text" in bn or "synthetic" in bn.lower():
                continue

            adj, deg, n, m = load_graph(path)
            if n > args.max_nodes or n < 50:
                continue

            print(f"\n{'='*60}")
            print(f"{bn} — n={n}, m={m}")
            print(f"{'='*60}")

            # Get exact BC (C++)
            bc_dict, _ = get_exact_bc(path)
            if bc_dict is None:
                print("  Skipped (could not compute BC)")
                continue

            # GNN
            print(f"  GNN...", end=" ")
            gnn_eff, gnn_rho, gnn_t = eval_gnn(model, adj, deg, n, bc_dict, rng, args.trials)
            print(f"eff={gnn_eff*100:.2f}% rho={gnn_rho:.4f} time={gnn_t*1000:.1f}ms")
            f.write(f"{bn},{n},{m},GNN-DrBC,{gnn_eff*100:.2f},{gnn_rho:.4f},{gnn_t*1000:.1f}\n")
            results.append(("GNN-DrBC", bn, n, gnn_eff, gnn_rho, gnn_t*1000))

            # BOLT
            if n <= 15000:
                print(f"  BOLT...", end=" ")
                bolt_eff, bolt_err, bolt_t = eval_bolt(path, 25, args.trials)
                if bolt_eff is not None:
                    print(f"eff={bolt_eff:.2f}% err={bolt_err:.2f}% time={bolt_t:.1f}ms")
                    f.write(f"{bn},{n},{m},BOLT-T25,{bolt_eff:.2f},,{bolt_t:.1f}\n")
                    results.append(("BOLT-T25", bn, n, bolt_eff/100, 0, bolt_t))
                else:
                    print("skipped")
            f.flush()

    # Summary
    print(f"\n\n{'='*75}")
    print(f"{'FINAL COMPARISON':^75}")
    print(f"{'='*75}")
    print(f"{'Dataset':<24} {'Model':<12} {'Eff%':>8} {'ρ':>7} {'Time(ms)':>10}")
    print(f"{'-'*62}")
    for m_name, ds, n, eff, rho, t in sorted(results, key=lambda x: (x[1], x[0])):
        rho_s = f"{rho:.4f}" if rho else ""
        print(f"{ds:<24} {m_name:<12} {eff*100:>7.2f} {rho_s:>7} {t:>9.1f}")

    print(f"\n{'Model':<12} {'Avg Eff%':>10} {'Avg Time(ms)':>14} {'Count':>6}")
    print(f"{'-'*44}")
    for m_name in sorted(set(r[0] for r in results)):
        mr = [r for r in results if r[0] == m_name]
        print(f"{m_name:<12} {np.mean([r[3] for r in mr])*100:>9.2f} "
              f"{np.mean([r[5] for r in mr]):>13.1f} {len(mr):>6}")

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
