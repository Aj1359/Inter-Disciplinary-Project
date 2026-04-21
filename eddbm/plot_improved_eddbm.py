import os
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"

def parse_output_file(path):
    result = {"path": path, "t_sweep": [], "prob_vs_t": []}
    
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    mode = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            mode = None if line == "" else mode
            continue
        
        if line.startswith("Dataset="):
            result["dataset"] = line.split("=")[1]
        elif line.startswith("Nodes="):
            parts = line.split()
            result["nodes"] = int(parts[0].split("=")[1])
            result["edges"] = int(parts[1].split("=")[1])
        elif "T,ClassicAcc%" in line:
            mode = "t_sweep"
        elif "T,ClassicEst" in line:
            mode = "prob_vs_t"
        elif mode == "t_sweep":
            try:
                parts = line.split(",")
                if len(parts) >= 7:
                    result["t_sweep"].append({
                        "T": int(parts[0]),
                        "c_acc": float(parts[1]),
                        "c_err": float(parts[2]),
                        "c_time": float(parts[3]),
                        "i_acc": float(parts[4]),
                        "i_err": float(parts[5]),
                        "i_time": float(parts[6]),
                    })
            except (ValueError, IndexError):
                pass
        elif mode == "prob_vs_t":
            try:
                parts = line.split(",")
                if len(parts) >= 4:
                    result["prob_vs_t"].append({
                        "T": int(parts[0]),
                        "classic": float(parts[1]),
                        "improved": float(parts[2]),
                        "exact": float(parts[3]),
                    })
            except (ValueError, IndexError):
                pass
    
    return result

def resample_data(x, y, num_points=100):
    if len(x) < 2:
        return x, y
    f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
    x_new = np.linspace(min(x), max(x), num_points)
    y_new = f(x_new)
    return x_new, y_new

def plot_prob_vs_samples(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    n_ds = len(all_results)
    cols = min(3, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_ds == 1:
        axes = [axes]
    else:
        axes = axes.flat
    
    for i, res in enumerate(all_results):
        ax = axes[i] if n_ds > 1 else axes
        pvt = res.get("prob_vs_t", [])
        if not pvt:
            ax.axis('off')
            continue
        T_vals    = [r["T"] for r in pvt]
        classic   = [r["classic"] for r in pvt]
        improved  = [r["improved"] for r in pvt]
        exact     = pvt[0]["exact"]
        
        ax.axhline(exact, color="#2ecc71", linewidth=2.5, linestyle="-",
                   label=f"Optimal (Exact BC={exact:.4f})")
        
        T_c, C_c = resample_data(T_vals, classic)
        T_i, C_i = resample_data(T_vals, improved)
        
        ax.plot(T_c, C_c, color="#e74c3c", linewidth=2, label="Classic EDDBM")
        ax.plot(T_i, C_i, color="#2196F3", linewidth=2, label="Improved EDDBM")
        ax.scatter(T_vals, classic, color="#e74c3c", marker='o', s=40)
        ax.scatter(T_vals, improved, color="#2196F3", marker='s', s=40)
        
        ax.set_xlabel("Samples T", fontsize=10)
        ax.set_ylabel("Estimated BC", fontsize=10)
        ax.set_title(res.get("dataset", ""), fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_ds, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle("Estimated BC vs Samples T\n(Optimal vs Classic vs Improved EDDBM)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "prob_vs_samples.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")

def plot_accuracy_vs_t(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    colors_c = plt.cm.Reds(np.linspace(0.5, 0.9, len(all_results)))
    colors_i = plt.cm.Blues(np.linspace(0.5, 0.9, len(all_results)))
    
    for (res, cc, ci) in zip(all_results, colors_c, colors_i):
        ts = res.get("t_sweep", [])
        if not ts:
            continue
        T_vals = [r["T"] for r in ts]
        c_acc  = [r["c_acc"] for r in ts]
        i_acc  = [r["i_acc"] for r in ts]
        ds     = res.get("dataset", "")
        
        T_c, A_c = resample_data(T_vals, c_acc)
        T_i, A_i = resample_data(T_vals, i_acc)
        
        ax.plot(T_c, A_c, color=cc, linewidth=1.8, label=f"{ds} Classic")
        ax.plot(T_i, A_i, color=ci, linewidth=1.8, label=f"{ds} Improved")
        ax.scatter(T_vals, c_acc, color=cc, marker='o', s=30)
        ax.scatter(T_vals, i_acc, color=ci, marker='s', s=30)
    
    ax.axhline(100, color="black", linestyle=":", alpha=0.5, label="Optimal (100%)")
    ax.axhline(50, color="gray", linestyle=":", alpha=0.3, label="Random (50%)")
    ax.set_xlabel("Samples T", fontsize=10)
    ax.set_ylabel("Pairwise Ordering Accuracy (%)", fontsize=10)
    ax.set_title("Ordering Accuracy vs T — Classic vs Improved EDDBM",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_vs_t.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")

def plot_error_vs_t(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    colors_c = plt.cm.Reds(np.linspace(0.5, 0.9, len(all_results)))
    colors_i = plt.cm.Blues(np.linspace(0.5, 0.9, len(all_results)))
    
    for (res, cc, ci) in zip(all_results, colors_c, colors_i):
        ts = res.get("t_sweep", [])
        if not ts:
            continue
        T_vals = [r["T"] for r in ts]
        c_err  = [r["c_err"] for r in ts]
        i_err  = [r["i_err"] for r in ts]
        ds     = res.get("dataset", "")
        
        T_c, E_c = resample_data(T_vals, c_err)
        T_i, E_i = resample_data(T_vals, i_err)
        
        ax.plot(T_c, E_c, color=cc, linewidth=1.8, label=f"{ds} Classic")
        ax.plot(T_i, E_i, color=ci, linewidth=1.8, label=f"{ds} Improved")
        ax.scatter(T_vals, c_err, color=cc, marker='o', s=30)
        ax.scatter(T_vals, i_err, color=ci, marker='s', s=30)
    
    ax.set_xlabel("Samples T", fontsize=10)
    ax.set_ylabel("Average Relative Error (%)", fontsize=10)
    ax.set_title("Estimation Error vs T — Classic vs Improved EDDBM",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "error_vs_t.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")

def plot_time_vs_t(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    colors_c = plt.cm.Reds(np.linspace(0.5, 0.9, len(all_results)))
    colors_i = plt.cm.Blues(np.linspace(0.5, 0.9, len(all_results)))
    
    for (res, cc, ci) in zip(all_results, colors_c, colors_i):
        ts = res.get("t_sweep", [])
        if not ts:
            continue
        T_vals = [r["T"] for r in ts]
        c_time = [r["c_time"] for r in ts]
        i_time = [r["i_time"] for r in ts]
        ds     = res.get("dataset", "")
        
        T_c, Time_c = resample_data(T_vals, c_time)
        T_i, Time_i = resample_data(T_vals, i_time)
        
        ax.plot(T_c, Time_c, color=cc, linewidth=1.8, label=f"{ds} Classic")
        ax.plot(T_i, Time_i, color=ci, linewidth=1.8, label=f"{ds} Improved")
        ax.scatter(T_vals, c_time, color=cc, marker='o', s=30)
        ax.scatter(T_vals, i_time, color=ci, marker='s', s=30)
    
    ax.set_xlabel("Samples T", fontsize=10)
    ax.set_ylabel("Runtime (seconds)", fontsize=10)
    ax.set_title("Runtime vs T — Classic vs Improved EDDBM",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "time_vs_t.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"ERROR: {RESULTS_DIR} directory not found.")
        return
    
    txt_files = glob.glob(os.path.join(RESULTS_DIR, "*.txt"))
    if not txt_files:
        print(f"No .txt files in {RESULTS_DIR}")
        return
    
    all_results = []
    for f in txt_files:
        try:
            res = parse_output_file(f)
            if res.get("t_sweep") or res.get("prob_vs_t"):
                all_results.append(res)
        except Exception as e:
            print(f"Error parsing {f}: {e}")
    
    if not all_results:
        print("No valid results found.")
        return
    
    print(f"Parsed {len(all_results)} result files.")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("Generating plots...")
    plot_prob_vs_samples(all_results, PLOTS_DIR)
    plot_accuracy_vs_t(all_results, PLOTS_DIR)
    plot_error_vs_t(all_results, PLOTS_DIR)
    plot_time_vs_t(all_results, PLOTS_DIR)
    
    print("Done!")

if __name__ == "__main__":
    main()


def parse_output_file(path):
    """Parse the structured output file from improved_eddbm.cpp."""
    result = {"path": path, "t_sweep": [], "prob_vs_t": []}
    
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    mode = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            mode = None if line == "" else mode
            continue
        
        if line.startswith("Dataset="):
            result["dataset"] = line.split("=")[1]
        elif line.startswith("Nodes="):
            parts = line.split()
            result["nodes"] = int(parts[0].split("=")[1])
            result["edges"] = int(parts[1].split("=")[1])
        elif "T,ClassicAcc%" in line:
            mode = "t_sweep"
        elif "T,ClassicEst" in line:
            mode = "prob_vs_t"
        elif mode == "t_sweep":
            try:
                parts = line.split(",")
                if len(parts) >= 7:
                    result["t_sweep"].append({
                        "T": int(parts[0]),
                        "c_acc": float(parts[1]),
                        "c_err": float(parts[2]),
                        "c_time": float(parts[3]),
                        "i_acc": float(parts[4]),
                        "i_err": float(parts[5]),
                        "i_time": float(parts[6]),
                    })
            except (ValueError, IndexError):
                pass
        elif mode == "prob_vs_t":
            try:
                parts = line.split(",")
                if len(parts) >= 4:
                    result["prob_vs_t"].append({
                        "T": int(parts[0]),
                        "classic": float(parts[1]),
                        "improved": float(parts[2]),
                        "exact": float(parts[3]),
                    })
            except (ValueError, IndexError):
                pass
    
    return result


def plot_prob_vs_samples(all_results, out_dir):
    """Plot 1: Probability estimate vs T samples [Fig 1 analogue]"""
    os.makedirs(out_dir, exist_ok=True)
    
    n_ds = len(all_results)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]
    
    for ax, res in zip(axes, all_results):
        pvt = res.get("prob_vs_t", [])
        if not pvt:
            continue
        T_vals    = [r["T"] for r in pvt]
        classic   = [r["classic"] for r in pvt]
        improved  = [r["improved"] for r in pvt]
        exact     = pvt[0]["exact"]  # same for all T
        
        ax.axhline(exact, color="#2ecc71", linewidth=2.5, linestyle="-",
                   label=f"Optimal (Exact BC={exact:.4f})")
        ax.plot(T_vals, classic, "o--", color="#e74c3c", linewidth=2,
                label="Classic EDDBM")
        ax.plot(T_vals, improved, "s-",  color="#2196F3", linewidth=2,
                label="Improved EDDBM")
        
        ax.set_xlabel("Samples T", fontsize=11)
        ax.set_ylabel("Estimated BC", fontsize=11)
        ax.set_title(res.get("dataset", ""), fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    fig.suptitle("Estimated BC vs Samples T\n(Optimal vs Classic vs Improved EDDBM)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "prob_vs_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")


def plot_accuracy_vs_t(all_results, out_dir):
    """Plot 2: Pairwise Ordering Accuracy vs T"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    
    colors_c = plt.cm.Reds(np.linspace(0.5, 0.9, len(all_results)))
    colors_i = plt.cm.Blues(np.linspace(0.5, 0.9, len(all_results)))
    
    for (res, cc, ci) in zip(all_results, colors_c, colors_i):
        ts = res.get("t_sweep", [])
        if not ts:
            continue
        T_vals = [r["T"] for r in ts]
        c_acc  = [r["c_acc"] for r in ts]
        i_acc  = [r["i_acc"] for r in ts]
        ds     = res.get("dataset", "")
        
        ax.plot(T_vals, c_acc, "o--", color=cc, linewidth=1.8, label=f"{ds} Classic")
        ax.plot(T_vals, i_acc, "s-",  color=ci, linewidth=1.8, label=f"{ds} Improved")
    
    ax.axhline(100, color="black", linestyle=":", alpha=0.5, label="Optimal (100%)")
    ax.axhline(50, color="gray", linestyle=":", alpha=0.3, label="Random (50%)")
    ax.set_xlabel("Samples T", fontsize=11)
    ax.set_ylabel("Pairwise Ordering Accuracy (%)", fontsize=11)
    ax.set_title("Ordering Accuracy vs T — Classic vs Improved EDDBM",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_vs_t.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")


def plot_error_vs_t(all_results, out_dir):
    """Plot 3: Average Relative Error vs T [Fig 2 analogue]"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    
    colors_c = plt.cm.Reds(np.linspace(0.5, 0.9, len(all_results)))
    colors_i = plt.cm.Blues(np.linspace(0.5, 0.9, len(all_results)))
    
    for (res, cc, ci) in zip(all_results, colors_c, colors_i):
        ts = res.get("t_sweep", [])
        if not ts:
            continue
        T_vals = [r["T"] for r in ts]
        c_err  = [r["c_err"] for r in ts]
        i_err  = [r["i_err"] for r in ts]
        ds     = res.get("dataset", "")
        
        ax.plot(T_vals, c_err, "o--", color=cc, linewidth=1.8, label=f"{ds} Classic")
        ax.plot(T_vals, i_err, "s-",  color=ci, linewidth=1.8, label=f"{ds} Improved")
    
    ax.set_xlabel("Samples T", fontsize=11)
    ax.set_ylabel("Average Relative Error (%)", fontsize=11)
    ax.set_title("Estimation Error vs T — Classic vs Improved EDDBM",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "avg_error_vs_t.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")


def plot_efficiency_vs_t(all_results, out_dir):
    """Plot 4: Efficiency (wall-clock time) vs T"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    
    colors_c = plt.cm.Reds(np.linspace(0.5, 0.9, len(all_results)))
    colors_i = plt.cm.Blues(np.linspace(0.5, 0.9, len(all_results)))
    
    for (res, cc, ci) in zip(all_results, colors_c, colors_i):
        ts = res.get("t_sweep", [])
        if not ts:
            continue
        T_vals = [r["T"] for r in ts]
        c_time = [r["c_time"] for r in ts]
        i_time = [r["i_time"] for r in ts]
        ds     = res.get("dataset", "")
        
        ax.plot(T_vals, c_time, "o--", color=cc, linewidth=1.8, label=f"{ds} Classic O(K·T·m)")
        ax.plot(T_vals, i_time, "s-",  color=ci, linewidth=1.8, label=f"{ds} Improved O(T·m)")
    
    ax.set_xlabel("Samples T", fontsize=11)
    ax.set_ylabel("Wall-Clock Time (seconds)", fontsize=11)
    ax.set_title("Efficiency vs T — Classic vs Improved EDDBM\n(Classic costs K× more time)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "efficiency_vs_t.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  => Saved: {path}")


def main():
    txt_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.txt")))
    
    if not txt_files:
        print(f"No result files found in {RESULTS_DIR}/")
        print("Run improved_eddbm.exe or compile and run the C++ first.")
        return
    
    print(f"Found {len(txt_files)} result files:")
    all_results = []
    for path in txt_files:
        print(f"  Parsing: {path}")
        res = parse_output_file(path)
        all_results.append(res)
        print(f"    Dataset: {res.get('dataset','?')}  |  T-sweep rows: {len(res.get('t_sweep',[]))}")
    
    print("\nGenerating plots...")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plot_prob_vs_samples(all_results, PLOTS_DIR)
    plot_accuracy_vs_t(all_results, PLOTS_DIR)
    plot_error_vs_t(all_results, PLOTS_DIR)
    plot_efficiency_vs_t(all_results, PLOTS_DIR)
    
    print(f"\nAll plots saved to: {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    main()
