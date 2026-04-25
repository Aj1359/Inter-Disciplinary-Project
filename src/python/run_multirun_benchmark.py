#!/usr/bin/env python3
"""
Run repeated GNN-v3 evaluations with different seeds, then aggregate and plot.

This reduces one-run randomness and produces smoother, presentation-ready results.
"""
import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_model_specs(model_specs):
    parsed = []
    for spec in model_specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --model spec '{spec}'. Use NAME:CHECKPOINT")
        name, path = spec.split(":", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --model spec '{spec}'. Use NAME:CHECKPOINT")
        parsed.append((name, path))
    return parsed


def ffloat(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def run_eval(project_dir, model_path, seed, trials, max_nodes, out_csv):
    cmd = [
        sys.executable,
        "gnn_v3_evaluate.py",
        "--data-dir", ".",
        "--gnn-model", model_path,
        "--trials", str(trials),
        "--max-nodes", str(max_nodes),
        "--seed", str(seed),
        "--out", out_csv,
    ]
    result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for model={model_path}, seed={seed}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def read_eval_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_csv(path, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(raw_rows):
    by_key = defaultdict(list)
    for r in raw_rows:
        key = (r["dataset"], r["model"])
        by_key[key].append(r)

    summary = []
    for (dataset, model), items in sorted(by_key.items()):
        eff_vals = [ffloat(x["efficiency_pct"]) for x in items]
        rho_vals = [ffloat(x["spearman_rho"]) for x in items]
        tau_vals = [ffloat(x["kendall_tau"]) for x in items]
        time_vals = [ffloat(x["time_ms"]) for x in items]

        eff_mean, eff_std = mean_std(eff_vals)
        rho_mean, rho_std = mean_std(rho_vals)
        tau_mean, tau_std = mean_std(tau_vals)
        time_mean, time_std = mean_std(time_vals)

        summary.append({
            "dataset": dataset,
            "model": model,
            "runs": len(items),
            "efficiency_mean": "" if eff_mean is None else f"{eff_mean:.4f}",
            "efficiency_std": "" if eff_std is None else f"{eff_std:.4f}",
            "spearman_mean": "" if rho_mean is None else f"{rho_mean:.6f}",
            "spearman_std": "" if rho_std is None else f"{rho_std:.6f}",
            "kendall_mean": "" if tau_mean is None else f"{tau_mean:.6f}",
            "kendall_std": "" if tau_std is None else f"{tau_std:.6f}",
            "time_ms_mean": "" if time_mean is None else f"{time_mean:.4f}",
            "time_ms_std": "" if time_std is None else f"{time_std:.4f}",
        })

    overall = []
    by_model = defaultdict(list)
    for row in summary:
        by_model[row["model"]].append(row)

    for model, items in sorted(by_model.items()):
        eff_vals = [ffloat(x["efficiency_mean"]) for x in items]
        rho_vals = [ffloat(x["spearman_mean"]) for x in items]
        time_vals = [ffloat(x["time_ms_mean"]) for x in items]

        eff_mean, eff_std = mean_std(eff_vals)
        rho_mean, rho_std = mean_std(rho_vals)
        time_mean, time_std = mean_std(time_vals)

        overall.append({
            "model": model,
            "datasets": len(items),
            "avg_efficiency": "" if eff_mean is None else f"{eff_mean:.4f}",
            "std_across_datasets_efficiency": "" if eff_std is None else f"{eff_std:.4f}",
            "avg_spearman": "" if rho_mean is None else f"{rho_mean:.6f}",
            "std_across_datasets_spearman": "" if rho_std is None else f"{rho_std:.6f}",
            "avg_time_ms": "" if time_mean is None else f"{time_mean:.4f}",
            "std_across_datasets_time_ms": "" if time_std is None else f"{time_std:.4f}",
        })

    return summary, overall


def grouped_bar_plot(summary_rows, out_path):
    dataset_order = sorted({r["dataset"] for r in summary_rows})
    model_order = sorted({r["model"] for r in summary_rows})

    values = {(r["dataset"], r["model"]): (ffloat(r["efficiency_mean"]), ffloat(r["efficiency_std"]))
              for r in summary_rows}

    x = list(range(len(dataset_order)))
    width = 0.8 / max(1, len(model_order))

    plt.figure(figsize=(14, 6))
    for mi, model in enumerate(model_order):
        offs = [xi - 0.4 + width / 2 + mi * width for xi in x]
        ys = []
        es = []
        for ds in dataset_order:
            y, e = values.get((ds, model), (0.0, 0.0))
            ys.append(0.0 if y is None else y)
            es.append(0.0 if e is None else e)
        plt.bar(offs, ys, width=width, yerr=es, capsize=3, label=model)

    plt.xticks(x, dataset_order, rotation=20, ha="right")
    plt.ylabel("Efficiency (%)")
    plt.title("Multi-run Efficiency (mean ± std)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def overall_plot(overall_rows, out_path):
    models = [r["model"] for r in overall_rows]
    ys = [ffloat(r["avg_efficiency"]) or 0.0 for r in overall_rows]
    es = [ffloat(r["std_across_datasets_efficiency"]) or 0.0 for r in overall_rows]

    plt.figure(figsize=(8, 5))
    plt.bar(models, ys, yerr=es, capsize=4)
    plt.ylabel("Average Efficiency (%)")
    plt.title("Overall Efficiency by Model")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run repeated GNN evaluations and aggregate results")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--model", action="append", required=True,
                        help="Model spec NAME:CHECKPOINT. Repeat for multiple models.")
    parser.add_argument("--seeds", default="11,23,37",
                        help="Comma-separated seeds for repeated runs")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--max-nodes", type=int, default=12000)
    parser.add_argument("--out-dir", default="benchmark_results")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    model_specs = parse_model_specs(args.model)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise SystemExit("No seeds provided")

    raw_dir = os.path.join(args.out_dir, "raw")
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    all_rows = []

    for model_name, model_path in model_specs:
        if not os.path.isabs(model_path):
            model_path = os.path.join(args.project_dir, model_path)
        if not os.path.exists(model_path):
            raise SystemExit(f"Model checkpoint not found: {model_path}")

        for run_idx, seed in enumerate(seeds, start=1):
            tmp_csv = os.path.join(raw_dir, f"{model_name}_seed{seed}.csv")
            print(f"[RUN] model={model_name} seed={seed}")
            run_eval(args.project_dir, model_path, seed, args.trials, args.max_nodes, tmp_csv)

            rows = read_eval_csv(tmp_csv)
            for r in rows:
                row_model = r["model"].strip()
                if row_model.upper().startswith("GNN"):
                    row_model = model_name

                all_rows.append({
                    "run_id": run_idx,
                    "seed": seed,
                    "model_group": model_name,
                    "dataset": r["dataset"],
                    "n": r["n"],
                    "m": r["m"],
                    "model": row_model,
                    "efficiency_pct": r["efficiency_pct"],
                    "spearman_rho": r["spearman_rho"],
                    "kendall_tau": r["kendall_tau"],
                    "time_ms": r["time_ms"],
                })

            if not args.keep_temp:
                try:
                    os.remove(tmp_csv)
                except OSError:
                    pass

    raw_out = os.path.join(args.out_dir, "multirun_raw.csv")
    write_csv(
        raw_out,
        [
            "run_id", "seed", "model_group", "dataset", "n", "m", "model",
            "efficiency_pct", "spearman_rho", "kendall_tau", "time_ms"
        ],
        all_rows,
    )

    summary_rows, overall_rows = aggregate(all_rows)

    summary_out = os.path.join(args.out_dir, "multirun_summary.csv")
    overall_out = os.path.join(args.out_dir, "multirun_overall.csv")

    write_csv(
        summary_out,
        [
            "dataset", "model", "runs",
            "efficiency_mean", "efficiency_std",
            "spearman_mean", "spearman_std",
            "kendall_mean", "kendall_std",
            "time_ms_mean", "time_ms_std",
        ],
        summary_rows,
    )

    write_csv(
        overall_out,
        [
            "model", "datasets",
            "avg_efficiency", "std_across_datasets_efficiency",
            "avg_spearman", "std_across_datasets_spearman",
            "avg_time_ms", "std_across_datasets_time_ms",
        ],
        overall_rows,
    )

    grouped_bar_plot(summary_rows, os.path.join(plot_dir, "efficiency_by_dataset_mean_std.png"))
    overall_plot(overall_rows, os.path.join(plot_dir, "overall_efficiency.png"))

    print("\nSaved:")
    print(f"  {raw_out}")
    print(f"  {summary_out}")
    print(f"  {overall_out}")
    print(f"  {os.path.join(plot_dir, 'efficiency_by_dataset_mean_std.png')}")
    print(f"  {os.path.join(plot_dir, 'overall_efficiency.png')}")


if __name__ == "__main__":
    main()
