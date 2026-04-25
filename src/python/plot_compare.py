#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="model_compare.csv")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    rows = []
    with open(args.inp, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["efficiency"] or not row["avg_error"]:
                continue
            rows.append({
                "graph": row["graph"],
                "model": row["model"],
                "T": int(row["T"]),
                "efficiency": float(row["efficiency"]),
                "avg_error": float(row["avg_error"]),
            })

    if not rows:
        raise SystemExit("No usable rows found in input")

    os.makedirs(args.out_dir, exist_ok=True)

    by_graph = defaultdict(list)
    for r in rows:
        by_graph[r["graph"]].append(r)

    for graph, items in by_graph.items():
        items.sort(key=lambda x: (x["model"], x["T"]))

        for metric, ylabel, suffix in [
            ("efficiency", "Efficiency (%)", "eff"),
            ("avg_error", "Avg Error (%)", "err"),
        ]:
            plt.figure(figsize=(6, 4))
            by_model = defaultdict(list)
            for r in items:
                by_model[r["model"]].append(r)

            for model, series in by_model.items():
                series.sort(key=lambda x: x["T"])
                xs = [r["T"] for r in series]
                ys = [r[metric] for r in series]
                plt.plot(xs, ys, marker="o", label=model)

            plt.title(f"{graph} - {ylabel}")
            plt.xlabel("T")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.legend()
            out_path = os.path.join(args.out_dir, f"{graph}_{suffix}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
