#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys
import re


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def parse_metrics(output):
    eff = None
    err = None
    for line in output.splitlines():
        if re.match(r"^\s*\d+\s+\d+\.\d+", line):
            parts = line.split()
            if len(parts) >= 2:
                eff = float(parts[1])
        if line.strip().startswith("AvgError"):
            continue
    for line in output.splitlines():
        if re.match(r"^\s*\d+\s+\d+\.\d+", line):
            parts = line.split()
            if len(parts) >= 2:
                err = float(parts[1])
    return eff, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=10)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--ml-weight", type=float, default=0.5)
    parser.add_argument("--compute-exact", type=int, default=1)
    parser.add_argument("--out", default="fusion_ml_summary.csv")
    args = parser.parse_args()

    txt_files = sorted(glob.glob("*.txt"))
    if not txt_files:
        print("No .txt files found.", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("graph,efficiency,avg_error\n")
        for path in txt_files:
            base = os.path.splitext(os.path.basename(path))[0]
            ml_csv = f"{base}_ml_probs.csv"

            run([sys.executable, "ml_infer.py", path, "--out", ml_csv])
            res = run([
                "./fusion-ml",
                path,
                str(args.t),
                str(args.trials),
                str(args.compute_exact),
                ml_csv,
                str(args.ml_weight),
            ])
            eff, err = parse_metrics(res.stdout)
            if eff is None:
                eff = 0.0
            if err is None:
                err = 0.0
            f.write(f"{base},{eff},{err}\n")
            print(f"{base}: eff={eff} err={err}")

    print(f"Saved summary to {args.out}")


if __name__ == "__main__":
    main()
