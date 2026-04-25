#!/usr/bin/env python3
import argparse
import glob
import os
import re
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def parse_eff_err(output):
    eff = None
    err = None
    mode = None
    for line in output.splitlines():
        if "Efficiency vs T" in line:
            mode = "eff"
            continue
        if "Average Error vs T" in line:
            mode = "err"
            continue
        if re.match(r"^\s*\d+\s+\d+\.\d+", line):
            parts = line.split()
            if len(parts) >= 2:
                val = float(parts[1])
                if mode == "eff":
                    eff = val
                elif mode == "err":
                    err = val
    return eff, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", default="5,10,15")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--ml-weight", type=float, default=0.5)
    parser.add_argument("--compute-exact", type=int, default=1)
    parser.add_argument("--out", default="model_compare.csv")
    args = parser.parse_args()

    t_vals = [int(x.strip()) for x in args.t.split(",") if x.strip()]
    if not t_vals:
        raise SystemExit("No T values provided")

    txt_files = sorted(glob.glob("*.txt"))
    if not txt_files:
        raise SystemExit("No .txt files found")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("graph,model,T,efficiency,avg_error\n")
        for path in txt_files:
            base = os.path.splitext(os.path.basename(path))[0]
            ml_csv = f"{base}_ml_probs.csv"

            run([sys.executable, "ml_infer.py", path, "--out", ml_csv])

            for t in t_vals:
                # fusion-ml
                res = run([
                    "./fusion-ml",
                    path,
                    str(t),
                    str(args.trials),
                    str(args.compute_exact),
                    ml_csv,
                    str(args.ml_weight),
                ])
                eff, err = parse_eff_err(res.stdout)
                eff_val = "" if eff is None else f"{eff:.2f}"
                err_val = "" if err is None else f"{err:.2f}"
                f.write(f"{base},fusion-ml,{t},{eff_val},{err_val}\n")

                # eddbm baseline (bolt)
                res = run([
                    "./bolt",
                    path,
                    str(t),
                    str(args.trials),
                    str(args.compute_exact),
                ])
                eff, err = parse_eff_err(res.stdout)
                eff_val = "" if eff is None else f"{eff:.2f}"
                err_val = "" if err is None else f"{err:.2f}"
                f.write(f"{base},eddbm,{t},{eff_val},{err_val}\n")

            print(f"Done: {base}")

    print(f"Saved summary to {args.out}")


if __name__ == "__main__":
    main()
