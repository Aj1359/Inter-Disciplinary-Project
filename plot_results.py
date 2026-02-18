#!/usr/bin/env python3
"""
plot_results.py
---------------
Visualise BOLT (EDDBM) results from all tested datasets.
Generates:
  1. Efficiency vs T  (per graph)
  2. Average Error vs T  (per graph)
  3. Combined comparison
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ─── datasets that have efficiency/error CSVs ───────────────────────────────
DATASETS = {
    "Wiki-Vote"     : ("Wiki-Vote_efficiency.csv",       "Wiki-Vote_error.csv"),
    "as20000102"    : ("as20000102_efficiency.csv",      "as20000102_error.csv"),
}

COLORS = plt.cm.tab10.colors

# ─── 1. Load data ─────────────────────────────────────────────────────────
eff_frames = {}
err_frames = {}

for label, (eff_f, err_f) in DATASETS.items():
    if os.path.exists(eff_f):
        df = pd.read_csv(eff_f)
        df["Efficiency"] *= 100  # convert to %
        eff_frames[label] = df
    if os.path.exists(err_f):
        df = pd.read_csv(err_f)
        err_frames[label] = df

# ─── 2. Efficiency vs T ──────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

ax = axs[0]
for i, (label, df) in enumerate(eff_frames.items()):
    ax.plot(df["T"], df["Efficiency"], marker='o', label=label, color=COLORS[i], linewidth=2)

ax.set_xlabel("Number of Samples (T)", fontsize=12)
ax.set_ylabel("Efficiency (%)", fontsize=12)
ax.set_title("Betweenness-Ordering Efficiency vs T (BOLT)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
ax.set_ylim(50, 102)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

# ─── 3. Average Error vs T ──────────────────────────────────────────────
ax = axs[1]
for i, (label, df) in enumerate(err_frames.items()):
    ax.plot(df["T"], df["AvgError"], marker='s', label=label, color=COLORS[i], linewidth=2, linestyle='--')

ax.set_xlabel("Number of Samples (T)", fontsize=12)
ax.set_ylabel("Average Error (%)", fontsize=12)
ax.set_title("Betweenness Estimation Avg Error vs T (BOLT)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

plt.tight_layout()
plt.savefig("bolt_results.png", dpi=150, bbox_inches='tight')
print("Saved: bolt_results.png")
plt.show()

# ─── 4. Summary table ─────────────────────────────────────────────────────
print("\n=== Efficiency at T=25 ===")
print(f"{'Dataset':<20} {'Efficiency@T=25':>16}")
print("-" * 38)
for label, df in eff_frames.items():
    row = df[df["T"] == 25]
    if not row.empty:
        print(f"{label:<20} {row['Efficiency'].values[0]:>15.2f}%")

print("\n=== Average Error at T=25 ===")
print(f"{'Dataset':<20} {'AvgError@T=25':>16}")
print("-" * 38)
for label, df in err_frames.items():
    row = df[df["T"] == 25]
    if not row.empty:
        print(f"{label:<20} {row['AvgError'].values[0]:>15.2f}%")
