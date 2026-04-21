import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

datasets = [
    ("as20000102.txt", "as20000102"),
    ("Wiki-Vote.txt", "Wiki-Vote"),
    ("CA-HepTh.txt", "CA-HepTh"),
    ("oregon1_010331.txt", "oregon1_010331"),
    ("p2p-Gnutella30.txt", "p2p-Gnutella30")
]

# Run the compiled C++ script using 500 samples for high smoothness and max_nodes=2000 to keep it manageable
for fname, dname in datasets:
    if os.path.exists(fname):
        print(f"Running smoothed evaluations on {dname}...")
        subprocess.run(["./task5_smooth.exe", fname, dname, "500", "2000"])
    else:
        print(f"Skipping {dname}, not found")

# We plot the EDDBM variants to show efficiency and error vs datasets just like Figure 2 but cleaner
plt.figure(figsize=(12, 14))

plt.subplot(2, 1, 1)
# We plot standard EDDBM for comparison across datasets, using rolling window 5 for extra smoothing
for fname, dname in datasets:
    f_res = f"{dname}_smoothed_results.csv"
    if os.path.exists(f_res):
        df = pd.read_csv(f_res)
        df_ed = df[df['Model'] == 'EDDBM']
        smoothed = df_ed['AvgError'].rolling(window=5, min_periods=1).mean()
        plt.plot(df_ed['T'], smoothed, linewidth=2, label=dname)

plt.xlabel("T (samples)")
plt.ylabel("Average Error (%)")
plt.ylim(0, 160)
plt.xlim(0, 100)
plt.grid(True, linestyle="-", color='gray', alpha=0.5)
plt.title("(a) Smoothed Average Error (EDDBM)", y=-0.15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

plt.subplot(2, 1, 2)
for fname, dname in datasets:
    f_res = f"{dname}_smoothed_results.csv"
    if os.path.exists(f_res):
        df = pd.read_csv(f_res)
        df_ed = df[df['Model'] == 'EDDBM']
        smoothed = df_ed['Efficiency'].rolling(window=5, min_periods=1).mean()
        plt.plot(df_ed['T'], smoothed, linewidth=2, label=dname)

plt.xlabel("T (samples)")
plt.ylabel("Average Efficiency (%)")
plt.ylim(55, 105)
plt.xlim(0, 100)
plt.grid(True, linestyle="-", color='gray', alpha=0.5)
plt.title("(b) Smoothed Average Efficiency (EDDBM)", y=-0.15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

plt.tight_layout()
plt.subplots_adjust(right=0.75) 

if not os.path.exists("plots"):
    os.makedirs("plots")
    
plt.savefig("plots/Figure2_Smoothed.png", dpi=300)
print("Finished saving smooth graphs to plots/Figure2_Smoothed.png")

# Notice that the CSVs are already generated with EDDBM, CAEDDBM, PDEDDBM inside the root dir as per user request.
