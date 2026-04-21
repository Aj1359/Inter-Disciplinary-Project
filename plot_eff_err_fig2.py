import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

datasets = [
    ("as20000102.txt", "as20000102"),
    ("Wiki-Vote.txt", "Wiki-Vote"),
    ("CA-HepTh.txt", "CA-HepTh"),
    ("oregon1_010331.txt", "oregon1_010331"),
    ("CA-AstroPh.txt", "CA-AstroPh"),
    ("CA-CondMat.txt", "CA-CondMat"),
    ("p2p-Gnutella30.txt", "p2p-Gnutella30")
]

for fname, dname in datasets:
    if os.path.exists(fname):
        print(f"Running continuous T evaluation on {dname}...")
        # Restrict to 2000 nodes for speed since we just need the curve trends over T
        subprocess.run(["./task4_eff.exe", fname, dname, "2000"])
    else:
        print(f"Skipping {dname}, not found")

plt.figure(figsize=(10, 12))

# Subplot 1: Average Error
plt.subplot(2, 1, 1)
for fname, dname in datasets:
    f_res = f"{dname}_100.csv"
    if os.path.exists(f_res):
        df = pd.read_csv(f_res)
        # Smoothing just for visual clarity of lines as requested in Figure 2
        smoothed = df['AvgError'].rolling(window=3, min_periods=1).mean()
        plt.plot(df['T'], smoothed, linewidth=2, label=dname)

plt.xlabel("T (samples)")
plt.ylabel("Average Error (%)")
plt.ylim(0, 160)
plt.xlim(0, 100)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(20))
plt.grid(True, linestyle="-", color='gray', alpha=0.5)
plt.title("(a) Average Error", y=-0.2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

# Subplot 2: Average Efficiency
plt.subplot(2, 1, 2)
for fname, dname in datasets:
    f_res = f"{dname}_100.csv"
    if os.path.exists(f_res):
        df = pd.read_csv(f_res)
        smoothed = df['Efficiency'].rolling(window=3, min_periods=1).mean()
        plt.plot(df['T'], smoothed, linewidth=2, label=dname)

plt.xlabel("T (samples)")
plt.ylabel("Average Efficiency (%)")
plt.ylim(55, 105)
plt.xlim(0, 100)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
plt.grid(True, linestyle="-", color='gray', alpha=0.5)
plt.title("(b) Average Efficiency", y=-0.2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

plt.tight_layout()
plt.subplots_adjust(right=0.75, bottom=0.1) # leave room for legend outside 
plt.savefig("plots/Figure2_Redo.png", dpi=300)
print("Figure 2 has been re-created at plots/Figure2_Redo.png")
