import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os

datasets = [
    ("CA-HepTh.txt", "ca_hepth", 5000), # 10k nodes
    ("Wiki-Vote.txt", "wiki_vote", 5000), # 7k nodes
]

# Run experiments
for d_file, d_name, max_nodes in datasets:
    if os.path.exists(d_file):
        print(f"Running experiments on {d_name}...")
        subprocess.run(["./task1_experiments.exe", d_file, d_name, "100", str(max_nodes)])
    else:
        print(f"Skipping {d_name}, file not found.")

if not os.path.exists("plots"):
    os.makedirs("plots")

# Plot 1: Probabilities
for _, d_name, _ in datasets:
    prob_file = f"{d_name}_prob.csv"
    if os.path.exists(prob_file):
        df = pd.read_csv(prob_file)
        plt.figure()
        # Group by distance and mean probability
        grouped = df.groupby('Distance').mean('Probability')
        plt.plot(grouped.index, grouped['Probability'], marker='o')
        plt.xlabel('Distance to source')
        plt.ylabel('P(Distance)')
        plt.title(f'Probability of Dependency vs Distance ({d_name})')
        plt.yscale('log')
        plt.grid()
        plt.savefig(f"plots/{d_name}_prob.png")
        plt.close()

# Plot 2: Efficiency vs T and Plot 3: Avg Error vs T
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
for _, d_name, _ in datasets:
    res_file = f"{d_name}_results.csv"
    if os.path.exists(res_file):
        df = pd.read_csv(res_file)
        plt.plot(df['T'], df['Efficiency'], marker='o', label=d_name)
plt.xlabel('T (Iterations)')
plt.ylabel('Efficiency (%)')
plt.title('Approximation Efficiency')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
for _, d_name, _ in datasets:
    res_file = f"{d_name}_results.csv"
    if os.path.exists(res_file):
        df = pd.read_csv(res_file)
        plt.plot(df['T'], df['AvgError'], marker='s', label=d_name)
plt.xlabel('T (Iterations)')
plt.ylabel('Average Absolute Error')
plt.title('Average Error vs T')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f"plots/Efficiency_Error.png")
plt.close()
print("Plots generated in plots/ directory.")
