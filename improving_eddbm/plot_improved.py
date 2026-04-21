import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os

datasets = [
    ("../as20000102.txt", "as20000102", 2000),
    ("../Wiki-Vote.txt", "Wiki-Vote", 2000),
    ("../CA-HepTh.txt", "CA-HepTh", 2000),
    ("../oregon1_010331.txt", "oregon1_010331", 2000),
    ("../CA-AstroPh.txt", "CA-AstroPh", 2000),
    ("../CA-CondMat.txt", "CA-CondMat", 2000),
    ("../p2p-Gnutella30.txt", "p2p-Gnutella30", 2000),
]

# Run experiments
for d_file, d_name, max_nodes in datasets:
    if os.path.exists(d_file):
        print(f"Running improved experiments on {d_name}...")
        subprocess.run(["./task3_experiments.exe", d_file, d_name, "100", str(max_nodes)])

if not os.path.exists("plots"):
    os.makedirs("plots")

# Plot 2: Efficiency vs T and Plot 3: Avg Error vs T
plt.figure(figsize=(10,5))

# Efficiency Plot
plt.subplot(1,2,1)
for _, d_name, _ in datasets:
    res_file = f"{d_name}_improved_results.csv"
    orig_res = f"../{d_name}_results.csv"
    
    if os.path.exists(res_file):
        df = pd.read_csv(res_file)
        for model in ["CAEDDBM", "PDEDDBM"]:
            m_df = df[df['Model'] == model]
            plt.plot(m_df['T'], m_df['Efficiency'], marker='o', label=f'{d_name} {model}')
            
    # Also plot original EDDBM if it exists
    if os.path.exists(orig_res):
        df_orig = pd.read_csv(orig_res)
        plt.plot(df_orig['T'], df_orig['Efficiency'], marker='x', linestyle='--', label=f'{d_name} EDDBM (Orig)')

plt.xlabel('T (Iterations)')
plt.ylabel('Efficiency (%)')
plt.title('Approximation Efficiency')
plt.legend()
plt.grid()

# Error Plot
plt.subplot(1,2,2)
for _, d_name, _ in datasets:
    res_file = f"{d_name}_improved_results.csv"
    orig_res = f"../{d_name}_results.csv"
    
    if os.path.exists(res_file):
        df = pd.read_csv(res_file)
        for model in ["CAEDDBM", "PDEDDBM"]:
            m_df = df[df['Model'] == model]
            plt.plot(m_df['T'], m_df['AvgError'], marker='o', label=f'{d_name} {model}')
            
    # Also plot original EDDBM if it exists
    if os.path.exists(orig_res):
        df_orig = pd.read_csv(orig_res)
        plt.plot(df_orig['T'], df_orig['AvgError'], marker='x', linestyle='--', label=f'{d_name} EDDBM (Orig)')

plt.xlabel('T (Iterations)')
plt.ylabel('Average Absolute Error')
plt.title('Average Error vs T')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(f"plots/Improved_Efficiency_Error.png")
plt.close()
print("Plots generated in improved_eddbm/plots/ directory.")
