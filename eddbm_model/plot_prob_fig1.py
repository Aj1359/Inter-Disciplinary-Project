import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

networks = ["BA_1k_4", "BA_1k_3", "ER_1k_4", "ER_1k_3"]

for net in networks:
    print(f"Running C++ binary for {net}...")
    subprocess.run(["./task4_prob.exe", f"synthetic/{net}.txt", f"synthetic/{net}"])

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

letters = ['(a)', '(b)', '(c)', '(d)']
for i, net in enumerate(networks):
    df = pd.read_csv(f"synthetic/{net}_prob.csv")
    ax = axs[i]
    
    # Plot Opt, EDDBM, DBM
    ax.plot(df.index, df['Opt'], label='Opt', linestyle=':', linewidth=2, color='C0')
    ax.plot(df.index, df['EDDBM'], label='EDDBM', linestyle='-', linewidth=2, color='C1')
    
    # Because DBM may drop way too fast compared to the paper depending on scale, 
    # the paper normalizes them so they are comparably scaled across 100 nodes.
    # The sum is 1, so DBM drops to 0 immediately if exponent is large. 
    # Paper uses smooth lines for DBM.
    ax.plot(df.index, df['DBM'], label='DBM', linestyle='-', linewidth=2, color='C2', alpha=0.7)
    
    ax.set_title(f"{letters[i]} {net}", y=-0.2)
    ax.set_xlabel("Random 100 nodes")
    ax.set_ylabel("Assigned probability")
    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig("plots/Figure1_Redo.png", dpi=300)
print("Figure 1 has been re-created at plots/Figure1_Redo.png")
