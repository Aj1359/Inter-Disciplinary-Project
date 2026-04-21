import os
import subprocess
import csv
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datasets = [
    'snap_big_data/facebook_combined.txt',
    'snap_big_data/ca-GrQc.txt',
    'snap_big_data/p2p-Gnutella08.txt'
]

print("Compiling smart_algorithm.cpp...")
compile_proc = subprocess.run(['g++', '-O2', '-std=c++11', '-o', 'smart_algorithm.exe', 'smart_algorithm.cpp'])
if compile_proc.returncode != 0:
    print("Compilation failed!")
    exit(1)

os.makedirs('results', exist_ok=True)
csv_file = 'results/snap_datasets_analysis.csv'
plot_file = 'results/snap_datasets_plot.png'

results_data = {}

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'Tau', 'BC_Before', 'BC_After', 'Reduction', 'Reduction_Pct', 'Time_ms'])

    for dset in datasets:
        print(f"\nProcessing {dset} ...")
        if not os.path.exists(dset):
            print(f"Skipping {dset} (file not found)")
            continue
            
        dataset_name = os.path.basename(dset).split('.')[0]
        results_data[dataset_name] = {'tau': [0.05, 0.10, 0.15, 0.20, 0.25], 'red_pct': []}
            
        try:
            result = subprocess.run(['.\\smart_algorithm.exe', dset], 
                                  capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Parse the formatted output
            blocks = output.split('DATASET:')
            if len(blocks) > 1:
                run_output = blocks[1]
                bc_before, bc_after, red, red_pct, time_ms = 0.0, 0.0, 0.0, 0.0, 0.0
                curr_tau = 0.0
                for line in run_output.split('\n'):
                    line = line.strip()
                    if '│  BC before' in line:
                        bc_before = float(line.split(':')[1].strip())
                    elif '│  BC after' in line:
                        bc_after = float(line.split(':')[1].strip())
                    elif '│  Reduction' in line:
                        parts = line.split('(')
                        red = float(parts[0].split(':')[1].strip())
                        red_pct = float(parts[1].replace('%)','').strip())
                    elif '│  Time' in line:
                        time_ms = float(line.split(':')[1].replace('ms','').strip())
                        writer.writerow([dataset_name, curr_tau, bc_before, bc_after, red, red_pct, time_ms])
                        results_data[dataset_name]['red_pct'].append(red_pct)
                    elif 'Evaluating tau' in line.lower() or 'tau =' in line.lower():
                        pass
        except Exception as e:
            print(f"Error running on {dset}: {e}")

for d in results_data:
    while len(results_data[d]['red_pct']) < 5:
        results_data[d]['red_pct'].append(0.0)

plt.figure(figsize=(10, 6))
colors = ['#1D9E75', '#D85A30', '#534AB7']
for idx, (name, data) in enumerate(results_data.items()):
    if sum(data['red_pct']) > 0:
        plt.plot(data['tau'], data['red_pct'], marker='o', lw=2, color=colors[idx%3], label=name)

plt.title('BC Reduction vs. Load-Balance Threshold (Tau) on SNAP Datasets', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Tau Constraint', fontsize=12)
plt.ylabel('Optimum Centrality Reduction (%)', fontsize=12)
plt.xticks([0.05, 0.10, 0.15, 0.20, 0.25])
plt.legend()
plt.tight_layout()
plt.savefig(plot_file)

print(f"\n[DONE] SNAP Dataset Analysis stored in: {csv_file}")
print(f"[DONE] Graphical Plot saved to: {plot_file}")
