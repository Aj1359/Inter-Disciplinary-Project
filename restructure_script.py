import os
import shutil
import glob

# The 5 primary folders to create
FOLDERS = [
    "eddbm_model",
    "datasets",
    "ml_models_all",
    "improving_eddbm_all",
    "other_models"
]

for f in FOLDERS:
    os.makedirs(f, exist_ok=True)

def safe_move(src, dest_folder):
    if os.path.exists(src):
        try:
            dest_path = os.path.join(dest_folder, os.path.basename(src))
            if os.path.exists(dest_path):
                # Don't overwrite, just remove the src if it's the exact same or rename
                pass
            else:
                shutil.move(src, dest_folder)
                print(f"Moved {src} to {dest_folder}/")
        except Exception as e:
            print(f"Error moving {src}: {e}")

# 1. DATASETS
for ext in ["*.txt", "*.txt.gz", "*_100.csv"]:
    for f in glob.glob(ext):
        safe_move(f, "datasets")

# 2. EDDBM MODEL
eddbm_files = [
    "eddm_wiki.cpp", "eddm.exe", "wiki.exe",
    "task1_experiments.cpp", "task1_experiments.exe",
    "plot_prob_fig1.py", "plot_experiments.py",
    "wiki_vote_results.csv", "wiki_vote_prob.csv",
    "ca_hepth_results.csv", "ca_hepth_prob.csv"
]
for f in eddbm_files:
    safe_move(f, "eddbm_model")

# 3. ML MODEL
# Merge 'MLmodel' and 'ml_model' into 'ml_models_all'
# First bring out the advanced options from MLmodel
advanced_opt_paths = [
    "MLmodel/ml_regressor",
    "MLmodel/neighborhood_smooth",
    "MLmodel/top_t_sampling"
]
for opt in advanced_opt_paths:
    safe_move(opt, "improving_eddbm_all")

# Put remaining MLmodel contents into ml_models_all
if os.path.exists("MLmodel"):
    for item in os.listdir("MLmodel"):
        safe_move(os.path.join("MLmodel", item), "ml_models_all")
    # Clean empty MLmodel dir
    try: os.rmdir("MLmodel")
    except: pass

if os.path.exists("ml_model"): # Old folder
    for item in os.listdir("ml_model"):
        safe_move(os.path.join("ml_model", item), "ml_models_all")
    try: os.rmdir("ml_model")
    except: pass

# 4. IMPROVING EDDBM (Also include task5 smoothing and improved_eddbm cpp files)
improving = [
    "task5_smooth.cpp", "task5_smooth.exe", "plot_smooth.py",
    "improved_eddbm/task3_experiments.cpp", "improved_eddbm/plot_improved.py"
]
for f in improving:
    if f.startswith("improved_eddbm/"):
        safe_move(f, "improving_eddbm_all")
    else:
        safe_move(f, "improving_eddbm_all")

if os.path.exists("improved_eddbm"):
    for item in os.listdir("improved_eddbm"):
        safe_move(os.path.join("improved_eddbm", item), "improving_eddbm_all")
    try: os.rmdir("improved_eddbm")
    except: pass

# 5. OTHER MODELS
others = [
    "brandes.cpp", "brandes.exe",
    "efficiency.cpp", "eff.exe", "efficiency.csv",
    "task4_eff.exe", "task4_eff_error.cpp", "task4_prob.exe", "task4_prob_experiments.cpp",
    "produce_synthetic.py", "synthetic", "a.out", "plot.py"
]
for f in others:
    safe_move(f, "other_models")

# Rename the final aggregated folders back to requested names
def safe_rename(src, dest):
    if os.path.exists(src) and not os.path.exists(dest):
        os.rename(src, dest)

safe_rename("ml_models_all", "ml_model")
safe_rename("improving_eddbm_all", "improving_eddbm")

print("Restructuring Complete.")
