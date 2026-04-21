# EDDBM Execution Guide

## Prerequisites
- **C++ Compiler**: g++ (MinGW on Windows, or system g++)
- **Python 3.7+** with packages: scipy, matplotlib, numpy

## Installation (One-time)

### Windows (PowerShell)
```powershell
# Install Python packages
pip install scipy matplotlib numpy

# Verify g++ is installed
g++ --version
```

### Linux/macOS
```bash
# Install dependencies
sudo apt-get install g++ python3-pip
pip3 install scipy matplotlib numpy
```

## Compilation

### Windows (PowerShell)
```powershell
# Navigate to eddbm folder
cd "c:\Users\ranik\OneDrive\Documents\Inter-disciplinary\eddbm"

# Compile C++ code
g++ -O2 -std=c++17 -o improved_eddbm.exe improved_eddbm.cpp

# Verify compilation
dir improved_eddbm.exe
```

### Linux/macOS
```bash
cd /path/to/Inter-disciplinary/eddbm

# Compile
g++ -O2 -std=c++17 -o improved_eddbm improved_eddbm.cpp

# Verify
ls -la improved_eddbm
```

## Execution

### 1. Run EDDBM on a Dataset

#### Windows (PowerShell)
```powershell
cd "c:\Users\ranik\OneDrive\Documents\Inter-disciplinary\improved_eddbm_v2"

# Example: Run on Wiki-Vote dataset with K=15 targets, T_max=100
.\improved_eddbm.exe ..\datasets\Wiki-Vote.txt 15 100 wiki_output.txt 1500

# Run on multiple datasets
.\improved_eddbm.exe ..\datasets\as20000102.txt 15 100 as_output.txt 1500
.\improved_eddbm.exe ..\datasets\CA-HepTh.txt 15 100 ca_hepth_output.txt 1500
.\improved_eddbm.exe ..\datasets\p2p-Gnutella30.txt 15 100 p2p_output.txt 1500
```

#### Linux/macOS
```bash
cd /path/to/Inter-disciplinary/improved_eddbm_v2

# Example: Run on Wiki-Vote dataset
./improved_eddbm ../datasets/Wiki-Vote.txt 15 100 wiki_output.txt 1500

# Run on multiple datasets
./improved_eddbm ../datasets/as20000102.txt 15 100 as_output.txt 1500
./improved_eddbm ../datasets/CA-HepTh.txt 15 100 ca_hepth_output.txt 1500
./improved_eddbm ../datasets/p2p-Gnutella30.txt 15 100 p2p_output.txt 1500
```

### Command Format
```
improved_eddbm.exe <dataset.txt> <K> <T_max> <output.txt> <max_nodes>
```

**Parameters:**
- `<dataset.txt>`: Path to edge list file (SNAP format)
- `<K>`: Number of target nodes to evaluate (default: 15)
- `<T_max>`: Maximum number of samples to test (default: 100)
- `<output.txt>`: Output file for results
- `<max_nodes>`: Maximum nodes to load from dataset (default: 1500)

### Example Outputs
The command will generate an output file containing:
- Exact BC values for target nodes
- T-sweep results (accuracy, error, runtime)
- Probability estimates vs. T
- Best T value and speedup metrics

### 2. Generate Plots

#### Windows (PowerShell)
```powershell
cd "c:\Users\ranik\OneDrive\Documents\Inter-disciplinary\improved_eddbm_v2"

# Ensure results/ folder exists with .txt output files
mkdir -Force results

# Run plot script (auto-finds .txt files in results/ folder)
python plot_improved_eddbm.py
```

#### Linux/macOS
```bash
cd /path/to/Inter-disciplinary/improved_eddbm_v2

# Create results folder if needed
mkdir -p results

# Run plot script
python3 plot_improved_eddbm.py
```

### Expected Plot Output
The script generates 4 plots in `plots/` folder:
1. **prob_vs_samples.png** - Estimated BC vs T samples
2. **accuracy_vs_t.png** - Pairwise ordering accuracy vs T
3. **error_vs_t.png** - Average relative error vs T
4. **time_vs_t.png** - Runtime vs T

## Quick Start Script

### Windows (PowerShell)
```powershell
# Full pipeline
$WorkDir = "c:\Users\ranik\OneDrive\Documents\Inter-disciplinary"

# Compile
cd "$WorkDir\improved_eddbm_v2"
g++ -O2 -std=c++17 -o improved_eddbm.exe improved_eddbm.cpp

# Create results folder
mkdir -Force results

# Run experiments
.\improved_eddbm.exe ..\datasets\Wiki-Vote.txt 15 100 results\wiki.txt 1500
.\improved_eddbm.exe ..\datasets\as20000102.txt 15 100 results\as.txt 1500
.\improved_eddbm.exe ..\datasets\CA-HepTh.txt 15 100 results\ca.txt 1500

# Generate plots
python plot_improved_eddbm.py

# View plots
mkdir -Force plots
Write-Host "Plots saved to: $WorkDir\improved_eddbm_v2\plots"
```

### Linux/macOS (Bash)
```bash
#!/bin/bash
WorkDir="/path/to/Inter-disciplinary"

# Compile
cd "$WorkDir/improved_eddbm_v2"
g++ -O2 -std=c++17 -o improved_eddbm improved_eddbm.cpp

# Create results folder
mkdir -p results

# Run experiments
./improved_eddbm ../datasets/Wiki-Vote.txt 15 100 results/wiki.txt 1500
./improved_eddbm ../datasets/as20000102.txt 15 100 results/as.txt 1500
./improved_eddbm ../datasets/CA-HepTh.txt 15 100 results/ca.txt 1500

# Generate plots
python3 plot_improved_eddbm.py

echo "Plots saved to: $WorkDir/improved_eddbm_v2/plots"
```

## Troubleshooting

### C++ Compilation Issues
```powershell
# Verify g++ installation
g++ --version

# If not found, install MinGW or use:
choco install mingw  # Windows with Chocolatey
```

### Python Package Issues
```powershell
# Install missing packages
pip install --upgrade scipy matplotlib numpy

# Verify installation
python -c "import scipy, matplotlib, numpy; print('All packages OK')"
```

### Dataset Not Found
```powershell
# List available datasets
dir ..\datasets\

# Use correct path relative to working directory
.\improved_eddbm.exe ..\datasets\Wiki-Vote.txt 15 100 output.txt 1500
```

### No Plots Generated
```powershell
# Verify results folder has .txt files
dir results\

# Check plot folder was created
dir plots\
```

## Dataset Availability
Available datasets in `../datasets/`:
- `Wiki-Vote.txt`
- `as20000102.txt`
- `CA-HepTh.txt`
- `p2p-Gnutella30.txt`
- `Email-Enron.txt`
- `facebook_combined.txt`

## Performance Notes
- **Small graphs** (1000 nodes): ~1-2 seconds per experiment
- **Medium graphs** (5000 nodes): ~5-10 seconds per experiment
- **Large datasets**: Limit with `max_nodes` parameter for speed
