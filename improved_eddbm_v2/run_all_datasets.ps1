# run_all_datasets.ps1
# =====================
# Compiles improved_eddbm.cpp and runs it on all available datasets.
# Stores all output files in results/, then calls the Python plotter.
#
# Usage (from improved_eddbm_v2/ folder):
#   powershell -ExecutionPolicy Bypass -File run_all_datasets.ps1

$ErrorActionPreference = "Continue"

# Config
$K      = 20
$T_max  = 100
$MaxN   = 1500
$ResDir = "results"
$DataDir = "..\datasets"

# Create output directory
New-Item -ItemType Directory -Force -Path $ResDir | Out-Null

# Compile C++
Write-Host "`n[1] Compiling improved_eddbm.cpp..." -ForegroundColor Cyan
g++ -O2 -std=c++17 -o improved_eddbm.exe improved_eddbm.cpp
if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "    Compiled OK." -ForegroundColor Green

# Run on all datasets
$Datasets = @(
    "Wiki-Vote.txt",
    "CA-HepTh.txt",
    "p2p-Gnutella08.txt",
    "as20000102.txt"
)

Write-Host "`n[2] Running on all datasets..." -ForegroundColor Cyan

$AllSuccess = $true
foreach ($ds in $Datasets) {
    $path = Join-Path $DataDir $ds
    if (-not (Test-Path $path)) {
        Write-Host "    SKIP: $ds (not found at $path)" -ForegroundColor Yellow
        continue
    }
    
    $dsBase = [System.IO.Path]::GetFileNameWithoutExtension($ds)
    $outFile = Join-Path $ResDir "$($dsBase)_results.txt"
    
    Write-Host "    Running: $ds -> $outFile" -ForegroundColor White
    
    $start = Get-Date
    .\improved_eddbm.exe $path $K $T_max $outFile $MaxN
    $elapsed = (Get-Date) - $start
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    Done in $([math]::Round($elapsed.TotalSeconds, 1))s" -ForegroundColor Green
    } else {
        Write-Host "    FAILED for $ds" -ForegroundColor Red
        $AllSuccess = $false
    }
}

# Plot
Write-Host "`n[3] Generating plots with Python..." -ForegroundColor Cyan
$env:PYTHONIOENCODING = "utf-8"
python plot_improved_eddbm.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "    Plots generated in plots/" -ForegroundColor Green
} else {
    Write-Host "    Plot generation failed!" -ForegroundColor Red
}

Write-Host "`n[OK] All done! Results in results/   Plots in plots/" -ForegroundColor Green
