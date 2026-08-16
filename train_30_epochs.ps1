# ==============================================================================
# Agent Risk Grounding & Visual Risk Inspection (ARG-VRI / SENTINEL-Vision)
# 30-Epoch Master Training & Results Generator Script (PowerShell)
# ==============================================================================

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " Starting 30-Epoch Training for ARG-VRI on Real Trajectory Data..." -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan

# 1. Ensure results directory structure exists
New-Item -ItemType Directory -Force -Path results\epoch_logs, results\metrics, results\plots, results\visual_predictions, checkpoints | Out-Null

# 2. Run the 30-Epoch Master Training and Results Generation Pipeline
python scripts\train_30_epochs_and_save_all_results.py

Write-Host "`n==============================================================================" -ForegroundColor Cyan
Write-Host " Training Complete! All logs, plots, metrics, and images saved to .\results\" -ForegroundColor Green
Write-Host " Check .\results\final_training_report.md for full performance summary." -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor Cyan
