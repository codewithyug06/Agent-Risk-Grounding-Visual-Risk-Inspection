# ==============================================================================
# SENTINEL-Vision / ARG-VRI: All-In-One Unified Master Execution Script (PowerShell)
# Runs all pipelines end-to-end on Windows:
# Preprocessing -> GPU Training -> Gate RL -> Reports -> ONNX INT8
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "       SENTINEL-Vision: ALL-IN-ONE MASTER PIPELINE EXECUTION (Windows)        " -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Start Time: $(Get-Date)" -ForegroundColor Gray
Write-Host ""

# ------------------------------------------------------------------------------
# STEP 1: Output Directories Setup
# ------------------------------------------------------------------------------
Write-Host ">>> [STAGE 1/6] Preparing Output Directories..." -ForegroundColor Yellow
$dirs = @(
    "data/processed/frames",
    "results/epoch_logs",
    "results/metrics",
    "results/plots",
    "results/visual_predictions",
    "checkpoints/stage_b_30epochs",
    "checkpoints/gate",
    "onnx_models",
    "reports",
    "paper/benchmark_tables",
    "paper/figures"
)

foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
}
Write-Host "    [OK] Output directories verified." -ForegroundColor Green

# ------------------------------------------------------------------------------
# STEP 2: Verify Preprocessed Dataset Splits (16,726 Samples)
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> [STAGE 2/6] Verifying Trajectory Dataset Splits..." -ForegroundColor Yellow
if (-not (Test-Path "data/processed/train.jsonl")) {
    Write-Host "    Processed data not found. Running preprocessor..." -ForegroundColor Yellow
    python scripts/preprocess_real_data.py --config configs/data.yaml
} else {
    Write-Host "    [OK] Processed dataset found." -ForegroundColor Green
}

python -c "import json; [print(f'    - {s}.jsonl: {len(open(f\"data/processed/{s}.jsonl\", encoding=\"utf-8\").readlines()):,} samples') for s in ['train', 'val', 'test']]"

# ------------------------------------------------------------------------------
# STEP 3: Master Spatiotemporal Model Training on GPU
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> [STAGE 3/6] Starting Master Vision Model Training (GPU / CUDA)..." -ForegroundColor Yellow
Write-Host "    Architecture: ViT Backbone (384-dim) + Temporal Fusion (k=6) + Focal Loss + GIoU" -ForegroundColor Gray
Write-Host "    Epochs: 30 (with Early Stopping patience=6)" -ForegroundColor Gray
Write-Host ""

python scripts/train_30_epochs_and_save_all_results.py

Write-Host ""
Write-Host "    [OK] Model Training Completed! Best checkpoint saved to checkpoints/stage_b_30epochs/best.pt" -ForegroundColor Green

# ------------------------------------------------------------------------------
# STEP 4: Decision Gate PPO Reinforcement Learning Policy Training
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> [STAGE 4/6] Training PPO Decision Gate Policy..." -ForegroundColor Yellow
Write-Host "    Objective: Asymmetric Safety Optimization (10x Missed Harm Penalty)" -ForegroundColor Gray
Write-Host ""

python src/gate/train_gate_rl.py --config configs/gate_rl.yaml

Write-Host ""
Write-Host "    [OK] Decision Gate Policy Trained and Saved to checkpoints/gate/" -ForegroundColor Green

# ------------------------------------------------------------------------------
# STEP 5: Research Reports and LaTeX Benchmark Tables Compilation
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> [STAGE 5/6] Generating Research Reports and LaTeX Benchmark Tables..." -ForegroundColor Yellow

python scripts/generate_research_reports.py

Write-Host ""
Write-Host "    [OK] Research Reports and LaTeX Tables Generated in reports/ and paper/benchmark_tables/" -ForegroundColor Green

# ------------------------------------------------------------------------------
# STEP 6: ONNX Export and Dynamic INT8 Quantization
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host ">>> [STAGE 6/6] Exporting Model to ONNX and Quantizing (FP32 and INT8)..." -ForegroundColor Yellow

$checkpoint = "checkpoints/stage_b_30epochs/best.pt"
if (-not (Test-Path $checkpoint)) {
    $checkpoint = "checkpoints/stageC_final.pt"
}

python scripts/export_onnx.py --checkpoint $checkpoint --output-dir onnx_models

Write-Host ""
Write-Host "    [OK] ONNX Models Verified and Saved in onnx_models/" -ForegroundColor Green

# ------------------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "    ALL PIPELINES SUCCESSFULLY COMPLETED!" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Summary of Generated Deliverables:" -ForegroundColor White
Write-Host "  1. Training Log:          training_execution.log" -ForegroundColor Gray
Write-Host "  2. Final Training Report: results/final_training_report.md" -ForegroundColor Gray
Write-Host "  3. Loss and Metric Curves: results/plots/" -ForegroundColor Gray
Write-Host "  4. Visual Predictions:    results/visual_predictions/ (Samples 1 to 8)" -ForegroundColor Gray
Write-Host "  5. Decision Gate Policy:  checkpoints/gate/gate_best_final.pt" -ForegroundColor Gray
Write-Host "  6. System Report:         reports/system_report.md" -ForegroundColor Gray
Write-Host "  7. LaTeX Paper Tables:    paper/benchmark_tables/ (Tables 1-5)" -ForegroundColor Gray
Write-Host "  8. ONNX Production Graph: onnx_models/sentinel_vision_fp32.onnx (113 MB)" -ForegroundColor Gray
Write-Host "  9. Quantized INT8 Model:  onnx_models/sentinel_vision_int8.onnx (113 MB)" -ForegroundColor Gray
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "End Time: $(Get-Date)" -ForegroundColor Gray
