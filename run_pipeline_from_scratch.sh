#!/usr/bin/env bash
# ==============================================================================
# SENTINEL-Vision / ARG-VRI: End-to-End Master Pipeline Script (Bash)
# Runs the entire lifecycle from raw data to trained model & ONNX deployment
# ==============================================================================

set -e

echo "=============================================================================="
echo "    SENTINEL-Vision: Starting Full Pipeline Execution From Scratch"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Step 1: Directory Setup
# ------------------------------------------------------------------------------
echo ""
echo ">>> [1/5] Creating Directory Structure..."
mkdir -p data/processed/frames \
         results/epoch_logs \
         results/metrics \
         results/plots \
         results/visual_predictions \
         checkpoints/stage_b_30epochs \
         onnx_models \
         reports \
         paper/benchmark_tables \
         paper/figures

# ------------------------------------------------------------------------------
# Step 2: Data Preprocessing
# (Processes Multimodal-Mind2Web, ScreenSpot-v1, ScreenSpot-v2 into 16,726 samples)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [2/5] Running Real-Data Preprocessing Pipeline..."
python scripts/preprocess_real_data.py --config configs/data.yaml

# Quick check on preprocessed data splits
python -c "
import json
for split in ['train', 'val', 'test']:
    lines = len(open(f'data/processed/{split}.jsonl').readlines())
    print(f'  [DATA] {split}.jsonl: {lines} samples')
"

# ------------------------------------------------------------------------------
# Step 3: Model Training (GPU Accelerated, Mixed Precision)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [3/5] Starting Master Model Training Loop on GPU..."
python scripts/train_30_epochs_and_save_all_results.py

# ------------------------------------------------------------------------------
# Step 4: Research Benchmarks & Paper Tables Generation
# ------------------------------------------------------------------------------
echo ""
echo ">>> [4/5] Generating Benchmark Tables & Performance Reports..."
python scripts/generate_research_reports.py

# ------------------------------------------------------------------------------
# Step 5: ONNX INT8 Quantization & Latency Verification
# ------------------------------------------------------------------------------
echo ""
echo ">>> [5/5] Exporting and Quantizing Model to ONNX INT8..."
CHECKPOINT="checkpoints/stage_b_30epochs/best.pt"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="checkpoints/stageC_final.pt"
fi

python scripts/export_onnx.py --checkpoint "$CHECKPOINT" --output-dir onnx_models

echo ""
echo "=============================================================================="
echo "    All Stages Successfully Completed!"
echo "    - Training Log:        training_execution.log"
echo "    - Final Report:        results/final_training_report.md"
echo "    - Research Report:     reports/system_report.md"
echo "    - Benchmark Tables:    paper/benchmark_tables/"
echo "    - Visual Predictions:  results/visual_predictions/"
echo "    - ONNX Models:         onnx_models/sentinel_vision_int8.onnx"
echo "=============================================================================="
