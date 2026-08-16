#!/usr/bin/env bash
# ==============================================================================
# Agent Risk Grounding & Visual Risk Inspection (ARG-VRI / SENTINEL-Vision)
# 30-Epoch Master Training & Results Generator Script (Bash)
# ==============================================================================

set -e

echo "=============================================================================="
echo " Starting 30-Epoch Training for ARG-VRI on Real Trajectory Data..."
echo "=============================================================================="

# 1. Ensure output results directories exist
mkdir -p results/epoch_logs results/metrics results/plots results/visual_predictions checkpoints

# 2. Run the 30-Epoch Master Training and Results Generation Pipeline
python scripts/train_30_epochs_and_save_all_results.py

echo ""
echo "=============================================================================="
echo " Training Complete! All logs, plots, metrics, and images saved to ./results/"
echo " Check ./results/final_training_report.md for full performance summary."
echo "=============================================================================="
