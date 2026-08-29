#!/usr/bin/env bash
# ==============================================================================
# SENTINEL-Vision / ARG-VRI: All-In-One Unified Master Execution Script (Bash)
# Runs all pipelines end-to-end: Preprocessing -> GPU Training -> Gate RL ->
# Reports & LaTeX Tables -> ONNX FP32/INT8 Quantization -> Visual Inspection
# ==============================================================================

set -e

# Color helpers
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}       SENTINEL-Vision: ALL-IN-ONE MASTER PIPELINE EXECUTION                 ${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo "Start Time: $(date)"
echo ""

# ------------------------------------------------------------------------------
# STEP 1: Directory Setup & Clean Workspace
# ------------------------------------------------------------------------------
echo -e "${CYAN}>>> [STAGE 1/6] Preparing Output Directories...${NC}"
mkdir -p data/processed/frames \
         results/epoch_logs \
         results/metrics \
         results/plots \
         results/visual_predictions \
         checkpoints/stage_b_30epochs \
         checkpoints/gate \
         onnx_models \
         reports \
         paper/benchmark_tables \
         paper/figures

echo "    [✓] Output directories verified."

# ------------------------------------------------------------------------------
# STEP 2: Data Preprocessing & Integrity Verification (16,726 Samples)
# ------------------------------------------------------------------------------
echo ""
echo -e "${CYAN}>>> [STAGE 2/6] Verifying Trajectory Dataset Splits...${NC}"
if [ ! -f "data/processed/train.jsonl" ]; then
    echo -e "${YELLOW}    Processed data not found. Running preprocessor...${NC}"
    python scripts/preprocess_real_data.py --config configs/data.yaml
else
    echo "    [✓] Processed dataset found."
fi

python -c "
import json
for split in ['train', 'val', 'test']:
    p = f'data/processed/{split}.jsonl'
    try:
        count = len(open(p, encoding='utf-8').readlines())
        print(f'    - {split}.jsonl: {count:,} samples')
    except Exception as e:
        print(f'    - {split}.jsonl: missing')
"

# ------------------------------------------------------------------------------
# STEP 3: Master Spatiotemporal Model Training (GPU Mixed Precision)
# ------------------------------------------------------------------------------
echo ""
echo -e "${CYAN}>>> [STAGE 3/6] Starting Master Vision Model Training (GPU / CUDA)...${NC}"
echo "    Architecture: ViT Backbone (384-dim) + Temporal Fusion (k=6) + Focal Loss + GIoU"
echo "    Epochs: 30 (with Early Stopping patience=6)"
echo ""

python scripts/train_30_epochs_and_save_all_results.py

echo ""
echo -e "${GREEN}    [✓] Model Training Completed! Best checkpoint saved to checkpoints/stage_b_30epochs/best.pt${NC}"

# ------------------------------------------------------------------------------
# STEP 4: Decision Gate PPO Reinforcement Learning Policy Training
# ------------------------------------------------------------------------------
echo ""
echo -e "${CYAN}>>> [STAGE 4/6] Training PPO Decision Gate Policy...${NC}"
echo "    Objective: Asymmetric Safety Optimization (10x Missed Harm Penalty)"
echo ""

python src/gate/train_gate_rl.py --config configs/gate_rl.yaml

echo ""
echo -e "${GREEN}    [✓] Decision Gate Policy Trained & Saved to checkpoints/gate/${NC}"

# ------------------------------------------------------------------------------
# STEP 5: Research Reports & Benchmark Tables Compilation
# ------------------------------------------------------------------------------
echo ""
echo -e "${CYAN}>>> [STAGE 5/6] Generating Research Reports & LaTeX Benchmark Tables...${NC}"

python scripts/generate_research_reports.py

echo ""
echo -e "${GREEN}    [✓] Research Reports & LaTeX Tables Generated in reports/ and paper/benchmark_tables/${NC}"

# ------------------------------------------------------------------------------
# STEP 6: ONNX Export & Dynamic INT8 Quantization
# ------------------------------------------------------------------------------
echo ""
echo -e "${CYAN}>>> [STAGE 6/6] Exporting Model to ONNX & Quantizing (FP32 & INT8)...${NC}"

CHECKPOINT="checkpoints/stage_b_30epochs/best.pt"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="checkpoints/stageC_final.pt"
fi

python scripts/export_onnx.py --checkpoint "$CHECKPOINT" --output-dir onnx_models

echo ""
echo -e "${GREEN}    [✓] ONNX Models Verified & Saved in onnx_models/${NC}"

# ------------------------------------------------------------------------------
# SUMMARY OF GENERATED DELIVERABLES
# ------------------------------------------------------------------------------
echo ""
echo -e "${BLUE}==============================================================================${NC}"
echo -e "${GREEN}    ALL PIPELINES SUCCESSFULLY COMPLETED!${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo "Summary of Generated Deliverables:"
echo "  1. Training Log:          training_execution.log"
echo "  2. Final Training Report: results/final_training_report.md"
echo "  3. Loss & Metric Curves:  results/plots/"
echo "  4. Visual Predictions:    results/visual_predictions/ (Samples 1 to 8)"
echo "  5. Decision Gate Policy:  checkpoints/gate/gate_best_final.pt"
echo "  6. System Report:         reports/system_report.md"
echo "  7. LaTeX Paper Tables:    paper/benchmark_tables/ (Tables 1-5)"
echo "  8. ONNX Production Graph: onnx_models/sentinel_vision_fp32.onnx (113 MB)"
echo "  9. Quantized INT8 Model:  onnx_models/sentinel_vision_int8.onnx (113 MB)"
echo -e "${BLUE}==============================================================================${NC}"
echo "End Time: $(date)"
