#!/usr/bin/env bash
# Final training run: Stage A -> B -> C, then gold (MisActBench) and test evaluations.
# Run from the project root in Git Bash:  bash scripts/run_final_training.sh
# Moves any old checkpoints aside (nothing is deleted).
set -euo pipefail

STAMP=$(date +%Y%m%d_%H%M)
if [ -d checkpoints ]; then mv checkpoints "checkpoints_run_${STAMP}"; fi
mkdir -p checkpoints

python -m src.training.train_stageA device=cuda training.batch_size=4 +synthetic_injection_dir=data/__no_synthetic__
python -m src.training.train_stageB device=cuda training.batch_size=4 +synthetic_injection_dir=data/__no_synthetic__
python -m src.training.train_stageC device=cuda training.batch_size=4 +synthetic_injection_dir=data/__no_synthetic__

python scripts/eval_misactbench_gold.py --checkpoint checkpoints/stage_c/best.pt --out results/misactbench_gold_eval.json
python scripts/eval_screenspot.py --checkpoint checkpoints/stage_c/best.pt --out results/screenspot_v2_point_accuracy.json
python scripts/evaluate_final.py --checkpoint checkpoints/stage_c/best_loc.pt --out results/final_test_metrics_best_loc.json
python scripts/evaluate_final.py --checkpoint checkpoints/stage_c/best.pt --out results/final_test_metrics_best.json

echo "Done. Results: results/misactbench_gold_eval.json, results/screenspot_v2_point_accuracy.json, results/final_test_metrics_*.json"
