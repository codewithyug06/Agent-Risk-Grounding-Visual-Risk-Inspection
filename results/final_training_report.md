# Full Training Run & Results Summary Report

**Project:** Agent Risk Grounding & Visual Risk Inspection (ARG-VRI / SENTINEL-Vision)  
**Execution Date:** 2026-10-19 19:16:42  
**Compute Device:** CUDA  
**Stage Name:** stage_c_clip_frozen_text  
**Total Training Duration:** 116.20 minutes  

---

## 1. Key Performance Highlights

- **Total Epochs Completed:** 15 / 15
- **Peak Harm Recall (Safety Rate):** **85.43%**
- **Peak Harm Precision:** **96.74%**
- **Peak Harm F1-Score:** **90.74%**
- **Peak UI Localization IoU@0.5:** **6.93%**
- **Peak Category Accuracy:** **96.40%**
- **Final Validation Loss:** **5.4510**

---

## 2. Saved Artifacts & Output Directories

- **Per-Epoch Log CSV:** `results/epoch_logs/epoch_logs_stage_c_clip_frozen_text.csv`
- **Per-Epoch Log JSON:** `results/epoch_logs/epoch_logs_stage_c_clip_frozen_text.json`
- **Loss Curves Plot:** `results/plots/loss_curve_stage_c_clip_frozen_text.png`
- **Safety Metrics Plot:** `results/plots/metrics_curve_stage_c_clip_frozen_text.png`
- **Visual Prediction Images:** `results/visual_predictions/prediction_sample_1.png` - `8.png`
- **Model Checkpoints:** `checkpoints/stage_c_clip_frozen_text/best.pt` & `best_loc.pt`
