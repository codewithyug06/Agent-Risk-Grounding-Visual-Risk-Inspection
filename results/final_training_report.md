# Full Training Run & Results Summary Report

**Project:** Agent Risk Grounding & Visual Risk Inspection (ARG-VRI / SENTINEL-Vision)  
**Execution Date:** 2026-10-14 13:42:53  
**Compute Device:** CUDA (NVIDIA GeForce RTX 4050 Laptop GPU)  
**Stage Name:** stage_c_clip_frozen  
**Total Training Duration:** 284.16 minutes  

---

## 1. Key Performance Highlights

- **Total Epochs Completed:** 15 / 15
- **Peak Harm Recall (Safety Rate):** **53.06%**
- **Peak Harm Precision:** **47.38%**
- **Peak Harm F1-Score:** **45.60%**
- **Peak UI Localization IoU@0.5:** **13.74%**
- **Peak Category Accuracy:** **84.87%**
- **Final Train Loss:** **3.4656**
- **Final Validation Loss:** **5.1802**

---

## 2. Saved Artifacts & Output Directories

- **Per-Epoch Log CSV:** `results/epoch_logs/epoch_logs_stage_c_clip_frozen.csv`
- **Per-Epoch Log JSON:** `results/epoch_logs/epoch_logs_stage_c_clip_frozen.json`
- **Loss Curves Plot:** `results/plots/loss_curve_stage_c_clip_frozen.png`
- **Safety Metrics Plot:** `results/plots/metrics_curve_stage_c_clip_frozen.png`
- **Visual Prediction Images:** `results/visual_predictions/prediction_sample_1.png` - `8.png`
- **Model Checkpoints:** `checkpoints/stage_c_clip_frozen/best.pt` & `best_loc.pt`
