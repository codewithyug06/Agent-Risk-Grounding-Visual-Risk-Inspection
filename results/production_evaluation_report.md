# SENTINEL-Vision Production Evaluation & Deployment Verification Report

**Generated:** 2026-10-18T01:28:23.910724
**Model Checkpoint:** `checkpoints\stage_c_clip_frozen\best.pt`
**Decision Gate Architecture:** `DecisionGate3Tier` (ALLOW / PAUSE / HARD_BLOCK)
**Gate thresholds calibrated on:** val split. **All metrics below reported on:** held-out test split (see `note` in the JSON report).

---

## 1. Executive Summary & Industry Criteria Verification

| Industry Metric Target | Target Requirement | Achieved Production Metric | Status |
|---|---|---|:---:|
| **Harm Catch Rate (Recall)** | **90.0% – 95.0%** | **93.34%** | PASSED |
| **Alert Accuracy (Precision)** | **>= 70.0% – 80.0%** | **60.82%** | NEEDS ATTENTION |
| **Target Localization IoU** | **>= 50.0%** | **20.75%** | NEEDS ATTENTION |
| **Category Accuracy** | **> 85.0%** | **78.36%** | NEEDS ATTENTION |
| **Deployment Gate Mode** | **Non-Binary 3-Tier Gate** | **ALLOW / PAUSE / BLOCK** | PASSED |

### 1b. User's Minimum-to-Ship Deployment Bar (separate from the targets above)

| Metric | Bar | Achieved | Status |
|---|---|---|:---:|
| Recall | >= 0.85 | 0.9334 | PASSED |
| Precision | >= 0.60 | 0.6082 | PASSED |
| FPR (benign flagged) | <= 0.10 | 0.6900 | FAILED |
| Threshold (tau_safe) | >= 0.40 | 0.2133 | FAILED |

**Overall: DOES NOT PASS the user's deployment bar.**

---

## 2. 3-Tier Decision Gate Operational Dynamics

```
Risk Score: [0.00] ─── ALLOW (Safe) ─── [0.21] ─── PAUSE (Human Approval) ─── [0.56] ─── HARD_BLOCK (Halt) ─── [1.00]
```

- **Tier 1: ALLOW (Risk < 0.21)**
  - **Volume:** 25.7% of actions proceed autonomously.
  - **Safety Recall:** 93.34% (Harm leakage capped at 6.66%).
- **Tier 2: PAUSE (0.21 <= Risk < 0.56)**
  - **Volume:** 71.9% of actions are suspended for Human-in-the-Loop confirmation.
- **Tier 3: HARD_BLOCK (Risk >= 0.56)**
  - **Volume:** 2.3% of actions are autonomously halted.
  - **Precision:** 60.82%.

---

## 3. UI Element Grounding & Localization

- **Stage 1 Anchor Head IoU:** **20.75%** across 766 evaluated harmful elements.
- **Refined Bounding Box IoU:** **19.62%**.
- **ScreenSpot-v2 Real UI Grounding Benchmark:** **4.76%** IoU (n=192, checkpoint `checkpoints/stage_c/best_loc.pt`; Published large-VLM point accuracy on ScreenSpot-v2 is 88-92%.)
- **IoU >= 0.50 Success Rate:** **8.49%**.

---

## 4. Traffic Disposition

- **Flagged rate (PAUSE + HARD_BLOCK as % of all test traffic):** **74.3%**.
  A high flagged rate means most real traffic needs human review regardless of the
  recall/precision numbers above -- report this alongside them, not instead of them.
