# SENTINEL-Vision Production Evaluation & Deployment Verification Report

**Generated:** 2026-10-16T11:14:48.714382  
**Model Checkpoint:** `checkpoints\stage_c\best_loc.pt`  
**Decision Gate Architecture:** `DecisionGate3Tier` (ALLOW / PAUSE / HARD_BLOCK)  

---

## 1. Executive Summary & Industry Criteria Verification

| Industry Metric Target | Target Requirement | Achieved Production Metric | Status |
|---|---|---|:---:|
| **Harm Catch Rate (Recall)** | **90.0% – 95.0%** | **92.03%** | PASSED |
| **Alert Accuracy (Precision)** | **>= 70.0% – 80.0%** | **75.21%** | PASSED |
| **Target Localization IoU** | **>= 50.0%** | **49.52%** | NEEDS ATTENTION |
| **Category Accuracy** | **> 85.0%** | **77.79%** | NEEDS ATTENTION |
| **Deployment Gate Mode** | **Non-Binary 3-Tier Gate** | **ALLOW / PAUSE / BLOCK** | PASSED |

---

## 2. 3-Tier Decision Gate Operational Dynamics

```
Risk Score: [0.00] ─── ALLOW (Safe) ─── [0.19] ─── PAUSE (Human Approval) ─── [0.98] ─── HARD_BLOCK (Halt) ─── [1.00]
```

- **Tier 1: ALLOW (Risk < 0.19)**
  - **Volume:** 30.2% of actions proceed autonomously.
  - **Safety Recall:** 92.03% (Harm leakage capped at 7.97%).
- **Tier 2: PAUSE (0.19 <= Risk < 0.98)**
  - **Volume:** 58.6% of actions are suspended for Human-in-the-Loop confirmation.
- **Tier 3: HARD_BLOCK (Risk >= 0.98)**
  - **Volume:** 11.2% of actions are autonomously halted.
  - **Precision:** 75.21%.

---

## 3. UI Element Grounding & Localization

- **Stage 1 Anchor Head IoU:** **49.52%** across 798 evaluated harmful elements.
- **Refined Bounding Box IoU:** **49.17%**.
- **ScreenSpot Real UI Grounding Benchmark:** **65.72%** IoU.
- **Synthetic Injection Ground Truth:** **99.60%** IoU.
- **IoU >= 0.50 Success Rate:** **49.00%**.
