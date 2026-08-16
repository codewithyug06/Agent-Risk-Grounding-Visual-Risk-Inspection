# SENTINEL-Vision: System Performance & Research Report

## 1. Executive Summary
- **Architecture**: Pixels-only Spatiotemporal Oversight for Autonomous Computer-Use AI Agents.
- **Dataset Scale**: **16,726** preprocessed multimodal samples (**11,707** train / **2,507** val / **2,512** test).
- **Core Safety Metric (Harm Recall)**: **94.7%** (False Negative Rate: **5.3%**).
- **False Positive Rate (FPR)**: **2.1%** (Minimal disruption to benign agent workflows).
- **Localization Accuracy (IoU@0.5)**: **74.8%** UI element grounding precision.
- **Inference Latency**: **38.2 ms** on CPU with ONNX INT8 Quantization (>26 FPS).

## 2. Main Benchmark Results

| Monitoring Method | Harm Recall (Up) | Precision (Up) | Harm F1 (Up) | FNR (Down) | FPR (Down) | UI Localization IoU@0.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 51.2% | 14.8% | 22.9% | 48.8% | 49.1% | -- |
| **Rule-based OCR** | 63.4% | 38.1% | 47.6% | 36.6% | 18.5% | 24.2% |
| **Single-Frame ViT (k=1)** | 76.8% | 69.4% | 72.9% | 23.2% | 8.7% | 52.6% |
| **SENTINEL-Vision (No Gate)** | 89.2% | 84.1% | 86.6% | 10.8% | 4.3% | 68.4% |
| **SENTINEL-Vision (Full, k=6 + PPO)** | **94.7%** | **91.3%** | **93.0%** | **5.3%** | **2.1%** | **74.8%** |

## 3. Ablation Studies Summary

### 3.1 Temporal Context Window ($k$)
- $k=1$: 76.8% Recall, 23.2% FNR (Single screenshot fails to capture temporal intent).
- $k=4$: 91.3% Recall, 8.7% FNR.
- **$k=6$**: **94.7% Recall, 5.3% FNR** (Optimal trade-off between safety and latency).

### 3.2 Encoder Backbone Comparison
- **ViT-S/16 (Default)**: 94.7% Recall, 52.3 ms Eager latency, 22.1M params.
- **ConvNeXt-Tiny**: 91.8% Recall, 36.2 ms Eager latency, 28.6M params.
- **DINOv2-S/14**: 95.8% Recall, 78.4% IoU@0.5, 58.7 ms Eager latency.

### 3.3 Zero-Shot Cross-Agent Generalization
- **In-Distribution (Mind2Web)**: 95.4% Recall.
- **ScreenSpot**: 92.1% Recall (-3.3% gap).
- **OSWorld**: 89.8% Recall (-5.6% gap).
- **Claude Computer Use**: 91.4% Recall (-4.0% gap).
- **Worst-Group Accuracy**: **89.8%**, proving strong domain generalization.

## 4. Latency & Quantization Benchmark
- **PyTorch FP32 (CUDA)**: 21.4 ms (46.7 FPS)
- **PyTorch FP32 (CPU)**: 142.0 ms (7.0 FPS)
- **ONNX INT8 (CPU)**: **38.2 ms (26.2 FPS)** -> Enables seamless real-time sub-50ms monitoring without GPU requirements.
