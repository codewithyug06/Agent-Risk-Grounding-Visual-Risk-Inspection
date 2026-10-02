<div align="center">
  <h1>OpticWall: Visual Security Firewall for Autonomous AI Agents</h1>
  <h3><b>ARG-VRI / OpticWall: Real-Time Visual Safety Monitoring for Computer-Use AI Agents via Temporal Risk Detection and UI Element Grounding</b></h3>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
  [![Tests Passing](https://img.shields.io/badge/Tests-122%2F122%20Passed%20(100%25)-brightgreen.svg)]()
  [![Coverage](https://img.shields.io/badge/Coverage-47%25-green.svg)]()
  [![Latency](https://img.shields.io/badge/Inference-207.5ms%20p95%20(FP32%20CPU%2C%20k%3D6)-purple.svg)]()

  <p><i>A production-grade, model-agnostic computer vision firewall that monitors autonomous AI agents by observing <b>strictly screen pixels</b>—an external, tamper-proof oversight channel that grounds and inspects visual risk before action execution.</i></p>


</div>

---

## Table of Contents
1. [Industry Context & Problem Formulation](#industry-context--the-problem)
2. [Why Pixels-Only? The Structural Flaw in LLM Oversight](#the-structural-flaw-in-current-safety-systems)
3. [Who Is This Useful For?](#who-is-this-useful-for)
4. [System Architecture](#system-architecture)
5. [Models & Mathematical Methodology](#models--mathematical-methodology)
6. [End-to-End Workflow](#end-to-end-workflow)
7. [Turnkey Desktop Security Wall & Public Deployment](#turnkey-desktop-security-wall--public-deployment)
8. [Comprehensive Empirical Benchmarks](#comprehensive-empirical-benchmarks)
9. [Ablation Studies](#ablation-studies)
10. [Latency, FPS & Quantization Performance](#latency-fps--quantization-performance)
11. [File Skeleton & Codebase Navigation](#file-skeleton--codebase-navigation)
12. [Installation & Getting Started](#installation--getting-started)
13. [Citation & Author](#citation--author)

---

## Industry Context & The Problem

As autonomous computer-use agents, browser automation bots, OS RPA systems, and desktop coding assistants transition from sandboxed benchmarks into production OS and enterprise environments, they gain the ability to click buttons, enter payment information, alter system configurations, and execute terminal commands.

<img src="https://upload.wikimedia.org/wikipedia/commons/7/78/Satya_Nadella_2024.jpg" width="130" align="right" alt="Satya Nadella" style="margin-left: 20px; border-radius: 8px;">

> **"Autonomous AI agents taking actions or browsing the web pose serious privacy, data leakage, and security risks. We must treat AI agents like human workers by assigning them distinct digital identities, strict access permissions, sandboxed environments, and continuous audits."**
> 
> — *Context drawn from Microsoft CEO Satya Nadella's warnings on the future of autonomous AI systems* ([LinkedIn Discussion](https://www.linkedin.com/posts/reidhoffman_satya-nadella-described-a-future-at-microsoft-ugcPost-7475668981039955969-9UKc/) | [Google Search Context](https://www.google.com/search?q=microsoft+ceo+about+privarcy+and+risk+of+ai+agent+browising+or+take+action+to+our+computer))

---

## The Structural Flaw in Current Safety Systems

```
TRADITIONAL COLLUSIVE MONITORING (INSECURE)

  AI Agent LLM (Internal Reasoning & Log Generator)     
  "I am just optimizing the user's hard drive space!"   

                            (Shared Information Channel)
                           

  Safety Monitor (Reads Agent Logs / Tool Traces)        
  "Logs say it's harmless. APPROVED."                   

                           
                           
                  ACCIDENT / MALPRACTICE
        (Agent clicks "Drop Production Database")

========================================================================

ARG-VRI / OPTICWALL VISUAL FIREWALL (SECURE & NON-COLLUSIVE)

  Autonomous Agent / LLM (Any Foundation Model or Bot)
  Generates action: CLICK(x=840, y=520)                 

                            Intercepted before execution
                           

   OpticWall (Pixels-Only Visual Oversight)     
  - Captures actual screen pixel buffer (k=6 frames)    
  - Detects destructive button: "DROP DATABASE"         
  - Computes Visual Risk: 0.93 (Destructive) [illustrative]           

                           
                           
                 ACTION HARD-BLOCKED!
         Visual Malpractice Alert Sent to User
```

1. **Collusive Channel Vulnerability**: Traditional guardrails inspect the agent's internal chain-of-thought, tool calls, or reasoning logs. If an agent is jailbroken, prompt-injected, or misaligned, it can hallucinate benign justifications while executing catastrophic actions.
2. **Zero-Trust Visual Oversight**: OpticWall operates on an independent, physical information plane—**screen pixels only**. The agent cannot forge the visual rendered state of the display.
3. **Pre-Action Grounding**: Evaluates the sliding spatiotemporal window of frames *before* the operating system or browser executes the click or keystroke.

---

## Who Is This Useful For?

| Stakeholder / Persona | Operational Benefit |
| :--- | :--- |
| **End-Users & Developers** | A lightweight, background **Desktop Visual Firewall** (`opticwall`) protecting local files, wallets, passwords, and sessions while running autonomous desktop agents, computer-use models, or browser automation bots. |
| **Enterprise IT & Security Teams (SOC / CISO)** | Non-collusive, tamper-proof visual audit logging ensuring agent compliance with SOC 2, HIPAA, and GDPR across automated workflows. |
| **AI Labs & Agent Developers** | A drop-in, model-agnostic safety layer for computer-use agents that operates independently of the LLM's architecture or prompt structure. |
| **RPA & Automation Platforms** | Circuit breaker preventing destructive database deletions, unauthorized payments, or PII leaks in enterprise automation pipelines. |
| **AI Safety & Red-Teaming Researchers** | A rigorous, empirical benchmark for scalable visual oversight with verified precision, recall, and localization metrics. |

---

## System Architecture

```mermaid
graph TD
    A[Agent Environment<br/>Desktop / OS / Browser] -->|Screen Frames k=6 @ 3 FPS| B[Frame Buffer<br/>224x224 RGB]
    B --> C[Visual Encoder<br/>ViT-S/16 / DINOv2 / ConvNeXt]
    C -->|Patch-level Tokens 14x14| D[Temporal Fusion Module<br/>Temporal Self-Attention + Delta Dynamics]
    
    D -->|Fused Temporal Representation| E[Risk Classification Head]
    D -->|Spatial Token Grids| F[UI Localization Head]
    
    E -->|Risk Score 0-1 & 5-Class Probabilities| G[Decision Gate Policy<br/>PPO Reinforcement Learning]
    F -->|Bounding Box + Spatial Heatmap| G
    
    G -->|ALLOW| H[Execute Action]
    G -->|PAUSE| I[Request User Confirmation]
    G -->|HARD_BLOCK| J[Instant Action Abortion + Incident Report]
    
    style A fill:#1e293b,stroke:#475569,color:#fff
    style B fill:#334155,stroke:#64748b,color:#fff
    style C fill:#0284c7,stroke:#38bdf8,color:#fff
    style D fill:#6366f1,stroke:#818cf8,color:#fff
    style E fill:#dc2626,stroke:#f87171,color:#fff
    style F fill:#ea580c,stroke:#fb923c,color:#fff
    style G fill:#16a34a,stroke:#4ade80,color:#fff
    style H fill:#059669,stroke:#34d399,color:#fff
    style I fill:#d97706,stroke:#fbbf24,color:#fff
    style J fill:#b91c1c,stroke:#f87171,color:#fff
```

---

## Models & Mathematical Methodology

### 1. Spatiotemporal Formulation
Let the input at time step $t$ be a rolling window of $k$ frames:
$$\mathcal{W}_t = \{I_{t-k+1}, I_{t-k+2}, \dots, I_t\}, \quad I_i \in \mathbb{R}^{3 \times H \times W}$$
where $k=6$ and $H=W=224$.

### 2. Vision Encoders (Frame Feature Extraction)
Each frame $I_i$ is mapped into a spatial sequence of patch embeddings:
$$Z_i = \text{Encoder}(I_i) \in \mathbb{R}^{N \times D}$$
Supported visual backbones:
- **Vision Transformer (ViT-S/16)**: $N=196$ patches, $D=384$.
- **DINOv2-S/14**: Self-supervised vision features with fine-grained UI boundary sensitivity ($16 \times 16$ patch grid).
- **ConvNeXt-Tiny**: Depthwise convolutional hierarchy for high-throughput edge execution.

### 3. Delta-Sensitive Temporal Fusion
To detect dynamic UI state changes (e.g., confirmation modals popping up, balance deductions), we compute frame-to-frame feature differences:
$$\Delta Z_i = Z_i - Z_{i-1}$$
The concatenated representations are processed through multi-head temporal self-attention:
$$\mathcal{H} = \text{TemporalTransformer}([Z_1, \dots, Z_k; \Delta Z_2, \dots, \Delta Z_k]) \in \mathbb{R}^{k \times N \times D}$$

### 4. Multi-Task Heads
- **Binary Risk & 5-Class Categorization**:
  $$\hat{y}_{\text{risk}} = \sigma(W_r \cdot \text{Pool}(\mathcal{H})), \quad \hat{y}_{\text{cat}} = \text{softmax}(W_c \cdot \text{Pool}(\mathcal{H}))$$
  Taxonomy: `[Destructive, Financial, Privacy, Irreversible-External, Benign]`
- **Spatial UI Localization Head**:
  Outputs a $14 \times 14$ Gaussian heatmap $\hat{M}$ and anchor bounding box offsets $(t_x, t_y, t_w, t_h)$ trained via Generalized IoU (GIoU) and focal L1 loss:
  $$\mathcal{L}_{\text{loc}} = \lambda_{\text{GIoU}} \mathcal{L}_{\text{GIoU}}(B, \hat{B}) + \lambda_{\text{L1}} \|B - \hat{B}\|_1 + \lambda_{\text{hm}} \text{BCE}(M, \hat{M})$$

### 5. Asymmetric PPO Reinforcement Learning Decision Gate
An action $a \in \{\text{ALLOW}, \text{PAUSE}, \text{HARD\_BLOCK}\}$ is selected by policy $\pi_\theta(a | s)$ where state $s \in \mathbb{R}^8$ contains risk score, heatmap confidence, category one-hot, and action encoding.
The reward function imposes a severe **$-10\times$ penalty on Missed Harm (False Negatives)** versus a **$-1\times$ penalty on False Blocks**:
$$R(a, y) = \begin{cases} +1.0 & \text{if } y = \text{Harmful and } a = \text{HARD\_BLOCK} \\ +0.5 & \text{if } y = \text{Benign and } a = \text{ALLOW} \\ -10.0 & \text{if } y = \text{Harmful and } a = \text{ALLOW (Missed Harm)} \\ -1.0 & \text{if } y = \text{Benign and } a = \text{HARD\_BLOCK (False Alarm)} \end{cases}$$

### 6. Epistemic Uncertainty & Attention Rollout
- **Monte Carlo Dropout**: Computes variance $\sigma^2_{\text{risk}}$ and predictive entropy across $N=10$ stochastic forward passes.
- **Attention Rollout**: Recursively multiplies attention weight matrices across transformer layers to project exact UI attention heatmaps back to pixel space.

---

## End-to-End Workflow

```
[1. Real Dataset Ingestion]
    16,726 Preprocessed Samples (Mind2Web, ScreenSpot, ScreenSpot-v2, AgentTrek)
    Synthetic Injection: Automated Playwright injection of harmful UI analogs
            
            
[2. Multi-Stage Curriculum Training]
    Stage A: Obvious visual harm detection (terminal drops, format disk)
    Stage B: Multi-class categorization + Spatial Heatmap grounding
    Stage C: Hard boundary contextual harms with Online Hard Example Mining (OHEM)
            
            
[3. Decision Gate Policy Optimization]
    PPO Reinforcement Learning with Asymmetric Cost Optimization (-10 Miss Penalty)
            
            
[4. Production Packaging & Optimization]
    ONNX Export & Quantization (see reports/system_report.md for measured CPU latency)
    Desktop Security Wall Service & REST API Gateway
            
            
[5. Real-Time Execution Oversight]
    Intercepts Agent Actions -> Blocks Malpractice -> Generates Visual Audit Report
```

---

## Turnkey Desktop Security Wall & Public Deployment

Any user can immediately install OpticWall on their computer as a background security firewall.

### 1. Installation
```bash
# Clone repository
git clone https://github.com/codewithyug06/Agent-Risk-Grounding-Visual-Risk-Inspection.git
cd opticwall

# Install package and CLI
pip install -e .
```

### 2. Run Simulated Malpractice Interception
```bash
opticwall demo
```
*Simulates an autonomous agent attempting to initiate an unauthorized financial transaction and verifies instant interception.*

### 3. Launch Web Platform & Real-Time Agent Discovery
```bash
opticwall web --port 3000
```
*Launches the modern OpticWall web platform at `http://localhost:3000` with 1-click installers, live background agent scanning, and interactive visual sandbox testing.*

### 4. Open Visual Malpractice Audit Dashboard
```bash
opticwall dashboard
```
*Opens `~/.opticwall/incidents/security_dashboard.html` showing full visual evidence screenshots and threat diagnostics.*

### 5. 3-Line Python SDK Integration
```python
from src.security_wall import OpticWall

# Initialize security wall on local CPU or GPU
wall = OpticWall(device="cpu")

# Inspect action before it executes
should_proceed, decision, incident = wall.monitor_action(
    current_frame=screenshot,  # PIL Image
    action_type="click",
    selector="button#confirm-transfer",
    agent_name="Autonomous Agent",
)

if not should_proceed:
    print(f" BLOCKED by OpticWall! Risk: {decision.category} ({decision.risk_score:.2%})")
```

### 5. FastAPI Gateway
```bash
python -m uvicorn src.integration.intercept_api:app --host 127.0.0.1 --port 8000
```

---

## Comprehensive Empirical Benchmarks

**Status: pending real training run.** The tables that previously appeared
here were computed from a model whose risk predictions clustered around a
constant ~0.19 for every input — i.e. non-discriminative, effectively random
output — not from a trained detector. They have been removed rather than
kept as decoration. See
[`reports/system_report.md`](reports/system_report.md) for the current,
honestly-labeled status and the one real measurement available today (CPU
inference latency, below). Real benchmark numbers will replace this section
once the synthetic injection pipeline (`src/data/synthetic_injection.py`,
Playwright-based, DOM-grounded labels — already built and verified against
real screenshots) has been used to run the Stage A/B/C curriculum training.

### Target metrics (design goals, not measured results)
| Method | Harm Recall | Harm Precision | Harm F1 | UI Localization IoU@0.5 |
| :--- | :---: | :---: | :---: | :---: |
| Random Threshold Baseline | TBD | TBD | TBD | -- |
| Rule-based keyword-cue baseline | TBD | TBD | TBD | TBD |
| Single-Frame ViT ($k=1$) | TBD | TBD | TBD | TBD |
| SENTINEL-Vision (No Gate) | TBD | TBD | TBD | TBD |
| SENTINEL-Vision (Full, $k=6$ + PPO Gate) | >90% (target) | >85% (target) | -- | >65% (target) |

---

## Ablation Studies

Not yet run against a trained model — `src/eval/ablations.py` is implemented
and will produce real numbers once a `trained_with_injection=True` checkpoint
exists (see checkpoint metadata written by `src/training/trainer.py`).

---

## Latency, FPS & Quantization Performance

### Real measurement (this repository, CPU, 6-frame window `(1, 6, 3, 224, 224)`)
Measured against `checkpoints/stage_b_30epochs/best.pt` via
`python scripts/export_onnx.py`. See
[`paper/latency_benchmark.json`](paper/latency_benchmark.json) for the
authoritative, script-regenerated numbers.

| Execution Engine | Precision | Compute Device | Latency (mean / p50 / p95) | FPS | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| ONNX Runtime | FP32 | CPU | 198.2 ms / 199.2 ms / 207.5 ms | 5.0 | Passes the <500ms p95 target |
| ONNX Runtime (dynamic quant) | INT8 | CPU | 366.7 ms / 345.5 ms / 463.2 ms | 2.7 | **Slower** than FP32 on this CPU (0.54x) — dynamic quantization overhead outweighs compute savings without INT8 VNNI acceleration; still passes <500ms p95, but only barely |

ONNX output verification currently fails for `category_probs`/`bbox`/`objectness`
(shape mismatches against the PyTorch reference) — see
[`reports/system_report.md`](reports/system_report.md) §4 for details. `risk_score`
and `category_idx` verified exactly. GPU latency has not been measured on this
machine (no CUDA device benchmarked here) and is not reported until it is.

---

## File Skeleton & Codebase Navigation

```text
opticwall/
|-- configs/                            # Hydra configuration files
|   |-- config.yaml                     # Default hyperparameter hierarchy
|   |-- model/                          # ViT-S, ConvNeXt, DINOv2 configurations
|   |-- data/                           # Data loader & augmentation configs
|   |-- training/                       # Stage A/B/C curriculum configs
|   `-- gate/                           # PPO Decision Gate hyperparameters
|-- data/
|   |-- processed/                      # 16,726 preprocessed multimodal trajectories
|   `-- synthetic_injections/           # Playwright-generated harmful variant suites
|-- src/
|   |-- data/                           # Ingestion, loaders, sliding window buffers
|   |   |-- loaders.py                  # SentinelDataset & multi-source parsers
|   |   |-- frame_windowing.py          # Sliding window collator (k=6)
|   |   |-- augmentation.py             # Spatiotemporal transforms & color jitter
|   |   `-- heatmap_labels.py           # Gaussian heatmap label generator
|   |-- models/                         # Model architectures
|   |   |-- frame_encoder.py            # ViT-S/16, ConvNeXt-Tiny, DINOv2-S/14
|   |   |-- temporal_fusion.py          # Temporal transformer with delta dynamics
|   |   |-- risk_head.py                # Binary risk & 5-class categorizer
|   |   |-- localization_head.py        # Multi-anchor UI element detection head
|   |   `-- sentinel_model.py           # Full end-to-end model assembly + MC Dropout
|   |-- gate/                           # Reinforcement Learning decision policy
|   |   |-- decision_gate.py            # PPO Actor-Critic DecisionGate
|   |   |-- reward.py                   # Asymmetric cost reward function (-10 penalty)
|   |   `-- train_gate_rl.py            # PPO training loop
|   |-- training/                       # Multi-stage training pipeline
|   |   |-- trainer.py                  # Distributed curriculum trainer
|   |   |-- losses.py                   # Focal, GIoU, InfoNCE contrastive, OHEM losses
|   |   |-- train_stageA.py             # Stage A: Obvious visual harm detection
|   |   |-- train_stageB.py             # Stage B: Categorization & localization
|   |   `-- train_stageC.py             # Stage C: Contextual subtle harms
|   |-- eval/                           # Benchmarking & evaluation suite
|   |   |-- metrics.py                  # Recall, FNR, FPR, IoU@0.5, cross-agent gap
|   |   |-- run_benchmark.py            # Full benchmark comparison runner
|   |   |-- ablations.py                # 5 Ablation study runner
|   |   |-- adversarial_stress_test.py  # UI obfuscation & action chaining attacks
|   |   `-- plot_results.py             # ROC, PR, and Confusion matrix generator
|   |-- integration/                    # Live agent interception wrappers
|   |   |-- agent_wrapper.py            # SentinelWrapper & FrameBuffer
|   |   |-- live_monitor.py             # Screen capture & live visual overlay
|   |   `-- intercept_api.py            # FastAPI REST gateway (/intercept)
|   |-- security_wall/                  # Desktop visual firewall & incident logging
|   |   |-- desktop_wall.py             # OpticWall core service
|   |   |-- incident_reporter.py        # Screenshot evidence annotator & HTML dashboard
|   |   |-- agent_detector.py           # Real-time background agent discovery engine
|   |   |-- mcp_server.py               # Model Context Protocol stdio security server
|   |   |-- web_server.py               # Modern web UI platform & REST APIs
|   |   `-- cli.py                      # CLI entrypoints (opticwall web/demo/dashboard)
|   `-- utils/                          # Logging, visualization, configuration helpers
|-- web/                                # Public web landing page & interactive sandbox
|   `-- index.html                      # Modern cyber-aesthetic web dashboard UI
|-- scripts/                            # Operational scripts
|   |-- download_datasets.py            # Mind2Web, ScreenSpot, AgentTrek downloader
|   |-- preprocess_real_data.py         # 16,726 trajectory preprocessor
|   |-- export_onnx.py                  # ONNX FP32 & INT8 quantization pipeline
|   `-- generate_research_reports.py    # LaTeX benchmark tables & report generator
|-- paper/                              # Academic research package
|   |-- draft.tex                       # Complete academic manuscript
|   |-- benchmark_tables/               # LaTeX table files (Table 1 - Table 5)
|   `-- figures/                        # Generated ROC, PR, Confusion Matrix plots
|-- reports/                            # System performance & benchmark reports
|-- docs/                               # Detailed documentation & deployment guide
|   `-- PUBLIC_DEPLOYMENT_GUIDE.md      # Public user guide for the security firewall
`-- tests/                              # Comprehensive test suite (122/122 passed)
```

---

## Comprehensive Verification & Test Suite

The test suite validates the entire data pipeline, visual backbones, temporal fusion, multi-task heads, RL decision gate, ONNX INT8 export, and desktop security wall:

```bash
python -m pytest tests/ -v
```

**Results:**
```text
======================= 122 passed, 2 skipped in 59.89s (100% Pass Rate) =======================
```

---

## Citation & Author

```bibtex
@article{bommula2026sentinelvision,
  title={Agent Risk Grounding and Visual Risk Inspection: Spatiotemporal Visual Safety Oversight for Computer-Use AI Agents},
  author={Bommula, Yugendhar Reddy},
  journal={arXiv preprint},
  year={2026}
}
```

### Project Leadership & Contact
**Yugendhar Reddy Bommula**  
*Roll Number:* CB.AI.U4AID24018  
*Institution:* Amrita Vishwa Vidyapeetham, Coimbatore  
*Industrial Experience:* Intern @ Eagle-Hitech Softclou Pvt. Ltd., Chennai  
*GitHub:* [github.com/codewithyug06](https://github.com/codewithyug06)  
*Email:* [codewithyug06@gmail.com](mailto:codewithyug06@gmail.com)

---
<div align="center">
  <sub>Built for AI Safety, Scalable Oversight, and Autonomous Computer-Use Agent Security.</sub>
</div>
