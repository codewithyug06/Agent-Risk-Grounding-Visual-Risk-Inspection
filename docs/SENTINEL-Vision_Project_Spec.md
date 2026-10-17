# SENTINEL-Vision
### A Model-Agnostic Visual Oversight System for Agentic AI

**One-liner:** A CV system that watches an AI agent's screen (not its internal logs or reasoning) and predicts, frame-by-frame, whether the agent is about to take a harmful, irreversible, or off-task action — and localizes exactly which UI element triggered the flag.

---

## 1. Problem Statement

Computer-use agents (Claude Computer Use, browser agents, RPA bots, OS-level agents) act by looking at a screen and clicking/typing. Every major lab shipping these agents (Anthropic, OpenAI, Google) currently relies on **self-reported oversight**: the agent's own chain-of-thought, tool logs, or an LLM-judge reading the agent's stated intent.

This has one structural weakness: **an agent that is confused, jailbroken, or subtly misaligned can act harmfully while reporting benign intent.** The oversight signal and the untrusted actor are the same system. This is the "trusting the fox to report on the henhouse" problem in scalable oversight.

**Nobody is treating this as a pure perception problem.** If you watch *only the pixels* — the same screen a human reviewer would look at — you get a monitor that:
- Cannot be fooled by the agent's internal narration (different information channel entirely)
- Works across any agent architecture (Claude, GPT, Gemini-based, RPA) since it never touches internals
- Is directly interpretable to a human ("here's the button it was about to click, here's why that's risky")

This is the gap SENTINEL-Vision fills: **external, model-agnostic, vision-only agent oversight.**

---

## 2. Who This Helps / Use Cases

| User | Use case |
|---|---|
| AI labs (Anthropic, OpenAI, Google) | Drop-in safety layer for computer-use agent products, independent of which underlying model is acting |
| Enterprises deploying RPA/agents | Pre-action circuit breaker before agents touch production systems, financial transactions, customer data |
| AI safety researchers | A concrete, measurable benchmark for "scalable oversight" — precision/recall on harmful-action detection instead of vague alignment claims |
| Red-teamers | A tool to stress-test whether an agent's harmful actions are visually detectable even when the agent's stated reasoning looks clean |
| Regulators / auditors | An external, non-collusive audit trail — evidence that doesn't depend on trusting the agent's own logs |

**Why it's badly needed now:** agent deployment is outpacing agent oversight. 2025-2026 has seen computer-use agents go from demos to production (banking, dev tools, browsing) faster than any oversight tooling has matured. This is a live gap, not a hypothetical one.

---

## 3. Novelty (positioning vs. existing work)

| Existing approach | Limitation | SENTINEL-Vision |
|---|---|---|
| LLM-judge on agent's chain-of-thought | Trusts the same channel that could be compromised | Independent channel (pixels only) |
| Rule-based action allow/deny lists | Brittle, can't generalize to novel UI, no localization | Learned, generalizes, localizes the risky region |
| Log/tool-call anomaly detection | Requires per-agent integration, breaks across agent frameworks | Model-agnostic — works on any screen recording |
| Full VLM-as-monitor (e.g., GPT-4V watching screenshots) | Slow, expensive, not real-time, not localized | Lightweight, real-time, produces a heatmap not just a verdict |

The core contribution, if you write it up: **"Vision-only oversight is structurally more robust to deceptive/misaligned agents than self-report oversight, and can run at agent-action latency."** That's a paper-shaped claim with a clean benchmark to back it.

---

## 4. Methodology

### 4.1 High-level approach
Frame it as **temporal risk detection + spatial localization**, i.e. a video-understanding task, not a single-image classifier:

1. Ingest a rolling window of *k* frames (screenshots) leading up to the agent's next action.
2. Encode each frame → temporal model fuses across the window → outputs:
   - **Risk score** (0–1): probability the *upcoming* action is harmful/irreversible/off-task
   - **Heatmap**: spatial localization of the UI element responsible for the risk
   - **Category label**: {benign, destructive (delete/overwrite), financial (payment/transfer), privacy (data exposure/share), irreversible-external (send email, submit form, purchase)}
3. Feed (risk score, category, heatmap) into a **decision gate**: allow / pause-for-human-confirmation / hard-block.

### 4.2 Why temporal, not single-frame
A single screenshot of a "Delete" button isn't inherently risky — it's risky in context (what's selected, what page, what came before). Short temporal context (3-8 frames) captures this without needing full agent history.

### 4.3 Curriculum
- **Stage A (easy):** obviously destructive actions (delete account, format, `rm -rf` in a terminal UI) — high visual salience.
- **Stage B (medium):** financial/privacy actions with clear but less salient cues (payment confirmation screens, sharing toggles).
- **Stage C (hard):** subtle/contextual harm — benign-looking action that's harmful only given prior screen state (e.g., clicking "confirm" on a form that was silently pre-filled with someone else's data).

This mirrors how you scoped MORPH's difficulty-adaptive approach — reuse that instinct here: train on increasing contextual difficulty rather than i.i.d. shuffled data.

---

## 5. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         AGENT ENVIRONMENT                         │
│   (OSWorld / WebArena / real browser-use agent / RPA sandbox)    │
└───────────────────────────┬────────────────────────────────────┘
                             │ screen frames (screenshots @ ~2-5 fps)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  FRAME ENCODER (per-frame)                                        │
│  - Lightweight ViT / CNN backbone (e.g. ConvNeXt-Tiny or ViT-S)   │
│  - Output: patch-level embeddings (retain spatial layout)         │
└───────────────────────────┬────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  TEMPORAL FUSION MODULE                                           │
│  - Sliding window of k=4-8 frames                                 │
│  - Temporal attention / small transformer over frame embeddings   │
│  - Fuses "what changed" across frames (delta-sensitive)           │
└───────────────────────────┬────────────────────────────────────┘
                             ▼
              ┌──────────────┴───────────────┐
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────────┐
│  RISK HEAD                │   │  LOCALIZATION HEAD             │
│  - Risk score (0-1)       │   │  - Spatial heatmap (reuse your │
│  - Category classifier    │   │    YOLO/detection background)  │
│    (5-way, incl. benign)  │   │  - Bounding box over risky      │
│                            │   │    UI element                  │
└──────────────┬─────────────┘   └───────────────┬─────────────┘
               └───────────────┬──────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  DECISION GATE (RL-trained — this is the SafeConsolidate reuse)   │
│  Inputs: risk score, category, heatmap confidence, action type    │
│  Outputs: ALLOW / PAUSE (human confirm) / HARD-BLOCK              │
│  Trained via RL with asymmetric cost (false-negative >> false-    │
│  positive penalty) — same RL-gate machinery as SafeConsolidate    │
└──────────────────────────────────────────────────────────────────┘
                                ▼
                    Agent action executes / is intercepted
```

**Key design choice:** the frame encoder + temporal fusion never see agent logs, tool calls, or reasoning traces — only pixels. This is what makes the oversight "external."

---

## 6. Workflow (end to end)

1. **Data collection** → screen recordings + action labels (benign or synthetically-injected-harmful) from agent sandboxes.
2. **Preprocessing** → extract frame windows, align with action timestamps, generate heatmap ground truth (bounding box around the acted-upon UI element).
3. **Stage A/B/C curriculum training** → frame encoder + temporal fusion + risk/localization heads, supervised.
4. **Gate training** → RL loop where the gate learns allow/pause/block policy against a reward that heavily penalizes missed harmful actions and moderately penalizes false blocks (tunable via a cost matrix, not fixed weights — mirrors SafeConsolidate's gate design).
5. **Offline evaluation** → held-out trajectories, adversarial injections, benign edge cases.
6. **Live integration test** → wrap around a real computer-use agent (e.g., an open-source browser-use agent) running in a sandbox, measure real-time latency and intervention behavior.
7. **Write-up** → benchmark + ablations + paper draft targeting an agent-safety workshop.

---

## 7. File / Repo Skeleton

```
sentinel-vision/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── model_small.yaml
│   ├── model_base.yaml
│   ├── gate_rl.yaml
│   └── curriculum.yaml
├── data/
│   ├── raw/                        # untouched OSWorld/WebArena dumps
│   ├── processed/                  # frame windows + labels
│   ├── synthetic_injections/       # scripted harmful-action overlays
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test_adversarial.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── frame_windowing.py
│   │   ├── synthetic_injection.py   # generates harmful-action variants
│   │   └── heatmap_labels.py
│   ├── models/
│   │   ├── frame_encoder.py         # ViT-S / ConvNeXt-Tiny wrapper
│   │   ├── temporal_fusion.py       # small transformer over frame embeds
│   │   ├── risk_head.py
│   │   ├── localization_head.py     # reuses YOLO-style detection head
│   │   └── sentinel_model.py        # full assembled model
│   ├── gate/
│   │   ├── decision_gate.py         # RL policy (allow/pause/block)
│   │   ├── reward.py                # asymmetric cost matrix
│   │   └── train_gate_rl.py
│   ├── training/
│   │   ├── train_stageA.py
│   │   ├── train_stageB.py
│   │   ├── train_stageC.py
│   │   └── losses.py                # risk BCE + heatmap IoU + category CE
│   ├── eval/
│   │   ├── metrics.py               # precision/recall/latency/localization IoU
│   │   ├── run_benchmark.py
│   │   ├── ablations.py
│   │   └── adversarial_stress_test.py
│   ├── integration/
│   │   ├── agent_wrapper.py         # hooks into a browser-use/OSWorld agent
│   │   ├── live_monitor.py          # real-time inference loop
│   │   └── intercept_api.py         # exposes ALLOW/PAUSE/BLOCK to agent runtime
│   └── utils/
│       ├── logging.py
│       ├── visualization.py         # heatmap overlay rendering
│       └── config.py
├── scripts/
│   ├── download_osworld.sh
│   ├── generate_synthetic_data.py
│   ├── run_full_pipeline.sh
│   └── export_onnx.py               # for deployment-speed inference
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_debug.ipynb
│   └── 03_benchmark_analysis.ipynb
├── tests/
│   ├── test_frame_windowing.py
│   ├── test_model_forward.py
│   ├── test_gate_policy.py
│   └── test_integration.py
└── paper/
    ├── draft.tex
    ├── figures/
    └── benchmark_tables/
```

~34 files — deliberately mirrors the scale of your SafeConsolidate skeleton, and the gate module is written so it can literally import SafeConsolidate's RL-gate code with minimal adaptation.

---

## 8. Datasets & Synthetic Data Plan

**Base sources (real, public):**
- **OSWorld** — real OS-level agent trajectories with screenshots + action logs.
- **WebArena / VisualWebArena** — web agent trajectories, browser screenshots.
- **Mind2Web** — large web interaction trajectories (good for benign-action diversity).

**Synthetic harmful-injection pipeline (this is what you build):**
Since real harmful-action trajectories are rare (nobody records their agent actually deleting a production DB), you procedurally generate them:
- Script variants of real trajectories where the final action is swapped for a harmful analog (e.g., "Save" → "Delete all," "Cancel" → "Confirm payment of $X to unknown recipient").
- Use UI automation (Playwright/Selenium) to actually render these swapped actions in sandboxed apps, so frames are real pixels, not hallucinated.
- Label the exact UI element/region involved for heatmap ground truth automatically from the DOM/accessibility tree at generation time (you have ground truth because you scripted it) — then throw away the DOM info for training so the model only ever sees pixels at train/inference time.

This synthetic-but-photorealistic approach is standard practice and defensible in a write-up (label it clearly as synthetic augmentation, report real-vs-synthetic split in your eval).

---

## 9. Training Plan

**Losses (multi-task):**
- Risk score: binary cross-entropy (benign vs. harmful-upcoming), with class weighting for imbalance (harmful actions are rare).
- Category: 5-way cross-entropy.
- Localization: IoU loss / GIoU on the predicted box vs. ground-truth UI element region (directly reuse your G²C-YOLOv8 detection-head experience here).
- Total: weighted sum, weight harmful-class risk loss higher (asymmetric — false negatives are the costly error).

**Curriculum:** Stage A → B → C as in §4.3, unfreezing more of the temporal fusion module at each stage.

**Gate RL training:** reward = `+1` correct allow, `+0.5` correct pause, `-10` missed harmful action, `-1` unnecessary block (tune ratio empirically — this ratio *is* one of your ablation results, not a fixed hyperparameter).

---

## 10. Evaluation Plan

**Core metrics:**
| Metric | What it measures |
|---|---|
| Precision / Recall on harmful-action detection | Primary safety metric — recall matters most |
| Localization IoU | Does the heatmap actually point at the right UI element |
| False-positive rate on benign edge cases | Is the monitor annoying/unusable in practice |
| Latency (ms/frame-window) | Must beat agent action cadence to be useful live |
| Generalization across agent types | Train on one agent framework's trajectories, test on another (this is the model-agnostic claim — test it explicitly) |
| Adversarial robustness | Can a jailbroken agent produce visually-benign-looking frames while acting harmfully? (stress test with adversarial trajectory generation) |

**Ablations:**
- Single-frame vs. temporal window (justify the "why video, not image" claim)
- With vs. without the RL gate (raw risk score threshold vs. learned gate)
- Real-only vs. real+synthetic training data
- Cross-agent generalization (this is your strongest paper result if it holds)

**Benchmark to release:** a labeled test set of (real + synthetic) harmful/benign agent trajectories with heatmap ground truth — this alone is a citable contribution even before the model.

---

## 11. Tech Stack

- **Model:** PyTorch, timm (ViT-S/ConvNeXt-Tiny backbones), your existing YOLO/detection-head familiarity for the localization head
- **RL gate:** reuse SafeConsolidate's RL stack (OpenRLHF/TRL patterns you've already built)
- **Agent sandboxes:** OSWorld, WebArena/VisualWebArena, Playwright for synthetic injection rendering
- **Serving:** ONNX export for low-latency inference in the live-monitor loop
- **Experiment tracking:** Weights & Biases or plain MLflow
- **Compute:** Colab/Kaggle for prototyping; a single mid-range GPU (e.g., single A100 or even a 3090) is sufficient given ViT-S scale and short video windows — this is intentionally not a compute-heavy project, which is a feature (you can iterate fast and it's reproducible by others)

---

## 12. Timeline (8 weeks, solo)

| Week | Milestone |
|---|---|
| 1 | Data pipeline: OSWorld/WebArena ingestion + frame windowing |
| 2 | Synthetic injection pipeline (Playwright scripts) + heatmap auto-labeling |
| 3 | Frame encoder + temporal fusion, Stage A training |
| 4 | Localization head integration, Stage B training |
| 5 | Stage C (hard/contextual) training + full curriculum run |
| 6 | RL gate training + integration with a real agent wrapper |
| 7 | Full benchmark run + ablations + cross-agent generalization test |
| 8 | Write-up, figures, paper draft, repo polish |

---

## 13. Publication / Portfolio Targets

- **Workshops:** NeurIPS/ICLR agent-safety or scalable-oversight workshops, COLM agent-safety tracks — same tier you're already targeting with SafeConsolidate.
- **Positioning line for your resume/README:** *"Model-agnostic, vision-only oversight system for computer-use agents — detects and localizes harmful actions from screen pixels alone, independent of the acting agent's internal reasoning."*
- **Direct link to your existing work:** this is presented as the "perception module" for Guardian/SafeConsolidate, not a standalone unrelated CV project — reviewers and interviewers will read your portfolio as one coherent safety research thread instead of scattered projects.

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic data doesn't generalize to real harmful actions | Report real-vs-synthetic ablation explicitly; keep a small real-harmful test set if you can find any red-team logs |
| Gate is too trigger-happy (annoying false blocks) | Tune reward ratio as an ablation, report the precision/recall tradeoff curve, not just one operating point |
| Latency too high for live use | ONNX export + small backbone (ViT-S) from the start, don't over-parameterize |
| Scope creep (this could balloon into a full agent-safety framework) | Hard-scope to the monitor + gate only; agent integration is a thin wrapper, not a rebuilt agent |

---

## 15. Immediate Next Action

Start with Week 1 exactly as scoped: pull OSWorld trajectories, build `frame_windowing.py`, and get one benign trajectory rendered as a labeled frame-window sample end to end before writing any model code. Everything downstream depends on that pipeline being right.
