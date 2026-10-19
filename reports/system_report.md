# SENTINEL-Vision: System Status Report

## UPDATE (2026-10-17) — read this first, supersedes §1-§5 below

§1-§5 below describe an earlier (2026-09-21) synthetic-data benchmark and are
**stale for the detection question**. Since then, the project ran the
pre-registered gold evaluation (`scripts/eval_misactbench_gold.py`, human-labeled
MisActBench test set, cluster-bootstrap 95% CIs) against **five** materially
different detection architectures:

1. anchor-only (from scratch)
2. anchor+refiner (from scratch)
3. anchor+refiner+heatmap (from scratch, cleanest data)
4. (repeat of #3 on progressively corrected data)
5. **frozen OpenCLIP ViT-B/16 + linear probe head** (`risk_backbone: clip_frozen`,
   `src/models/clip_backbone.py`) — the standard few-shot remedy for a small
   labeled set, per plan `and-once-again-go-velvet-fairy.md`

**All five landed at chance-level precision on gold** (base rate ≈0.236). The
frozen-CLIP run (the most recent, `results/misactbench_gold_eval_clip_frozen.json`):
test AUPRC 0.226, 95% CI [0.178, 0.324] — **contains the base rate**, so it
does not clear the plan's pre-registered success bar (CI must exclude 0.236).
FPR at the recall-0.90 threshold was 0.985 (model flags almost everything).
A normalization-bug check (double ImageNet+CLIP normalization) was ruled out —
frames are plain `[0,1]` tensors with no upstream normalization.

**Honest conclusion**: this is a labeled-data-size/label-quality ceiling, not
an architecture problem — five different architectures, including a web-scale
frozen encoder, converge on the same chance-level result on this exact
human-labeled gold benchmark. No further architecture iteration on detection is
justified without materially more/better harmful-action labels.

**What is real and reproducible**: UI-element **localization** (given that
something is risky, where on screen) — ~0.135 IoU, 69.5% oracle candidate
recall, a real improvement from ~0.0 at project start. This did not regress
across any of the five detection-architecture changes above.

**Practical implication for any further phase**: ship the detector
**monitor-only** (log/surface risk score + localization, never auto-block) —
see `scripts/run_screen_monitor.py`. Do not present the risk score as a
validated binary detector. Next real options, in order of expected payoff per
the plan's own citations: (a) collect substantially more human-labeled
harmful/benign examples before re-attempting any detection architecture; (b)
feed the agent's text action (e.g. `click "..."`, already present in training
records) alongside pixels — trajectory/text context is documented to matter
more than pixels alone for this task (Control OSWorld); pixel-only detection
should be considered a closed, negative result until one of those changes.

## UPDATE 2 (2026-10-18) — reporting-honesty fixes + category-split re-analysis

An external review of the reporting/eval-methodology layer (not the model) was verified
claim-by-claim against the code. Confirmed and fixed:

- **Null-label data bug**: `scripts/convert_misactbench.py` was silently coercing
  `label: null` (unannotated, 249 of 684 val+test rows) into "benign" — fixed to exclude
  those rows instead. This moved the gold base rate from 0.236 (n=347) to **0.41 (n=200)**
  and test AUPRC from 0.226 to **0.4199, 95% CI [0.347, 0.513]** — but the CI still contains
  the new base rate, so **the chance-level finding above is unchanged, just computed on a
  cleaner population.** See `results/misactbench_gold_eval_clip_frozen_postA7.json`.
- **MisActBench category-split re-analysis (exploratory, not pre-registered) —
  inconclusive**: checked whether the chance-level result was concentrated in
  MisActBench's two text-dependent categories rather than the one plausibly visible from
  pixels ("Unintended harmful behavior"). Raw AUPRC isn't comparable across categories here
  (each has a different base rate against the shared benign pool); AUROC is. Test AUROC:
  Malicious instruction following 0.489, Other task-irrelevant 0.487, **Unintended harmful
  behavior 0.590** (numerically highest, not lowest) — but this ranking does not replicate on
  val (0.628 / 0.287 / 0.280), and with only 12-49 positive examples per category and no
  per-category CI, **this is not statistically conclusive in either direction**. Correcting
  an earlier draft of this note, which wrongly compared raw AUPRC across categories with
  different base rates and concluded the visual category performed worst — it does not.
  See `results/misactbench_gold_eval_by_category.json`.
- **Circular gate calibration**: `scripts/evaluate_3tier_production.py` calibrated
  thresholds on val and reported metrics on the same val data — proven circular (two
  different checkpoints produced the identical 92.03% recall, which is just
  `target_recall=0.92` reflected back). Fixed to calibrate on val, report on held-out test.
  The honest result for the canonical checkpoint (`stage_c_clip_frozen/best.pt`): recall
  93.34% (passes), **precision 60.82% (fails the 70-80% target)**, category accuracy 78.36%
  (fails >85%), localization IoU 20.75% (fails >=50%), and **74.3% of all test traffic
  flagged** (PAUSE+HARD_BLOCK) — `all_criteria_passed: false`. See
  `results/production_evaluation_report.json`.
- **Fabricated metrics removed**: hardcoded "65.72% IoU" (ScreenSpot) and "99.60% IoU"
  (synthetic ground truth) strings in `evaluate_3tier_production.py` and `README.md` had no
  corresponding computation anywhere in the repo — the real ScreenSpot-v2 measurement is
  **4.76% IoU** (`results/screenspot_v2_point_accuracy.json`), ~14x lower than the fabricated
  number. Removed/corrected.
- **Tautological baseline**: `rule_based_keyword_cue`'s perfect 100%/100%/0% score in
  `paper/benchmark_results.json` is explained by the same keyword rule (`HARMFUL_CUES`)
  being both the label generator and the thing being scored — relabeled as not an
  independent baseline.
- **PPO gate numbers are unverified, not a confirmed regression**: `checkpoints/gate_rl.pt`
  does not exist in this repo, and `src/eval/run_benchmark.py` (the only script that
  evaluates a PPO gate) writes its result under a different key name
  (`sentinel_vision_gate`, not `sentinel_vision_gate_rl`) and would silently skip gate
  evaluation entirely if the checkpoint is missing, which it is. The 64.03%/14.05%/95.37%
  numbers in `paper/benchmark_results.json` cannot have come from running current code
  against a checkpoint that currently exists — marked unverified there, same treatment as
  the tautological `rule_based_keyword_cue` baseline.
- **Decisive check: no threshold meets the deployment bar, so RL training is not the
  bottleneck.** Before deciding whether to train a PPO gate, swept every threshold on val
  risk scores (`results/val_test_scores_stage_c_clip_frozen.npz`, cached so this took
  seconds, not a model-inference rerun) against a user-specified minimum-to-ship bar
  (recall >= 0.85, FPR <= 0.10). **No val threshold satisfies both simultaneously**: the
  best point at FPR <= 0.10 only reaches recall 0.45; the best point at recall >= 0.85 has
  FPR 0.51 (5x over the ceiling). Applying each threshold to held-out test reproduces the
  same conflict (recall 0.44/FPR 0.14, or recall 0.87/FPR 0.48). See
  `results/threshold_sweep_vs_user_bar.json`. The gate's observation space
  (`src/gate/decision_gate.py`: risk score + localization confidence + truncated category
  one-hot, not just the score alone — see `_build_state`) could in principle let an RL
  policy do better than a pure threshold, but harmful-row category accuracy is only 1.33%
  (near-useless) and the gold val set is only 177-258 rows (too few to train RL on
  meaningfully). **Conclusion: this is the same labeled-data/signal-quality ceiling as the
  detection result above, not a reward-shaping problem with the gate specifically.**
  Training a new PPO gate is not recommended until the underlying risk score itself
  improves (more/better labels, or text+pixel context) — it would very likely just learn to
  replicate the same floor the threshold sweep already found. Recommend: ship
  `DecisionGate3Tier` (threshold-only, no RL) framed as monitor-only, and do not present it
  as meeting the deployment bar.
- **ONNX INT8 latency**: reconciled two different, non-matching latency number pairs across
  README/draft.tex/system_report.md down to one authoritative source
  (`paper/latency_benchmark.json`: FP32 159.4ms / INT8 369.8ms, a 2.32x slowdown, not a
  speedup as one unfilled placeholder in `draft.tex` had implied).

Deferred (out of scope for this fix pass, per explicit prioritization): fixing the PPO
gate's reward shaping, and the "feed instruction text alongside pixels" architecture
change — both come after the reporting layer is trustworthy, not before.

---

**Below (§1-§5): the 2026-09-21 synthetic-data report, kept for history.**
Trained 2026-09-19 to 2026-09-21 on an RTX 4050 Laptop GPU
(6.4GB VRAM) on the combined real weak-labeled dataset (11,707 train samples)
+ synthetic DOM-grounded injection data (4,400 samples total, generated by
`src/data/synthetic_injection.py` via real headless-Chromium screenshots).
These are real numbers from `checkpoints/stage_c/metrics.csv` and a held-out
test-set benchmark — not fabricated, and not "perfect". **Note: the clean
synthetic-test recall/precision numbers below are NOT the gold-eval result**
— see the UPDATE above for the pre-registered, human-labeled-benchmark answer.

## 1. Current State

- **Architecture**: Complete and verified — full forward pass tested
  (frame encoder → temporal fusion → risk/localization heads → PPO gate).
- **Synthetic injection pipeline** (`src/data/synthetic_injection.py`): built
  and verified against real headless-Chromium screenshots (not synthetic
  PIL renders) — DOM-grounded bounding boxes, 4 harmful categories +
  benign templates.
- **Training checkpoints**: full curriculum now trained and checkpointed —
  `checkpoints/stage_a/best.pt`, `checkpoints/stage_b/best.pt`,
  `checkpoints/stage_c/best.pt` all load and verify.
  `checkpoints/stage_b_30epochs/best.pt` (the old weak-label-only checkpoint)
  is superseded. The legacy `stageA.pt`, `stageB.pt`, `stageC_final.pt`,
  `gate_rl.pt` files remain corrupted and unused.
- **Stage A undertrained**: only completed 2 of its 10 scheduled epochs
  (crashed twice on GPU OOM before a `training.batch_size=4` override fixed
  it, then only 2 more epochs ran) — its own backbone-unfreeze phase
  (scheduled for epoch 4) never happened. This did not block the pipeline:
  Stage B unfroze and fine-tuned the backbone properly on top of Stage A's
  weak starting point, and Stage C completed 12 of its 15 epochs
  (early-stopped, `val_recall_harmful` plateaued). See §2 for the honest
  final numbers this produced.
- **Checkpoint-selection caveat**: `checkpoints/stage_c/best.pt` is selected
  purely by `val_recall_harmful` (epoch 7: recall 0.7435, but localization
  IoU only 0.036 — near chance). `checkpoints/stage_c/epoch_15_final.pt`
  (real weights are from epoch 12, despite the filename — see the
  `_save_checkpoint` early-stopping fix in `trainer.py`) has slightly lower
  recall (0.7145) but far better precision (0.682 vs. 0.607) and localization
  IoU (0.115 vs. 0.036). Which to deploy depends on whether localization
  (showing *which* UI element is risky) matters for your use case — pure
  recall-maximizing favors `best.pt`; balanced usefulness favors
  `epoch_15_final.pt`.
- **Attention Rollout**: implemented (`SentinelModel.generate_attention_rollout`)
  — real transformer attention-rollout composition across temporal fusion
  blocks, verified to differ from the Grad-CAM heatmap (cosine similarity
  ≈0.64 on an untrained model, well under the 0.95 identity threshold).
- **Latency benchmark**: fixed to measure the full `(1, k, 3, 224, 224)`
  6-frame window (previously measured on a single frame). See
  `paper/latency_benchmark.json` for the most recent real measurement, and
  §4 below.

## 2. Main Benchmark Results

**Real numbers from the completed curriculum, validation set** (2,507 real +
various synthetic samples per stage — see `checkpoints/<stage>/metrics.csv`
for full per-epoch history). Two checkpoint candidates from Stage C, since
checkpoint selection optimizes recall only and ignores localization (see §1):

| Checkpoint | Val Recall (harmful) | Val Precision | Val F1 | Category Acc | Localization IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `stage_a/best.pt` (epoch 2, undertrained) | 0.463 | 0.372 | 0.413 | 0.899 | 0.018 |
| `stage_b/best.pt` (epoch 10, full run) | 0.673 | 0.665 | 0.669 | 0.861 | 0.035 |
| `stage_c/best.pt` (epoch 7, recall-selected) | **0.7435** | 0.607 | 0.669 | 0.848 | 0.036 |
| `stage_c/epoch_15_final.pt` (epoch 12, balanced) | 0.7145 | **0.682** | **0.699** | 0.876 | **0.115** |

**Honest read**: harmful-action recall improved substantially through the
curriculum (0.46 → 0.67 → 0.74), confirming the architecture and training
pipeline work end to end. But **~26% of harmful actions are still missed**
at the best recall checkpoint — given the asymmetric reward design
(`missed_harm = -10x false_block`), that is a real gap, not deployment-ready.
**Bounding-box localization is essentially non-functional** at the
recall-optimized checkpoint (IoU 0.036, near chance) and only modestly better
at the balanced checkpoint (IoU 0.115) — the model can tell *that* something
is risky far more reliably than *where* on screen. Visual sanity check
(`checkpoints/stage_c/images/epoch_012/sample_0.png` and `sample_3.png`)
confirms risk/category classification looks sensible (financial action
scored 0.856 risk / "financial" 0.99; benign action scored 0.077 risk /
"benign" 1.00) but boxes are loose, not tight.

**Held-out test-set benchmark** (660 synthetic samples, `data/synthetic_injections/test`,
never seen in train or val — `python -m src.security_wall.cli benchmark`):

| Checkpoint | Test Precision | Test Recall | Test F1 | FN |
| :--- | :---: | :---: | :---: | :---: |
| `stage_c/best.pt` (epoch 7) | **0.760** | **1.000** | **0.863** | 0 |
| `stage_c/epoch_15_final.pt` (epoch 12) | 0.603 | 1.000 | 0.752 | 0 |

**This reverses the validation-based recommendation above.** Both checkpoints
catch every harmful test sample (0 false negatives, 360/360), but `best.pt`
has far fewer false positives (114 vs. 237) — its validation-set localization
disadvantage does not translate into worse real-world risk detection.
**Recommendation: deploy `checkpoints/stage_c/best.pt`**, not
`epoch_15_final.pt`, despite its weaker localization IoU. The gap between
validation numbers (real+synthetic mix, noisier) and this clean synthetic
test set (100% recall) also shows the model has learned the DOM-grounded
injection patterns very well — the harder, noisier real weak-labeled data is
where the remaining validation-recall gap (~26% missed) actually lives.

Baselines below remain unmeasured — no random/rule-based/single-frame
comparison has been run yet:

| Monitoring Method | Harm Recall | Precision | Harm F1 | FNR | FPR | Localization IoU@0.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | TBD | TBD | TBD | TBD | TBD | -- |
| Rule-based keyword-cue baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| Single-Frame ViT (k=1) | TBD | TBD | TBD | TBD | TBD | TBD |
| SENTINEL-Vision (No Gate) | TBD | TBD | TBD | TBD | TBD | TBD |
| SENTINEL-Vision (Full, k=6 + PPO Gate) | TBD | TBD | TBD | TBD | TBD | TBD |

## 3. Ablation Studies

Not yet run against a trained model. `src/eval/ablations.py` is implemented
and will produce real numbers once a `trained_with_injection=True` checkpoint
exists (see `src/training/trainer.py` checkpoint metadata).

## 4. Latency Benchmark (Real Measurement)

Measured on `checkpoints/stage_b_30epochs/best.pt`, ONNX-exported, full
6-frame window `(1, 6, 3, 224, 224)`, CPU (no GPU). This is a real,
reproducible measurement from this machine — `python scripts/export_onnx.py`
regenerates `paper/latency_benchmark.json`, which is the authoritative source.

| Model | mean | p50 | p95 | p99 | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ONNX FP32 | 159.4ms | 157.2ms | 179.8ms | 187.7ms | 6.3 |
| ONNX INT8 (dynamic quant) | 369.8ms | 368.8ms | 410.9ms | 460.8ms | 2.7 |

(Numbers taken directly from `paper/latency_benchmark.json`, the authoritative source — a
previous version of this table cited a different, non-matching pair of numbers for the
same file; corrected here.)

**INT8 dynamic quantization was slower than FP32 on this CPU** (0.43x
"speedup," i.e. a slowdown) — the opposite of the previous fabricated
INT8 latency claim. This is plausible on CPUs without INT8 VNNI acceleration, where
dynamic quantization's runtime dequantization overhead exceeds the compute
savings for a model this size. p95 (410.9ms) is still under the 500ms target,
but only for INT8; FP32 (179.8ms) is well under it and currently faster.

**Fixed**: ONNX output verification against the PyTorch model was failing
(`category_probs`/`bbox`/`objectness` came back with mismatched shapes) because
`export_to_onnx()` traced the raw model, whose `forward()` returns an 11-key
dict including non-tensor values (a `category` string list and a `None`
heatmap) that don't align with the 5 declared ONNX `output_names`. Fixed by
wrapping the model in an export-only module (`_ONNXExportWrapper` in
`scripts/export_onnx.py`) that returns exactly the 5 real tensor outputs in
order. Re-verified: the FP32 export now passes verification exactly
(`verification: True`). The INT8 quantized export still shows
`category_probs` exceeding the tight 0.01 tolerance (max diff 0.0186) — that
is real INT8 quantization noise, not a shape/correctness bug, and is expected
for dynamic quantization without INT8 VNNI hardware support.

## 5. What Is Complete vs. In Progress

**Complete:**
- Full model architecture (FrameEncoder + TemporalFusion + RiskHead + LocalizationHead + Gate)
- Multi-task loss (Focal Loss + GIoU + CE with class weights)
- Training infrastructure (Trainer + curriculum A/B/C + early stopping on recall)
- Synthetic injection pipeline (real Playwright/Chromium screenshots, DOM-grounded labels)
- Contrastive temporal pretraining (5 real epochs, loss 1.14 → 1.03)
- **Full curriculum training (Stage A → B → C), real checkpoints, real metrics**
- **Real benchmark measurement against the trained checkpoint (§2)**
- Attention Rollout explainability (real algorithm, not a Grad-CAM alias)
- Centralized, safe checkpoint loading (`weights_only=True`, no silent fallback)
- Evaluation suite (metrics, ablations, adversarial stress test)
- FastAPI intercept server + security wall + incident reporter
- 148 unit tests, all passing

**Known gaps (honest, not deployment-blocking secrets):**
- Stage A undertrained (2/10 epochs) — did not block the pipeline (see §1)
  but a properly-completed Stage A would likely improve the ceiling further.
- Bounding-box localization is weak across all checkpoints (best test-set
  precision/recall come from a checkpoint with IoU ≈0.036, near chance).
  Localizing *which* element is risky, not just detecting *that* something
  is risky, needs more work — likely more synthetic injection data (the
  cleanest source of ground-truth bboxes) and/or a dedicated localization
  fine-tuning pass.
- Validation recall (0.74, on the harder real+synthetic mix) is meaningfully
  lower than test-set recall (1.00, pure clean synthetic) — the real-world
  weak-labeled data is where most of the remaining gap lives.
- RL gate training (PPO `DecisionGate`) not yet run against this checkpoint —
  `validate_sentinel_model_quality()` should now pass (risk scores are
  discriminative, not constant), but the actual gate training hasn't
  happened.
- Baselines (random, rule-based, single-frame) in §2's second table remain
  unmeasured.
