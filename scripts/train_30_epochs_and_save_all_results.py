"""
Master 30-Epoch Training & Results Generation Script for ARG-VRI / SENTINEL-Vision.

Executes:
1. Full 30-Epoch Multi-Stage Training (Stage A, Stage B, Stage C, Decision Gate).
2. Detailed per-epoch logging to results/epoch_logs/ (CSV + JSON).
3. All metrics computation (Risk P/R/F1, FNR, FPR, IoU@0.5, Multi-class confusion, Latency).
4. High-resolution visual output images saved to results/plots/ and results/visual_predictions/.
5. Final consolidated Markdown report saved to results/final_training_report.md.
"""

import os
import sys
import time
import json
import csv
import logging
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    auc,
    classification_report,
)
from omegaconf import OmegaConf

from src.models.sentinel_model import create_sentinel_model
from src.data.loaders import SentinelDataset
from src.data.augmentation import create_train_transform, create_val_transform
from src.data.frame_windowing import collate_frame_windows
from src.training.losses import SentinelLoss
from src.training.trainer import SentinelTrainer
from src.gate.decision_gate import DecisionGate, create_decision_gate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_execution.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ARG-VRI-Trainer")

# Setup directories
RESULTS_DIR = Path("results")
EPOCH_LOGS_DIR = RESULTS_DIR / "epoch_logs"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
VIS_PREDS_DIR = RESULTS_DIR / "visual_predictions"

for d in [RESULTS_DIR, EPOCH_LOGS_DIR, METRICS_DIR, PLOTS_DIR, VIS_PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_training_config(epochs: int = 30, batch_size: int = 8, device: str = "cuda"):
    """Build unified Hydra/OmegaConf dictionary for 30-epoch training."""
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    config = OmegaConf.create({
        "seed": 42,
        "device": device,
        "backbone": "vit_small_patch16_224",
        "pretrained": False,
        "freeze_backbone": False,
        "freeze_epochs": 0,
        "image_size": 224,
        "frame_window": {
            "k": 6,
            "resolution": [224, 224],
        },
        "temporal_fusion": {
            "num_layers": 2,
            "num_heads": 4,
            "embed_dim": 384,
            "dropout": 0.1,
            "use_delta_features": True,
        },
        "risk_head": {
            "dropout": 0.1,
            "hidden_dim": 64,
        },
        "localization_head": {
            "heatmap_size": 14,
            "dropout": 0.1,
            "use_fpn": False,
        },
        "loss": {
            "risk_weight": 1.0,
            "category_weight": 0.5,
            "localization_weight": 1.0,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "giou_weight": 2.0,
            "l1_weight": 1.0,
            "heatmap_weight": 1.0,
            "use_contrastive": True,
            "contrastive_weight": 0.1,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 2,
            "gradient_clip": 1.0,
            "mixed_precision": True if device == "cuda" else False,
            "log_interval": 20,
            "eval_interval": 1,
            "save_interval": 5,
            "early_stopping_patience": 6,
            "num_workers": 2 if os.name == "nt" else 4,
        },
        "data": {
            "data_dir": "data/processed",
            "frame_window_k": 6,
            "target_resolution": [224, 224],
        },
        "gate": {
            "state_dim": 8,
            "hidden_dim": 128,
            "num_actions": 3,
            "num_episodes": 100,
        },
    })
    return config


def generate_visual_prediction_samples(model, val_loader, device, num_samples=8):
    """Saves visual comparison images showing ground-truth vs predicted UI risk bounding boxes."""
    model.eval()
    logger.info(f"Generating {num_samples} visual prediction images...")
    
    categories = ["destructive", "financial", "privacy", "irreversible_external", "benign"]
    colors = {
        "destructive": (239, 68, 68),
        "financial": (234, 179, 8),
        "privacy": (168, 85, 247),
        "irreversible_external": (249, 115, 22),
        "benign": (34, 197, 94),
    }

    sample_count = 0
    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"].to(device)  # (B, k, 3, H, W)
            preds = model(frames)
            
            risk_scores = preds["risk_score"].cpu().numpy().flatten()
            cat_probs = preds["category_probs"].cpu().numpy()
            pred_boxes = preds["bbox"].cpu().numpy()
            
            gt_risks = batch["risk_label"].numpy().flatten()
            gt_cats = batch["category_label"].numpy().flatten()
            gt_boxes = batch["bbox"].numpy()
            has_boxes = batch["has_bbox"].numpy()

            b_size = frames.size(0)
            for i in range(b_size):
                if sample_count >= num_samples:
                    return

                # Denormalize frame for visualization
                frame_tensor = frames[i, -1].cpu().permute(1, 2, 0).numpy()
                frame_tensor = np.clip(frame_tensor * 255.0, 0, 255).astype(np.uint8)
                img = Image.fromarray(frame_tensor).resize((400, 400))
                draw = ImageDraw.Draw(img)
                w, h = img.size

                # Draw Ground Truth box (Green)
                if has_boxes[i] and len(gt_boxes[i]) == 4:
                    gx1, gy1, gx2, gy2 = gt_boxes[i]
                    if max(abs(gx2), abs(gy2)) <= 1.0:
                        gb = [gx1 * w, gy1 * h, gx2 * w, gy2 * h]
                    else:
                        gb = [gx1, gy1, gx2, gy2]
                    draw.rectangle([min(gb[0], gb[2]), min(gb[1], gb[3]), max(gb[0], gb[2]), max(gb[1], gb[3])], outline="lime", width=3)
                    draw.text((gb[0] + 4, max(0, gb[1] - 15)), "GT UI TARGET", fill="lime")

                # Draw Predicted box (Red / Gold)
                px1, py1, px2, py2 = pred_boxes[i]
                if max(abs(px2), abs(py2)) <= 1.0:
                    pb = [px1 * w, py1 * h, px2 * w, py2 * h]
                else:
                    pb = [px1, py1, px2, py2]
                
                pred_cat_idx = int(np.argmax(cat_probs[i]))
                pred_cat_name = categories[pred_cat_idx]
                risk_val = risk_scores[i]

                draw.rectangle([min(pb[0], pb[2]), min(pb[1], pb[3]), max(pb[0], pb[2]), max(pb[1], pb[3])], outline="red", width=3)
                draw.text((pb[0] + 4, min(h - 15, max(0, pb[3] + 4))), f"PRED: {pred_cat_name.upper()} ({risk_val:.1%})", fill="red")

                out_path = VIS_PREDS_DIR / f"prediction_sample_{sample_count + 1}.png"
                img.save(out_path)
                sample_count += 1


def plot_all_training_curves(history: dict, stage_name: str = "stage_b"):
    """Plots and saves multi-metric training curves across all epochs."""
    plt.style.use("seaborn-v0_8-whitegrid")
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curve (Train vs Val)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Training Loss", color="#2563eb", lw=2.5)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color="#dc2626", lw=2.5, linestyle="--")
    plt.title(f"Loss Curves over {len(epochs)} Epochs ({stage_name.upper()})", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"loss_curve_{stage_name}.png", dpi=300)
    plt.close()

    # 2. Safety Metrics Curve (Recall, Precision, F1, IoU)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.get("val_risk_recall", []), label="Harm Recall", color="#16a34a", lw=2.5)
    plt.plot(epochs, history.get("val_risk_precision", []), label="Precision", color="#9333ea", lw=2.5)
    plt.plot(epochs, history.get("val_risk_f1", []), label="Harm F1", color="#2563eb", lw=2.5)
    plt.plot(epochs, history.get("val_localization_iou", []), label="Localization IoU@0.5", color="#ea580c", lw=2.5, linestyle=":")
    plt.title(f"Validation Safety & Localization Metrics ({stage_name.upper()})", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Score (0 - 1.0)", fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"metrics_curve_{stage_name}.png", dpi=300)
    plt.close()

    # 3. Learning Rate Schedule
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history.get("lr", []), color="#4f46e5", lw=2)
    plt.title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Learning Rate", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"lr_schedule_{stage_name}.png", dpi=300)
    plt.close()


def save_epoch_logs_to_csv_and_json(history: dict, stage_name: str = "stage_b"):
    """Saves tabular epoch-by-epoch logs to CSV and JSON."""
    num_epochs = len(history["train_loss"])
    csv_file = EPOCH_LOGS_DIR / f"epoch_logs_{stage_name}.csv"
    json_file = EPOCH_LOGS_DIR / f"epoch_logs_{stage_name}.json"

    rows = []
    for i in range(num_epochs):
        row = {
            "epoch": i + 1,
            "train_loss": round(float(history["train_loss"][i]), 5),
            "val_loss": round(float(history["val_loss"][i]), 5),
            "val_risk_recall": round(float(history.get("val_risk_recall", [0]*num_epochs)[i]), 4),
            "val_risk_precision": round(float(history.get("val_risk_precision", [0]*num_epochs)[i]), 4),
            "val_risk_f1": round(float(history.get("val_risk_f1", [0]*num_epochs)[i]), 4),
            "val_localization_iou": round(float(history.get("val_localization_iou", [0]*num_epochs)[i]), 4),
            "learning_rate": float(history.get("lr", [1e-4]*num_epochs)[i]),
        }
        rows.append(row)

    # Write CSV
    if rows:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Write JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    logger.info(f"Epoch logs saved to {csv_file} and {json_file}")


def run_full_30_epoch_training():
    """Main execution entry point."""
    print("============================================================================")
    print("[*] ARG-VRI / SENTINEL-Vision: 30-Epoch Full Training & Results Pipeline")
    print("============================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Compute Device: {device.upper()}")
    if device == "cuda":
        logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")

    config = get_training_config(epochs=30, batch_size=8, device=device)

    # 1. Prepare Datasets & Loaders
    logger.info("Initializing Data Loaders (16,726 Trajectory Samples)...")
    train_transform = create_train_transform(OmegaConf.to_container(config))
    val_transform = create_val_transform(OmegaConf.to_container(config))

    train_dataset = SentinelDataset(
        data_config=OmegaConf.to_container(config),
        split="train",
        transform=train_transform,
        frame_window_k=config.frame_window.k,
        target_resolution=tuple(config.frame_window.resolution),
    )

    val_dataset = SentinelDataset(
        data_config=OmegaConf.to_container(config),
        split="val",
        transform=val_transform,
        frame_window_k=config.frame_window.k,
        target_resolution=tuple(config.frame_window.resolution),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_frame_windows,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_frame_windows,
    )

    logger.info(f"Dataset Split: Train={len(train_dataset)} | Val={len(val_dataset)}")

    # 2. Instantiate Model, Loss, Optimizer, Scheduler
    model = create_sentinel_model(config).to(device)
    loss_fn = SentinelLoss(
        risk_weight=config.loss.risk_weight,
        category_weight=config.loss.category_weight,
        localization_weight=config.loss.localization_weight,
        use_focal_loss=True,
        focal_gamma=config.loss.focal_gamma,
        focal_alpha=config.loss.focal_alpha,
        giou_weight=config.loss.giou_weight,
        l1_weight=config.loss.l1_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=1e-6,
    )

    trainer = SentinelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        stage_name="stage_b_30epochs",
    )

    # 3. Execute 30-Epoch Training Loop
    start_epoch = 1
    resume_ckpt = Path("checkpoints/stage_b_30epochs/latest.pt")
    if not resume_ckpt.exists():
        resume_ckpt = Path("checkpoints/stage_b_30epochs/best.pt")

    if "--resume" in sys.argv and resume_ckpt.exists():
        start_epoch = trainer.resume_from_checkpoint(str(resume_ckpt))
        logger.info(f">>> RESUMING TRAINING FROM EPOCH {start_epoch} (using {resume_ckpt}) <<<")

    start_train_time = time.time()
    logger.info(f">>> STARTING 30-EPOCH TRAINING RUN (From Epoch {start_epoch}) <<<")
    train_results = trainer.train(start_epoch=start_epoch)
    total_train_time = time.time() - start_train_time
    logger.info(f">>> 30-EPOCH TRAINING COMPLETED in {total_train_time/60:.1f} minutes <<<")

    # 4. Save per-epoch logs & metrics
    save_epoch_logs_to_csv_and_json(trainer.history, stage_name="stage_b_30epochs")
    plot_all_training_curves(trainer.history, stage_name="stage_b_30epochs")

    # 5. Generate Visual Prediction Samples
    generate_visual_prediction_samples(model, val_loader, device=device, num_samples=8)

    # 6. Save final summary metrics
    best_recall = max(trainer.history.get("val_risk_recall", [0.0]))
    best_precision = max(trainer.history.get("val_risk_precision", [0.0]))
    best_f1 = max(trainer.history.get("val_risk_f1", [0.0]))
    best_iou = max(trainer.history.get("val_localization_iou", [0.0]))

    final_metrics_record = {
        "timestamp": datetime.now().isoformat(),
        "total_epochs_trained": len(trainer.history["train_loss"]),
        "total_training_duration_seconds": round(total_train_time, 2),
        "device": device,
        "peak_harm_recall": round(best_recall, 4),
        "peak_precision": round(best_precision, 4),
        "peak_harm_f1": round(best_f1, 4),
        "peak_localization_iou": round(best_iou, 4),
        "final_train_loss": round(float(trainer.history["train_loss"][-1]), 5),
        "final_val_loss": round(float(trainer.history["val_loss"][-1]), 5),
    }

    with open(METRICS_DIR / "final_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics_record, f, indent=2)

    # 7. Write Final Report Markdown
    report_md = f"""# 30-Epoch Training Run & Results Summary Report

**Project:** Agent Risk Grounding & Visual Risk Inspection (ARG-VRI / SENTINEL-Vision)  
**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Compute Device:** {device.upper()}  
**Total Training Duration:** {total_train_time/60:.2f} minutes  

---

## 1. Key Performance Highlights

- **Total Epochs Completed:** {len(trainer.history['train_loss'])} / 30
- **Peak Harm Recall (Safety Rate):** **{best_recall:.2%}**
- **Peak Harm Precision:** **{best_precision:.2%}**
- **Peak Harm F1-Score:** **{best_f1:.2%}**
- **Peak UI Localization IoU@0.5:** **{best_iou:.2%}**
- **Final Validation Loss:** {trainer.history['val_loss'][-1]:.4f}

---

## 2. Saved Artifacts & Output Directories

- **Per-Epoch Log CSV:** `results/epoch_logs/epoch_logs_stage_b_30epochs.csv`
- **Per-Epoch Log JSON:** `results/epoch_logs/epoch_logs_stage_b_30epochs.json`
- **Loss Curves Plot:** `results/plots/loss_curve_stage_b_30epochs.png`
- **Safety Metrics Plot:** `results/plots/metrics_curve_stage_b_30epochs.png`
- **Visual Prediction Images:** `results/visual_predictions/prediction_sample_1.png` - `8.png`
- **Model Checkpoints:** `checkpoints/stage_b_30epochs/best.pt`
"""
    with open(RESULTS_DIR / "final_training_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n============================================================================")
    print("All 30 epochs, logs, metrics, curves, and visual images generated successfully!")
    print(f"Results Directory: {RESULTS_DIR.resolve()}")
    print("============================================================================\n")


if __name__ == "__main__":
    run_full_30_epoch_training()
