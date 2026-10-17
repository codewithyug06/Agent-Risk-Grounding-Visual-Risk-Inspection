"""
Shared training loop for SENTINEL-Vision.
Handles Hydra config, W&B logging, mixed precision, checkpointing, early stopping.
"""

import os
import shutil
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
import logging
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import hydra

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models.sentinel_model import SentinelModel
from ..data.frame_windowing import collate_frame_windows
from .losses import SentinelLoss, create_loss_function

logger = logging.getLogger(__name__)



def _loc_mask(batch, device):
    """Supervise localization on all samples that provide a valid bounding box."""
    return batch["has_bbox"].float().to(device)

class SentinelTrainer:
    """
    Main training loop for SENTINEL-Vision.
    Supports curriculum stages A/B/C with different configurations.
    """

    def __init__(
        self,
        config: DictConfig,
        model: SentinelModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: SentinelLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda",
        stage_name: str = "stage_a",
        trained_with_injection: bool = False,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ):
        self.trained_with_injection = trained_with_injection
        self.dataset_stats = dataset_stats or {}
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.stage_name = stage_name

        # Training config
        train_config = config.get("training", {})
        self.epochs = train_config.get("epochs", 35)
        self.warmup_epochs = train_config.get("warmup_epochs", 3)
        self.gradient_clip = train_config.get("gradient_clip", 1.0)
        self.mixed_precision = train_config.get("mixed_precision", True)
        self.log_interval = train_config.get("log_interval", 50)
        self.eval_interval = train_config.get("eval_interval", 1)
        self.gradient_accumulation_steps = max(1, train_config.get("gradient_accumulation_steps", 1))

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", f"checkpoints/{stage_name}"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("-inf")
        self.checkpoint_metric = config.get("checkpoint_metric", "recall_harmful")
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.early_stopping_counter = 0
        self.best_loc_iou = float("-inf")

        # Mixed precision: prefer bfloat16 on Ampere/Ada GPUs for rock-solid stability
        self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        self.use_bf16 = self.mixed_precision and self.device_type == "cuda" and torch.cuda.is_bf16_supported()
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.use_scaler = self.mixed_precision and not self.use_bf16
        try:
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_scaler)
        except Exception:
            self.scaler = GradScaler(enabled=self.use_scaler)

        # W&B
        self.use_wandb = config.get("wandb", {}).get("enabled", False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb", {}).get("project", "sentinel-vision"),
                name=f"{stage_name}-{config.get('wandb', {}).get('run_name', 'run')}",
                config=OmegaConf.to_container(config, resolve=True),
            )
            wandb.watch(self.model, log="all", log_freq=100)

        # Metrics tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_risk_precision": [],
            "val_risk_recall": [],
            "val_risk_f1": [],
            "val_recall_harmful": [],
            "val_localization_iou": [],
            "lr": [],
        }

        # Set seed
        self._set_seed(config.get("seed", 42))

        logger.info(f"Trainer initialized for {stage_name}: "
                    f"epochs={self.epochs}, device={device}, "
                    f"mixed_precision={self.mixed_precision}")

    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self, start_epoch: int = 1) -> Dict[str, float]:
        """
        Run full training loop.
        Returns best validation metrics.
        """
        logger.info(f"Starting training for {self.stage_name} (Epochs {start_epoch} to {self.epochs})")

        for epoch in range(start_epoch, self.epochs + 1):
            self.model.set_epoch(epoch)

            # Training epoch
            train_metrics = self._train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Validation
            if epoch % self.eval_interval == 0:
                val_metrics = self._validate(epoch)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_risk_precision"].append(val_metrics["risk_precision"])
                self.history["val_risk_recall"].append(val_metrics["risk_recall"])
                self.history["val_risk_f1"].append(val_metrics["risk_f1"])
                self.history["val_recall_harmful"].append(val_metrics["recall_harmful"])
                self.history["val_localization_iou"].append(val_metrics.get("localization_iou", 0.0))
                self.history.setdefault("val_category_accuracy", []).append(val_metrics.get("category_accuracy", 0.0))

                self._save_metrics_csv(epoch, train_metrics, val_metrics)
                self._save_sample_visualizations(epoch)

                # Checkpointing: recall drives best.pt; localization IoU drives
                # best_loc.pt. Either improving resets patience, so localization
                # is not cut off while recall plateaus.
                metric_key = self.checkpoint_metric
                if metric_key not in val_metrics and metric_key.startswith("val_"):
                    metric_key = metric_key[4:]
                metric_value = val_metrics.get(metric_key, val_metrics.get("recall_harmful", val_metrics["loss"]))
                loc_iou = val_metrics.get("localization_iou", 0.0)
                improved = False
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                    logger.info(f"New best {self.checkpoint_metric} ({metric_key}): {metric_value:.4f}")
                    improved = True
                if loc_iou > self.best_loc_iou:
                    self.best_loc_iou = loc_iou
                    self._save_checkpoint(epoch, val_metrics, is_best=True, suffix="_loc")
                    logger.info(f"New best localization_iou: {loc_iou:.4f}")
                    improved = True
                if improved:
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    logger.info(f"No improvement for {self.early_stopping_counter} epochs")

                # Early stopping
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

            # Regular checkpoint
            if epoch % self.config.get("save_interval", 5) == 0:
                self._save_checkpoint(epoch, val_metrics if 'val_metrics' in locals() else {}, is_best=False)

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final checkpoint. Use the loop's actual last `epoch` (which may
        # be less than self.epochs if early stopping broke the loop early)
        # and that epoch's own metrics dict, not the full multi-epoch history
        # — a checkpoint's `metrics` field must describe that one checkpoint.
        self._save_checkpoint(
            epoch,
            val_metrics if 'val_metrics' in locals() else {},
            is_best=False,
            suffix="_final",
        )

        logger.info(f"Training completed. Best {self.checkpoint_metric}: {self.best_metric:.4f}")
        return {"best_metric": self.best_metric, "history": self.history}

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        loss_components = {}

        start_time = time.time()
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            frames = batch["frames"]  # (B, k, C, H, W)
            targets = {
                "risk_label": batch["risk_label"],
                "category_label": batch["category_label"],
                "bbox": batch["bbox"],
                "has_bbox": _loc_mask(batch, self.device),
            }

            try:
                autocast_ctx = torch.amp.autocast(self.device_type, dtype=self.amp_dtype, enabled=self.mixed_precision)
            except Exception:
                autocast_ctx = autocast(enabled=self.mixed_precision)

            with autocast_ctx:
                predictions = self.model(
                    frames,
                    gt_bbox_pixel=targets["bbox"] * self.model.image_size,
                )
                loss_dict = self.loss_fn(predictions, targets)
                loss = loss_dict["total_loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Encountered NaN/Inf loss ({loss.item() if hasattr(loss, 'item') else loss}), skipping step."
                )
                self.model.zero_grad(set_to_none=True)
                continue

            scaled_loss = loss / self.gradient_accumulation_steps
            if self.use_scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_accum_step = ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or ((batch_idx + 1) == len(self.train_loader))
            if is_accum_step:
                if self.use_scaler:
                    self.scaler.unscale_(self.optimizer)
                if self.gradient_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.warning(
                            f"Epoch {epoch} [{batch_idx}] Non-finite gradient norm "
                            f"({grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm}), skipping optimizer step."
                        )
                        self.model.zero_grad(set_to_none=True)
                        if self.use_scaler:
                            self.scaler.update()
                        continue
                if self.use_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.model.zero_grad(set_to_none=True)

            # Accumulate
            batch_size = frames.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += val.item() * batch_size

            # Logging
            if batch_idx % self.log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"({elapsed:.1f}s elapsed)"
                )

                if self.use_wandb:
                    wandb.log({
                        f"{self.stage_name}/train_batch_loss": loss.item(),
                        f"{self.stage_name}/train_batch_lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "step": epoch * len(self.train_loader) + batch_idx,
                    })

        avg_loss = total_loss / total_samples
        avg_components = {k: v / total_samples for k, v in loss_components.items()}

        logger.info(f"Epoch {epoch} Train Loss: {avg_loss:.4f} | "
                    f"Risk: {avg_components.get('risk_loss', 0):.4f} "
                    f"Cat: {avg_components.get('category_loss', 0):.4f} "
                    f"Loc: {avg_components.get('localization_loss', 0):.4f}")

        if self.use_wandb:
            wandb.log({
                f"{self.stage_name}/train_epoch_loss": avg_loss,
                **{f"{self.stage_name}/train_{k}": v for k, v in avg_components.items()},
                "epoch": epoch,
            })

        return {"loss": avg_loss, **avg_components}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        all_preds = {"risk_score": [], "category_probs": [], "bbox": []}
        all_targets = {"risk_label": [], "category_label": [], "bbox": [], "has_bbox": []}

        total_loss = 0.0
        total_samples = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            frames = batch["frames"]
            targets = {
                "risk_label": batch["risk_label"],
                "category_label": batch["category_label"],
                "bbox": batch["bbox"],
                "has_bbox": _loc_mask(batch, self.device),
            }

            try:
                autocast_ctx = torch.amp.autocast(self.device_type, dtype=self.amp_dtype, enabled=self.mixed_precision)
            except Exception:
                autocast_ctx = autocast(enabled=self.mixed_precision)

            with autocast_ctx:
                predictions = self.model(frames)
                loss_dict = self.loss_fn(predictions, targets)

            batch_size = frames.size(0)
            total_loss += loss_dict["total_loss"].item() * batch_size
            total_samples += batch_size

            # Collect predictions
            all_preds["risk_score"].append(predictions["risk_score"].cpu())
            all_preds["category_probs"].append(predictions["category_probs"].cpu())
            all_preds["bbox"].append(predictions.get("bbox_refined", predictions["bbox"]).cpu())

            all_targets["risk_label"].append(targets["risk_label"].cpu())
            all_targets["category_label"].append(targets["category_label"].cpu())
            all_targets["bbox"].append(targets["bbox"].cpu())
            all_targets["has_bbox"].append(targets["has_bbox"].cpu())

        avg_loss = total_loss / total_samples

        # Concatenate
        for key in all_preds:
            all_preds[key] = torch.cat(all_preds[key], dim=0)
        for key in all_targets:
            all_targets[key] = torch.cat(all_targets[key], dim=0)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_targets)
        metrics["loss"] = avg_loss

        logger.info(
            f"Epoch {epoch} Val Loss: {avg_loss:.4f} | "
            f"Risk P/R/F1: {metrics['risk_precision']:.4f}/{metrics['risk_recall']:.4f}/{metrics['risk_f1']:.4f} | "
            f"Harmful Recall: {metrics['recall_harmful']:.4f} | "
            f"Loc IoU: {metrics.get('localization_iou', 0):.4f}"
        )

        if self.use_wandb:
            wandb.log({
                f"{self.stage_name}/val_loss": avg_loss,
                f"{self.stage_name}/val_risk_precision": metrics["risk_precision"],
                f"{self.stage_name}/val_risk_recall": metrics["risk_recall"],
                f"{self.stage_name}/val_risk_f1": metrics["risk_f1"],
                f"{self.stage_name}/val_recall_harmful": metrics["recall_harmful"],
                f"{self.stage_name}/val_localization_iou": metrics.get("localization_iou", 0.0),
                "epoch": epoch,
            })

        return metrics

    def _compute_metrics(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute validation metrics."""
        risk_scores = preds["risk_score"].squeeze(-1)  # (N,)
        risk_labels = targets["risk_label"].float()  # (N,)

        # Binary predictions
        risk_preds = (risk_scores > threshold).float()

        # Precision, Recall, F1
        tp = (risk_preds * risk_labels).sum()
        fp = (risk_preds * (1 - risk_labels)).sum()
        fn = ((1 - risk_preds) * risk_labels).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Harmful recall (recall on positive class only)
        recall_harmful = recall.item()

        # Dynamic optimal threshold via Precision-Recall curve
        best_f1 = f1.item()
        best_precision = precision.item()
        best_recall = recall.item()
        best_thresh = threshold
        auc_score = 0.5

        if risk_labels.sum() > 0:
            try:
                import numpy as np
                from sklearn.metrics import precision_recall_curve, roc_auc_score
                y_true = risk_labels.cpu().numpy()
                y_scores = risk_scores.cpu().numpy()
                p_curve, r_curve, th_curve = precision_recall_curve(y_true, y_scores)
                f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-6)
                if len(f1_curve) > 0 and not np.isnan(f1_curve).all():
                    best_idx = int(np.nanargmax(f1_curve))
                    best_f1 = float(f1_curve[best_idx])
                    best_precision = float(p_curve[best_idx])
                    best_recall = float(r_curve[best_idx])
                    if len(th_curve) > 0:
                        best_thresh = float(th_curve[min(best_idx, len(th_curve) - 1)])
                auc_score = float(roc_auc_score(y_true, y_scores))
            except Exception:
                pass

        # Category accuracy
        cat_probs = preds["category_probs"]
        cat_preds = cat_probs.argmax(dim=-1)
        cat_labels = targets["category_label"]
        cat_acc = (cat_preds == cat_labels).float().mean().item()

        # Localization IoU (only for harmful with bbox)
        has_bbox = targets["has_bbox"] > 0.5
        harmful_mask = (targets["risk_label"] == 1) & has_bbox

        localization_iou = 0.0
        if harmful_mask.any():
            pred_bboxes = preds["bbox"][harmful_mask]
            gt_bboxes = targets["bbox"][harmful_mask]

            ious = []
            for i in range(len(pred_bboxes)):
                iou = self._compute_iou(pred_bboxes[i], gt_bboxes[i])
                ious.append(iou)
            localization_iou = sum(ious) / len(ious) if ious else 0.0

        # Choose the best metric set between optimal calibrated threshold and fixed 0.5
        use_optimal = best_f1 >= f1.item()
        reported_precision = best_precision if use_optimal else precision.item()
        reported_recall = best_recall if use_optimal else recall.item()
        reported_f1 = best_f1 if use_optimal else f1.item()

        return {
            "risk_precision": reported_precision,
            "risk_recall": reported_recall,
            "risk_f1": reported_f1,
            "recall_harmful": reported_recall,
            "category_accuracy": cat_acc,
            "localization_iou": localization_iou,
            "auc_roc": auc_score,
            "optimal_threshold": best_thresh,
            "risk_precision_05": precision.item(),
            "risk_recall_05": recall.item(),
            "risk_f1_05": f1.item(),
        }

    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two normalized boxes."""
        x1_i = max(box1[0].item(), box2[0].item())
        y1_i = max(box1[1].item(), box2[1].item())
        x2_i = min(box1[2].item(), box2[2].item())
        y2_i = min(box1[3].item(), box2[3].item())

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        inter = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
        area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def _save_metrics_csv(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Append this epoch's metrics as one row to <checkpoint_dir>/metrics.csv."""
        import csv

        csv_path = self.checkpoint_dir / "metrics.csv"
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", ""),
            "val_loss": val_metrics.get("loss", ""),
            "val_risk_precision": val_metrics.get("risk_precision", ""),
            "val_risk_recall": val_metrics.get("risk_recall", ""),
            "val_risk_f1": val_metrics.get("risk_f1", ""),
            "val_recall_harmful": val_metrics.get("recall_harmful", ""),
            "val_category_accuracy": val_metrics.get("category_accuracy", ""),
            "val_localization_iou": val_metrics.get("localization_iou", ""),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_sample_visualizations(self, epoch: int, n_samples: int = 8):
        """
        Save prediction-overlay PNGs (risk score, category, bbox) for a
        handful of validation samples to
        <checkpoint_dir>/images/epoch_{epoch:03d}/sample_{i}.png, so training
        progress can be inspected visually, not just via scalar metrics.
        """
        from ..utils.visualization import visualize_predictions

        out_dir = self.checkpoint_dir / "images" / f"epoch_{epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        category_names = ["destructive", "financial", "privacy", "irreversible_external", "benign"]

        self.model.eval()
        saved = 0
        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch["frames"].to(self.device)  # (B, k, C, H, W)
                predictions = self.model(frames)

                last_frame = frames[:, -1]  # (B, C, H, W) — most recent frame in the window
                risk_scores = predictions["risk_score"].squeeze(-1).cpu()
                category_probs = predictions["category_probs"].cpu()
                bboxes = predictions["bbox"].cpu()

                for i in range(last_frame.shape[0]):
                    if saved >= n_samples:
                        break

                    img_t = last_frame[i].cpu()
                    img_min, img_max = img_t.min(), img_t.max()
                    img_norm = (img_t - img_min) / (img_max - img_min + 1e-8)
                    img = Image.fromarray((img_norm.permute(1, 2, 0).numpy() * 255).astype("uint8"))

                    cat_idx = int(category_probs[i].argmax().item())
                    # Early-training bbox regressions can have x2<x1 or y2<y1
                    # (or fall outside [0,1]) before the model has converged;
                    # sort and clip so the overlay draw never crashes on them.
                    bx1, by1, bx2, by2 = bboxes[i].tolist()
                    bx1, bx2 = sorted((max(0.0, min(1.0, bx1)), max(0.0, min(1.0, bx2))))
                    by1, by2 = sorted((max(0.0, min(1.0, by1)), max(0.0, min(1.0, by2))))
                    if bx2 <= bx1:
                        bx2 = min(1.0, bx1 + 1e-3)
                    if by2 <= by1:
                        by2 = min(1.0, by1 + 1e-3)

                    r_val = float(risk_scores[i].item())
                    if math.isnan(r_val) or math.isinf(r_val):
                        r_val = 0.0
                    c_conf = float(category_probs[i, cat_idx].item())
                    if math.isnan(c_conf) or math.isinf(c_conf):
                        c_conf = 0.0

                    vis = visualize_predictions(
                        image=img,
                        risk_score=r_val,
                        category=category_names[cat_idx],
                        category_conf=c_conf,
                        bbox=(bx1, by1, bx2, by2),
                        normalized=True,
                    )
                    vis.save(out_dir / f"sample_{saved}.png")
                    saved += 1

                if saved >= n_samples:
                    break
        self.model.train()
        logger.info(f"Saved {saved} sample prediction images to {out_dir}")

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        suffix: str = "",
    ):
        """Save model checkpoint."""
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        import hashlib
        config_hash = hashlib.sha256(str(sorted(config_dict.items())).encode("utf-8")).hexdigest()[:16]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "config": config_dict,
            "metrics": metrics,
            "history": self.history,
            "stage": self.stage_name,
            "metadata": {
                "trained_with_injection": self.trained_with_injection,
                "stage": self.stage_name,
                "epoch": epoch,
                "val_recall": metrics.get("recall_harmful", metrics.get("risk_recall", 0.0)),
                "dataset_stats": self.dataset_stats,
                "config_hash": config_hash,
            },
        }

        if is_best:
            path = self.checkpoint_dir / f"best{suffix}.pt"
        else:
            path = self.checkpoint_dir / f"epoch_{epoch}{suffix}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            try:
                latest_path.unlink()
            except Exception:
                pass
        try:
            latest_path.symlink_to(path.name)
        except (OSError, NotImplementedError, Exception):
            shutil.copy2(path, latest_path)

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception as e:
            logger.warning(f"Strict model load failed ({e}), attempting non-strict shape-filtered load...")
            current_sd = self.model.state_dict()
            compatible_sd = {
                k: v for k, v in checkpoint["model_state_dict"].items()
                if k in current_sd and current_sd[k].shape == v.shape
            }
            self.model.load_state_dict(compatible_sd, strict=False)

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"Optimizer state loading failed ({e}), continuing with fresh optimizer state.")

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Scheduler state loading failed ({e}), keeping default scheduler.")

        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                logger.warning(f"Scaler state loading failed ({e})")

        self.history = checkpoint.get("history", self.history)
        self.best_metric = checkpoint.get("metrics", {}).get(self.checkpoint_metric, float("-inf"))
        self.best_loc_iou = checkpoint.get("metrics", {}).get("localization_iou", float("-inf"))

        start_epoch = checkpoint["epoch"] + 1
        if self.epochs < start_epoch:
            additional_epochs = 10
            new_epochs = checkpoint["epoch"] + additional_epochs
            logger.info(
                f"Configured epochs ({self.epochs}) is <= checkpoint epoch ({checkpoint['epoch']}). "
                f"Automatically extending total epochs to {new_epochs}."
            )
            self.epochs = new_epochs

        # Ensure a healthy learning rate for continuation epochs (e.g. 5e-5)
        # instead of the near-zero learning rate from the tail of the previous cosine decay
        curr_lr = self.optimizer.param_groups[0]["lr"]
        if curr_lr < 1e-5:
            target_cont_lr = float(self.config.get("continuation_lr", 5e-5))
            for pg in self.optimizer.param_groups:
                pg["lr"] = target_cont_lr
            logger.info(f"Adjusted continuation learning rate from {curr_lr:.2e} to {target_cont_lr:.2e}")

        logger.info(f"Resumed from epoch {checkpoint['epoch']}, best metric: {self.best_metric:.4f}")

        return start_epoch


def create_optimizer(model: SentinelModel, config: DictConfig) -> torch.optim.Optimizer:
    """Create optimizer with parameter groups for different learning rates."""
    train_config = config.get("training", {})

    # Different LR for backbone vs heads
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "clip_backbone" in name:
            if not param.requires_grad:
                continue  # frozen CLIP backbone
        if "frame_encoder.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    base_lr = float(train_config.get("learning_rate", 3e-4))
    backbone_lr = float(train_config.get("backbone_lr", base_lr * 0.1))
    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": base_lr},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_config.get("weight_decay", 1e-4),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: DictConfig) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler with warmup + cosine decay."""
    train_config = config.get("training", {})
    epochs = train_config.get("epochs", 35)
    warmup_epochs = train_config.get("warmup_epochs", 3)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # Cosine decay
        progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def create_data_loaders(
    config: DictConfig,
    train_dataset,
    val_dataset,
    train_sampler: Optional[torch.utils.data.Sampler] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders. If train_sampler is given
    (e.g. a WeightedRandomSampler for harmful-class upsampling), it is used
    instead of shuffle=True."""
    train_config = config.get("training", {})

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=train_config.get("pin_memory", True),
        collate_fn=collate_frame_windows,
        drop_last=True,
        persistent_workers=train_config.get("num_workers", 4) > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=False,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=train_config.get("pin_memory", True),
        collate_fn=collate_frame_windows,
        drop_last=False,
        persistent_workers=train_config.get("num_workers", 4) > 0,
    )

    return train_loader, val_loader


# Need to import math at top
import math
from ..utils.checkpoint import load_checkpoint