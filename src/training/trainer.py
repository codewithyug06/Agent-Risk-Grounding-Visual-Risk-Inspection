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
    ):
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

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", f"checkpoints/{stage_name}"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("-inf")
        self.checkpoint_metric = config.get("checkpoint_metric", "recall_harmful")
        self.early_stopping_patience = config.get("early_stopping_patience", 5)
        self.early_stopping_counter = 0

        # Mixed precision
        self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        try:
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.mixed_precision)
        except Exception:
            self.scaler = GradScaler(enabled=self.mixed_precision)

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

                # Checkpointing
                metric_value = val_metrics.get(self.checkpoint_metric, val_metrics["loss"])
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                    self.early_stopping_counter = 0
                    logger.info(f"New best {self.checkpoint_metric}: {metric_value:.4f}")
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

        # Save final checkpoint
        self._save_checkpoint(self.epochs, self.history, is_best=False, suffix="_final")

        logger.info(f"Training completed. Best {self.checkpoint_metric}: {self.best_metric:.4f}")
        return {"best_metric": self.best_metric, "history": self.history}

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        loss_components = {}

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            frames = batch["frames"]  # (B, k, C, H, W)
            targets = {
                "risk_label": batch["risk_label"],
                "category_label": batch["category_label"],
                "bbox": batch["bbox"],
                "has_bbox": batch["has_bbox"],
            }

            self.optimizer.zero_grad()

            try:
                autocast_ctx = torch.amp.autocast(self.device_type, enabled=self.mixed_precision)
            except Exception:
                autocast_ctx = autocast(enabled=self.mixed_precision)

            with autocast_ctx:
                predictions = self.model(frames)
                loss_dict = self.loss_fn(predictions, targets)
                loss = loss_dict["total_loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

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
                "has_bbox": batch["has_bbox"],
            }

            try:
                autocast_ctx = torch.amp.autocast(self.device_type, enabled=self.mixed_precision)
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
            all_preds["bbox"].append(predictions["bbox"].cpu())

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

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Harmful recall (recall on positive class only)
        recall_harmful = recall.item()

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

        return {
            "risk_precision": precision.item(),
            "risk_recall": recall.item(),
            "risk_f1": f1.item(),
            "recall_harmful": recall_harmful,
            "category_accuracy": cat_acc,
            "localization_iou": localization_iou,
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

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        suffix: str = "",
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "config": OmegaConf.to_container(self.config, resolve=True),
            "metrics": metrics,
            "history": self.history,
            "stage": self.stage_name,
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_metric = checkpoint.get("metrics", {}).get(self.checkpoint_metric, float("-inf"))

        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, best metric: {self.best_metric:.4f}")

        return start_epoch


def create_optimizer(model: SentinelModel, config: DictConfig) -> torch.optim.Optimizer:
    """Create optimizer with parameter groups for different learning rates."""
    train_config = config.get("training", {})

    # Different LR for backbone vs heads
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "frame_encoder.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": train_config.get("backbone_lr", train_config.get("learning_rate", 3e-4) * 0.1)},
        {"params": head_params, "lr": train_config.get("learning_rate", 3e-4)},
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
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    train_config = config.get("training", {})

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=True,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=train_config.get("pin_memory", True),
        collate_fn=collate_frame_windows,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=False,
        num_workers=train_config.get("num_workers", 4),
        pin_memory=train_config.get("pin_memory", True),
        collate_fn=collate_frame_windows,
        drop_last=False,
        persistent_workers=True,
    )

    return train_loader, val_loader


# Need to import math at top
import math