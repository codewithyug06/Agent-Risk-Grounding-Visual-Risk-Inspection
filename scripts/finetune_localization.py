#!/usr/bin/env python3
"""
Dedicated localization fine-tuning pass for SENTINEL-Vision.

The full curriculum (Stage A/B/C) leaves bounding-box localization weak
(IoU 0.03-0.12 on validation — see reports/system_report.md SS2) because the
risk/category classification loss dominates training and the only bbox
supervision comes from a small slice of the synthetic injection data.

This script fine-tunes ONLY the localization pathway (frame_encoder +
temporal_fusion + localization_head; risk_head is frozen so classification
knowledge from the curriculum isn't disturbed) using a combined dataset:

  1. ScreenSpot (bevaya/ScreenSpot, HF Hub, already cached locally) — 1,272
     real UI screenshots with human-annotated (instruction, bbox) pairs
     across windows/web/mobile. Bboxes are already normalized [0,1],
     matching this project's convention exactly. Since it provides single
     static screenshots (not a temporal window), each image is repeated k
     times to form a degenerate static k-frame window — a standard, valid
     way to feed single-image grounding data through a video-window model
     (the temporal-delta signal is simply zero for these samples).
  2. SyntheticInjectionDataset's harmful categories (destructive/financial/
     privacy/irreversible_external) — real k=6 windows with DOM-grounded
     bboxes, already used in Stage C but a small fraction of that stage's
     mixed classification+localization objective.

Only a pure GIoU + L1 bbox loss is backpropagated — no risk/category signal
touches this pass, so it cannot regress the classification quality already
achieved by curriculum training.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from src.models.sentinel_model import SentinelModel
from src.data.loaders import SyntheticInjectionDataset
from src.data.augmentation import ValidationAugmentation
from src.data.frame_windowing import collate_frame_windows
from src.training.losses import AnchorMatchedLocalizationLoss
from src.utils.visualization import visualize_predictions
from src.utils.logging import setup_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HARMFUL_CATEGORIES = ["destructive", "financial", "privacy", "irreversible_external"]
CATEGORY_NAMES = ["destructive", "financial", "privacy", "irreversible_external", "benign"]


class ScreenSpotDataset(Dataset):
    """
    Wraps bevaya/ScreenSpot (HF Hub) as a SENTINEL-Vision-compatible dataset:
    each single screenshot is repeated k times to form a degenerate static
    frame window. No risk/category labels — has_bbox=1, bbox=real, everything
    else is a placeholder that the (frozen) risk_head output is never
    supervised against.
    """

    def __init__(self, indices, k: int = 6, target_resolution=(224, 224)):
        from datasets import load_dataset
        self._hf_dataset = load_dataset("bevaya/ScreenSpot", split="test")
        self.indices = indices
        self.k = k
        self.transform = ValidationAugmentation(target_resolution=target_resolution)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self._hf_dataset[self.indices[idx]]
        image = item["image"].convert("RGB")
        frame_tensor = self.transform(image)  # (C, H, W), [0,1]
        frames = frame_tensor.unsqueeze(0).repeat(self.k, 1, 1, 1)  # (k, C, H, W)

        bbox = torch.tensor(item["bbox"], dtype=torch.float32).clamp(0.0, 1.0)
        if bbox[2] <= bbox[0]:
            bbox[2] = min(1.0, bbox[0] + 1e-3)
        if bbox[3] <= bbox[1]:
            bbox[3] = min(1.0, bbox[1] + 1e-3)

        return {
            "frames": frames,
            "risk_label": torch.tensor(0),
            "category_label": torch.tensor(4),  # unused (risk_head frozen)
            "bbox": bbox,
            "has_bbox": torch.tensor(1.0),
        }


def build_datasets(k: int, target_resolution, screenspot_val_frac: float = 0.15, seed: int = 42):
    n_screenspot = len(__import__("datasets").load_dataset("bevaya/ScreenSpot", split="test"))
    indices = list(range(n_screenspot))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n_screenspot * screenspot_val_frac))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    ss_train = ScreenSpotDataset(train_idx, k=k, target_resolution=target_resolution)
    ss_val = ScreenSpotDataset(val_idx, k=k, target_resolution=target_resolution)

    syn_train = SyntheticInjectionDataset(
        "data/synthetic_injections", split="train", k=k,
        target_resolution=target_resolution, categories=HARMFUL_CATEGORIES,
    )
    syn_val = SyntheticInjectionDataset(
        "data/synthetic_injections", split="val", k=k,
        target_resolution=target_resolution, categories=HARMFUL_CATEGORIES,
    )

    train_dataset = ConcatDataset([ss_train, syn_train])
    val_dataset = ConcatDataset([ss_val, syn_val])
    logger.info(f"Train: {len(ss_train)} ScreenSpot + {len(syn_train)} synthetic = {len(train_dataset)}")
    logger.info(f"Val: {len(ss_val)} ScreenSpot + {len(syn_val)} synthetic = {len(val_dataset)}")
    return train_dataset, val_dataset


def compute_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    x1 = max(pred[0].item(), gt[0].item())
    y1 = max(pred[1].item(), gt[1].item())
    x2 = min(pred[2].item(), gt[2].item())
    y2 = min(pred[3].item(), gt[3].item())
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_p = (pred[2] - pred[0]).item() * (pred[3] - pred[1]).item()
    area_g = (gt[2] - gt[0]).item() * (gt[3] - gt[1]).item()
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0


def _clamp_bbox(bbox):
    """Sort/clip a possibly-invalid (x2<x1 or y2<y1, or outside [0,1]) box
    into a drawable one. Early-training predictions can be invalid this
    way; visualize_predictions/PIL's rectangle() crashes otherwise."""
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-3)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-3)
    return (x1, y1, x2, y2)


def save_sample_images(model, val_loader, device, out_dir: Path, n_samples: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"].to(device)
            gt_bbox = batch["bbox"]
            out = model(frames)
            pred_bbox = out["bbox"].cpu()

            last_frame = frames[:, -1].cpu()
            for i in range(last_frame.shape[0]):
                if saved >= n_samples:
                    model.train()
                    return
                img_t = last_frame[i]
                img = Image.fromarray((img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8"))
                vis = visualize_predictions(
                    image=img, risk_score=0.0, category="bbox_only", category_conf=0.0,
                    bbox=_clamp_bbox(pred_bbox[i].tolist()), normalized=True,
                )
                vis.save(out_dir / f"sample_{saved}_pred.png")
                gt_vis = visualize_predictions(
                    image=img, risk_score=0.0, category="ground_truth", category_conf=0.0,
                    bbox=_clamp_bbox(gt_bbox[i].tolist()), normalized=True,
                )
                gt_vis.save(out_dir / f"sample_{saved}_gt.png")
                saved += 1
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Dedicated localization fine-tuning for SENTINEL-Vision.")
    parser.add_argument("--checkpoint", default="checkpoints/stage_c/best.pt")
    parser.add_argument("--output-dir", default="checkpoints/localization_finetune")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    setup_logging("INFO")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = SentinelModel.from_pretrained(args.checkpoint).to(device)
    for p in model.risk_head.parameters():
        p.requires_grad = False
    logger.info("risk_head frozen — this pass only touches localization.")

    train_dataset, val_dataset = build_datasets(k=model.k, target_resolution=(model.image_size, model.image_size))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, collate_fn=collate_frame_windows, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=1, collate_fn=collate_frame_windows,
    )

    loss_fn = AnchorMatchedLocalizationLoss(giou_weight=1.0, l1_weight=0.5, obj_weight=1.0)
    anchors = model.localization_head.anchors  # (H*W*A, 4) fixed pixel-coord geometry
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            frames = batch["frames"].to(device)
            gt_bbox = batch["bbox"].to(device)
            has_bbox = batch["has_bbox"].to(device)

            optimizer.zero_grad()
            out = model(frames)
            gt_bbox_pixel = gt_bbox * model.image_size
            loss_dict = loss_fn(
                out["bbox_pred_grid"], out["objectness_logits_grid"], anchors,
                gt_bbox_pixel, has_bbox=has_bbox, image_size=model.image_size,
            )
            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                gt_bbox = batch["bbox"]
                out = model(frames)
                pred_bbox = out["bbox"].cpu()
                for i in range(pred_bbox.shape[0]):
                    ious.append(compute_iou(pred_bbox[i], gt_bbox[i]))
        val_iou = sum(ious) / max(len(ious), 1)

        logger.info(f"Epoch {epoch}/{args.epochs} — train_loss={train_loss:.4f} val_iou={val_iou:.4f}")

        write_header = not metrics_csv.exists()
        with open(metrics_csv, "a") as f:
            if write_header:
                f.write("epoch,train_loss,val_iou\n")
            f.write(f"{epoch},{train_loss},{val_iou}\n")

        save_sample_images(model, val_loader, device, out_dir / "images" / f"epoch_{epoch:03d}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "metrics": {"val_iou": val_iou, "train_loss": train_loss},
                # Store as a plain dict, not a raw OmegaConf DictConfig — the
                # latter isn't in torch's weights_only=True safe-globals
                # allowlist and breaks the safe checkpoint loader used
                # throughout this project (src/utils/checkpoint.py).
                "config": (
                    OmegaConf.to_container(model.config, resolve=True)
                    if hasattr(model, "config") and isinstance(model.config, DictConfig)
                    else (model.config if hasattr(model, "config") else {})
                ),
                "metadata": {
                    "stage": "localization_finetune",
                    "base_checkpoint": args.checkpoint,
                    "epoch": epoch,
                    "val_iou": val_iou,
                },
            }, out_dir / "best.pt")
            logger.info(f"New best val_iou: {val_iou:.4f} — saved {out_dir / 'best.pt'}")

    logger.info(f"Localization fine-tuning complete. Best val_iou: {best_iou:.4f}")


if __name__ == "__main__":
    main()
