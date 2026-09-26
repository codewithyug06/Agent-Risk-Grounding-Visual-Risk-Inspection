#!/usr/bin/env python3
"""
Pre-trains the temporal fusion module with an InfoNCE contrastive objective
before supervised curriculum training (Stage A/B/C).

Pulls temporally adjacent frames (positive pairs) from the same trajectory
close together in representation space and pushes frames from different
trajectories (negatives, via the InfoNCE batch) apart. Uses
ContrastiveTemporalLoss from src/training/losses.py.

The frame encoder backbone stays frozen; only temporal_fusion parameters are
optimized. Runs on the real dataset only (benign-only trajectories are fine —
contrastive pretraining doesn't need risk labels).
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.models.sentinel_model import create_sentinel_model
from src.data.loaders import SentinelDataset
from src.data.augmentation import create_train_transform
from src.data.frame_windowing import collate_frame_windows
from src.training.losses import ContrastiveTemporalLoss
from src.utils.config import load_config
from src.utils.logging import setup_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_contrastive_pretrain(
    output_path: str = "checkpoints/contrastive_pretrain/temporal_fusion.pt",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_config = load_config("data.yaml")
    model_config = load_config("model_small")
    config = OmegaConf.merge(model_config, data_config)

    model = create_sentinel_model(config).to(device)
    model.freeze_backbone()
    for p in model.risk_head.parameters():
        p.requires_grad = False
    for p in model.localization_head.parameters():
        p.requires_grad = False
    logger.info("Frame encoder backbone, risk head, and localization head frozen. "
                "Only temporal_fusion parameters will be optimized.")

    transform = create_train_transform(OmegaConf.to_container(config))
    train_dataset = SentinelDataset(
        data_config=OmegaConf.to_container(config),
        split="train",
        transform=transform,
        frame_window_k=config.frame_window.k,
        target_resolution=tuple(config.frame_window.resolution),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.training.get("num_workers", 0),
        collate_fn=collate_frame_windows,
        drop_last=True,
    )
    logger.info(f"Train samples: {len(train_dataset)}")

    contrastive_loss = ContrastiveTemporalLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.temporal_fusion.parameters(), lr=learning_rate)

    model.temporal_fusion.train()
    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            frames = batch["frames"].to(device)  # (B, k, C, H, W)

            with torch.no_grad():
                frame_embeddings = model.frame_encoder(frames)  # (B, k, N, D) — frozen backbone

            optimizer.zero_grad()

            # Run the actual temporal_fusion transformer blocks (the module
            # being optimized) per-frame, without collapsing across time yet,
            # so the contrastive loss's gradient flows into temporal_fusion's
            # parameters instead of bypassing them.
            tf = model.temporal_fusion
            x = tf.temporal_pos_enc(frame_embeddings)
            if tf.use_delta_features and frame_embeddings.shape[1] > 1:
                delta = tf.delta_extractor(frame_embeddings)
                delta = torch.nn.functional.pad(delta, (0, 0, 0, 0, 1, 0))
                x = torch.cat([x, delta], dim=-1)
                x = tf.delta_proj(x)
            for block in tf.blocks:
                x = block(x)
            x = tf.norm(x)  # (B, k, N, D)

            pooled = x.mean(dim=2)  # (B, k, D) — mean-pool patches per frame

            loss = contrastive_loss(pooled)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        logger.info(f"Epoch {epoch}/{epochs} — contrastive loss: {avg_loss:.4f}")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "temporal_fusion_state_dict": model.temporal_fusion.state_dict(),
        "epochs": epochs,
        "history": history,
        "metadata": {
            "trained_with_injection": False,
            "stage": "contrastive",
            "epoch": epochs,
        },
    }, output_path_obj)
    logger.info(f"Saved pretrained temporal fusion weights to {output_path_obj}")
    return str(output_path_obj)


def main():
    parser = argparse.ArgumentParser(description="Contrastive temporal pretraining for SENTINEL-Vision.")
    parser.add_argument("--output", default="checkpoints/contrastive_pretrain/temporal_fusion.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    setup_logging("INFO")
    run_contrastive_pretrain(
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
