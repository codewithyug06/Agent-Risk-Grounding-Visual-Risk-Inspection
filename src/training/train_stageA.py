"""
Stage A Training: Obvious Harm Detection.
Trains on destructive actions with high visual salience (delete, format, rm -rf).
Freezes backbone, trains temporal fusion + heads.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path

from ..models.sentinel_model import SentinelModel, create_sentinel_model
from ..data.loaders import SentinelDataset
from ..data.augmentation import create_train_transform, create_val_transform
from ..training.trainer import (
    SentinelTrainer,
    create_optimizer,
    create_scheduler,
    create_data_loaders,
)
from ..training.losses import create_loss_function
from ..utils.logging import setup_logging
from ..utils.config import load_config

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
def train_stage_a(config: DictConfig) -> DictConfig:
    """Main entry point for Stage A training."""
    # Setup logging
    setup_logging(config.get("log_level", "INFO"))
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Stage A Training: Obvious Harm")
    logger.info("=" * 60)

    # Load curriculum config
    curriculum_config = load_config("curriculum.yaml")
    stage_config = curriculum_config["stage_a"]

    # Load data config (paths to processed real dataset) and merge it in so
    # SentinelDataset can read `processed_dir`, `frame_window`, etc.
    data_config = load_config("data.yaml")

    # Merge stage-specific config
    config = OmegaConf.merge(config, data_config, stage_config)
    config.stage_name = "stage_a"

    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Loading datasets...")
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

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    logger.info("Creating model...")
    model = create_sentinel_model(config)
    model = model.to(device)

    # Freeze backbone for Stage A
    if config.get("freeze_backbone", True):
        model.freeze_backbone()
        logger.info("Backbone frozen for Stage A")

    # Create loss
    loss_fn = create_loss_function(OmegaConf.to_container(config))

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, train_dataset, val_dataset)

    # Create trainer
    trainer = SentinelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        stage_name="stage_a",
    )

    # Resume if checkpoint provided
    resume_path = config.get("resume_from", None)
    if resume_path:
        start_epoch = trainer.resume_from_checkpoint(resume_path)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Train
    results = trainer.train()

    logger.info("Stage A training completed!")
    logger.info(f"Best val_recall_harmful: {results['best_metric']:.4f}")

    return results


if __name__ == "__main__":
    train_stage_a()