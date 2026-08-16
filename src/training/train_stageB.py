"""
Stage B Training: Financial/Privacy Actions.
Adds financial and privacy categories, unfreezes backbone.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging

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
def train_stage_b(config: DictConfig) -> DictConfig:
    """Main entry point for Stage B training."""
    setup_logging(config.get("log_level", "INFO"))
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Stage B Training: Financial/Privacy")
    logger.info("=" * 60)

    # Load curriculum config
    curriculum_config = load_config("curriculum.yaml")
    stage_config = curriculum_config["stage_b"]

    # Load data config (paths to processed real dataset) and merge it in so
    # SentinelDataset can read `processed_dir`, `frame_window`, etc.
    data_config = load_config("data.yaml")

    # Merge stage-specific config
    config = OmegaConf.merge(config, data_config, stage_config)
    config.stage_name = "stage_b"

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Stage A checkpoint
    stage_a_checkpoint = config.get("stage_a_checkpoint", "checkpoints/stage_a/best.pt")
    logger.info(f"Loading Stage A checkpoint: {stage_a_checkpoint}")

    # Create datasets with new target categories
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

    # Create model and load Stage A weights
    logger.info("Creating model...")
    model = create_sentinel_model(config)
    model = model.to(device)

    # Load Stage A checkpoint
    checkpoint = torch.load(stage_a_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    logger.info("Stage A weights loaded")

    # Unfreeze backbone for Stage B
    if not config.get("freeze_backbone", False):
        model.unfreeze_backbone()
        logger.info("Backbone unfrozen for Stage B")

    # Create loss (may have updated class weights)
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
        stage_name="stage_b",
    )

    # Train
    results = trainer.train()

    logger.info("Stage B training completed!")
    logger.info(f"Best val_recall_harmful: {results['best_metric']:.4f}")

    return results


if __name__ == "__main__":
    train_stage_b()