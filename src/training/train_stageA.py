"""
Stage A Training: Obvious Harm Detection.
Trains on destructive actions with high visual salience (delete, format, rm -rf)
using the real weak-labeled dataset combined with DOM-grounded synthetic
injection samples (destructive category only). Backbone stays frozen for the
first `freeze_backbone_epochs` epochs, then unfreezes.
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..models.sentinel_model import create_sentinel_model
from ..data.augmentation import create_train_transform, create_val_transform
from ..training.trainer import (
    SentinelTrainer,
    create_optimizer,
    create_scheduler,
    create_data_loaders,
)
from ..training.losses import create_loss_function
from ..training.curriculum_data import build_combined_dataset, dataset_stats_dict
from ..utils.logging import setup_logging
from ..utils.config import load_config

logger = logging.getLogger(__name__)


class _UnfreezeBackboneCallback:
    """Unfreezes the frame-encoder backbone at a given epoch during training."""

    def __init__(self, model, unfreeze_at_epoch: int):
        self.model = model
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self._done = False

    def maybe_unfreeze(self, epoch: int):
        if not self._done and epoch >= self.unfreeze_at_epoch:
            self.model.unfreeze_backbone()
            logger.info(f"Backbone unfroze at epoch {epoch}")
            self._done = True


@hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
def train_stage_a(config: DictConfig) -> DictConfig:
    """Main entry point for Stage A training."""
    setup_logging(config.get("log_level", "INFO"))
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Stage A Training: Obvious Harm")
    logger.info("=" * 60)

    curriculum_config = load_config("curriculum.yaml")
    stage_config = curriculum_config["stage_a"]

    data_config = load_config("data.yaml")
    # Hydra loads `config` in struct mode (strict schema), which rejects
    # merging in keys that aren't already present — such as `processed_dir`
    # from data.yaml. Disable struct mode for this merge only.
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, data_config, stage_config)
    config.stage_name = "stage_a"
    # curriculum.yaml's per-stage `num_epochs` must be propagated into
    # `training.epochs`, which is what SentinelTrainer actually reads —
    # without this, every stage silently trains for model_small.yaml's
    # default of 35 epochs instead of the curriculum's intended count.
    if "num_epochs" in stage_config:
        config.training.epochs = stage_config["num_epochs"]

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    train_transform = create_train_transform(OmegaConf.to_container(config))
    val_transform = create_val_transform(OmegaConf.to_container(config))

    synthetic_categories = list(config.get("target_categories", ["destructive"]))
    synthetic_categories = [c for c in synthetic_categories if c != "benign"] or ["destructive"]

    train_dataset = build_combined_dataset(config, "train", train_transform, synthetic_categories)
    val_dataset = build_combined_dataset(config, "val", val_transform, synthetic_categories)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    logger.info("Creating model...")
    model = create_sentinel_model(config)
    model = model.to(device)

    freeze_backbone_epochs = config.get("freeze_backbone_epochs", 3)
    if config.get("freeze_backbone", True):
        model.freeze_backbone()
        logger.info(f"Backbone frozen for Stage A (unfreezes at epoch {freeze_backbone_epochs + 1})")
    unfreeze_cb = _UnfreezeBackboneCallback(model, freeze_backbone_epochs + 1)

    loss_fn = create_loss_function(OmegaConf.to_container(config))
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    train_sampler = train_dataset.get_weighted_sampler()
    train_loader, val_loader = create_data_loaders(config, train_dataset, val_dataset, train_sampler=train_sampler)

    dataset_stats = dataset_stats_dict(train_dataset, val_dataset)

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
        trained_with_injection=dataset_stats["train_synthetic"] > 0,
        dataset_stats=dataset_stats,
    )

    # Wire the unfreeze callback into the per-epoch hook the trainer already calls.
    original_set_epoch = model.set_epoch

    def set_epoch_with_unfreeze(epoch):
        original_set_epoch(epoch)
        unfreeze_cb.maybe_unfreeze(epoch)

    model.set_epoch = set_epoch_with_unfreeze

    resume_path = config.get("resume_from", None)
    if resume_path:
        start_epoch = trainer.resume_from_checkpoint(resume_path)
        logger.info(f"Resumed from epoch {start_epoch}")

    results = trainer.train()

    logger.info("Stage A training completed!")
    logger.info(f"Best val_recall_harmful: {results['best_metric']:.4f}")

    return results


if __name__ == "__main__":
    train_stage_a()
