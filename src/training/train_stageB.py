"""
Stage B Training: Financial/Privacy Actions.
Adds financial and privacy categories to the destructive-only Stage A curriculum,
using the real weak-labeled dataset combined with DOM-grounded synthetic
injection samples. Unfreezes the backbone fully. Adds Online Hard Example
Mining (OHEM) on top of the risk-classification loss.
"""

import logging

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from ..models.sentinel_model import SentinelModel
from ..training.trainer import (
    SentinelTrainer,
    create_optimizer,
    create_scheduler,
    create_data_loaders,
)
from ..training.losses import SentinelLoss, OHEMLoss, create_loss_function
from ..training.curriculum_data import build_combined_dataset, dataset_stats_dict
from ..data.augmentation import create_train_transform, create_val_transform
from ..utils.logging import setup_logging
from ..utils.config import load_config

logger = logging.getLogger(__name__)


class OHEMAugmentedLoss(nn.Module):
    """
    Wraps SentinelLoss and adds an Online Hard Example Mining term on the
    risk-classification logits: within each batch, only the hardest
    keep_ratio fraction of samples (by per-sample BCE loss) contribute to
    this extra term, focusing gradient on the examples the model currently
    gets most wrong.
    """

    def __init__(self, base_loss: SentinelLoss, keep_ratio: float = 0.7, ohem_weight: float = 0.5):
        super().__init__()
        self.base_loss = base_loss
        self.ohem = OHEMLoss(loss_fn=nn.BCEWithLogitsLoss(reduction="none"), keep_ratio=keep_ratio)
        self.ohem_weight = ohem_weight

    def forward(self, predictions, targets):
        loss_dict = self.base_loss(predictions, targets)
        risk_logits = predictions["risk_logits"].squeeze(-1)
        risk_labels = targets["risk_label"].float()
        ohem_term = self.ohem(risk_logits, risk_labels)
        loss_dict["ohem_loss"] = ohem_term
        loss_dict["total_loss"] = loss_dict["total_loss"] + self.ohem_weight * ohem_term
        return loss_dict


@hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
def train_stage_b(config: DictConfig) -> DictConfig:
    """Main entry point for Stage B training."""
    setup_logging(config.get("log_level", "INFO"))
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Stage B Training: Financial/Privacy")
    logger.info("=" * 60)

    curriculum_config = load_config("curriculum.yaml")
    stage_config = curriculum_config["stage_b"]

    data_config = load_config("data.yaml")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, data_config, stage_config)
    config.stage_name = "stage_b"
    if "num_epochs" in stage_config:
        config.training.epochs = stage_config["num_epochs"]

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    stage_a_checkpoint = config.get("stage_a_checkpoint", "checkpoints/stage_a/best.pt")
    logger.info(f"Loading Stage A checkpoint: {stage_a_checkpoint}")

    # Load the model architecture from the Stage A checkpoint's own saved
    # config, not the caller-supplied config — the two can diverge (e.g.
    # risk_head hidden_dim) and building the model from the wrong config
    # produces a cryptic state_dict size-mismatch error instead of loading.
    model = SentinelModel.from_pretrained(stage_a_checkpoint)
    model = model.to(device)
    logger.info("Stage A weights loaded")

    if not config.get("freeze_backbone", False):
        model.unfreeze_backbone()
        logger.info("Backbone unfrozen for Stage B")

    logger.info("Loading datasets...")
    train_transform = create_train_transform(OmegaConf.to_container(config))
    val_transform = create_val_transform(OmegaConf.to_container(config))

    synthetic_categories = list(config.get("target_categories", ["destructive", "financial", "privacy"]))
    synthetic_categories = [c for c in synthetic_categories if c != "benign"] or ["destructive", "financial", "privacy"]

    train_dataset = build_combined_dataset(config, "train", train_transform, synthetic_categories)
    val_dataset = build_combined_dataset(config, "val", val_transform, synthetic_categories)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    base_loss = create_loss_function(OmegaConf.to_container(config))
    loss_fn = OHEMAugmentedLoss(
        base_loss,
        keep_ratio=config.get("ohem_keep_ratio", 0.7),
        ohem_weight=config.get("ohem_weight", 0.5),
    )

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
        stage_name="stage_b",
        trained_with_injection=dataset_stats["train_synthetic"] > 0,
        dataset_stats=dataset_stats,
    )

    resume_path = config.get("resume_from", None)
    if resume_path:
        start_epoch = trainer.resume_from_checkpoint(resume_path)
        logger.info(f"Resumed from epoch {start_epoch}")

    results = trainer.train()

    logger.info("Stage B training completed!")
    logger.info(f"Best val_recall_harmful: {results['best_metric']:.4f}")

    return results


if __name__ == "__main__":
    train_stage_b()
