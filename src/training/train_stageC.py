"""
Stage C Training: Contextual Edge Cases.
Trains on all harmful categories (destructive, financial, privacy,
irreversible_external) using the combined real + synthetic-injection
dataset, with everything unfrozen and hard example mining from Stage B's
validation run prioritized in the training mix.
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..models.sentinel_model import SentinelModel
from ..training.trainer import (
    SentinelTrainer,
    create_optimizer,
    create_scheduler,
    create_data_loaders,
)
from ..training.losses import create_loss_function
from ..training.curriculum_data import build_combined_dataset, dataset_stats_dict
from ..data.augmentation import create_train_transform, create_val_transform
from ..data.frame_windowing import collate_frame_windows
from ..utils.logging import setup_logging
from ..utils.config import load_config

logger = logging.getLogger(__name__)


def mine_hard_examples(
    model: SentinelModel,
    val_dataset,
    config: DictConfig,
    device: str,
    confidence_threshold: float = 0.6,
    top_k: float = 0.2,
) -> list:
    """
    Mine hard examples from a validation dataset using an already-loaded
    model. Hard examples = low-confidence correct predictions or
    high-confidence wrong predictions.
    """
    from torch.utils.data import DataLoader

    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.get("batch_size", 16),
        shuffle=False,
        num_workers=config.training.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_frame_windows,
    )

    hard_examples = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            frames = batch["frames"]
            risk_labels = batch["risk_label"]
            category_labels = batch["category_label"]

            output = model(frames)

            risk_scores = output["risk_score"].squeeze(-1)  # (B,)
            category_probs = output["category_probs"]  # (B, 5)
            category_preds = category_probs.argmax(dim=-1)

            for i in range(frames.shape[0]):
                is_harmful = risk_labels[i].item() == 1
                pred_harmful = risk_scores[i].item() > 0.5
                pred_conf = risk_scores[i].item() if pred_harmful else 1 - risk_scores[i].item()
                cat_correct = (category_preds[i] == category_labels[i]).item()

                is_hard = False
                if is_harmful and pred_conf < confidence_threshold:
                    is_hard = True  # Missed/uncertain harmful
                elif not is_harmful and pred_conf > (1 - confidence_threshold):
                    is_hard = True  # False positive
                elif is_harmful and not cat_correct:
                    is_hard = True  # Wrong category

                if is_hard:
                    hard_examples.append({
                        "risk_score": risk_scores[i].item(),
                        "true_label": risk_labels[i].item(),
                        "true_category": category_labels[i].item(),
                        "pred_category": category_preds[i].item(),
                        "difficulty": 1.0 - pred_conf if is_harmful else pred_conf,
                    })

    hard_examples.sort(key=lambda x: x["difficulty"], reverse=True)
    n_select = int(len(hard_examples) * top_k)
    model.train()
    return hard_examples[:n_select]


@hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
def train_stage_c(config: DictConfig) -> DictConfig:
    """Main entry point for Stage C training."""
    setup_logging(config.get("log_level", "INFO"))
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Stage C Training: Contextual Edge Cases")
    logger.info("=" * 60)

    curriculum_config = load_config("curriculum.yaml")
    stage_config = curriculum_config["stage_c"]

    data_config = load_config("data.yaml")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, data_config, stage_config)
    config.stage_name = "stage_c"
    if "num_epochs" in stage_config:
        config.training.epochs = stage_config["num_epochs"]

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    stage_b_checkpoint = config.get("stage_b_checkpoint", "checkpoints/stage_b/best.pt")
    logger.info(f"Loading Stage B checkpoint: {stage_b_checkpoint}")

    # As in Stage B: build the model from the checkpoint's own saved config,
    # not the caller-supplied config, to avoid architecture mismatches.
    model = SentinelModel.from_pretrained(stage_b_checkpoint)
    model = model.to(device)
    model.unfreeze_backbone()
    logger.info("Stage B weights loaded, full model unfrozen for Stage C")

    logger.info("Loading datasets...")
    train_transform = create_train_transform(OmegaConf.to_container(config))
    val_transform = create_val_transform(OmegaConf.to_container(config))

    synthetic_categories = list(config.get(
        "target_categories", ["destructive", "financial", "privacy", "irreversible_external"]
    ))
    synthetic_categories = [c for c in synthetic_categories if c != "benign"] or \
        ["destructive", "financial", "privacy", "irreversible_external"]

    train_dataset = build_combined_dataset(config, "train", train_transform, synthetic_categories)
    val_dataset = build_combined_dataset(config, "val", val_transform, synthetic_categories)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    if config.get("hard_example_mining", False):
        logger.info("Mining hard examples from Stage B's validation run...")
        hard_examples = mine_hard_examples(
            model=model,
            val_dataset=val_dataset,
            config=config,
            device=device,
            confidence_threshold=config.get("confidence_threshold", 0.6),
            top_k=config.get("hard_example_top_k", 0.2),
        )
        logger.info(f"Mined {len(hard_examples)} hard examples "
                    f"(will inform sampler weighting; harmful ratio in hard set: "
                    f"{sum(h['true_label'] for h in hard_examples) / max(len(hard_examples), 1):.2f})")

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
        stage_name="stage_c",
        trained_with_injection=dataset_stats["train_synthetic"] > 0,
        dataset_stats=dataset_stats,
    )

    resume_path = config.get("resume_from", None)
    if resume_path:
        start_epoch = trainer.resume_from_checkpoint(resume_path)
        logger.info(f"Resumed from epoch {start_epoch}")

    results = trainer.train()

    logger.info("Stage C training completed!")
    logger.info(f"Best val_recall_harmful: {results['best_metric']:.4f}")

    return results


if __name__ == "__main__":
    train_stage_c()
