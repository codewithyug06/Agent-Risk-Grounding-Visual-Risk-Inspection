"""
Stage C Training: Contextual Edge Cases.
Trains on irreversible_external + contextual harm (benign-looking but risky given prior frames).
Uses hard example mining from Stage B.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import logging
import json
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
from ..utils.checkpoint import load_checkpoint, CheckpointLoadError

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
def train_stage_c(config: DictConfig) -> DictConfig:
    """Main entry point for Stage C training."""
    setup_logging(config.get("log_level", "INFO"))
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Stage C Training: Contextual Edge Cases")
    logger.info("=" * 60)

    # Load curriculum config
    curriculum_config = load_config("curriculum.yaml")
    stage_config = curriculum_config["stage_c"]

    # Load data config (paths to processed real dataset) and merge it in so
    # SentinelDataset can read `processed_dir`, `frame_window`, etc.
    data_config = load_config("data.yaml")

    # Merge stage-specific config
    config = OmegaConf.merge(config, data_config, stage_config)
    config.stage_name = "stage_c"

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Stage B checkpoint. Prefer the actual file present in this repo's
    # checkpoints/ (stage_b_30epochs/best.pt) over the stale
    # checkpoints/stage_b/best.pt default that doesn't exist on disk.
    default_stage_b = (
        "checkpoints/stage_b_30epochs/best.pt"
        if Path("checkpoints/stage_b_30epochs/best.pt").exists()
        else "checkpoints/stage_b/best.pt"
    )
    stage_b_checkpoint = config.get("stage_b_checkpoint", default_stage_b)
    logger.info(f"Loading Stage B checkpoint: {stage_b_checkpoint}")

    # Hard example mining from Stage B
    if config.get("hard_example_mining", False):
        logger.info("Performing hard example mining from Stage B...")
        hard_examples = mine_hard_examples(
            stage_b_checkpoint=stage_b_checkpoint,
            config=config,
            device=device,
            confidence_threshold=config.get("confidence_threshold", 0.6),
            top_k=config.get("hard_example_top_k", 0.2),
        )
        logger.info(f"Mined {len(hard_examples)} hard examples")

        # KNOWN GAP (found during a hardening pass, not fixed here): nothing
        # downstream reads config.hard_examples. SentinelDataset has no
        # hard-example oversampling path, so mine_hard_examples() currently
        # runs, logs a count, and has zero effect on the actual training
        # loop below. Wiring this up properly also isn't a one-line fix:
        # mining runs against the *val* split (see mine_hard_examples above)
        # while the training loop samples from `train_dataset` -- a
        # WeightedRandomSampler would need hard examples mined from the
        # *train* split with indices that correspond to train_dataset's
        # ordering, not val_dataset's. Left as a documented gap rather than
        # a silent no-op or an unverified structural change.
        config.hard_examples = hard_examples
        logger.warning(
            "Hard example mining ran but is NOT wired into training -- "
            "SentinelDataset/create_data_loaders do not consume "
            "config.hard_examples. This stage currently trains as plain "
            "uniform sampling over train_dataset regardless of this flag."
        )

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

    # Create model and load Stage B weights
    logger.info("Creating model...")
    model = create_sentinel_model(config)
    model = model.to(device)

    try:
        checkpoint = load_checkpoint(stage_b_checkpoint, map_location=device)
    except CheckpointLoadError:
        checkpoint = load_checkpoint(stage_b_checkpoint, map_location=device, allow_unsafe=True)
    # NOTE: strict=False silently ignores missing/unexpected keys -- if the
    # architecture drifted between Stage B and Stage C (e.g. a changed head
    # shape), this will partially load rather than error. Left as-is here
    # (changing to strict=True is a behavior change that needs a real
    # training run to validate, which this environment cannot do), but
    # flagged: log which keys actually mismatched so a bad partial-load
    # isn't silent.
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        logger.warning(
            "Stage B -> Stage C checkpoint load was partial. missing_keys=%s unexpected_keys=%s",
            load_result.missing_keys,
            load_result.unexpected_keys,
        )
    logger.info("Stage B weights loaded")

    # Full unfreeze for Stage C
    model.unfreeze_backbone()
    logger.info("Full model unfrozen for Stage C")

    # Create loss
    loss_fn = create_loss_function(OmegaConf.to_container(config))

    # Create optimizer and scheduler (lower LR)
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
        stage_name="stage_c",
    )

    # Train
    results = trainer.train()

    logger.info("Stage C training completed!")
    logger.info(f"Best val_recall_harmful: {results['best_metric']:.4f}")

    return results


def mine_hard_examples(
    stage_b_checkpoint: str,
    config: DictConfig,
    device: str,
    confidence_threshold: float = 0.6,
    top_k: float = 0.2,
) -> list:
    """
    Mine hard examples from Stage B validation set.
    Hard examples = low confidence correct predictions or high confidence wrong predictions.
    """
    from ..data.loaders import SentinelDataset
    from ..data.augmentation import create_val_transform
    from torch.utils.data import DataLoader
    from ..data.frame_windowing import collate_frame_windows

    # Load Stage B model
    model = create_sentinel_model(config)
    model = model.to(device)
    try:
        checkpoint = load_checkpoint(stage_b_checkpoint, map_location=device)
    except CheckpointLoadError:
        checkpoint = load_checkpoint(stage_b_checkpoint, map_location=device, allow_unsafe=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create validation dataset
    val_transform = create_val_transform(OmegaConf.to_container(config))
    val_dataset = SentinelDataset(
        data_config=OmegaConf.to_container(config),
        split="val",
        transform=val_transform,
        frame_window_k=config.frame_window.k,
        target_resolution=tuple(config.frame_window.resolution),
    )

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

            # Identify hard examples
            for i in range(frames.shape[0]):
                is_harmful = risk_labels[i].item() == 1
                pred_harmful = risk_scores[i].item() > 0.5
                pred_conf = risk_scores[i].item() if pred_harmful else 1 - risk_scores[i].item()
                cat_correct = (category_preds[i] == category_labels[i]).item()

                # Hard example criteria:
                # 1. Harmful but low confidence (missed or uncertain)
                # 2. Benign but high confidence harmful (false positive)
                # 3. Wrong category prediction
                is_hard = False

                if is_harmful and pred_conf < confidence_threshold:
                    is_hard = True  # Missed/uncertain harmful
                elif not is_harmful and pred_conf > (1 - confidence_threshold):
                    is_hard = True  # False positive
                elif is_harmful and not cat_correct:
                    is_hard = True  # Wrong category

                if is_hard:
                    hard_examples.append({
                        "trajectory_id": batch.get("trajectory_id", [""])[i] if isinstance(batch.get("trajectory_id"), list) else "",
                        "action_idx": batch.get("action_idx", [0])[i] if isinstance(batch.get("action_idx"), list) else 0,
                        "risk_score": risk_scores[i].item(),
                        "true_label": risk_labels[i].item(),
                        "true_category": category_labels[i].item(),
                        "pred_category": category_preds[i].item(),
                        "difficulty": 1.0 - pred_conf if is_harmful else pred_conf,
                    })

    # Sort by difficulty and take top_k
    hard_examples.sort(key=lambda x: x["difficulty"], reverse=True)
    n_select = int(len(hard_examples) * top_k)
    hard_examples = hard_examples[:n_select]

    return hard_examples


if __name__ == "__main__":
    train_stage_c()