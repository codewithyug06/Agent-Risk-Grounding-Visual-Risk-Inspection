"""
Shared dataset-construction helper for the Stage A/B/C curriculum scripts.
Builds a CombinedSentinelDataset (real weak-labeled + DOM-grounded synthetic
injection samples) per curriculum stage, filtered to that stage's target
harmful categories, and falls back gracefully if either data source is absent.
"""

import logging
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from ..data.loaders import SentinelDataset, SyntheticInjectionDataset, CombinedSentinelDataset

logger = logging.getLogger(__name__)


def build_combined_dataset(
    config: DictConfig,
    split: str,
    transform,
    synthetic_categories: Optional[List[str]] = None,
) -> CombinedSentinelDataset:
    """Build a CombinedSentinelDataset from real (if available) + synthetic (if
    available) data. Raises FileNotFoundError only if BOTH sources are absent."""
    real_dataset = None
    try:
        real_dataset = SentinelDataset(
            data_config=OmegaConf.to_container(config),
            split=split,
            transform=transform,
            frame_window_k=config.frame_window.k,
            target_resolution=tuple(config.frame_window.resolution),
        )
        logger.info(f"[{split}] Real dataset: {len(real_dataset)} samples")
    except FileNotFoundError as e:
        logger.warning(f"[{split}] Real dataset unavailable: {e}")

    synthetic_dataset = None
    injection_dir = config.get("synthetic_injection_dir", "data/synthetic_injections")
    try:
        synthetic_dataset = SyntheticInjectionDataset(
            injection_dir=injection_dir,
            split=split,
            transform=transform,
            k=config.frame_window.k,
            target_resolution=tuple(config.frame_window.resolution),
            categories=synthetic_categories,
        )
        logger.info(f"[{split}] Synthetic dataset ({synthetic_categories}): {len(synthetic_dataset)} samples")
    except FileNotFoundError as e:
        logger.warning(f"[{split}] Synthetic injection dataset unavailable: {e}")

    if real_dataset is None and synthetic_dataset is None:
        raise FileNotFoundError(
            f"No data available for split={split}. Run scripts/preprocess_real_data.py "
            "and/or scripts/generate_synthetic_data.py first."
        )

    return CombinedSentinelDataset(
        real_dataset=real_dataset,
        synthetic_dataset=synthetic_dataset,
        synthetic_weight=config.get("synthetic_weight", 3.0),
    )


def dataset_stats_dict(train_dataset: CombinedSentinelDataset, val_dataset: CombinedSentinelDataset) -> dict:
    return {
        "train_total": len(train_dataset),
        "val_total": len(val_dataset),
        "train_real": getattr(train_dataset, "_real_len", 0),
        "train_synthetic": getattr(train_dataset, "_synthetic_len", 0),
    }
