"""
Smoke test: real-data pipeline end-to-end (load -> batch -> single forward pass).
Mirrors the config-merge done in train_stageA.py but does NOT train.
"""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from src.utils.config import load_config
from src.data.loaders import SentinelDataset
from src.data.augmentation import create_train_transform
from src.models.sentinel_model import create_sentinel_model


def main():
    # Build config exactly like the training entry points do.
    config = load_config("model_small.yaml")
    data_config = load_config("data.yaml")
    curriculum_config = load_config("curriculum.yaml")
    stage_config = curriculum_config["stage_a"]
    config = OmegaConf.merge(config, data_config, stage_config)
    config.stage_name = "stage_a"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[config] device={config.device} "
          f"processed_dir={config.processed_dir} k={config.frame_window.k} "
          f"res={list(config.frame_window.resolution)}")

    transform = create_train_transform(OmegaConf.to_container(config))

    dataset = SentinelDataset(
        data_config=OmegaConf.to_container(config),
        split="train",
        transform=transform,
        frame_window_k=config.frame_window.k,
        target_resolution=tuple(config.frame_window.resolution),
    )
    print(f"[data] train samples (frame windows): {len(dataset)}")

    # Pull a single sample, batch it.
    sample = dataset[0]
    print(f"[sample] frames shape={tuple(sample['frames'].shape)} "
          f"risk={int(sample['risk_label'])} cat={int(sample['category_label'])} "
          f"has_bbox={float(sample['has_bbox'])}")

    batch = {
        k: sample[k].unsqueeze(0) if isinstance(sample[k], torch.Tensor) else sample[k]
        for k in sample
    }

    model = create_sentinel_model(config)
    model = model.to(config.device)
    model.train()

    frames = batch["frames"].to(config.device)
    out = model(frames)

    print(f"[forward] risk_logits={tuple(out['risk_logits'].shape)} "
          f"category_logits={tuple(out['category_logits'].shape)} "
          f"bbox={tuple(out['bbox'].shape)}")
    print("[OK] data loading + single forward pass succeeded.")


if __name__ == "__main__":
    main()
