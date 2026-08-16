#!/usr/bin/env python3
"""
Preprocess real SENTINEL-Vision datasets into a unified, train-ready format.

Reads the downloaded raw datasets from ../../Datasets and produces:
  data/processed/
    train.jsonl       # one record per frame-action sample
    val.jsonl
    test.jsonl
    frames/           # extracted screenshot PNGs, resized to target resolution
    stats.json        # class/category counts + split sizes

Each JSONL record:
  {
    "task_id": str,            # groups samples into a pseudo-trajectory
    "frame_path": str,         # relative to data/processed/
    "action": str,             # human-action text (weak-labeled)
    "label": int,              # 0 benign, 1 harmful (weak)
    "category": int,           # 0 destructive,1 financial,2 privacy,3 irreversible_external,4 benign
    "bbox": [x1,y1,x2,y2]|None, # normalized 0-1, from real grounding GT
    "source": str,             # dataset origin
    "metadata": {...}
  }

Real-data-only: no synthetic injection. Harmful positives come from weak
keyword labeling of REAL action text (see src/data/loaders.py: HARMFUL_CUES).

Usage:
  python scripts/preprocess_real_data.py --config configs/data.yaml
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml
from PIL import Image
from datasets import load_from_disk

# Allow running as a script: add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import (
    CATEGORY_MAP,
    HARMFUL_CUES,
)

logger = logging.getLogger(__name__)

# Local raw dataset roots (downloaded by user)
RAW_ROOT = ROOT / "data"

# Category index for "benign" (5th class)
BENIGN_IDX = CATEGORY_MAP["benign"]


def _weak_label(text: str) -> Tuple[int, int]:
    t = text.lower()
    for cat, kws in HARMFUL_CUES.items():
        for kw in kws:
            if kw in t:
                return 1, CATEGORY_MAP[cat]
    return 0, BENIGN_IDX


def _bbox_from_candidate(candidate: Dict[str, Any], size: Tuple[int, int]) -> Optional[List[float]]:
    try:
        attrs = json.loads(candidate.get("attributes", "{}"))
        rect = attrs.get("bounding_box_rect")
        if not rect:
            return None
        x, y, w, h = map(float, rect.split(","))
        W, H = size
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 0.005 or (y2 - y1) < 0.005:
            return None
        return [x1, y1, x2, y2]
    except Exception:
        return None


def _load_image(example: Dict, key: str) -> Optional[Image.Image]:
    """Load a screenshot/instruction image from a HF example.

    Returns None when the field is missing or empty (some Multimodal-Mind2Web
    rows have null `screenshot` values) so the caller can skip them.
    """
    d = example.get(key)
    if d is None:
        return None
    try:
        if isinstance(d, dict) and "bytes" in d:
            return Image.open(__import__("io").BytesIO(d["bytes"])).convert("RGB")
        if isinstance(d, str):
            return Image.open(d).convert("RGB")
        if isinstance(d, Image.Image):
            return d.convert("RGB")
        if isinstance(d, np.ndarray):
            return Image.fromarray(d).convert("RGB")
    except Exception:
        return None
    return None


def _iter_multimodal_mind2web(resolution: Tuple[int, int]):
    """Yield (image, action_text, source, meta) for Multimodal-Mind2Web."""
    path = RAW_ROOT / "Multimodal-Mind2Web"
    ds = load_from_disk(path)
    for split in ["train", "test_domain", "test_task", "test_website"]:
        if split not in ds:
            continue
        for ex in ds[split]:
            img = _load_image(ex, "screenshot")
            if img is None:
                continue  # some test rows have null screenshots
            img = img.resize(resolution, Image.LANCZOS)
            text = (
                ex.get("target_action_reprs")
                or ex.get("operation")
                or " ".join(ex.get("action_reprs", []))
            )
            bbox = None
            pos = ex.get("pos_candidates", [])
            if pos:
                bbox = _bbox_from_candidate(pos[0], ex["screenshot"].size)
            meta = {
                "website": ex.get("website", ""),
                "domain": ex.get("domain", ""),
                "task": ex.get("confirmed_task", ""),
                "split": split,
            }
            yield img, text, "multimodal_mind2web", meta


def _iter_screenspot(version: str, resolution: Tuple[int, int]):
    name = "ScreenSpot" if version == "v1" else "ScreenSpot-v2"
    split = "test" if version == "v1" else "train"
    path = RAW_ROOT / name
    ds = load_from_disk(path)
    if split not in ds:
        return
    for ex in ds[split]:
        img = _load_image(ex, "image")
        if img is None:
            continue
        img = img.resize(resolution, Image.LANCZOS)
        instr = ex.get("instruction", "")
        bbox = ex.get("bbox", [0, 0, 1, 1])
        if max(bbox) > 1.0:
            W, H = img.size
            bbox = [bbox[0] / W, bbox[1] / H, bbox[2] / W, bbox[3] / H]
        meta = {
            "data_type": ex.get("data_type", ""),
            "data_source": ex.get("data_source", ""),
            "split": split,
        }
        yield img, instr, f"screenspot_{version}", meta


def preprocess(
    out_dir: Path,
    resolution: Tuple[int, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full preprocessing with resume support.

    Checkpoint file: out_dir/_samples.jsonl stores all raw samples (before split).
    On restart, reads the checkpoint to skip re-iterating source datasets and
    re-writing already-existing frame files.
    """
    random.seed(seed)
    np.random.seed(seed)

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "_samples.jsonl"

    # --- resume: load existing samples from checkpoint ---
    samples: List[Dict[str, Any]] = []
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                samples.append(json.loads(line))
        # Validate that all checkpointed frames still exist on disk.
        for rec in samples:
            if not (frames_dir / Path(rec["frame_path"]).name).exists():
                logger.warning("Checkpoint frame missing; discarding checkpoint.")
                samples.clear()
                break
        if samples:
            logger.info(f"Resuming from checkpoint: {len(samples)} samples")
    else:
        logger.info("No checkpoint found, starting fresh")

    # frame_counter = next frame index to write; equals count of done samples.
    frame_counter = len(samples)

    def _add(img: Image.Image, text: str, source: str, meta: Dict):
        nonlocal frame_counter
        label, category = _weak_label(text)
        fname = f"{frame_counter:08d}.png"
        img.save(frames_dir / fname)
        frame_counter += 1
        task_id = f"{source}_{meta.get('website', meta.get('data_type', 'x'))}_{frame_counter}"
        rec = {
            "task_id": task_id,
            "frame_path": f"frames/{fname}",
            "action": text,
            "label": label,
            "category": category,
            "bbox": meta.pop("bbox", None),
            "source": source,
            "metadata": meta,
        }
        samples.append(rec)
        # Persist checkpoint every 100 samples
        if frame_counter % 100 == 0:
            _write_checkpoint(samples, checkpoint_file)

    # Re-run all source iterators; skip yields already captured in a prior run
    # (their frames are already on disk). This lets the job continue across
    # restarts instead of restarting from zero.
    skip = len(samples)  # number of already-processed yields

    def _iter_all():
        yield from _iter_multimodal_mind2web(resolution)
        yield from _iter_screenspot("v1", resolution)
        yield from _iter_screenspot("v2", resolution)

    for i, (img, text, src, meta) in enumerate(_iter_all()):
        if i < skip:
            continue  # already saved in a prior run
        _add(img, text, src, meta)

    # Final checkpoint write
    _write_checkpoint(samples, checkpoint_file)

    # Deterministic split by source so each dataset contributes proportionally,
    # and so the same image never lands in two splits.
    by_source: Dict[str, List[Dict]] = {}
    for s in samples:
        by_source.setdefault(s["source"], []).append(s)

    splits = {"train": [], "val": [], "test": []}
    for src, items in by_source.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])

    # Reverse map: integer category -> name (category is stored as an int 0-4)
    idx_to_name = {v: k for k, v in CATEGORY_MAP.items()}

    stats = {"total": len(samples), "resolution": list(resolution), "sources": {}}
    for split_name, items in splits.items():
        with open(out_dir / f"{split_name}.jsonl", "w") as f:
            for rec in items:
                f.write(json.dumps(rec) + "\n")
        # Stats per split
        n_harm = sum(1 for r in items if r["label"] == 1)
        cat_counts = {k: 0 for k in CATEGORY_MAP}
        for r in items:
            cat_counts[idx_to_name[int(r["category"])]] += 1
        stats.setdefault(split_name, {
            "n": len(items),
            "harmful": n_harm,
            "benign": len(items) - n_harm,
            "category_counts": cat_counts,
        })

    for src, items in by_source.items():
        stats["sources"][src] = len(items)

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Preprocessing complete: {len(samples)} samples -> {out_dir}")
    logger.info(f"  train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    logger.info(f"  harmful total={sum(1 for r in samples if r['label']==1)}")
    return stats


def _write_checkpoint(samples: List[Dict[str, Any]], checkpoint_file: Path):
    """Atomically write samples checkpoint."""
    tmp = checkpoint_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for rec in samples:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(checkpoint_file)


def main():
    parser = argparse.ArgumentParser(description="Preprocess real SENTINEL-Vision datasets")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--resolution", nargs=2, type=int, default=None,
                        help="Resize (H W). Overrides config.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    fw = cfg.get("frame_window", {})
    if args.resolution:
        resolution = tuple(args.resolution)
    else:
        resolution = tuple(fw.get("resolution", [224, 224]))

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocess(
        out_dir=out_dir,
        resolution=resolution,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
