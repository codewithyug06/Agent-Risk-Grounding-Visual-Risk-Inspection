"""
Point-in-box grounding accuracy (the standard GUI-grounding metric) on ScreenSpot-v2.

A prediction is correct if its predicted point lies inside the ground-truth box.
Reports point-in-box with a 95% cluster bootstrap CI (clusters = templates),
plus IoU as a secondary number. Uses the processed test split rows whose
task_id starts with 'screenspot_v2_'.

Usage:
    python scripts/eval_screenspot.py --checkpoint checkpoints/stage_c/best_loc.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.augmentation import create_val_transform
from src.data.frame_windowing import collate_frame_windows
from src.data.loaders import SentinelDataset
from src.models.sentinel_model import create_sentinel_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config

PREFIX = "screenspot_v2_"
BOOT_N = 2000
SEED = 42


def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return float(x1 <= px <= x2 and y1 <= py <= y2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/stage_c/best_loc.pt")
    ap.add_argument("--out", default="results/screenspot_v2_point_accuracy.json")
    args = ap.parse_args()

    config = OmegaConf.merge(load_config("model_small.yaml"), load_config("data.yaml"),
                             load_config("curriculum.yaml")["stage_c"])
    model = create_sentinel_model(config)
    model.load_state_dict(load_checkpoint(args.checkpoint)["model_state_dict"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(dev)
    S = model.image_size

    ds = SentinelDataset(data_config=OmegaConf.to_container(config), split="test",
                         transform=create_val_transform(OmegaConf.to_container(config)),
                         frame_window_k=config.frame_window.k,
                         target_resolution=tuple(config.frame_window.resolution))
    rows = []  # (template_group, point_hit, iou)
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_frame_windows):
            keep = [i for i, t in enumerate(b["trajectory_ids"]) if str(t).startswith(PREFIX)]
            if not keep:
                continue
            out = model(b["frames"].to(dev))
            box_pix = out["bbox_pixel"].cpu()
            points = out["point_pixel"].cpu() if "point_pixel" in out else None
            gt = b["bbox"].cpu() * S
            for i in keep:
                pb = box_pix[i].tolist()
                if points is not None:
                    px, py = points[i].tolist()
                else:
                    px, py = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                g = gt[i].tolist()
                group = str(b["trajectory_ids"][i]).rsplit("_", 1)[0]
                rows.append((group, point_in_box(px, py, g), iou(pb, g)))

    if not rows:
        raise SystemExit("no ScreenSpot-v2 rows found in the test split")

    hits = [r[1] for r in rows]
    groups = {}
    for g, h, _ in rows:
        groups.setdefault(g, []).append(h)
    keys = list(groups)
    rng = random.Random(SEED)
    boots = []
    for _ in range(BOOT_N):
        sample = [v for g in (rng.choice(keys) for _ in keys) for v in groups[g]]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    result = {
        "checkpoint": args.checkpoint,
        "benchmark": "ScreenSpot-v2 (processed test split rows)",
        "n": len(rows),
        "point_in_box_accuracy": round(sum(hits) / len(hits), 4),
        "point_in_box_ci95": [round(boots[int(0.025 * BOOT_N)], 4), round(boots[int(0.975 * BOOT_N)], 4)],
        "mean_iou": round(sum(r[2] for r in rows) / len(rows), 4),
        "note": "Published large-VLM point accuracy on ScreenSpot-v2 is 88-92%.",
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
