"""
Final held-out evaluation for SENTINEL-Vision.

Evaluates a checkpoint on the TEST split (never used for checkpoint selection):
  - harmful detection: recall / precision / F1 / accuracy at threshold 0.5
  - localization: IoU of the refined box, the first-stage argmax box, and the
    oracle ceiling (best candidate); plus proposal recall (ceiling for the refiner)

Usage:
    python scripts/evaluate_final.py --checkpoint checkpoints/stage_c/best.pt --out results/final_test_metrics.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.data.augmentation import create_val_transform
from src.data.frame_windowing import collate_frame_windows
from src.data.loaders import SentinelDataset
from src.models.sentinel_model import create_sentinel_model
from src.training.losses import AnchorMatchedLocalizationLoss as A
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def iou_pair(a, b):
    return A._pairwise_iou(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/stage_c/best.pt")
    ap.add_argument("--out", default="results/final_test_metrics.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    config = OmegaConf.merge(load_config("model_small.yaml"), load_config("data.yaml"),
                             load_config("curriculum.yaml")["stage_c"])
    ck = load_checkpoint(args.checkpoint)
    model = create_sentinel_model(config)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(dev)

    ds = SentinelDataset(data_config=OmegaConf.to_container(config), split="test",
                         transform=create_val_transform(OmegaConf.to_container(config)),
                         frame_window_k=config.frame_window.k,
                         target_resolution=tuple(config.frame_window.resolution))
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_frame_windows)

    tp = fp = fn = tn = 0
    refined, argmax, oracle, proposal_hit = [], [], [], []
    S = model.image_size
    with torch.no_grad():
        for batch in dl:
            y = batch["risk_label"].to(dev)
            out = model(batch["frames"].to(dev))
            pred = (out["risk_score"].squeeze(-1) > args.threshold).float()
            tp += ((pred == 1) & (y == 1)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()

            mask = (y == 1) & (batch["has_bbox"].to(dev) > 0.5)
            if not mask.any() or "refine_proposals" not in out:
                continue
            gt_all = batch["bbox"].to(dev) * S
            grid = out["bbox_pred_grid"].reshape(batch["frames"].shape[0], -1, 4)
            for i in torch.nonzero(mask).flatten().tolist():
                gt = gt_all[i:i + 1]
                refined.append(iou_pair(out["bbox_refined_pixel"][i:i + 1], gt).item())
                argmax.append(iou_pair(out.get("bbox_pixel_stage1", out["bbox_pixel"])[i:i + 1], gt).item())
                cand = iou_pair(grid[i], gt.expand(grid.shape[1], 4))
                oracle.append(cand.max().item())
                props = out["refine_proposals"][i]
                proposal_hit.append(float(iou_pair(props, gt.expand_as(props)).max() >= 0.5))

    def summarize(vals):
        if not vals:
            return {"n": 0}
        t = torch.tensor(vals)
        return {"n": len(vals), "mean_iou": round(t.mean().item(), 4),
                "median_iou": round(t.median().item(), 4),
                "frac_iou_ge_0.5": round((t >= 0.5).float().mean().item(), 4)}

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    result = {
        "checkpoint": args.checkpoint,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "split": "test",
        "detection": {
            "threshold": args.threshold,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "accuracy": round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "localization": {
            "refined_box": summarize(refined),
            "first_stage_argmax_box": summarize(argmax),
            "oracle_best_candidate": summarize(oracle),
            "proposal_recall_iou_ge_0.5": round(sum(proposal_hit) / max(len(proposal_hit), 1), 4),
        },
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
