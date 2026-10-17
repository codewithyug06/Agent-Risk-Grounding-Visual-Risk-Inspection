"""
Merge silver synthetic records into data/processed as a separate tier.

Split by template family (never by sample):
  train: modal, inline   val: card   test: toolbar
Harmful examples are subsampled to 25% of each split so synthetic base rate
does not dominate the real data's base rate. Benign examples are kept.

Idempotent: removes any earlier 'synthetic_v2' records before appending.
Usage:
    python scripts/merge_silver.py --src data/synthetic_v2 --seed 42
"""

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_OF = {"modal": "train", "inline": "train", "card": "val", "toolbar": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["data/synthetic_v2", "data/synthetic_v3"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--harmful_fraction", type=float, default=0.25)
    args = ap.parse_args()

    proc = ROOT / "data" / "processed"
    dst_img = proc / "synthetic" / "images"
    dst_img.mkdir(parents=True, exist_ok=True)

    recs = []
    for s in args.src:
        src_dir = ROOT / s
        for l in open(src_dir / "records.jsonl", encoding="utf-8"):
            r = json.loads(l)
            r["_src"] = src_dir
            r["_set"] = src_dir.name
            recs.append(r)
    rng = random.Random(args.seed)
    by_split = {"train": [], "val": [], "test": []}
    for r in recs:
        by_split[SPLIT_OF[r["metadata"]["template"]]].append(r)

    for split, items in by_split.items():
        harm = [r for r in items if r["label"] == 1]  # balanced per split
        ben = [r for r in items if r["label"] == 0]
        keep_harm = int(len(ben) * args.harmful_fraction / (1 - args.harmful_fraction))
        rng.shuffle(harm)
        by_split[split] = ben + harm[:keep_harm]

    for split in ("train", "val", "test"):
        path = proc / f"{split}.jsonl"
        kept = []
        if path.exists():
            for line in open(path, encoding="utf-8"):
                if line.strip() and not str(json.loads(line).get("source", "")).startswith("synthetic"):
                    kept.append(line)
        added = 0
        for r in by_split[split]:
            name = Path(r["frame_path"]).name
            dst_name = f"{r['_set']}_{name}"
            shutil.copy2(r["_src"] / "images" / name, dst_img / dst_name)
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            out["frame_path"] = f"synthetic/images/{dst_name}"
            out["tier"] = "silver"
            kept.append(json.dumps(out) + "\n")
            added += 1
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(kept)
        print(f"{split}: added {added} silver records")


if __name__ == "__main__":
    main()
