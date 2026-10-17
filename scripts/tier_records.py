"""
Tag every processed record with an explicit label tier and keep gold out of training.

Tiers:
  gold    MisActBench (human labels). Removed from the TRAIN split so it is never
          fit. Kept in val (threshold selection, pre-registered) and test (one-shot).
  silver  Synthetic pages with labels correct by construction (merge_silver.py).
  bronze  Everything else: keyword-matched labels from Mind2Web and ScreenSpot.

Headline metrics use gold (detection) and ScreenSpot-v2 (grounding) only.
Usage:
    python scripts/tier_records.py
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def tier_of(rec):
    src = str(rec.get("source", ""))
    if src == "misactbench":
        return "gold"
    if src.startswith("synthetic"):
        return "silver"
    return "bronze"


def main():
    counts = {}
    dropped = Counter()
    for split in ("train", "val", "test"):
        path = PROC / f"{split}.jsonl"
        out_lines = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["tier"] = tier_of(rec)
            if split == "train" and rec["tier"] == "gold":
                dropped["gold_from_train"] += 1
                continue
            out_lines.append(json.dumps(rec) + "\n")
            counts[(split, rec["tier"])] = counts.get((split, rec["tier"]), 0) + 1
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
    for k in sorted(counts):
        print(k, counts[k])
    print("dropped:", dict(dropped))


if __name__ == "__main__":
    main()
