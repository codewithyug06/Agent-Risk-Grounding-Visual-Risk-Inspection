"""Recomputes data/processed/stats.json from scratch by scanning the
actual {train,val,test}.jsonl files. Safer than manually incrementing/
decrementing counters when splits are edited by conversion scripts.

Usage:
    python scripts/recompute_stats.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

CATEGORY_NAMES = {0: "destructive", 1: "financial", 2: "privacy", 3: "irreversible_external", 4: "benign"}


def main() -> None:
    stats = {"resolution": [224, 224], "sources": {}, "train": {}, "val": {}, "test": {}}
    total = 0

    for split in ("train", "val", "test"):
        path = PROCESSED_DIR / f"{split}.jsonl"
        n = 0
        harmful = 0
        benign = 0
        cat_counts = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n += 1
                total += 1
                if rec["label"] == 1:
                    harmful += 1
                else:
                    benign += 1
                cat_name = CATEGORY_NAMES.get(rec["category"], str(rec["category"]))
                cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
                src = rec.get("source", "unknown")
                stats["sources"][src] = stats["sources"].get(src, 0) + 1
        stats[split] = {"n": n, "harmful": harmful, "benign": benign, "category_counts": cat_counts}

    stats["total"] = total
    ordered = {
        "total": stats["total"],
        "resolution": stats["resolution"],
        "sources": stats["sources"],
        "train": stats["train"],
        "val": stats["val"],
        "test": stats["test"],
    }
    out_path = PROCESSED_DIR / "stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(ordered, indent=2))


if __name__ == "__main__":
    main()
