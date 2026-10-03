"""
Converts osunlp/MisActBench (real computer-use-agent trajectories with
human-annotated harmful/misaligned-action labels) into SENTINEL-Vision's
data/processed/{train,val,test}.jsonl schema, appending to the existing
real-data splits (Multimodal-Mind2Web + ScreenSpot).

Unlike the weak keyword-based labeling used for Mind2Web/ScreenSpot, the
harmful/benign label here comes directly from MisActBench's human
annotation (`step["label"]`) -- real, non-weak supervision. Category
(destructive/financial/privacy/irreversible_external) is still assigned
via keyword matching against the step's action code + trajectory
instruction, since MisActBench's own categories (malicious-instruction-
following / harmful-unintended-behavior / task-irrelevant-behavior) don't
map onto SENTINEL-Vision's UI-risk taxonomy. Harmful steps with no
keyword match fall back to "irreversible_external" (the catch-all for
actions that should have paused for confirmation).

Usage:
    python scripts/convert_misactbench.py
"""

import json
import random
import zipfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MISACTBENCH_DIR = REPO_ROOT / "data" / "MisActBench"
JSON_PATH = MISACTBENCH_DIR / "misactbench.json"
ZIP_PATH = MISACTBENCH_DIR / "trajectories.zip"
EXTRACT_DIR = MISACTBENCH_DIR  # zip already contains "trajectories/..." prefix

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FRAMES_OUT_DIR = PROCESSED_DIR / "frames_misactbench"
TARGET_RESOLUTION = (224, 224)

CATEGORY_MAP = {
    "destructive": 0,
    "financial": 1,
    "privacy": 2,
    "irreversible_external": 3,
    "benign": 4,
}

HARMFUL_CUES = {
    "destructive": [
        "delete", "remove", "clear", "reset", "format", "erase", "wipe",
        "purge", "cancel", "close", "discard", "terminate", "revoke", "rm ",
        "rmdir", "del ", "drop table", "uninstall",
    ],
    "financial": [
        "pay", "purchase", "buy", "checkout", "subscribe", "donate", "transfer",
        "order", "price", "cost", "$", "billing", "payment", "check out",
        "charge", "invoice", "complete order", "confirm payment", "venmo",
        "wire", "bank",
    ],
    "privacy": [
        "share", "export", "upload", "send", "download", "sync", "backup",
        "publish", "public", "invite", "forward", "copy", "attach",
        "grant access", "permission", "visibility", "external", "email",
    ],
}

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def ensure_extracted() -> None:
    marker = MISACTBENCH_DIR / "trajectories"
    if marker.exists() and any(marker.iterdir()):
        print(f"Already extracted: {marker}")
        return
    print(f"Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        z.extractall(EXTRACT_DIR, members=members)
    print("Extraction complete.")


def categorize(text: str) -> int:
    t = text.lower()
    for cat_name, keywords in HARMFUL_CUES.items():
        for kw in keywords:
            if kw in t:
                return CATEGORY_MAP[cat_name]
    return CATEGORY_MAP["irreversible_external"]


def resize_and_save(src_path: Path, dst_path: Path) -> bool:
    if not src_path.exists():
        return False
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(TARGET_RESOLUTION, Image.BILINEAR)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"  WARN: failed to process {src_path}: {e}")
        return False


def main() -> None:
    ensure_extracted()
    FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    traj_ids = list(data.keys())
    random.Random(RANDOM_SEED).shuffle(traj_ids)

    n = len(traj_ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    split_of_traj = {}
    for i, tid in enumerate(traj_ids):
        if i < n_train:
            split_of_traj[tid] = "train"
        elif i < n_train + n_val:
            split_of_traj[tid] = "val"
        else:
            split_of_traj[tid] = "test"

    records = {"train": [], "val": [], "test": []}
    counts = {"train": {"harmful": 0, "benign": 0}, "val": {"harmful": 0, "benign": 0}, "test": {"harmful": 0, "benign": 0}}
    cat_counts = {"train": {}, "val": {}, "test": {}}
    skipped_missing_image = 0

    for tid, traj in data.items():
        split = split_of_traj[tid]
        instruction = traj.get("instruction", "")
        steps = traj.get("steps", {})
        for step_key, step in sorted(steps.items(), key=lambda kv: int(kv[0])):
            label = 1 if step.get("label") else 0
            agent_output = step.get("agent_output", "")
            if label == 1:
                category = categorize(agent_output + " " + instruction)
            else:
                category = CATEGORY_MAP["benign"]

            rel_screenshot = step.get("screenshot_path", "")
            # screenshot_path in JSON is like "MisActBench/trajectories/<id>/step_0.png";
            # our extracted zip root already is data/MisActBench/, so strip the leading
            # "MisActBench/" component.
            parts = Path(rel_screenshot).parts
            if parts and parts[0] == "MisActBench":
                rel_screenshot_local = Path(*parts[1:])
            else:
                rel_screenshot_local = Path(rel_screenshot)
            src_path = MISACTBENCH_DIR / rel_screenshot_local

            out_name = f"{tid}_step{step.get('step_idx', step_key)}.png"
            dst_rel = Path("frames_misactbench") / out_name
            dst_path = PROCESSED_DIR / dst_rel

            if not resize_and_save(src_path, dst_path):
                skipped_missing_image += 1
                continue

            rec = {
                "task_id": f"misactbench_{tid}",
                "frame_path": str(dst_rel).replace("\\", "/"),
                "action": agent_output[:500],
                "label": label,
                "category": category,
                "bbox": None,
                "source": "misactbench",
                "metadata": {
                    "trajectory_id": tid,
                    "step_idx": step.get("step_idx", step_key),
                    "misactbench_category": step.get("category"),
                    "instruction": instruction[:300],
                    "agent": traj.get("metadata", {}).get("agent"),
                    "source_env": traj.get("metadata", {}).get("source"),
                },
            }
            records[split].append(rec)
            counts[split]["harmful" if label == 1 else "benign"] += 1
            cat_name = {v: k for k, v in CATEGORY_MAP.items()}[category]
            cat_counts[split][cat_name] = cat_counts[split].get(cat_name, 0) + 1

    for split in ("train", "val", "test"):
        out_path = PROCESSED_DIR / f"{split}.jsonl"
        with open(out_path, "a", encoding="utf-8") as f:
            for rec in records[split]:
                # ensure_ascii=True (default): loaders.py opens JSONL files
                # without specifying an encoding, so on Windows it falls back
                # to cp1252, which can't decode raw non-ASCII UTF-8 bytes.
                # Escaping non-ASCII as \uXXXX keeps the file ASCII-safe.
                f.write(json.dumps(rec) + "\n")
        print(
            f"{split}: appended {len(records[split])} records "
            f"(harmful={counts[split]['harmful']}, benign={counts[split]['benign']}), "
            f"categories={cat_counts[split]}"
        )

    if skipped_missing_image:
        print(f"Skipped {skipped_missing_image} steps with missing/unreadable screenshots.")

    stats_path = PROCESSED_DIR / "stats.json"
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        stats.setdefault("sources", {})["misactbench"] = sum(len(records[s]) for s in records)
        for split in ("train", "val", "test"):
            stats[split]["n"] += len(records[split])
            stats[split]["harmful"] += counts[split]["harmful"]
            stats[split]["benign"] += counts[split]["benign"]
            for cat_name, c in cat_counts[split].items():
                stats[split]["category_counts"][cat_name] = (
                    stats[split]["category_counts"].get(cat_name, 0) + c
                )
        stats["total"] = stats.get("total", 0) + sum(len(records[s]) for s in records)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Updated {stats_path}")


if __name__ == "__main__":
    main()
