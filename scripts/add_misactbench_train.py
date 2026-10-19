"""
Adds the MisActBench TRAIN split (446 trajectories, ~2,897 steps) to
data/processed/train.jsonl, while strictly keeping the 55 val trajectories
and 57 test trajectories held out (zero leakage).

This provides real computer-use-agent supervision (both harmful and benign)
during training, breaking the Clever Hans domain shortcut where real web
screenshots were 100% benign in train.
"""

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MISACTBENCH_DIR = REPO_ROOT / "data" / "MisActBench"
JSON_PATH = MISACTBENCH_DIR / "misactbench.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "val.jsonl"
TEST_FILE = PROCESSED_DIR / "test.jsonl"
FRAMES_DIR = PROCESSED_DIR / "frames_misactbench"

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


def categorize(text: str) -> int:
    t = text.lower()
    for cat_name, keywords in HARMFUL_CUES.items():
        for kw in keywords:
            if kw in t:
                return CATEGORY_MAP[cat_name]
    return CATEGORY_MAP["irreversible_external"]


def main():
    # 1. Check idempotency
    existing_misact = 0
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source") == "misactbench":
                existing_misact += 1
    if existing_misact > 0:
        print(f"train.jsonl already contains {existing_misact} misactbench records. Skipping append.")
        return

    # 2. Load misactbench.json and compute splits
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    traj_ids = list(data.keys())
    random.Random(42).shuffle(traj_ids)
    n = len(traj_ids)
    n_train = int(n * 0.8)
    train_traj_ids = set(traj_ids[:n_train])

    # 3. Check val and test overlap (must be strictly 0)
    val_trajs = set()
    for line in open(VAL_FILE, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("source") == "misactbench":
            val_trajs.add(r["metadata"]["trajectory_id"])

    test_trajs = set()
    for line in open(TEST_FILE, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("source") == "misactbench":
            test_trajs.add(r["metadata"]["trajectory_id"])

    assert len(train_traj_ids.intersection(val_trajs)) == 0, "Leakage detected between train and val!"
    assert len(train_traj_ids.intersection(test_trajs)) == 0, "Leakage detected between train and test!"
    print(f"Disjointness verified: 0 overlap with val ({len(val_trajs)} trajs) or test ({len(test_trajs)} trajs).")

    # 4. Build train records
    records_to_append = []
    harmful_count = 0
    benign_count = 0

    for tid in sorted(train_traj_ids):
        traj = data[tid]
        instruction = traj.get("instruction", "")
        steps = traj.get("steps", {})
        for step_key, step in sorted(steps.items(), key=lambda kv: int(kv[0])):
            label = 1 if step.get("label") else 0
            agent_output = step.get("agent_output", "")
            category = categorize(agent_output + " " + instruction) if label == 1 else CATEGORY_MAP["benign"]

            out_name = f"{tid}_step{step.get('step_idx', step_key)}.png"
            dst_rel = Path("frames_misactbench") / out_name
            dst_path = PROCESSED_DIR / dst_rel
            if not dst_path.exists():
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
                "tier": "gold",
            }
            records_to_append.append(rec)
            if label == 1:
                harmful_count += 1
            else:
                benign_count += 1

    # 5. Append to train.jsonl
    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        for rec in records_to_append:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully appended {len(records_to_append)} MisActBench records to {TRAIN_FILE}")
    print(f"  Harmful: {harmful_count}, Benign: {benign_count}")

    # 6. Recompute stats.json
    import subprocess
    import sys
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "recompute_stats.py")], check=True)


if __name__ == "__main__":
    main()
