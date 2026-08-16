"""
Dataset loaders for SENTINEL-Vision (real-data only).
Supports: Multimodal-Mind2Web, ScreenSpot, ScreenSpot-v2.
All loaders return pixels-only data — no agent internals, logs, or reasoning traces.
AgentTrek (no images), Mind2Web (no images), XD-Violence (not downloaded) are excluded.
"""

import os
import json
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import load_from_disk, Dataset as HFDataset
import cv2


@dataclass
class FrameActionPair:
    """A single frame-action pair from a trajectory."""
    frame: Image.Image
    action: str
    label: int  # 0=benign, 1=harmful (weak-labeled from real action text)
    category: int  # 0=destructive, 1=financial, 2=privacy, 3=irreversible_external, 4=benign
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] normalized 0-1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Trajectory:
    """A full trajectory of frame-action pairs."""
    pairs: List[FrameActionPair]
    trajectory_id: str
    source_dataset: str


CATEGORY_MAP = {
    "destructive": 0,
    "financial": 1,
    "privacy": 2,
    "irreversible_external": 3,
    "benign": 4,
}

# Weak harmful-cue keywords mined from REAL action text only.
# No synthetic data — these are cues that appear in real benign trajectories.
HARMFUL_CUES = {
    "destructive": [
        "delete", "remove", "clear", "reset", "format", "erase", "wipe",
        "purge", "cancel", "close", "discard", "terminate", "revoke"
    ],
    "financial": [
        "pay", "purchase", "buy", "checkout", "subscribe", "donate", "transfer",
        "order", "price", "cost", "$", "billing", "payment", "check out",
        "charge", "invoice", "complete order", "confirm payment"
    ],
    "privacy": [
        "share", "export", "upload", "send", "download", "sync", "backup",
        "publish", "public", "invite", "forward", "copy", "attach",
        "grant access", "permission", "visibility", "external"
    ],
    "irreversible_external": [
        "submit", "post", "confirm", "place order", "book", "save",
        "reply", "comment", "register", "sign up", "log in", "login",
        "signin", "send email", "send message", "submit form",
        "publish", "schedule", "deploy", "release", "merge", "push"
    ],
}


def parse_target_bounding_box(row: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract [x1, y1, x2, y2] bounding box coordinates from a dataset row if present.
    Supports formats from Multimodal-Mind2Web, ScreenSpot, AgentTrek, etc.
    """
    if "bbox" in row and row["bbox"]:
        b = row["bbox"]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    if "pos_candidates" in row and isinstance(row["pos_candidates"], list):
        for candidate in row["pos_candidates"]:
            if isinstance(candidate, dict) and "bbox" in candidate and candidate["bbox"]:
                b = candidate["bbox"]
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    if "target_bounding_box" in row and row["target_bounding_box"]:
        b = row["target_bounding_box"]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    if "box" in row and row["box"]:
        b = row["box"]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])

    return None


def _extract_bbox_from_candidate(candidate: Dict[str, Any], image_size: Tuple[int, int]) -> Optional[List[float]]:
    """
    Extract normalized bbox from Multimodal-Mind2Web pos_candidate.
    Candidate has 'attributes' JSON string with 'bounding_box_rect': "x,y,w,h" in pixels.
    """
    try:
        attrs_str = candidate.get("attributes", "{}")
        attrs = json.loads(attrs_str)
        rect_str = attrs.get("bounding_box_rect")
        if not rect_str:
            return None
        x, y, w, h = map(float, rect_str.split(","))
        W, H = image_size
        x1 = x / W
        y1 = y / H
        x2 = (x + w) / W
        y2 = (y + h) / H
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return [float(x1), float(y1), float(x2), float(y2)]
    except Exception:
        return None


def _weak_label_from_action(action_text: str) -> Tuple[int, int]:
    """
    Classify a real action string into (harmful_binary, category) using keyword cues.
    This is a WEAK label — real benign actions may contain cues (e.g., "submit" on a login form).
    Returns (0, 4) for benign by default.
    """
    t = action_text.lower()
    for cat_name, keywords in HARMFUL_CUES.items():
        for kw in keywords:
            if kw in t:
                return 1, CATEGORY_MAP[cat_name]
    return 0, CATEGORY_MAP["benign"]


def _load_image_from_hf(example: Dict, key: str) -> Image.Image:
    """Load image from HF dataset example (handles bytes dict, path, or PIL)."""
    img_data = example.get(key)
    if isinstance(img_data, dict) and "bytes" in img_data:
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    elif isinstance(img_data, str):
        return Image.open(img_data).convert("RGB")
    elif isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    else:
        return Image.fromarray(np.array(img_data)).convert("RGB")


def load_multimodal_mind2web(
    data_dir: str,
    split: str = "train",
) -> Iterator[FrameActionPair]:
    """
    Load Multimodal-Mind2Web dataset from local disk.
    Yields FrameActionPair with screenshot, action text, weak labels, and bbox from pos_candidates.
    """
    ds = load_from_disk(data_dir)
    if split in ds:
        ds = ds[split]

    for example in ds:
        frame = _load_image_from_hf(example, "screenshot")
        img_size = frame.size

        action_text = (
            example.get("target_action_reprs", "")
            or example.get("operation", "")
            or " ".join(example.get("action_reprs", []))
        )

        label, category = _weak_label_from_action(action_text)

        bbox = None
        pos_cands = example.get("pos_candidates", [])
        if pos_cands:
            bbox = _extract_bbox_from_candidate(pos_cands[0], img_size)

        yield FrameActionPair(
            frame=frame,
            action=action_text,
            label=label,
            category=category,
            bbox=bbox,
            metadata={
                "source": "multimodal_mind2web",
                "example_id": example.get("annotation_id", ""),
                "task": example.get("confirmed_task", ""),
                "website": example.get("website", ""),
                "domain": example.get("domain", ""),
            },
        )


def load_screenspot(
    data_dir: str,
    split: str = "train",
    version: str = "v1",
) -> Iterator[FrameActionPair]:
    """
    Load ScreenSpot or ScreenSpot-v2 dataset from local disk.
    Single-image grounding task — treat as benign localization samples.
    """
    ds = load_from_disk(data_dir)
    if split in ds:
        ds = ds[split]

    for example in ds:
        frame = _load_image_from_hf(example, "image")
        bbox = example.get("bbox", [0, 0, 1, 1])
        instruction = example.get("instruction", "")

        # Normalize bbox if it's pixel coordinates (ScreenSpot uses normalized)
        if bbox and max(bbox) > 1.0:
            W, H = frame.size
            bbox = [bbox[0]/W, bbox[1]/H, bbox[2]/W, bbox[3]/H]

        label, category = _weak_label_from_action(instruction)

        yield FrameActionPair(
            frame=frame,
            action=instruction,
            label=label,
            category=category,
            bbox=bbox,
            metadata={
                "source": f"screenspot_{version}",
                "example_id": example.get("img_filename", ""),
                "data_type": example.get("data_type", ""),
                "data_source": example.get("data_source", ""),
            },
        )


class SentinelDataset(Dataset):
    """
    Unified PyTorch Dataset wrapping all SENTINEL-Vision real data sources.
    Returns frame windows (k frames) with risk labels and localization targets.
    Loads from the preprocessed processed/ directory generated by preprocessing.py.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        split: str = "train",
        transform=None,
        frame_window_k: int = 6,
        target_resolution: Tuple[int, int] = (224, 224),
    ):
        self.data_config = data_config
        self.split = split
        self.transform = transform
        self.frame_window_k = frame_window_k
        self.target_resolution = target_resolution

        self.trajectories = self._load_processed_data()
        self.window_indices = self._build_window_indices()

    def _load_processed_data(self) -> List[Trajectory]:
        """
        Load preprocessed frame windows from data/processed/{split}.jsonl +
        associated frame images. Each JSONL record = one frame-action sample.
        Records sharing a task_id are grouped into a trajectory.
        """
        processed_dir = Path(self.data_config.get("processed_dir", "./data/processed"))
        split_file = processed_dir / f"{self.split}.jsonl"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {split_file}. "
                "Run scripts/preprocess_real_data.py first."
            )

        all_pairs: List[FrameActionPair] = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                frame_path = processed_dir / rec["frame_path"]
                if not frame_path.exists():
                    continue
                frame = Image.open(frame_path).convert("RGB")
                all_pairs.append(FrameActionPair(
                    frame=frame,
                    action=rec["action"],
                    label=rec["label"],
                    category=rec["category"],
                    bbox=rec.get("bbox"),
                    metadata={
                        "source": rec.get("source", "unknown"),
                        "task_id": rec.get("task_id", "unknown"),
                        **rec.get("metadata", {}),
                    },
                ))

        traj_dict: Dict[str, List[FrameActionPair]] = {}
        for pair in all_pairs:
            key = pair.metadata.get("task_id", "unknown")
            traj_dict.setdefault(key, []).append(pair)

        trajectories = [
            Trajectory(
                pairs=pairs,
                trajectory_id=task_id,
                source_dataset=pairs[0].metadata.get("source", "unknown") if pairs else "unknown",
            )
            for task_id, pairs in traj_dict.items()
        ]
        return trajectories

    def _build_window_indices(self) -> List[Tuple[int, int, int]]:
        indices = []
        k = self.frame_window_k
        half_k = k // 2

        for traj_idx, traj in enumerate(self.trajectories):
            n_frames = len(traj.pairs)
            if n_frames == 0:
                continue
            for action_idx in range(n_frames):
                start_idx = max(0, action_idx - half_k)
                end_idx = min(n_frames, start_idx + k)
                start_idx = max(0, end_idx - k)
                indices.append((traj_idx, start_idx, action_idx))

        return indices

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, start_idx, action_idx = self.window_indices[idx]
        traj = self.trajectories[traj_idx]
        k = self.frame_window_k

        frames = []
        for i in range(start_idx, min(start_idx + k, len(traj.pairs))):
            frame = traj.pairs[i].frame.resize(self.target_resolution, Image.LANCZOS)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        while len(frames) < k:
            frames.append(frames[-1] if frames else torch.zeros(3, *self.target_resolution))

        frames_tensor = torch.stack(frames)

        target_pair = traj.pairs[action_idx]

        return {
            "frames": frames_tensor,
            "risk_label": torch.tensor(target_pair.label, dtype=torch.long),
            "category_label": torch.tensor(target_pair.category, dtype=torch.long),
            "bbox": torch.tensor(target_pair.bbox if target_pair.bbox else [0, 0, 0, 0], dtype=torch.float32),
            "has_bbox": torch.tensor(1.0 if target_pair.bbox else 0.0, dtype=torch.float32),
            "trajectory_id": traj.trajectory_id,
            "action": target_pair.action,
        }


def load_real_dataset(
    data_config: Dict[str, Any],
    split: str = "train",
    transform=None,
    frame_window_k: int = 6,
    target_resolution: Tuple[int, int] = (224, 224),
) -> SentinelDataset:
    """Convenience factory: create a SentinelDataset over real preprocessed data."""
    return SentinelDataset(
        data_config=data_config,
        split=split,
        transform=transform,
        frame_window_k=frame_window_k,
        target_resolution=target_resolution,
    )
