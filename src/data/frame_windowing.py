"""
Frame windowing utilities for SENTINEL-Vision.
Extracts k-frame temporal windows centered on action timestamps with proper temporal alignment.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np
from .loaders import parse_target_bounding_box


@dataclass
class TimestampedFrame:
    """A frame with associated timestamp."""
    frame: Image.Image
    timestamp: float  # seconds since trajectory start
    frame_idx: int


def extract_frame_window(
    trajectory: List[TimestampedFrame],
    action_timestamp: float,
    k: int = 6,
    fps: int = 3,
    target_resolution: Tuple[int, int] = (224, 224),
) -> List[Image.Image]:
    """
    Extract a k-frame window centered on the action timestamp.

    Args:
        trajectory: List of TimestampedFrame objects (chronologically ordered)
        action_timestamp: Target timestamp in seconds
        k: Number of frames in window
        fps: Target sampling rate (frames per second)
        target_resolution: Output resolution for frames

    Returns:
        List of k PIL Images (padded with nearest neighbor if needed)
    """
    if not trajectory:
        # Return blank frames
        blank = Image.new("RGB", target_resolution, color=(128, 128, 128))
        return [blank] * k

    # Find the frame closest to action_timestamp
    timestamps = [tf.timestamp for tf in trajectory]
    action_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - action_timestamp))

    # Calculate window bounds
    half_k = k // 2
    start_idx = max(0, action_idx - half_k)
    end_idx = min(len(trajectory), start_idx + k)

    # Adjust if window is truncated at end
    if end_idx - start_idx < k:
        start_idx = max(0, end_idx - k)

    # Extract frames
    frames = []
    for i in range(start_idx, end_idx):
        frame = trajectory[i].frame.resize(target_resolution, Image.LANCZOS)
        frames.append(frame)

    # Pad with nearest neighbor (repeat last frame) if needed
    while len(frames) < k:
        frames.append(frames[-1] if frames else Image.new("RGB", target_resolution, color=(128, 128, 128)))

    return frames


def extract_frame_window_by_index(
    frames: List[Image.Image],
    action_idx: int,
    k: int = 6,
    target_resolution: Tuple[int, int] = (224, 224),
) -> List[Image.Image]:
    """
    Extract a k-frame window centered on action_idx (simpler version without timestamps).

    Args:
        frames: List of PIL Images in chronological order
        action_idx: Index of the action frame
        k: Number of frames in window
        target_resolution: Output resolution

    Returns:
        List of k PIL Images
    """
    if not frames:
        blank = Image.new("RGB", target_resolution, color=(128, 128, 128))
        return [blank] * k

    half_k = k // 2
    start_idx = max(0, action_idx - half_k)
    end_idx = min(len(frames), start_idx + k)

    if end_idx - start_idx < k:
        start_idx = max(0, end_idx - k)

    window_frames = []
    for i in range(start_idx, end_idx):
        frame = frames[i].resize(target_resolution, Image.LANCZOS)
        window_frames.append(frame)

    while len(window_frames) < k:
        window_frames.append(window_frames[-1] if window_frames else Image.new("RGB", target_resolution, color=(128, 128, 128)))

    return window_frames


class FrameWindowDataset(Dataset):
    """
    PyTorch Dataset that returns (window_tensor, risk_label, category_label, bbox_tensor).
    Handles variable-length trajectories with padding and temporal alignment.
    """

    def __init__(
        self,
        trajectories: List[List[Dict[str, Any]]],
        k: int = 6,
        target_resolution: Tuple[int, int] = (224, 224),
        transform=None,
        pad_mode: str = "repeat",  # "repeat", "zero", "mirror"
    ):
        """
        Args:
            trajectories: List of trajectories, each a list of frame dicts with keys:
                - 'frame': PIL.Image or path
                - 'timestamp': float (optional)
                - 'action': str
                - 'label': int (0=benign, 1=harmful)
                - 'category': int (0-4)
                - 'bbox': List[float] [x1,y1,x2,y2] normalized (optional)
            k: Number of frames per window
            target_resolution: Resize target
            transform: torchvision transforms
            pad_mode: How to pad short windows
        """
        self.trajectories = trajectories
        self.k = k
        self.target_resolution = target_resolution
        self.transform = transform
        self.pad_mode = pad_mode

        # Build index mapping: (traj_idx, action_idx) -> window
        self.window_indices = self._build_indices()

    def _build_indices(self) -> List[Tuple[int, int]]:
        """Build list of (trajectory_idx, action_idx) for all valid windows."""
        indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            n_frames = len(traj)
            for action_idx in range(n_frames):
                indices.append((traj_idx, action_idx))
        return indices

    def __len__(self) -> int:
        return len(self.window_indices)

    def _load_frame(self, frame_data: Any) -> Image.Image:
        """Load frame from various input formats."""
        if isinstance(frame_data, Image.Image):
            return frame_data.convert("RGB")
        elif isinstance(frame_data, str):
            return Image.open(frame_data).convert("RGB")
        elif isinstance(frame_data, np.ndarray):
            return Image.fromarray(frame_data).convert("RGB")
        elif isinstance(frame_data, dict) and "bytes" in frame_data:
            import io
            return Image.open(io.BytesIO(frame_data["bytes"])).convert("RGB")
        else:
            raise ValueError(f"Unsupported frame format: {type(frame_data)}")

    def _pad_frames(self, frames: List[Image.Image], target_k: int) -> List[Image.Image]:
        """Pad frames to target length using pad_mode."""
        if len(frames) >= target_k:
            return frames[:target_k]

        n_pad = target_k - len(frames)
        if self.pad_mode == "repeat":
            last_frame = frames[-1] if frames else Image.new("RGB", self.target_resolution, color=(128, 128, 128))
            return frames + [last_frame] * n_pad
        elif self.pad_mode == "zero":
            zero_frame = Image.new("RGB", self.target_resolution, color=(0, 0, 0))
            return frames + [zero_frame] * n_pad
        elif self.pad_mode == "mirror":
            # Mirror the sequence
            padded = frames[:]
            while len(padded) < target_k:
                remaining = target_k - len(padded)
                mirror_src = frames[-2::-1] if len(frames) > 1 else frames
                padded.extend(mirror_src[:remaining])
            return padded
        else:
            raise ValueError(f"Unknown pad_mode: {self.pad_mode}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, action_idx = self.window_indices[idx]
        traj = self.trajectories[traj_idx]
        k = self.k
        half_k = k // 2

        # Calculate window bounds
        start_idx = max(0, action_idx - half_k)
        end_idx = min(len(traj), start_idx + k)

        if end_idx - start_idx < k:
            start_idx = max(0, end_idx - k)

        # Extract and resize frames
        frames = []
        for i in range(start_idx, end_idx):
            frame = self._load_frame(traj[i]["frame"])
            frame = frame.resize(self.target_resolution, Image.LANCZOS)
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        # Pad to k frames
        frames = self._pad_frames(frames, k)

        # Stack to tensor: (k, C, H, W)
        frames_tensor = torch.stack(frames)

        # Get target from action_idx
        target = traj[action_idx]
        risk_label = target.get("label", 0)
        category_label = target.get("category", 4)  # default benign
        bbox = target.get("bbox", [0.0, 0.0, 0.0, 0.0])
        has_bbox = 1.0 if target.get("bbox") is not None else 0.0

        return {
            "frames": frames_tensor,                    # (k, C, H, W)
            "risk_label": torch.tensor(risk_label, dtype=torch.long),
            "category_label": torch.tensor(category_label, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "has_bbox": torch.tensor(has_bbox, dtype=torch.float32),
            "action": target.get("action", ""),
            "trajectory_idx": traj_idx,
            "action_idx": action_idx,
        }


def collate_frame_windows(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for FrameWindowDataset.
    Handles variable-length sequences and stacks tensors.
    """
    frames = torch.stack([item["frames"] for item in batch])  # (B, k, C, H, W)
    risk_labels = torch.stack([item["risk_label"] for item in batch])
    category_labels = torch.stack([item["category_label"] for item in batch])
    bboxes = torch.stack([item["bbox"] for item in batch])
    has_bbox = torch.stack([item["has_bbox"] for item in batch])

    return {
        "frames": frames,
        "risk_label": risk_labels,
        "category_label": category_labels,
        "bbox": bboxes,
        "has_bbox": has_bbox,
        "actions": [item.get("action", "") for item in batch],
        "trajectory_idxs": [item.get("trajectory_idx", item.get("trajectory_id", 0)) for item in batch],
        "trajectory_ids": [item.get("trajectory_id", str(item.get("trajectory_idx", ""))) for item in batch],
        "action_idxs": [item.get("action_idx", 0) for item in batch],
    }


def create_temporal_splits(
    trajectories: List[List[Dict]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split trajectories into train/val/test with temporal consistency.
    Ensures no trajectory appears in multiple splits.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    indices = list(range(len(trajectories)))
    random.shuffle(indices)

    n_train = int(len(trajectories) * train_ratio)
    n_val = int(len(trajectories) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]
    test_trajs = [trajectories[i] for i in test_idx]

    return train_trajs, val_trajs, test_trajs


def extract_frame_windows(
    hf_dataset,
    traj_indices: List[int],
    k: int = 6,
    target_resolution: Tuple[int, int] = (224, 224),
) -> List[Tuple[List[Image.Image], Optional[Tuple[float, float, float, float]], str, str]]:
    """
    Extract frame windows for a list of trajectory indices from a HuggingFace dataset.

    Compatible with src/data/real_dataset.py. Each row in the dataset is expected
    to have an 'image'/'images' field and a 'target_bounding_boxes' or 'bbox' field.

    Args:
        hf_dataset: HuggingFace Dataset with trajectory rows
        traj_indices: Row indices belonging to a single trajectory
        k: Number of frames per window
        target_resolution: Output resolution

    Yields/Returns:
        List of (window_frames, target_box, action_type, desc) tuples.
        target_box is (x, y, w, h) in original pixel coords or None.
    """
    from .loaders import parse_target_bounding_box

    results = []
    for action_idx in traj_indices:
        # Get the image(s) for this row
        row = hf_dataset[action_idx]
        img = row.get("image", None) or row.get("images", None)

        # Collect frames: support single image or list
        frames_for_row = []
        if isinstance(img, list):
            for im in img:
                if isinstance(im, dict) and "bytes" in im:
                    import io
                    frames_for_row.append(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))
                elif im is not None:
                    frames_for_row.append(im.convert("RGB") if hasattr(im, "convert") else Image.fromarray(np.array(im)).convert("RGB"))
        elif img is not None:
            if isinstance(img, dict) and "bytes" in img:
                import io
                frames_for_row.append(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
            else:
                frames_for_row.append(img.convert("RGB") if hasattr(img, "convert") else Image.fromarray(np.array(img)).convert("RGB"))

        # Extract target box in (x, y, w, h) form
        target_box = None
        parsed = parse_target_bounding_box(row)
        if parsed is not None:
            x1, y1, x2, y2 = parsed
            target_box = (x1, y1, x2 - x1, y2 - y1)

        action_type = row.get("action", "") or row.get("target_action", "")
        desc = row.get("target_action", "") or row.get("action", "") or row.get("utterance", "")

        # Use the row's single frame as the window (padding handled by FrameWindowDataset in real use);
        # here we return a single-frame window wrapped for compatibility.
        window_frames = frames_for_row if frames_for_row else [Image.new("RGB", target_resolution, color=(128, 128, 128))]
        # Pad to k if we have multiple frames available from the row
        while len(window_frames) < k:
            window_frames.append(window_frames[-1])

        results.append((window_frames[:k], target_box, action_type, desc))

    return results