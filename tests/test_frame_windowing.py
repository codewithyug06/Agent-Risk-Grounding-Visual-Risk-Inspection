"""
Tests for frame windowing functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data.frame_windowing import (
    TimestampedFrame,
    extract_frame_window,
    extract_frame_window_by_index,
    FrameWindowDataset,
    collate_frame_windows,
    create_temporal_splits,
)
from src.data.loaders import (
    FrameActionPair,
    Trajectory,
    SentinelDataset,
    load_real_dataset,
)
from src.data.augmentation import create_val_transform
from omegaconf import DictConfig, OmegaConf
from PIL import Image


class TestTimestampedFrame:
    """Tests for TimestampedFrame dataclass."""

    def test_creation(self):
        """Test TimestampedFrame creation."""
        from PIL import Image
        frame = Image.new("RGB", (224, 224), color="red")
        tf = TimestampedFrame(frame=frame, timestamp=1.5, frame_idx=3)
        assert tf.frame == frame
        assert tf.timestamp == 1.5
        assert tf.frame_idx == 3


class TestExtractFrameWindow:
    """Tests for extract_frame_window function (timestamped version)."""

    def test_basic_extraction(self):
        """Test basic window extraction centered on action timestamp."""
        from PIL import Image
        frames = [TimestampedFrame(
            frame=Image.new("RGB", (224, 224), color=(i*20, 0, 0)),
            timestamp=float(i),
            frame_idx=i
        ) for i in range(10)]

        window = extract_frame_window(frames, action_timestamp=5.0, k=6)

        assert len(window) == 6
        assert all(isinstance(f, Image.Image) for f in window)
        # Centered on frame 5 -> timestamps 2-7 (no padding needed)
        for i in range(6):
            assert window[i].getpixel((0,0))[0] == (2+i)*20

    def test_padding_at_start(self):
        """Test window near start is clamped to start (index 0-5)."""
        from PIL import Image
        frames = [TimestampedFrame(
            frame=Image.new("RGB", (224, 224), color=(i*20, 0, 0)),
            timestamp=float(i),
            frame_idx=i
        ) for i in range(10)]

        window = extract_frame_window(frames, action_timestamp=0.5, k=6)

        assert len(window) == 6
        # Closest frame is index 0; window spans indices 0-5
        assert window[0].getpixel((0,0))[0] == 0    # frame 0
        assert window[1].getpixel((0,0))[0] == 20   # frame 1
        assert window[2].getpixel((0,0))[0] == 40   # frame 2
        assert window[3].getpixel((0,0))[0] == 60   # frame 3
        assert window[4].getpixel((0,0))[0] == 80   # frame 4
        assert window[5].getpixel((0,0))[0] == 100  # frame 5

    def test_padding_at_end(self):
        """Test window near end repeats last frame when truncated."""
        from PIL import Image
        frames = [TimestampedFrame(
            frame=Image.new("RGB", (224, 224), color=(i*20, 0, 0)),
            timestamp=float(i),
            frame_idx=i
        ) for i in range(10)]

        window = extract_frame_window(frames, action_timestamp=9.5, k=6)

        assert len(window) == 6
        # Closest frame is index 9; window clamps to indices 4-9 (no padding needed)
        assert window[0].getpixel((0,0))[0] == 80   # frame 4
        assert window[1].getpixel((0,0))[0] == 100  # frame 5
        assert window[2].getpixel((0,0))[0] == 120  # frame 6
        assert window[3].getpixel((0,0))[0] == 140  # frame 7
        assert window[4].getpixel((0,0))[0] == 160  # frame 8
        assert window[5].getpixel((0,0))[0] == 180  # frame 9

    def test_empty_trajectory(self):
        """Test with empty trajectory."""
        from PIL import Image
        window = extract_frame_window([], action_timestamp=5.0, k=6)
        assert len(window) == 6
        assert all(f.getpixel((0,0)) == (128, 128, 128) for f in window)

    def test_custom_fps_and_resolution(self):
        """Test with custom fps and target resolution."""
        from PIL import Image
        frames = [TimestampedFrame(
            frame=Image.new("RGB", (448, 448), color=(i*20, 0, 0)),
            timestamp=float(i) * 0.5,  # 2 fps
            frame_idx=i
        ) for i in range(10)]

        window = extract_frame_window(frames, action_timestamp=2.0, k=4, fps=2, target_resolution=(112, 112))

        assert len(window) == 4
        assert window[0].size == (112, 112)


class TestExtractFrameWindowByIndex:
    """Tests for extract_frame_window_by_index function (simple index version)."""

    def test_basic_extraction(self):
        """Test basic window extraction by index (centered)."""
        from PIL import Image
        frames = [Image.new("RGB", (224, 224), color=(i*20, 0, 0)) for i in range(10)]

        window = extract_frame_window_by_index(frames, action_idx=5, k=6)

        assert len(window) == 6
        # Centered on index 5 -> frames 2-7
        for i in range(6):
            assert window[i].getpixel((0,0))[0] == (2+i)*20

    def test_padding_at_start(self):
        """Test window clamped to start when action_idx near start."""
        from PIL import Image
        frames = [Image.new("RGB", (224, 224), color=(i*20, 0, 0)) for i in range(10)]

        window = extract_frame_window_by_index(frames, action_idx=1, k=6)

        assert len(window) == 6
        # Window spans indices 0-5
        assert window[0].getpixel((0,0))[0] == 0    # frame 0
        assert window[1].getpixel((0,0))[0] == 20   # frame 1
        assert window[2].getpixel((0,0))[0] == 40   # frame 2
        assert window[3].getpixel((0,0))[0] == 60   # frame 3
        assert window[4].getpixel((0,0))[0] == 80   # frame 4
        assert window[5].getpixel((0,0))[0] == 100  # frame 5

    def test_padding_at_end(self):
        """Test window clamps/pads when action_idx near end."""
        from PIL import Image
        frames = [Image.new("RGB", (224, 224), color=(i*20, 0, 0)) for i in range(10)]

        window = extract_frame_window_by_index(frames, action_idx=8, k=6)

        assert len(window) == 6
        # Window clamps to indices 4-9 (no padding needed)
        assert window[0].getpixel((0,0))[0] == 80   # frame 4
        assert window[3].getpixel((0,0))[0] == 140  # frame 7
        assert window[4].getpixel((0,0))[0] == 160  # frame 8
        assert window[5].getpixel((0,0))[0] == 180  # frame 9

    def test_window_size_one(self):
        """Test window size of 1."""
        from PIL import Image
        frames = [Image.new("RGB", (224, 224), color=(i*20, 0, 0)) for i in range(10)]

        window = extract_frame_window_by_index(frames, action_idx=5, k=1)

        assert len(window) == 1
        assert window[0].getpixel((0,0))[0] == 100  # frame 5

    def test_large_window(self):
        """Test window larger than sequence."""
        from PIL import Image
        frames = [Image.new("RGB", (224, 224), color=(i*20, 0, 0)) for i in range(5)]

        window = extract_frame_window_by_index(frames, action_idx=2, k=10)

        assert len(window) == 10
        for i in range(5):
            assert window[i].getpixel((0,0))[0] == i*20
        for i in range(5, 10):
            assert window[i].getpixel((0,0))[0] == 80  # frame 4 (padded)


class TestFrameActionPair:
    """Tests for FrameActionPair dataclass (from loaders)."""

    def test_creation(self):
        """Test FrameActionPair creation."""
        from PIL import Image
        frame = Image.new("RGB", (224, 224), color="red")
        pair = FrameActionPair(
            frame=frame,
            action="click",
            label=1,
            category=0,
            bbox=[0.1, 0.1, 0.5, 0.5],
            metadata={"source": "test"}
        )
        assert pair.frame == frame
        assert pair.action == "click"
        assert pair.label == 1
        assert pair.category == 0
        assert pair.bbox == [0.1, 0.1, 0.5, 0.5]
        assert pair.metadata["source"] == "test"

    def test_defaults(self):
        """Test default values."""
        from PIL import Image
        frame = Image.new("RGB", (224, 224), color="red")
        pair = FrameActionPair(frame=frame, action="click", label=0, category=4)
        assert pair.bbox is None
        assert pair.metadata is None


class TestTrajectory:
    """Tests for Trajectory dataclass (from loaders)."""

    def test_creation(self):
        """Test Trajectory creation."""
        from PIL import Image
        pairs = [
            FrameActionPair(
                frame=Image.new("RGB", (224, 224), color=(i*20, 0, 0)),
                action="click",
                label=i % 2,
                category=0 if i % 2 == 1 else 4,
                bbox=[0.1, 0.1, 0.5, 0.5] if i % 2 == 1 else None,
            )
            for i in range(20)
        ]
        traj = Trajectory(pairs=pairs, trajectory_id="traj_001", source_dataset="test")
        assert len(traj.pairs) == 20
        assert traj.trajectory_id == "traj_001"
        assert traj.source_dataset == "test"


class TestFrameWindowDataset:
    """Tests for FrameWindowDataset."""

    def setup_method(self):
        """Create test config."""
        self.config = OmegaConf.create({
            "data": {
                "sources": [],
                "frame_window": {"k": 6, "fps": 3, "resolution": [224, 224]},
            },
            "frame_window": {"k": 6, "fps": 3, "resolution": [224, 224]},
        })
        self.transform = create_val_transform(OmegaConf.to_container(self.config))

    def test_creation(self):
        """Test dataset creation."""
        # FrameWindowDataset expects list of trajectory dicts
        traj = [
            {
                "frame": Image.new("RGB", (224, 224), color=(i*10, 0, 0)),
                "timestamp": float(i),
                "action": "click",
                "label": i % 2,
                "category": 0 if i % 2 == 1 else 4,
                "bbox": [0.1, 0.1, 0.5, 0.5] if i % 2 == 1 else None,
            }
            for i in range(20)
        ]

        dataset = FrameWindowDataset(
            trajectories=[traj],
            k=6,
            transform=None,
        )
        assert len(dataset) == 20

    def test_getitem(self):
        """Test __getitem__ returns correct keys and shapes."""
        traj = [
            {
                "frame": Image.new("RGB", (224, 224), color=(i*10, 0, 0)),
                "timestamp": float(i),
                "action": "click",
                "label": i % 2,
                "category": 0 if i % 2 == 1 else 4,
                "bbox": [0.1, 0.1, 0.5, 0.5],
            }
            for i in range(20)
        ]

        dataset = FrameWindowDataset(
            trajectories=[traj],
            k=6,
            transform=None,
        )

        item = dataset[0]
        assert "frames" in item
        assert item["frames"].shape == (6, 3, 224, 224)
        assert "risk_label" in item
        assert "category_label" in item
        assert "bbox" in item
        assert "has_bbox" in item
        assert "action" in item
        assert "trajectory_idx" in item
        assert "action_idx" in item


class TestCollateFrameWindows:
    """Tests for collate_frame_windows function."""

    def test_collate(self):
        """Test collation of batch."""
        from PIL import Image
        batch = []
        for i in range(4):
            frames = [Image.new("RGB", (224, 224), color=(i*10, 0, 0)) for _ in range(6)]
            tensors = [torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255.0 for f in frames]
            item = {
                "frames": torch.stack(tensors),
                "risk_label": torch.tensor(i % 2, dtype=torch.long),
                "category_label": torch.tensor(0 if i % 2 == 1 else 4, dtype=torch.long),
                "bbox": torch.tensor([0.1, 0.1, 0.5, 0.5], dtype=torch.float32),
                "has_bbox": torch.tensor(1.0 if i % 2 == 1 else 0.0, dtype=torch.float32),
                "action": "click",
                "trajectory_idx": 0,
                "action_idx": i,
            }
            batch.append(item)

        collated = collate_frame_windows(batch)

        assert collated["frames"].shape == (4, 6, 3, 224, 224)
        assert collated["risk_label"].shape == (4,)
        assert collated["category_label"].shape == (4,)
        assert collated["bbox"].shape == (4, 4)
        assert collated["has_bbox"].shape == (4,)
        assert len(collated["actions"]) == 4
        assert len(collated["trajectory_idxs"]) == 4
        assert len(collated["action_idxs"]) == 4


class TestCreateTemporalSplits:
    """Tests for create_temporal_splits function."""

    def test_basic_split(self):
        """Test basic temporal split."""
        trajectories = [{"task_id": f"task_{i}", "frames": [{"frame": None} for _ in range(10)]} for i in range(100)]

        train, val, test = create_temporal_splits(
            trajectories, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

        # No overlap
        train_ids = {t["task_id"] for t in train}
        val_ids = {t["task_id"] for t in val}
        test_ids = {t["task_id"] for t in test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_deterministic_split(self):
        """Test split is deterministic with same seed."""
        trajectories = [{"task_id": f"task_{i}"} for i in range(50)]

        train1, val1, test1 = create_temporal_splits(trajectories, seed=123)
        train2, val2, test2 = create_temporal_splits(trajectories, seed=123)

        assert [t["task_id"] for t in train1] == [t["task_id"] for t in train2]
        assert [t["task_id"] for t in val1] == [t["task_id"] for t in val2]
        assert [t["task_id"] for t in test1] == [t["task_id"] for t in test2]


class TestSentinelDatasetIntegration:
    """Integration tests with SentinelDataset (from loaders)."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        return OmegaConf.create({
            "data": {
                "sources": [],
                "frame_window": {"k": 6, "fps": 3, "resolution": [224, 224]},
            },
            "frame_window": {"k": 6, "fps": 3, "resolution": [224, 224]},
            "training": {"batch_size": 4, "num_workers": 0},
        })

    def test_dataset_creation(self, mock_config):
        """Test SentinelDataset creation with synthetic data."""
        transform = create_val_transform(OmegaConf.to_container(mock_config))
        try:
            dataset = SentinelDataset(
                data_config=OmegaConf.to_container(mock_config),
                split="val",
                transform=transform,
                frame_window_k=6,
                target_resolution=(224, 224),
            )
            assert dataset is not None
        except FileNotFoundError:
            # Expected if no preprocessed data
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])