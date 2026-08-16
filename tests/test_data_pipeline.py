"""
Tests for SENTINEL-Vision data pipeline.

Targets the implemented API: loaders.load_multimodal_mind2web,
loaders.SentinelDataset, loaders.load_real_dataset, and
frame_windowing.extract_frame_windows.
"""

import os
import sys
import json
import tempfile
import shutil

import pytest
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loaders import (
    FrameActionPair,
    Trajectory,
    SentinelDataset,
    load_multimodal_mind2web,
    load_screenspot,
    load_real_dataset,
)
from src.data.frame_windowing import extract_frame_windows


def _make_dummy_image(size=(224, 224), color=(128, 128, 128)):
    return Image.new("RGB", size, color)


def _write_synthetic_processed_dir(base_dir, split="train", n_trajs=2, n_pairs=4):
    """Write a synthetic processed_dir/{split}.jsonl + frame images."""
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Create a frame image reused across records
    frame_dir = os.path.join(processed_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    frame_path = os.path.join(frame_dir, "frame_0.png")
    _make_dummy_image().save(frame_path)

    rel_frame = os.path.relpath(frame_path, processed_dir)

    records = []
    for t in range(n_trajs):
        for p in range(n_pairs):
            has_bbox = (p % 2 == 1)
            records.append({
                "frame_path": rel_frame,
                "action": "click submit" if has_bbox else "scroll down",
                "label": 1 if has_bbox else 0,
                "category": 0 if has_bbox else 4,
                "bbox": [0.1, 0.1, 0.5, 0.5] if has_bbox else None,
                "source": "synthetic",
                "task_id": f"task_{t}",
                "metadata": {},
            })

    split_file = os.path.join(processed_dir, f"{split}.jsonl")
    with open(split_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return processed_dir


class TestLoaders:
    """Tests for dataset loaders."""

    def test_frame_action_pair(self):
        """Test FrameActionPair dataclass."""
        frame = _make_dummy_image()
        pair = FrameActionPair(
            frame=frame,
            action="click delete",
            label=1,
            category=0,
            bbox=[0.1, 0.1, 0.5, 0.5],
        )
        assert pair.label == 1
        assert pair.category == 0
        assert pair.bbox == [0.1, 0.1, 0.5, 0.5]

    def test_trajectory(self):
        """Test Trajectory dataclass."""
        pairs = [
            FrameActionPair(
                frame=_make_dummy_image(),
                action=f"action {i}",
                label=i % 2,
                category=0 if i % 2 else 4,
                bbox=[0.1, 0.1, 0.5, 0.5] if i % 2 else None,
            )
            for i in range(5)
        ]
        traj = Trajectory(pairs=pairs, trajectory_id="t1", source_dataset="synthetic")
        assert len(traj.pairs) == 5
        assert traj.trajectory_id == "t1"


class TestSentinelDataset:
    """Tests for SentinelDataset."""

    def test_dataset_from_synthetic_processed(self):
        """Test SentinelDataset loads from a synthetic processed dir."""
        tmp = tempfile.mkdtemp()
        try:
            processed_dir = _write_synthetic_processed_dir(tmp, split="train")
            data_config = {"processed_dir": processed_dir}

            dataset = SentinelDataset(
                data_config=data_config,
                split="train",
                frame_window_k=3,
                target_resolution=(224, 224),
            )

            assert len(dataset) > 0
            item = dataset[0]
            assert "frames" in item
            assert item["frames"].shape == (3, 3, 224, 224)
            assert "risk_label" in item
            assert "category_label" in item
            assert "bbox" in item
            assert "has_bbox" in item
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_real_dataset_factory(self):
        """Test load_real_dataset factory."""
        tmp = tempfile.mkdtemp()
        try:
            processed_dir = _write_synthetic_processed_dir(tmp, split="val")
            data_config = {"processed_dir": processed_dir}

            dataset = load_real_dataset(
                data_config=data_config,
                split="val",
                frame_window_k=4,
                target_resolution=(224, 224),
            )

            assert isinstance(dataset, SentinelDataset)
            assert len(dataset) > 0
            item = dataset[0]
            assert item["frames"].shape == (4, 3, 224, 224)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_processed_dir_raises(self):
        """Test FileNotFoundError when processed data missing."""
        with pytest.raises(FileNotFoundError):
            SentinelDataset(
                data_config={"processed_dir": "/nonexistent/path/xyz"},
                split="train",
            )


class TestExtractFrameWindows:
    """Tests for extract_frame_windows against a HuggingFace-style dataset."""

    def test_extract_with_synthetic_hf(self):
        """Test extract_frame_windows on a small in-memory HF dataset."""
        # Build a minimal HF dataset-like object supporting __getitem__ and __len__
        from datasets import Dataset as HFDataset

        n = 6
        rows = []
        for i in range(n):
            img = _make_dummy_image((300, 300))
            # Build an HF-compatible image dict (PIL under "image")
            rows.append({
                "image": img,
                "action": "click" if i % 2 == 1 else "scroll",
                "target_action_reprs": "click submit" if i % 2 == 1 else "scroll down",
                "pos_candidates": [
                    {"bbox": [10, 10, 50, 50]}
                ] if i % 2 == 1 else [],
                "annotation_id": f"a{i}",
                "confirmed_task": "task",
                "website": "w",
                "domain": "d",
            })

        hf_ds = HFDataset.from_list(rows)

        # Extract window around row indices 1..4 (a trajectory)
        traj_indices = list(range(1, 5))
        windows = list(extract_frame_windows(hf_ds, traj_indices, k=3))

        assert len(windows) == len(traj_indices)
        for window, target_box, action_type, desc in windows:
            assert isinstance(window, list)
            assert len(window) == 3  # k=3
            assert all(isinstance(f, Image.Image) for f in window)


class TestRealDatasetLoading:
    """Integration tests that only run when real data is available."""

    def test_load_multimodal_mind2web_available(self):
        """Test load_multimodal_mind2web if local dataset present (else skip)."""
        data_dir = r'D:\Computer Vision Project\Datasets\Multimodal-Mind2Web'
        if not os.path.isdir(data_dir):
            pytest.skip(f"Local dataset not available at {data_dir}")
        try:
            pairs = list(load_multimodal_mind2web(data_dir))
        except Exception as e:
            pytest.skip(f"Could not load dataset: {e}")
        assert len(pairs) > 0
        assert isinstance(pairs[0], FrameActionPair)

    def test_load_screenspot_available(self):
        """Test load_screenspot if local dataset present (else skip)."""
        data_dir = r'D:\Computer Vision Project\Datasets\ScreenSpot'
        if not os.path.isdir(data_dir):
            pytest.skip(f"Local dataset not available at {data_dir}")
        try:
            pairs = list(load_screenspot(data_dir))
        except Exception as e:
            pytest.skip(f"Could not load dataset: {e}")
        assert len(pairs) > 0
        assert isinstance(pairs[0], FrameActionPair)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])