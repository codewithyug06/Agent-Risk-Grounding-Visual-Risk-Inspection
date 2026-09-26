"""
Tests for the synthetic injection pipeline (src/data/synthetic_injection.py)
and its dataset integration (SyntheticInjectionDataset, CombinedSentinelDataset
in src/data/loaders.py).

Playwright/browser tests use lightweight mocks — they do not launch a real
browser, keeping this suite fast and deterministic. The real-browser path is
exercised separately by scripts/generate_synthetic_data.py.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.synthetic_injection import (
    PlaywrightInjector,
    HarmfulActionTemplates,
    BenignActionTemplates,
    CATEGORY_TO_IDX,
)
from src.data.loaders import SyntheticInjectionDataset, CombinedSentinelDataset


class TestPlaywrightInjector:
    @pytest.mark.asyncio
    async def test_capture_frame_returns_pil_image(self):
        injector = PlaywrightInjector()
        fake_page = MagicMock()
        # A minimal valid 1x1 PNG.
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        fake_page.screenshot = AsyncMock(return_value=png_bytes)

        frame = await injector.capture_frame(fake_page)
        assert isinstance(frame, Image.Image)
        assert frame.mode == "RGB"

    @pytest.mark.asyncio
    async def test_get_element_bbox_normalizes_to_0_1(self):
        injector = PlaywrightInjector(viewport={"width": 1280, "height": 720})
        fake_locator = MagicMock()
        fake_locator.count = AsyncMock(return_value=1)
        fake_first = MagicMock()
        fake_first.bounding_box = AsyncMock(return_value={"x": 128.0, "y": 72.0, "width": 256.0, "height": 144.0})
        fake_locator.first = fake_first

        fake_page = MagicMock()
        fake_page.locator = MagicMock(return_value=fake_locator)

        bbox = await injector.get_element_bbox(fake_page, "#some-btn")
        assert bbox is not None
        assert len(bbox) == 4
        assert all(0.0 <= v <= 1.0 for v in bbox)
        assert bbox == pytest.approx([0.1, 0.1, 0.3, 0.3], abs=1e-6)

    @pytest.mark.asyncio
    async def test_get_element_bbox_returns_none_when_missing(self):
        injector = PlaywrightInjector()
        fake_locator = MagicMock()
        fake_locator.count = AsyncMock(return_value=0)
        fake_page = MagicMock()
        fake_page.locator = MagicMock(return_value=fake_locator)

        bbox = await injector.get_element_bbox(fake_page, "#missing")
        assert bbox is None

    @pytest.mark.asyncio
    async def test_capture_frame_window_returns_k_frames(self):
        injector = PlaywrightInjector()
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        fake_page = MagicMock()
        fake_page.screenshot = AsyncMock(return_value=png_bytes)
        fake_page.wait_for_timeout = AsyncMock(return_value=None)

        frames = await injector.capture_frame_window(fake_page, k=6, delay_ms=1)
        assert len(frames) == 6
        assert all(isinstance(f, Image.Image) for f in frames)


class TestHarmfulActionTemplates:
    def test_all_categories_have_templates(self):
        for category in HarmfulActionTemplates.all_categories():
            templates = HarmfulActionTemplates.by_category(category)
            assert len(templates) > 0

    def test_each_template_has_selector(self):
        for category in HarmfulActionTemplates.all_categories():
            for template in HarmfulActionTemplates.by_category(category):
                assert template.selector.startswith("#") or template.selector.startswith(".")

    def test_templates_are_valid_html(self):
        for category in HarmfulActionTemplates.all_categories():
            for template in HarmfulActionTemplates.by_category(category):
                html = template.html_factory()
                assert "<html>" in html
                selector_id = template.selector.lstrip("#")
                assert f'id="{selector_id}"' in html

    def test_benign_templates_exist(self):
        assert len(BenignActionTemplates.TEMPLATES) > 0

    def test_category_to_idx_covers_all_categories(self):
        for category in HarmfulActionTemplates.all_categories():
            assert category in CATEGORY_TO_IDX
        assert "benign" in CATEGORY_TO_IDX


def _write_fake_injection_sample(split_dir, sample_id, risk_label, category, category_idx, bbox, k=6):
    sample_dir = split_dir / sample_id
    frames_dir = sample_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(k):
        Image.new("RGB", (32, 32), color=(i * 10, 0, 0)).save(frames_dir / f"frame_{i}.png")
    with open(sample_dir / "label.json", "w") as f:
        json.dump({
            "risk_label": risk_label,
            "category": category,
            "category_idx": category_idx,
            "bbox": bbox,
            "action_type": "click",
            "source_trajectory_id": f"synthetic_{sample_id}",
        }, f)


class TestSyntheticInjectionDataset:
    def test_loads_from_disk(self, tmp_path):
        split_dir = tmp_path / "train"
        _write_fake_injection_sample(split_dir, "destructive_00000", 1, "destructive", 0, [0.1, 0.1, 0.5, 0.5])
        _write_fake_injection_sample(split_dir, "benign_00000", 0, "benign", 4, None)

        ds = SyntheticInjectionDataset(str(tmp_path), split="train", k=6, target_resolution=(32, 32))
        assert len(ds) == 2

    def test_returns_correct_schema(self, tmp_path):
        split_dir = tmp_path / "train"
        _write_fake_injection_sample(split_dir, "destructive_00000", 1, "destructive", 0, [0.1, 0.1, 0.5, 0.5])

        ds = SyntheticInjectionDataset(str(tmp_path), split="train", k=6, target_resolution=(32, 32))
        item = ds[0]
        for key in ("frames", "risk_label", "category_label", "bbox", "has_bbox", "trajectory_id", "action"):
            assert key in item
        assert item["frames"].shape == (6, 3, 32, 32)
        assert item["risk_label"].item() == 1

    def test_bbox_is_normalized(self, tmp_path):
        split_dir = tmp_path / "train"
        _write_fake_injection_sample(split_dir, "destructive_00000", 1, "destructive", 0, [0.2, 0.3, 0.7, 0.8])

        ds = SyntheticInjectionDataset(str(tmp_path), split="train", k=6, target_resolution=(32, 32))
        bbox = ds[0]["bbox"]
        assert all(0.0 <= v.item() <= 1.0 for v in bbox)

    def test_split_counts_correct(self, tmp_path):
        for split, n in [("train", 3), ("val", 1), ("test", 1)]:
            split_dir = tmp_path / split
            for i in range(n):
                _write_fake_injection_sample(split_dir, f"benign_{i:05d}", 0, "benign", 4, None)

        for split, expected_n in [("train", 3), ("val", 1), ("test", 1)]:
            ds = SyntheticInjectionDataset(str(tmp_path), split=split, k=6, target_resolution=(32, 32))
            assert len(ds) == expected_n

    def test_missing_split_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SyntheticInjectionDataset(str(tmp_path), split="train", k=6)

    def test_category_filter(self, tmp_path):
        split_dir = tmp_path / "train"
        _write_fake_injection_sample(split_dir, "destructive_00000", 1, "destructive", 0, [0.1, 0.1, 0.5, 0.5])
        _write_fake_injection_sample(split_dir, "financial_00000", 1, "financial", 1, [0.1, 0.1, 0.5, 0.5])
        _write_fake_injection_sample(split_dir, "benign_00000", 0, "benign", 4, None)

        ds = SyntheticInjectionDataset(str(tmp_path), split="train", k=6, categories=["destructive"])
        assert len(ds) == 1


class TestCombinedSentinelDataset:
    def test_weighted_sampler_upsamples_harmful(self, tmp_path):
        split_dir = tmp_path / "train"
        _write_fake_injection_sample(split_dir, "destructive_00000", 1, "destructive", 0, [0.1, 0.1, 0.5, 0.5])
        for i in range(5):
            _write_fake_injection_sample(split_dir, f"benign_{i:05d}", 0, "benign", 4, None)

        synthetic_ds = SyntheticInjectionDataset(str(tmp_path), split="train", k=6, target_resolution=(32, 32))
        combined = CombinedSentinelDataset(real_dataset=None, synthetic_dataset=synthetic_ds, synthetic_weight=5.0)

        sampler = combined.get_weighted_sampler()
        harmful_idx = [i for i in range(len(combined)) if combined._is_harmful(i)][0]
        # Draw a large number of samples (with replacement) to get a stable
        # estimate of the sampler's draw frequency for the harmful sample.
        n_draws = 5000
        resampled = torch.utils.data.WeightedRandomSampler(sampler.weights, num_samples=n_draws, replacement=True)
        drawn = list(resampled)
        # With 5x weight on a single harmful sample among 6, it should be
        # drawn noticeably more than its natural 1/6 share.
        harmful_draws = sum(1 for i in drawn if i == harmful_idx)
        assert harmful_draws / len(drawn) > 1 / 6

    def test_len_is_sum_of_both(self, tmp_path):
        split_dir = tmp_path / "train"
        for i in range(2):
            _write_fake_injection_sample(split_dir, f"destructive_{i:05d}", 1, "destructive", 0, [0.1, 0.1, 0.5, 0.5])
        for i in range(3):
            _write_fake_injection_sample(split_dir, f"benign_{i:05d}", 0, "benign", 4, None)

        synthetic_ds = SyntheticInjectionDataset(str(tmp_path), split="train", k=6, target_resolution=(32, 32))
        combined = CombinedSentinelDataset(real_dataset=None, synthetic_dataset=synthetic_ds)
        assert len(combined) == 5

    def test_schema_consistent_with_sentineldataset(self, tmp_path):
        split_dir = tmp_path / "train"
        _write_fake_injection_sample(split_dir, "destructive_00000", 1, "destructive", 0, [0.1, 0.1, 0.5, 0.5])
        synthetic_ds = SyntheticInjectionDataset(str(tmp_path), split="train", k=6, target_resolution=(32, 32))
        combined = CombinedSentinelDataset(real_dataset=None, synthetic_dataset=synthetic_ds)

        item = combined[0]
        required_keys = {"frames", "risk_label", "category_label", "bbox", "has_bbox"}
        assert required_keys.issubset(item.keys())

    def test_requires_at_least_one_dataset(self):
        with pytest.raises(ValueError):
            CombinedSentinelDataset(real_dataset=None, synthetic_dataset=None)
