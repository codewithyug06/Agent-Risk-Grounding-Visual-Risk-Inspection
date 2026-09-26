"""
Tests for SentinelModel.generate_attention_rollout (real Attention Rollout,
not a Grad-CAM alias) in src/models/sentinel_model.py.
"""

import os
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sentinel_model import create_sentinel_model


@pytest.fixture
def small_model_config():
    return OmegaConf.create({
        "backbone": "vit_small_patch16_224",
        "pretrained": False,
        "freeze_backbone": True,
        "embed_dim": 384,
        "num_heads": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "fusion_mode": "last",
        "use_delta": True,
        "use_temp_pos": True,
        "risk_head": {"embed_dim": 384, "hidden_dim": 64, "num_categories": 5, "dropout": 0.0},
        "localization_head": {
            "embed_dim": 384, "num_anchors": 9, "anchor_sizes": [32, 64, 128],
            "anchor_ratios": [0.5, 1.0, 2.0], "hidden_dim": 64,
            "nms_threshold": 0.5, "score_threshold": 0.3,
        },
        "frame_window": {"k": 6, "resolution": [224, 224]},
    })


class TestAttentionRollout:
    def test_rollout_returns_correct_shape(self, small_model_config):
        model = create_sentinel_model(small_model_config)
        model.eval()
        frames = torch.randn(1, 6, 3, 224, 224)
        rollout = model.generate_attention_rollout(frames)
        assert isinstance(rollout, np.ndarray)
        assert rollout.shape == (224, 224)

    def test_rollout_in_0_1_range(self, small_model_config):
        model = create_sentinel_model(small_model_config)
        model.eval()
        frames = torch.randn(1, 6, 3, 224, 224)
        rollout = model.generate_attention_rollout(frames)
        assert rollout.min() >= 0.0 - 1e-5
        assert rollout.max() <= 1.0 + 1e-5

    def test_rollout_different_from_gradcam(self, small_model_config):
        model = create_sentinel_model(small_model_config)
        model.eval()
        frames = torch.randn(1, 6, 3, 224, 224)

        rollout = model.generate_attention_rollout(frames)
        gradcam = model.generate_heatmap(frames)

        a = rollout.flatten().astype(np.float64)
        b = gradcam.flatten().astype(np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        cos_sim = float(a @ b / denom) if denom > 1e-12 else 0.0
        assert cos_sim < 0.95

    def test_no_memory_leak_after_rollout(self, small_model_config):
        model = create_sentinel_model(small_model_config)
        model.eval()
        frames = torch.randn(1, 6, 3, 224, 224)

        n_hooks_before = sum(
            len(m._forward_hooks) + len(m._forward_pre_hooks)
            for m in model.modules()
        )
        model.generate_attention_rollout(frames)
        n_hooks_after = sum(
            len(m._forward_hooks) + len(m._forward_pre_hooks)
            for m in model.modules()
        )
        # generate_attention_rollout never registers forward hooks (it captures
        # attention via an explicit re-execution path), so hook counts must be
        # identical before and after — nothing to leak.
        assert n_hooks_before == n_hooks_after

    def test_single_batch_only(self, small_model_config):
        model = create_sentinel_model(small_model_config)
        model.eval()
        frames = torch.randn(2, 6, 3, 224, 224)
        with pytest.raises(ValueError):
            model.generate_attention_rollout(frames)

    def test_accepts_unbatched_input(self, small_model_config):
        model = create_sentinel_model(small_model_config)
        model.eval()
        frames = torch.randn(6, 3, 224, 224)  # no batch dim
        rollout = model.generate_attention_rollout(frames)
        assert rollout.shape == (224, 224)
