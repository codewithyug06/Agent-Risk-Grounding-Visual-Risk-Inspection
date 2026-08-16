"""
Pytest configuration and fixtures for SENTINEL-Vision tests.
"""

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf


@pytest.fixture
def device():
    """Get test device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model_config():
    """Base model configuration for tests."""
    return OmegaConf.create({
        "backbone": "vit_small_patch16_224",
        "pretrained": False,
        "freeze_backbone": True,
        "embed_dim": 384,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.1,
        "fusion_mode": "attn_pool",
        "use_delta": True,
        "use_temp_pos": True,
        "risk_head": {
            "embed_dim": 384,
            "hidden_dim": 256,
            "num_categories": 5,
            "dropout": 0.1,
        },
        "localization_head": {
            "embed_dim": 384,
            "num_anchors": 9,
            "anchor_sizes": [32, 64, 128],
            "anchor_ratios": [0.5, 1.0, 2.0],
            "hidden_dim": 256,
            "nms_threshold": 0.5,
            "score_threshold": 0.3,
        },
        "frame_window": {"k": 6, "resolution": [224, 224]},
    })


@pytest.fixture
def gate_config():
    """Gate configuration for tests."""
    return OmegaConf.create({
        "state_dim": 8,
        "hidden_dim": 128,
        "num_actions": 3,
    })


@pytest.fixture
def reward_config():
    """Reward configuration for tests."""
    return OmegaConf.create({
        "missed_harm_penalty": -10.0,
        "false_block_penalty": -1.0,
        "correct_block_reward": 1.0,
        "correct_allow_reward": 0.1,
        "pause_penalty": -0.5,
    })


@pytest.fixture
def sample_frames():
    """Generate sample frame tensors."""
    def _frames(batch=1, k=6, c=3, h=224, w=224):
        return torch.randn(batch, k, c, h, w)
    return _frames


@pytest.fixture
def sample_pil_frames():
    """Generate sample PIL frames."""
    def _frames(k=6, h=224, w=224):
        from PIL import Image
        return [Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)) for _ in range(k)]
    return _frames


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    torch.manual_seed(torch.initial_seed())


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")


# Skip GPU tests if no CUDA
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)