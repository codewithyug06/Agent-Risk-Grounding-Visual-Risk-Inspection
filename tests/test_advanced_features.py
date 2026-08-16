"""
Unit tests for SENTINEL-Vision Advanced Features:
- Uncertainty Quantification (MC Dropout)
- ONNX INT8 Quantization export
- Contrastive Temporal Loss
- Online Hard Example Mining (OHEM)
- Cross-Agent Generalization Transfer Metrics
- Attention Rollout visualization
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import tempfile
from omegaconf import OmegaConf

from src.models.sentinel_model import SentinelModel, create_sentinel_model
from src.training.losses import ContrastiveTemporalLoss, OHEMLoss
from src.eval.metrics import compute_cross_agent_metrics


@pytest.fixture
def minimal_config():
    return OmegaConf.create({
        "backbone": "vit_small_patch16_224",
        "pretrained": False,
        "freeze_backbone": False,
        "freeze_epochs": 0,
        "temporal_fusion": {
            "num_layers": 1,
            "num_heads": 2,
            "embed_dim": 384,
            "dropout": 0.1,
            "use_delta_features": True,
        },
        "risk_head": {
            "dropout": 0.1,
            "hidden_dim": 64,
        },
        "localization_head": {
            "heatmap_size": 14,
            "dropout": 0.1,
            "use_fpn": False,
        },
        "frame_window": {
            "k": 3,
            "resolution": [224, 224],
        },
        "image_size": 224,
    })


def test_mc_dropout_uncertainty(minimal_config):
    """Test Monte Carlo Dropout produces mean and variance."""
    model = create_sentinel_model(minimal_config)
    frames = [Image.new("RGB", (224, 224), color=(i * 40, 100, 150)) for i in range(3)]

    results = model.predict_with_uncertainty(frames, num_samples=4)

    assert "risk_score" in results
    assert "risk_variance" in results
    assert "epistemic_uncertainty" in results
    assert "predictive_entropy" in results
    assert 0.0 <= results["risk_score"] <= 1.0
    assert results["risk_variance"] >= 0.0
    assert results["num_samples"] == 4


def test_contrastive_temporal_loss():
    """Test InfoNCE contrastive temporal loss."""
    loss_fn = ContrastiveTemporalLoss(temperature=0.1)
    B, k, D = 4, 3, 64
    features = torch.randn(B, k, D, requires_grad=True)

    loss = loss_fn(features)
    assert loss.item() > 0.0
    loss.backward()
    assert features.grad is not None


def test_ohem_loss():
    """Test Online Hard Example Mining loss."""
    ce = nn.CrossEntropyLoss(reduction="none")
    ohem = OHEMLoss(loss_fn=ce, keep_ratio=0.5)

    B = 8
    logits = torch.randn(B, 5, requires_grad=True)
    targets = torch.randint(0, 5, (B,))

    loss = ohem(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_cross_agent_metrics():
    """Test cross-agent transfer generalization calculation."""
    agent_preds = {
        "claude_computer_use": np.array([0.9, 0.8, 0.1, 0.2]),
        "osworld_agent": np.array([0.7, 0.6, 0.4, 0.3]),
        "webarena_agent": np.array([0.95, 0.85, 0.05, 0.15]),
    }
    agent_labels = {
        "claude_computer_use": np.array([1, 1, 0, 0]),
        "osworld_agent": np.array([1, 1, 0, 0]),
        "webarena_agent": np.array([1, 1, 0, 0]),
    }

    metrics = compute_cross_agent_metrics(agent_preds, agent_labels, threshold=0.5)

    assert "worst_group_accuracy" in metrics
    assert "mean_accuracy" in metrics
    assert "generalization_gap" in metrics
    assert "per_agent" in metrics
    assert metrics["mean_accuracy"] == 1.0


def test_attention_rollout(minimal_config):
    """Test attention rollout heatmap output."""
    model = create_sentinel_model(minimal_config)
    dummy_tensor = torch.randn(1, 3, 3, 224, 224)

    heatmap = model.generate_attention_rollout(dummy_tensor)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-5
