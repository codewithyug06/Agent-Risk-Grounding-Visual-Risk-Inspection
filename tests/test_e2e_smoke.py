"""
End-to-end smoke test: 6 raw PIL images -> SentinelModel -> DecisionGate -> decision.
Does not require trained weights — verifies the full inference path is wired
correctly end to end, independent of model quality.
"""

import os
import sys

import pytest
from omegaconf import OmegaConf
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sentinel_model import create_sentinel_model
from src.gate.decision_gate import create_decision_gate


def _small_model_config():
    return OmegaConf.create({
        "backbone": "vit_small_patch16_224",
        "pretrained": False,
        "freeze_backbone": True,
        "embed_dim": 384,
        "num_heads": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "fusion_mode": "attn_pool",
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


def _gate_config():
    return OmegaConf.create({"state_dim": 8, "hidden_dim": 64, "num_actions": 3})


def test_full_pipeline_from_raw_frames_to_decision():
    frames = [Image.new("RGB", (224, 224), color=(i * 30, i * 20, i * 10)) for i in range(6)]

    model = create_sentinel_model(_small_model_config())
    gate = create_decision_gate(OmegaConf.to_container(_gate_config()))

    result = model.predict(frames)

    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["category"] in ["destructive", "financial", "privacy", "irreversible_external", "benign"]
    assert len(result["bbox"]) == 4
    assert all(isinstance(v, float) for v in result["bbox"])

    decision = gate.get_action(
        risk_score=result["risk_score"],
        category=result["category"],
        heatmap_conf=result["objectness"],
        action_type=0,
    )
    assert decision in ["ALLOW", "PAUSE", "HARD_BLOCK"]
