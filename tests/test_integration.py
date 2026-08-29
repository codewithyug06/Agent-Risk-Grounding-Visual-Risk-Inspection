"""
Integration tests for SENTINEL-Vision components.
Tests end-to-end workflows: data -> model -> gate -> decision.

Targets the implemented SENTINEL-Vision API.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf
from unittest.mock import patch
from PIL import Image
import io
import base64

from src.models.sentinel_model import create_sentinel_model
from src.gate.decision_gate import create_decision_gate
from src.gate.reward import create_reward_function
from src.data.augmentation import create_val_transform
from src.data.frame_windowing import collate_frame_windows
from src.integration.agent_wrapper import (
    SentinelWrapper,
    AgentAction,
    SentinelDecision,
    FrameBuffer,
)
from src.integration.intercept_api import (
    decode_frame,
    encode_heatmap,
    create_decision_response,
    APIState,
)


class TestEndToEndPipeline:
    """Test complete pipeline from frames to decision."""

    def setup_method(self):
        """Setup test config."""
        self.config = OmegaConf.create({
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
            "risk_head": {"embed_dim": 384, "hidden_dim": 256, "num_categories": 5, "dropout": 0.1},
            "localization": {
                "embed_dim": 384,
                "num_anchors": 9,
                "anchor_sizes": [32, 64, 128],
                "feature_stride": 16,
            },
            "frame_window": {"k": 6, "resolution": [224, 224]},
            "gate": {
                "state_dim": 8,
                "hidden_dim": 128,
                "num_actions": 3,
            },
            "reward": {
                "missed_harm": -10.0,
                "false_block": -1.0,
                "correct_block": 2.0,
                "correct_allow": 1.0,
                "correct_pause": 0.5,
                "false_pause": -0.5,
            },
        })

    def test_frames_to_decision(self):
        """Test full pipeline: frames -> model -> gate -> decision."""
        # Create models
        sentinel = create_sentinel_model(self.config)
        sentinel.eval()

        gate = create_decision_gate(self.config)
        gate.eval()

        # Create dummy frames
        frames = torch.randn(1, 6, 3, 224, 224)

        # SENTINEL forward
        with torch.no_grad():
            sentinel_out = sentinel(frames)

        # Gate decision
        risk_score = sentinel_out["risk_score"].item()
        category = sentinel_out["category_idx"].item()
        heatmap_conf = sentinel_out["objectness"].item()

        action = gate.get_action(risk_score, category, heatmap_conf, deterministic=True)

        assert action in ["ALLOW", "PAUSE", "HARD_BLOCK"]

    def test_wrapper_integration(self):
        """Test SentinelWrapper integration."""
        sentinel_checkpoint = "dummy.pt"
        gate_checkpoint = "dummy.pt"

        # Mock the model loading
        with patch('torch.load') as mock_load:
            mock_load.return_value = {
                "model_state_dict": create_sentinel_model(self.config).state_dict(),
                "gate_state_dict": create_decision_gate(self.config).state_dict(),
            }

            wrapper = SentinelWrapper(
                config=self.config,
                sentinel_checkpoint=sentinel_checkpoint,
                gate_checkpoint=gate_checkpoint,
                device="cpu",
            )

        # Add frames
        for _ in range(6):
            frame = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
            wrapper.add_frame(frame)

        # Predict
        prediction = wrapper.predict()
        assert "risk_score" in prediction
        assert "category_idx" in prediction

        # Create action
        action = AgentAction(action_type="click", selector="#submit")

        # Decide
        decision = wrapper.decide(prediction, action)
        assert isinstance(decision, SentinelDecision)
        assert decision.action in ["ALLOW", "PAUSE", "HARD_BLOCK"]

    def test_intercept_action(self):
        """Test action interception."""
        with patch('torch.load') as mock_load:
            mock_load.return_value = {
                "model_state_dict": create_sentinel_model(self.config).state_dict(),
                "gate_state_dict": create_decision_gate(self.config).state_dict(),
            }

            wrapper = SentinelWrapper(
                config=self.config,
                sentinel_checkpoint="dummy.pt",
                gate_checkpoint="dummy.pt",
                device="cpu",
            )

        # Add frames
        for _ in range(6):
            frame = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
            wrapper.add_frame(frame)

        # Intercept
        action = AgentAction(action_type="click", selector="#delete")
        decision, should_proceed = wrapper.intercept_action(action)

        assert isinstance(decision, SentinelDecision)
        assert isinstance(should_proceed, bool)


class TestFrameBuffer:
    """Test FrameBuffer functionality."""

    def test_buffer_operations(self):
        """Test adding and retrieving frames."""
        buffer = FrameBuffer(max_frames=6, target_size=(224, 224))

        # Add frames
        for i in range(10):
            frame = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype(np.uint8))
            buffer.add_frame(frame)

        # Should only keep last 6
        assert len(buffer) == 6

        # Get tensor
        config = OmegaConf.create({
            "frame_window": {"k": 6, "resolution": [224, 224]},
            "data": {"augmentation": {"val": {}}},
        })
        transform = create_val_transform(OmegaConf.to_container(config))
        tensor = buffer.get_tensor(transform)

        assert tensor.shape == (6, 3, 224, 224)

    def test_buffer_padding(self):
        """Test buffer padding when not full."""
        buffer = FrameBuffer(max_frames=6, target_size=(224, 224))

        # Add only 2 frames
        for i in range(2):
            frame = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype(np.uint8))
            buffer.add_frame(frame)

        config = OmegaConf.create({
            "frame_window": {"k": 6, "resolution": [224, 224]},
            "data": {"augmentation": {"val": {}}},
        })
        transform = create_val_transform(OmegaConf.to_container(config))
        tensor = buffer.get_tensor(transform)

        assert tensor.shape == (6, 3, 224, 224)


class TestAPIUtilities:
    """Test API utility functions."""

    def test_decode_frame(self):
        """Test base64 frame decoding."""
        # Create test image
        img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        decoded = decode_frame(b64)
        assert isinstance(decoded, Image.Image)
        assert decoded.size == (224, 224)

    def test_encode_heatmap(self):
        """Test heatmap encoding."""
        heatmap = np.random.rand(14, 14).astype(np.float32)
        b64 = encode_heatmap(heatmap)

        assert isinstance(b64, str)
        # Should be valid base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_create_decision_response(self):
        """Test decision response creation."""
        decision = SentinelDecision(
            action="HARD_BLOCK",
            risk_score=0.9,
            category="destructive",
            category_conf=0.95,
            bbox=(0.1, 0.1, 0.5, 0.5),
            reasoning="High risk destructive action",
            timestamp=1234567890.0,
        )

        heatmap = np.random.rand(14, 14).astype(np.float32)
        response = create_decision_response("req_123", decision, False, heatmap)

        assert response.request_id == "req_123"
        assert response.decision == "HARD_BLOCK"
        assert response.risk_score == 0.9
        assert response.category == "destructive"
        assert response.should_proceed == False
        assert response.heatmap_b64 is not None

    def test_api_state(self):
        """Test API state tracking."""
        state = APIState()

        assert state.request_count == 0
        assert state.get_stats()["total_requests"] == 0

        state.request_count = 5
        state.record_latency(10.0)
        state.record_latency(20.0)

        stats = state.get_stats()
        assert stats["total_requests"] == 5
        assert stats["avg_latency_ms"] == 15.0


class TestTrainingPipeline:
    """Test training pipeline components."""

    def test_loss_computation(self):
        """Test SentinelLoss computation."""
        from src.training.losses import SentinelLoss, FocalLoss

        loss_fn = SentinelLoss(
            risk_weight=1.0,
            category_weight=1.0,
            localization_weight=5.0,
            giou_weight=2.0,
            use_focal_loss=True,
            focal_gamma=2.0,
            focal_alpha=0.75,
        )

        # Create dummy outputs and targets
        batch_size = 4
        outputs = {
            "risk_logits": torch.randn(batch_size, 1),
            "category_logits": torch.randn(batch_size, 5),
            "bbox": torch.rand(batch_size, 4),
            "bbox_pixel": torch.rand(batch_size, 4),
            "objectness": torch.rand(batch_size),
        }
        targets = {
            "risk_label": torch.randint(0, 2, (batch_size,)).float(),
            "category_label": torch.randint(0, 5, (batch_size,)),
            "bbox": torch.rand(batch_size, 4),
            "has_bbox": torch.randint(0, 2, (batch_size,)).float(),
        }

        loss_dict = loss_fn(outputs, targets)

        assert "total_loss" in loss_dict
        assert "risk_loss" in loss_dict
        assert "category_loss" in loss_dict
        assert "localization_loss" in loss_dict
        assert loss_dict["total_loss"].item() >= 0

    def test_focal_loss(self):
        """Test FocalLoss."""
        from src.training.losses import FocalLoss

        focal = FocalLoss(gamma=2.0, alpha=0.75)

        inputs = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()

        loss = focal(inputs, targets)
        assert loss.item() >= 0

    def test_gradient_flow_training(self):
        """Test gradients flow during training step."""
        config = OmegaConf.create({
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
            "risk_head": {"embed_dim": 384, "hidden_dim": 256, "num_categories": 5, "dropout": 0.1},
            "localization": {
                "embed_dim": 384,
                "num_anchors": 9,
                "anchor_sizes": [32, 64, 128],
                "feature_stride": 16,
            },
            "frame_window": {"k": 6, "resolution": [224, 224]},
        })

        model = create_sentinel_model(config)
        model.train()

        # Only risk head and localization head should have gradients
        for name, param in model.named_parameters():
            if "backbone" in name:
                assert not param.requires_grad
            else:
                assert param.requires_grad

        # Forward
        x = torch.randn(2, 6, 3, 224, 224)
        out = model(x)

        # Loss
        from src.training.losses import SentinelLoss
        loss_fn = SentinelLoss(
            risk_weight=1.0,
            category_weight=1.0,
            localization_weight=5.0,
            giou_weight=2.0,
        )
        targets = {
            "risk_label": torch.tensor([1.0, 0.0]),
            "category_label": torch.tensor([0, 4]),
            "bbox": torch.rand(2, 4),
            "has_bbox": torch.tensor([1.0, 0.0]),
        }
        loss_dict = loss_fn(out, targets)
        loss = loss_dict["total_loss"]

        # Backward
        loss.backward()

        # Check gradients on trainable params
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
                    break

        assert has_grad


class TestDataPipeline:
    """Test data loading and preprocessing pipeline."""

    def test_augmentation_consistency(self):
        """Test spatial transforms are consistent across frames."""
        from src.data.augmentation import ScreenshotAugmentation

        config = OmegaConf.create({
            "data": {
                "augmentation": {
                    "train": {
                        "random_crop_scale": [0.8, 1.0],
                        "random_horizontal_flip": 0.5,
                        "color_jitter": 0.2,
                        "gaussian_blur": 0.1,
                        "jpeg_quality": [75, 95],
                    },
                    "val": {},
                }
            },
            "frame_window": {"resolution": [224, 224]},
        })

        aug = ScreenshotAugmentation(
            target_resolution=(224, 224),
            random_crop=True,
            crop_scale=(0.8, 1.0),
            brightness_jitter=0.2,
            gaussian_blur_prob=0.1,
            jpeg_noise=True,
        )

        # Create 6 frames
        frames = [Image.fromarray((np.random.rand(300, 300, 3) * 255).astype(np.uint8)) for _ in range(6)]

        # Augment
        augmented = aug(frames)

        # All should be same size (C, H, W) tensors
        for f in augmented:
            assert f.shape == (3, 224, 224)

        # Convert to tensors to check consistency
        assert len(augmented) == 6

    def test_collate_function(self):
        """Test collate_frame_windows handles variable inputs."""
        batch = []
        for i in range(3):
            item = {
                "frames": torch.randn(6, 3, 224, 224),
                "risk_label": torch.tensor(i % 2, dtype=torch.long),
                "category_label": torch.tensor(0 if i % 2 else 4, dtype=torch.long),
                "bbox": torch.rand(4),
                "has_bbox": torch.tensor(float(i % 2)),
                "action": "click",
                "trajectory_idx": i,
                "action_idx": i,
            }
            batch.append(item)

        collated = collate_frame_windows(batch)

        assert collated["frames"].shape == (3, 6, 3, 224, 224)
        assert collated["risk_label"].shape == (3,)
        assert collated["bbox"].shape == (3, 4)


class TestModelExport:
    """Test model export functionality."""

    def test_onnx_export(self, tmp_path):
        """Test ONNX export works."""
        config = OmegaConf.create({
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
            "risk_head": {"embed_dim": 384, "hidden_dim": 256, "num_categories": 5, "dropout": 0.1},
            "localization": {
                "embed_dim": 384,
                "num_anchors": 9,
                "anchor_sizes": [32, 64, 128],
                "feature_stride": 16,
            },
            "frame_window": {"k": 6, "resolution": [224, 224]},
        })

        model = create_sentinel_model(config)
        model.eval()

        onnx_path = tmp_path / "model.onnx"
        model.export_onnx(str(onnx_path))

        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])