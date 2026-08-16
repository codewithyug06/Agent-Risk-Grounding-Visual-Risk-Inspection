"""
Tests for localization head and bounding box utilities.
Targets the implemented SENTINEL-Vision API.
"""

import pytest
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.models.localization_head import LocalizationHead, create_localization_head
from src.data.heatmap_labels import (
    extract_bbox_from_dom,
    generate_heatmap_gt,
    generate_multiscale_heatmaps,
    validate_bbox,
    compute_iou,
    compute_giou,
    HeatmapLoss,
)


class TestBboxUtilities:
    """Tests for bounding box utilities in heatmap_labels."""

    def test_xywh_to_xyxy(self):
        """Test conversion from xywh to xyxy."""
        from src.eval.metrics import _xywh_to_xyxy

        boxes_xywh = np.array([[10, 20, 30, 40], [0, 0, 100, 100]])
        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

        expected = np.array([[10, 20, 40, 60], [0, 0, 100, 100]])
        assert np.allclose(boxes_xyxy, expected)

    def test_bbox_iou(self):
        """Test IoU computation."""
        from src.eval.metrics import _bbox_iou

        # Perfect overlap
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([0, 0, 10, 10])
        assert _bbox_iou(box1, box2) == 1.0

        # No overlap
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])
        assert _bbox_iou(box1, box2) == 0.0

        # Partial overlap
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])
        iou = _bbox_iou(box1, box2)
        # Intersection: 5x5=25, Union: 100+100-25=175, IoU=25/175=1/7
        assert abs(iou - 1/7) < 1e-6

    def test_validate_bbox(self):
        """Test bbox validation."""
        # Valid bbox
        assert validate_bbox([0.1, 0.1, 0.5, 0.5]) == True
        assert validate_bbox([0.0, 0.0, 0.9, 0.9]) == True

        # Invalid bboxes
        assert validate_bbox([0.5, 0.5, 0.1, 0.1]) == False  # x2 < x1
        assert validate_bbox([-0.1, 0.1, 0.5, 0.5]) == False  # negative
        assert validate_bbox([0.1, 0.1, 1.1, 0.5]) == False  # > 1
        assert validate_bbox([0.1, 0.1, 0.5]) == False  # wrong length
        # Too small
        assert validate_bbox([0.0, 0.0, 0.005, 0.005]) == False
        # Too large
        assert validate_bbox([0.0, 0.0, 0.99, 0.99]) == False

    def test_compute_iou(self):
        """Test IoU computation on lists."""
        # Same boxes
        box1 = [0.1, 0.1, 0.5, 0.5]
        box2 = [0.1, 0.1, 0.5, 0.5]
        iou = compute_iou(box1, box2)
        assert iou == 1.0

        # No overlap
        box1 = [0.0, 0.0, 0.3, 0.3]
        box2 = [0.7, 0.7, 1.0, 1.0]
        iou = compute_iou(box1, box2)
        assert iou == 0.0

        # Partial overlap
        box1 = [0.0, 0.0, 0.5, 0.5]
        box2 = [0.25, 0.25, 0.75, 0.75]
        iou = compute_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_compute_giou(self):
        """Test GIoU computation on lists."""
        # Same boxes -> GIoU = 1.0
        box1 = [0.1, 0.1, 0.5, 0.5]
        box2 = [0.1, 0.1, 0.5, 0.5]
        giou = compute_giou(box1, box2)
        assert giou == 1.0

        # No overlap -> GIoU negative
        box1 = [0.0, 0.0, 0.3, 0.3]
        box2 = [0.7, 0.7, 1.0, 1.0]
        giou = compute_giou(box1, box2)
        assert giou < 0

    def test_generate_heatmap_gt(self):
        """Test heatmap ground truth generation."""
        # Signature: generate_heatmap_gt(image_size, bbox, sigma_factor)
        bbox = [0.25, 0.25, 0.75, 0.75]  # Center box
        heatmap = generate_heatmap_gt((14, 14), bbox)

        assert heatmap.shape == (14, 14)
        assert heatmap.max() == 1.0
        assert heatmap.min() >= 0.0

        # Center should be hot
        assert heatmap[7, 7] > 0.5

    def test_generate_multiscale_heatmaps(self):
        """Test multiscale heatmap generation."""
        bbox = [0.25, 0.25, 0.75, 0.75]
        heatmaps = generate_multiscale_heatmaps((14, 14), bbox, scales=[14, 7])

        assert 14 in heatmaps
        assert 7 in heatmaps
        assert heatmaps[14].shape == (14, 14)
        assert heatmaps[7].shape == (7, 7)

    def test_heatmap_loss(self):
        """Test heatmap loss computation."""
        loss_fn = HeatmapLoss(scales=[14, 7], weights=[1.0, 0.5])

        pred_heatmaps = {
            14: torch.rand(2, 14, 14),
            7: torch.rand(2, 7, 7),
        }
        target_bbox = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]])

        loss = loss_fn(pred_heatmaps, target_bbox)
        assert loss.item() >= 0


class TestLocalizationHead:
    """Tests for LocalizationHead."""

    def setup_method(self):
        """Setup test config."""
        self.config = OmegaConf.create({
            "embed_dim": 384,
            "image_size": 224,
            "localization": {
                "num_anchors": 9,
                "anchor_sizes": [32, 64, 128, 256],
                "feature_stride": 16,
            },
        })

    def test_creation(self):
        """Test localization head creation."""
        head = create_localization_head(self.config)
        assert isinstance(head, LocalizationHead)

    def test_anchor_generation(self):
        """Test anchor generation (registered buffer)."""
        head = create_localization_head(self.config)
        # anchors: (H*W*A, 4)
        assert head.anchors.shape[1] == 4
        expected = head.fm_size * head.fm_size * head.num_anchors  # 14*14*9
        assert head.anchors.shape[0] == expected

    def test_forward_shape(self):
        """Test forward pass shapes."""
        head = create_localization_head(self.config)
        head.eval()

        # Spatial features: (B, N_patches, D)
        x = torch.randn(2, 196, 384)
        with torch.no_grad():
            out = head(x)

        assert "objectness" in out
        assert "bbox" in out
        assert "bbox_pixel" in out
        assert "objectness_max" in out
        assert "bbox_pred" in out
        assert "anchor_logits" in out
        # objectness: (B, H, W, A)
        assert out["objectness"].shape == (2, 14, 14, 9)
        # bbox: (B, 4) normalized
        assert out["bbox"].shape == (2, 4)
        # bbox_pixel: (B, 4) pixel coords
        assert out["bbox_pixel"].shape == (2, 4)
        # objectness_max: (B,)
        assert out["objectness_max"].shape == (2,)

    def test_forward_spatial_grid(self):
        """Test forward accepts (B, H, W, D) grid input."""
        head = create_localization_head(self.config)
        head.eval()

        x = torch.randn(1, 14, 14, 384)
        with torch.no_grad():
            out = head(x)
        assert out["bbox"].shape == (1, 4)

    def test_giou_loss_method(self):
        """Test GIoU loss via LocalizationHead.compute_giou_loss."""
        head = create_localization_head(self.config)

        pred_bboxes = torch.tensor([[[0.1, 0.1, 0.5, 0.5]]], dtype=torch.float32)
        gt_bboxes = torch.tensor([[[0.15, 0.15, 0.55, 0.55]]], dtype=torch.float32)

        loss = head.compute_giou_loss(pred_bboxes, gt_bboxes)
        assert loss.item() >= 0

    def test_decode_predictions(self):
        """Test prediction decoding via decode_predictions."""
        head = create_localization_head(self.config)
        head.eval()

        x = torch.randn(1, 196, 384)
        with torch.no_grad():
            out = head(x)

        decoded = head.decode_predictions(
            out["objectness"], out["bbox_pred"],
            conf_threshold=0.3, nms_threshold=0.5, top_k=10
        )

        # Returns one dict per batch item, each with "bbox" and "scores"
        assert isinstance(decoded, list)
        assert "bbox" in decoded[0]
        assert "scores" in decoded[0]
        assert isinstance(decoded[0]["bbox"], torch.Tensor)
        assert isinstance(decoded[0]["scores"], torch.Tensor)

    def test_grad_cam(self):
        """Test Grad-CAM generation."""
        head = create_localization_head(self.config)

        from src.models.sentinel_model import create_sentinel_model
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
                "anchor_sizes": [32, 64, 128, 256],
                "feature_stride": 16,
            },
            "frame_window": {"k": 6, "resolution": [224, 224]},
        })
        model = create_sentinel_model(config)
        model.eval()

        input_frames = torch.randn(1, 6, 3, 224, 224)

        # Use the last temporal fusion attention block as target layer
        # (same as SentinelModel.generate_heatmap uses) - this outputs a tensor
        target_layer = model.temporal_fusion.blocks[-1].attn

        assert target_layer is not None, "Could not find target layer in model"

        cam = head.generate_gradcam(
            model, input_frames, target_layer=target_layer, target_category=0
        )
        assert cam.shape[0] == 224
        assert cam.shape[1] == 224

    def test_multiple_anchor_configs(self):
        """Test with different anchor configurations."""
        for num_anchors in [5, 9, 15]:
            config = OmegaConf.create({
                "embed_dim": 384,
                "image_size": 224,
                "localization": {
                    "num_anchors": num_anchors,
                    "anchor_sizes": [32, 64, 128, 256],
                    "feature_stride": 16,
                },
            })
            head = create_localization_head(config)
            head.eval()
            x = torch.randn(1, 196, 384)
            with torch.no_grad():
                out = head(x)
            assert out["objectness"].shape[3] == num_anchors


class TestLocalizationIntegration:
    """Integration tests for localization with full model."""

    def test_sentinel_localization_output(self):
        """Test SENTINEL model produces localization outputs."""
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
                "anchor_sizes": [32, 64, 128, 256],
                "feature_stride": 16,
            },
            "frame_window": {"k": 6, "resolution": [224, 224]},
        })

        from src.models.sentinel_model import create_sentinel_model
        model = create_sentinel_model(config)
        model.eval()

        x = torch.randn(1, 6, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        assert "bbox" in out
        assert "objectness" in out
        assert out["bbox"].shape == (1, 4)
        assert out["objectness"].shape == (1,)


class TestLocalizationMetrics:
    """Tests for localization evaluation metrics."""

    def test_localization_iou_metric(self):
        """Test compute_localization_iou metric."""
        from src.eval.metrics import compute_localization_iou

        pred = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]])
        gt = np.array([[0.15, 0.15, 0.55, 0.55], [0.25, 0.25, 0.65, 0.65]])

        metrics = compute_localization_iou(pred, gt)

        assert "mean_iou" in metrics
        assert "median_iou" in metrics
        assert "iou_at_05" in metrics
        assert "iou_at_075" in metrics
        assert 0 <= metrics["mean_iou"] <= 1
        assert 0 <= metrics["iou_at_05"] <= 1

    def test_empty_inputs(self):
        """Test metrics with empty inputs."""
        from src.eval.metrics import compute_localization_iou

        pred = np.array([]).reshape(0, 4)
        gt = np.array([]).reshape(0, 4)

        metrics = compute_localization_iou(pred, gt)
        assert metrics["mean_iou"] == 0.0
        assert metrics["iou_at_05"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])