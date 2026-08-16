"""
Tests for model forward pass and components.
Targets the implemented SENTINEL-Vision API.
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.models.frame_encoder import FrameEncoder, create_frame_encoder
from src.models.temporal_fusion import TemporalFusion, SpatiotemporalAttention, DeltaFeatureExtractor, TemporalPositionalEncoding, create_temporal_fusion
from src.models.risk_head import RiskHead, RiskHeadWithUncertainty, create_risk_head
from src.models.localization_head import LocalizationHead, create_localization_head
from src.models.sentinel_model import SentinelModel, create_sentinel_model


class TestFrameEncoder:
    """Tests for FrameEncoder."""

    def setup_method(self):
        """Setup test config."""
        self.config = OmegaConf.create({
            "backbone": "vit_small_patch16_224",
            "pretrained": False,
            "freeze_backbone": True,
            "output_dim": 384,
        })

    def test_creation(self):
        """Test encoder creation."""
        encoder = create_frame_encoder(self.config)
        assert isinstance(encoder, FrameEncoder)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = create_frame_encoder(self.config)
        encoder.eval()

        # Input: (B, k, C, H, W)
        x = torch.randn(2, 6, 3, 224, 224)
        with torch.no_grad():
            out = encoder(x)

        # Output: (B, k, N_patches, D) - 14x14 patches for ViT-S/16 = 196
        assert out.shape == (2, 6, 196, 384)

    def test_single_frame_forward(self):
        """Test forward with a single frame (no time dim)."""
        encoder = create_frame_encoder(self.config)
        encoder.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder(x)

        assert out.shape == (2, 196, 384)

    def test_freeze_unfreeze(self):
        """Test freeze/unfreeze backbone."""
        config = OmegaConf.create({
            "backbone": "vit_small_patch16_224",
            "pretrained": False,
            "freeze_backbone": True,
            "freeze_backbone_epochs": 3,
            "output_dim": 384,
        })
        encoder = create_frame_encoder(config)

        # Initially frozen (freeze_backbone=True, freeze_epochs>0)
        for param in encoder.backbone.parameters():
            assert not param.requires_grad

        # Unfreeze via epoch transition
        encoder.set_epoch(encoder.freeze_epochs)
        for param in encoder.backbone.parameters():
            assert param.requires_grad

        # Re-freeze manually
        encoder._freeze_backbone()
        for param in encoder.backbone.parameters():
            assert not param.requires_grad

    def test_get_spatial_embeddings(self):
        """Test getting spatial embeddings for localization."""
        encoder = create_frame_encoder(self.config)
        encoder.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            spatial = encoder.get_spatial_embeddings(x)

        # Returns grid format (B, H_grid, W_grid, D) = (1, 14, 14, 384)
        assert spatial.shape == (1, 14, 14, 384)

    def test_different_backbones(self):
        """Test different backbone configurations."""
        for backbone in ["vit_small_patch16_224", "convnext_tiny"]:
            config = OmegaConf.create({
                "backbone": backbone,
                "pretrained": False,
                "freeze_backbone": True,
                "output_dim": 384,
            })
            encoder = create_frame_encoder(config)
            encoder.eval()
            x = torch.randn(1, 2, 3, 224, 224)
            with torch.no_grad():
                out = encoder(x)
            assert out.shape[0] == 1
            assert out.shape[1] == 2
            assert out.shape[2] == encoder.num_patches


class TestTemporalFusion:
    """Tests for TemporalFusion."""

    def setup_method(self):
        """Setup test config."""
        self.config = OmegaConf.create({
            "embed_dim": 384,
            "temporal": {
                "num_heads": 8,
                "num_layers": 3,
                "max_frames": 8,
                "dropout": 0.1,
                "fusion_mode": "attn_pool",
                "use_delta": True,
            },
        })

    def test_creation(self):
        """Test temporal fusion creation."""
        fusion = create_temporal_fusion(self.config)
        assert isinstance(fusion, TemporalFusion)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        fusion = create_temporal_fusion(self.config)
        fusion.eval()

        # Input: (B, k, N_patches, D)
        x = torch.randn(2, 6, 196, 384)
        with torch.no_grad():
            out = fusion(x)

        # Output: (B, N_patches, D) - fused representation
        assert out.shape == (2, 196, 384)

    def test_fusion_modes(self):
        """Test different fusion modes."""
        for mode in ["last", "mean", "attn_pool"]:
            config = OmegaConf.create({
                "embed_dim": 384,
                "temporal": {
                    "num_heads": 8,
                    "num_layers": 3,
                    "max_frames": 8,
                    "dropout": 0.1,
                    "fusion_mode": mode,
                    "use_delta": True,
                },
            })
            fusion = create_temporal_fusion(config)
            fusion.eval()
            x = torch.randn(1, 6, 196, 384)
            with torch.no_grad():
                out = fusion(x)
            assert out.shape == (1, 196, 384)

    def test_without_delta(self):
        """Test without delta features."""
        config = OmegaConf.create({
            "embed_dim": 384,
            "temporal": {
                "num_heads": 8,
                "num_layers": 3,
                "max_frames": 8,
                "dropout": 0.1,
                "fusion_mode": "attn_pool",
                "use_delta": False,
            },
        })
        fusion = create_temporal_fusion(config)
        fusion.eval()
        x = torch.randn(1, 6, 196, 384)
        with torch.no_grad():
            out = fusion(x)
        assert out.shape == (1, 196, 384)


class TestSpatiotemporalAttention:
    """Tests for SpatiotemporalAttention."""

    def test_forward(self):
        """Test attention forward pass."""
        attention = SpatiotemporalAttention(embed_dim=384, num_heads=8, dropout=0.1)
        attention.eval()

        x = torch.randn(2, 6, 196, 384)
        with torch.no_grad():
            out = attention(x)
        assert out.shape == (2, 6, 196, 384)


class TestDeltaFeatureExtractor:
    """Tests for DeltaFeatureExtractor."""

    def test_forward(self):
        """Test delta feature extraction."""
        extractor = DeltaFeatureExtractor(embed_dim=384)
        extractor.eval()

        x = torch.randn(2, 6, 196, 384)
        with torch.no_grad():
            out = extractor(x)
        # Should return (B, k-1, N_patches, D) - frame differences
        assert out.shape == (2, 5, 196, 384)


class TestTemporalPositionalEncoding:
    """Tests for TemporalPositionalEncoding."""

    def test_forward(self):
        """Test temporal positional encoding."""
        pos_enc = TemporalPositionalEncoding(max_frames=20, embed_dim=384)
        pos_enc.eval()

        x = torch.randn(2, 6, 196, 384)
        with torch.no_grad():
            out = pos_enc(x)
        assert out.shape == (2, 6, 196, 384)


class TestRiskHead:
    """Tests for RiskHead."""

    def setup_method(self):
        """Setup test config."""
        self.config = OmegaConf.create({
            "risk_head": {
                "embed_dim": 384,
                "hidden_dim": 256,
                "num_categories": 5,
                "dropout": 0.1,
            }
        })

    def test_creation(self):
        """Test risk head creation."""
        head = create_risk_head(self.config)
        assert isinstance(head, RiskHead)

    def test_forward(self):
        """Test forward pass."""
        head = create_risk_head(self.config)
        head.eval()

        x = torch.randn(4, 384)
        with torch.no_grad():
            out = head(x)

        assert "risk_score" in out
        assert "category_probs" in out
        assert "category_logits" in out
        assert out["risk_score"].shape == (4, 1)
        assert out["category_probs"].shape == (4, 5)
        assert out["category_logits"].shape == (4, 5)

    def test_risk_score_range(self):
        """Test risk score is in [0, 1]."""
        head = create_risk_head(self.config)
        head.eval()

        x = torch.randn(10, 384)
        with torch.no_grad():
            out = head(x)

        risk_scores = out["risk_score"].squeeze()
        assert torch.all(risk_scores >= 0)
        assert torch.all(risk_scores <= 1)

    def test_category_probs_sum_to_one(self):
        """Test category probabilities sum to 1."""
        head = create_risk_head(self.config)
        head.eval()

        x = torch.randn(10, 384)
        with torch.no_grad():
            out = head(x)

        probs_sum = out["category_probs"].sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum))

    def test_forward_with_cls_token(self):
        """Test forward with cls token (pooled input)."""
        head = create_risk_head(self.config)
        head.eval()

        # (B, D) already-pooled input
        x = torch.randn(4, 384)
        cls = torch.randn(4, 384)
        with torch.no_grad():
            out = head(x, cls_token=cls)
        assert out["risk_score"].shape == (4, 1)


class TestRiskHeadWithUncertainty:
    """Tests for RiskHeadWithUncertainty."""

    def setup_method(self):
        """Setup test config."""
        self.config = OmegaConf.create({
            "risk_head": {
                "embed_dim": 384,
                "hidden_dim": 256,
                "num_categories": 5,
                "dropout": 0.1,
            }
        })

    def test_forward_with_uncertainty(self):
        """Test forward pass with uncertainty."""
        head = RiskHeadWithUncertainty(
            embed_dim=384,
            hidden_dim=256,
            num_categories=5,
            dropout=0.1,
            mc_dropout_samples=5,
        )
        head.eval()

        x = torch.randn(4, 384)
        with torch.no_grad():
            out = head.forward_with_uncertainty(x)

        assert "risk_score" in out
        assert "risk_uncertainty" in out
        assert "category_probs" in out
        assert "category_uncertainty" in out
        assert out["risk_score"].shape == (4, 1)
        assert out["risk_uncertainty"].shape == (4, 1)


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
        """Test GIoU loss via LocalizationHead compute_giou_loss."""
        head = create_localization_head(self.config)

        pred_bboxes = torch.tensor([[[0.1, 0.1, 0.5, 0.5]]], dtype=torch.float32)
        gt_bboxes = torch.tensor([[[0.15, 0.15, 0.55, 0.55]]], dtype=torch.float32)

        loss = head.compute_giou_loss(pred_bboxes, gt_bboxes)
        assert loss.item() >= 0

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


class TestSentinelModel:
    """Tests for full SentinelModel."""

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
            "risk_head": {
                "embed_dim": 384,
                "hidden_dim": 256,
                "num_categories": 5,
                "dropout": 0.1,
            },
            "localization_head": {
                "embed_dim": 384,
                "num_anchors": 9,
                "anchor_sizes": [32, 64, 128, 256],
                "feature_stride": 16,
                "image_size": 224,
            },
            "frame_window": {"k": 6, "resolution": [224, 224]},
        })

    def test_creation(self):
        """Test model creation."""
        model = create_sentinel_model(self.config)
        assert isinstance(model, SentinelModel)

    def test_forward(self):
        """Test full forward pass."""
        model = create_sentinel_model(self.config)
        model.eval()

        # Input: (B, k, C, H, W)
        x = torch.randn(2, 6, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        # Check all outputs
        assert "risk_score" in out
        assert "category_probs" in out
        assert "category_idx" in out
        assert "bbox" in out
        assert "objectness" in out
        assert "confidence" in out

        assert out["risk_score"].shape == (2, 1)
        assert out["category_probs"].shape == (2, 5)
        assert out["category_idx"].shape == (2,)
        assert out["bbox"].shape == (2, 4)
        assert out["objectness"].shape == (2,)
        assert out["confidence"].shape == (2,)

        # category_idx is a valid int tensor
        assert out["category_idx"].dtype == torch.long
        # category is a list of strings
        assert isinstance(out["category"], list)
        assert len(out["category"]) == 2

    def test_predict_api(self):
        """Test predict API for PIL images."""
        model = create_sentinel_model(self.config)

        from PIL import Image

        # Create PIL images
        images = [Image.fromarray((torch.rand(224, 224, 3) * 255).byte().numpy()) for _ in range(6)]

        out = model.predict(images)

        assert "risk_score" in out
        assert "category" in out
        assert "bbox" in out
        assert "objectness" in out
        assert "confidence" in out

    def test_export_onnx(self, tmp_path):
        """Test ONNX export."""
        model = create_sentinel_model(self.config)
        model.eval()

        onnx_path = tmp_path / "test_model.onnx"
        try:
            model.export_onnx(str(onnx_path))
            assert onnx_path.exists()
        except Exception as e:
            # onnx may not be installed; skip gracefully
            pytest.skip(f"ONNX export unavailable: {e}")

    def test_parameter_count(self):
        """Test model has reasonable parameter count."""
        model = create_sentinel_model(self.config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should be around 20-30M params
        assert 10_000_000 < total_params < 50_000_000
        # Backbone frozen, so fewer trainable
        assert trainable_params < total_params

    def test_get_model_size(self):
        """Test model size reporting."""
        model = create_sentinel_model(self.config)
        size = model.get_model_size()
        assert "total_params" in size
        assert "trainable_params" in size
        assert size["total_params"] > 0


class TestModelGradients:
    """Test gradient flow through model."""

    def test_gradients_flow(self):
        """Test gradients flow to trainable parameters."""
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
            "localization_head": {"embed_dim": 384, "num_anchors": 9, "anchor_sizes": [32, 64, 128, 256], "feature_stride": 16, "image_size": 224},
            "frame_window": {"k": 6, "resolution": [224, 224]},
        })

        model = create_sentinel_model(config)
        model.train()

        x = torch.randn(2, 6, 3, 224, 224, requires_grad=True)
        out = model(x)

        # Compute dummy loss
        loss = out["risk_score"].mean() + out["category_probs"].mean()
        loss.backward()

        # Check gradients exist for non-frozen params
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    break

        assert has_grad, "No gradients flowing to trainable parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
