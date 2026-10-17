import pytest
import torch

from src.models.clip_backbone import ClipFrozenBackbone, ClipLinearRiskHead


@pytest.fixture(scope="module")
def backbone():
    return ClipFrozenBackbone(pretrained="openai", temporal_pool="mean")


def test_backbone_params_frozen(backbone):
    for p in backbone.visual.parameters():
        assert not p.requires_grad


def test_backbone_output_shape(backbone):
    B, k = 2, 3
    frame_window = torch.rand(B, k, 3, 224, 224)
    out = backbone(frame_window)
    assert out.shape == (B, backbone.embed_dim)


def test_backbone_forward_no_grad_accumulation(backbone):
    frame_window = torch.rand(1, 2, 3, 224, 224)
    out = backbone(frame_window)
    assert out.requires_grad is False


def test_attn_temporal_pool_shape():
    backbone = ClipFrozenBackbone(pretrained="openai", temporal_pool="attn")
    B, k = 2, 4
    frame_window = torch.rand(B, k, 3, 224, 224)
    out = backbone(frame_window)
    assert out.shape == (B, backbone.embed_dim)


def test_risk_head_linear_probe_shapes():
    embed_dim = 512
    head = ClipLinearRiskHead(embed_dim=embed_dim, num_categories=5, num_layers=0)
    pooled = torch.randn(4, embed_dim)
    out = head(pooled)
    assert out["risk_logits"].shape == (4, 1)
    assert out["risk_score"].shape == (4, 1)
    assert out["category_logits"].shape == (4, 5)
    assert out["category_probs"].shape == (4, 5)
    assert torch.all((out["risk_score"] >= 0) & (out["risk_score"] <= 1))


def test_risk_head_mlp_shapes():
    embed_dim = 512
    head = ClipLinearRiskHead(embed_dim=embed_dim, hidden_dim=128, num_categories=5, num_layers=2)
    pooled = torch.randn(4, embed_dim)
    out = head(pooled)
    assert out["risk_logits"].shape == (4, 1)
    assert out["category_logits"].shape == (4, 5)
