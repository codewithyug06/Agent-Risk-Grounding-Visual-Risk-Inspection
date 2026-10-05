import torch
from omegaconf import OmegaConf

from src.models.box_refiner import BoxRefiner
from src.models.sentinel_model import create_sentinel_model
from src.training.losses import SentinelLoss
from src.utils.config import load_config


def _grid(B=2):
    spatial = torch.randn(B, 14, 14, 384)
    xy = torch.rand(B, 14, 14, 32, 2) * 200
    wh = torch.rand(B, 14, 14, 32, 2) * 60 + 2
    bbox_pred = torch.cat([xy, xy + wh], dim=-1)
    obj = torch.randn(B, 14, 14, 32)
    return spatial, bbox_pred, obj


def test_train_adds_gt_jitter_eval_does_not():
    r = BoxRefiner(num_proposals=30, jitter_copies=3)
    spatial, bp, obj = _grid()
    gt = torch.tensor([[60.0, 100.0, 93.0, 103.0], [120.0, 40.0, 150.0, 50.0]])
    r.train()
    out = r(spatial, bp, obj, gt)
    assert out["refine_proposals"].shape == (2, 33, 4)
    r.eval()
    out = r(spatial, bp, obj, None)
    assert out["refine_proposals"].shape == (2, 30, 4)
    assert out["bbox_refined_pixel"].shape == (2, 4)


def test_zero_deltas_decode_to_proposal():
    props = torch.tensor([[[10.0, 20.0, 50.0, 30.0]]])
    deltas = torch.zeros(1, 1, 4)
    assert torch.allclose(BoxRefiner._decode(props, deltas), props, atol=1e-5)


def test_jitter_is_valid_box():
    r = BoxRefiner()
    gt = torch.tensor([[60.0, 100.0, 93.0, 103.0], [5.0, 5.0, 40.0, 20.0]])
    j = r._jitter_gt(gt)
    assert (j[..., 2] > j[..., 0]).all() and (j[..., 3] > j[..., 1]).all()
    assert (j >= 0).all() and (j <= 224).all()


def test_refiner_gradients_flow_to_regression_head():
    r = BoxRefiner()
    spatial, bp, obj = _grid()
    gt = torch.tensor([[60.0, 100.0, 93.0, 103.0], [120.0, 40.0, 150.0, 50.0]])
    r.train()
    out = r(spatial, bp, obj, gt)
    out["refine_boxes"].sum().backward()
    assert r.delta.weight.grad.abs().sum().item() > 0


def test_refiner_loss_zero_without_harmful_samples():
    loss = SentinelLoss()
    B = 2
    preds = {
        "risk_logits": torch.randn(B, 1),
        "category_logits": torch.randn(B, 5),
        "bbox": torch.rand(B, 4),
        "refine_proposals": torch.rand(B, 10, 4) * 100,
        "refine_scores": torch.randn(B, 10),
        "refine_boxes": torch.rand(B, 10, 4) * 100,
        "image_size": 224,
    }
    tg = {
        "risk_label": torch.zeros(B),
        "category_label": torch.tensor([4, 4]),
        "bbox": torch.rand(B, 4),
        "has_bbox": torch.zeros(B),
    }
    d = loss(preds, tg)
    assert d["refiner_loss"].item() == 0.0


def test_model_integration_train_and_eval_keys():
    cfg = OmegaConf.merge(load_config("model_small.yaml"), load_config("data.yaml"),
                          load_config("curriculum.yaml")["stage_c"])
    model = create_sentinel_model(cfg).train()
    x = torch.randn(2, 6, 3, 224, 224)
    gt = torch.tensor([[60.0, 100.0, 93.0, 103.0], [120.0, 40.0, 150.0, 50.0]])
    out = model(x, gt_bbox_pixel=gt)
    assert "refine_scores" in out and "bbox_refined" in out
    assert out["bbox_refined"].shape == (2, 4)
    assert out["bbox_refined"].min() >= 0 and out["bbox_refined"].max() <= 1
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out["refine_scores"].shape == (2, 30)
