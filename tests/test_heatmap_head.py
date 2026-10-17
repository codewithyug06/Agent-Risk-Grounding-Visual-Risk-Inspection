import torch

from src.models.heatmap_head import HeatmapHead, heatmap_loss, offset_loss


def _boxes():
    return torch.tensor([[60.0, 100.0, 93.0, 103.0], [120.0, 40.0, 150.0, 50.0],
                         [10.0, 150.0, 80.0, 158.0], [100.0, 20.0, 200.0, 30.0]])


def test_target_peak_is_nearest_cell_to_center():
    h = HeatmapHead()
    box = _boxes()
    t = h.gaussian_target(box)
    row, col, _ = h.offset_target(box)
    for b in range(box.shape[0]):
        assert int(t[b].reshape(-1).argmax()) == int(row[b]) * h.grid + int(col[b])


def test_decode_recovers_center_exactly_with_true_offsets():
    h = HeatmapHead()
    box = _boxes()
    row, col, off = h.offset_target(box)
    logits = h.gaussian_target(box)
    offsets = torch.zeros(box.shape[0], h.grid, h.grid, 2)
    offsets[torch.arange(4), row, col] = off
    pts = h.decode_point(logits, offsets)
    centers = (box[:, :2] + box[:, 2:]) / 2
    assert torch.allclose(pts, centers, atol=1e-3)


def test_heatmap_loss_zero_mask_is_zero():
    logits = torch.randn(2, 14, 14)
    target = torch.zeros(2, 14, 14)
    assert heatmap_loss(logits, target, torch.zeros(2)).item() == 0.0


def test_offset_loss_only_on_masked_samples():
    offsets = torch.zeros(2, 14, 14, 2)
    row = torch.tensor([1, 2])
    col = torch.tensor([3, 4])
    toff = torch.tensor([[0.2, 0.0], [5.0, 5.0]])
    loss = offset_loss(offsets, row, col, toff, torch.tensor([1.0, 0.0]))
    assert abs(loss.item() - 0.1) < 1e-6


def test_fixed_scene_point_in_box_reaches_full_accuracy():
    torch.manual_seed(0)
    h = HeatmapHead(dropout=0.0)
    box = _boxes()
    t = h.gaussian_target(box)
    row, col, toff = h.offset_target(box)
    spatial = torch.randn(4, 14, 14, 384) + torch.randn(1, 1, 1, 384) * 3
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3)
    mask = torch.ones(4)
    for _ in range(300):
        logits, off = h(spatial)
        loss = heatmap_loss(logits, t, mask) + offset_loss(off, row, col, toff, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    h.eval()
    with torch.no_grad():
        logits, off = h(spatial)
        p = h.decode_point(logits, off)
    inside = ((p[:, 0] >= box[:, 0]) & (p[:, 0] <= box[:, 2]) &
              (p[:, 1] >= box[:, 1]) & (p[:, 1] <= box[:, 3])).float().mean().item()
    assert inside == 1.0
