"""
Center-peaking heatmap head for GUI grounding (V2P-style).

One logit per cell of the spatial grid. Training target is a Gaussian centered
on the ground-truth click point (box center), sigma from element size. The
predicted point is the peak cell's center in pixels.
"""

import torch
import torch.nn as nn


class HeatmapHead(nn.Module):
    def __init__(self, embed_dim: int = 384, hidden: int = 192, feature_stride: int = 16,
                 image_size: int = 224, dropout: float = 0.1):
        super().__init__()
        self.stride = feature_stride
        self.image_size = image_size
        self.grid = image_size // feature_stride
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.constant_(self.net[-1].bias[0], -4.595)

    def forward(self, spatial: torch.Tensor):
        """
        spatial: (B, H, W, D). Returns logits (B, H, W) and sub-cell offsets (B, H, W, 2)
        in units of one cell (dx, dy from the cell center).
        """
        centred = spatial - spatial.mean(dim=(1, 2), keepdim=True)
        out = self.net(centred)
        return out[..., 0], out[..., 1:3]

    @torch.no_grad()
    def decode_point(self, logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Peak cell plus its sub-cell offset -> pixel point (B, 2)."""
        B, H, W = logits.shape
        idx = logits.reshape(B, -1).argmax(dim=1)
        row = torch.div(idx, W, rounding_mode="floor")
        col = idx % W
        off = offsets.reshape(B, -1, 2)[torch.arange(B, device=logits.device), idx]
        x = (col.float() + 0.5 + off[:, 0]) * self.stride
        y = (row.float() + 0.5 + off[:, 1]) * self.stride
        return torch.stack([x, y], dim=1)

    def offset_target(self, box_pixel: torch.Tensor):
        """Index of the cell containing the box center, and the sub-cell offset of the center."""
        # Nearest cell center, matching gaussian_target and decode_point.
        cx = (box_pixel[:, 0] + box_pixel[:, 2]) / 2 / self.stride - 0.5
        cy = (box_pixel[:, 1] + box_pixel[:, 3]) / 2 / self.stride - 0.5
        col = torch.round(cx).clamp(0, self.grid - 1)
        row = torch.round(cy).clamp(0, self.grid - 1)
        off = torch.stack([cx - col, cy - row], dim=1)
        return row.long(), col.long(), off

    def gaussian_target(self, box_pixel: torch.Tensor) -> torch.Tensor:
        """
        box_pixel: (B, 4) xyxy in pixels -> target heatmap (B, H, W) in [0, 1].
        Peak 1 at the box center cell; sigma (in cells) scales with box size.
        """
        cx = (box_pixel[:, 0] + box_pixel[:, 2]) / 2 / self.stride - 0.5
        cy = (box_pixel[:, 1] + box_pixel[:, 3]) / 2 / self.stride - 0.5
        w = (box_pixel[:, 2] - box_pixel[:, 0]).clamp(min=1.0) / self.stride
        h = (box_pixel[:, 3] - box_pixel[:, 1]).clamp(min=1.0) / self.stride
        sigma = torch.clamp(0.15 * torch.sqrt(w * h), min=0.5)
        ys = torch.arange(self.grid, device=box_pixel.device, dtype=box_pixel.dtype)
        xs = torch.arange(self.grid, device=box_pixel.device, dtype=box_pixel.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        d2 = (gx[None] - cx[:, None, None]) ** 2 + (gy[None] - cy[:, None, None]) ** 2
        return torch.exp(-d2 / (2 * sigma[:, None, None] ** 2))


def heatmap_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Listwise loss over cells: softmax over all cells, cross-entropy against the
    normalized Gaussian target. Stable peak selection (the sigmoid focal variant
    flips between neighboring cells on the fixed-scene check).
    logits, target: (B, H, W). mask: (B,) float, samples with a ground-truth point.
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0
    B = logits.shape[0]
    flat_logits = logits.reshape(B, -1)[mask > 0.5]
    t = target.reshape(B, -1)[mask > 0.5]
    t = t / t.sum(dim=1, keepdim=True).clamp(min=1e-6)
    logp = torch.log_softmax(flat_logits, dim=1)
    return torch.nan_to_num(-(t * logp).sum(dim=1).mean(), nan=0.0)


def offset_loss(offsets: torch.Tensor, row: torch.Tensor, col: torch.Tensor,
                target_off: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 on the sub-cell offset at the ground-truth center cell, for masked samples."""
    if mask.sum() == 0:
        return offsets.sum() * 0.0
    idx = torch.nonzero(mask > 0.5).flatten()
    pred = offsets[idx, row[idx], col[idx]]
    return (pred - target_off[idx]).abs().mean()
