"""
Second-stage box refiner for SENTINEL-Vision.

The first-stage localization head proposes boxes at every (cell, anchor). Its
argmax selection is unreliable, so the refiner scores a small set of NMS-
filtered proposals and regresses each one with RoI-pooled spatial features.
The highest-scoring refined box is the localization output.
"""

import torch
import torch.nn as nn
from torchvision.ops import nms, roi_align


class BoxRefiner(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        hidden: int = 256,
        pool: int = 7,
        num_proposals: int = 30,
        nms_iou: float = 0.5,
        image_size: int = 224,
        feature_stride: int = 16,
        jitter_copies: int = 3,
        jitter_frac: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_proposals = num_proposals
        self.nms_iou = nms_iou
        self.image_size = image_size
        self.stride = feature_stride
        self.pool = pool
        self.jitter_copies = jitter_copies
        self.jitter_frac = jitter_frac

        self.proj = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(hidden * pool * pool, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(hidden, 1)
        self.delta = nn.Linear(hidden, 4)
        nn.init.normal_(self.score.weight, std=0.01)
        nn.init.constant_(self.score.bias, -4.595)
        nn.init.normal_(self.delta.weight, std=0.01)
        nn.init.zeros_(self.delta.bias)

    def _select_proposals(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """NMS over one sample's candidates, then top-K indices (padded by score if NMS keeps fewer)."""
        K = self.num_proposals
        keep = nms(boxes, scores, self.nms_iou)[:K]
        if keep.numel() < K:
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[keep] = False
            order = scores.argsort(descending=True)
            rest = order[mask[order]][: K - keep.numel()]
            keep = torch.cat([keep, rest])
        return keep

    def _jitter_gt(self, gt: torch.Tensor) -> torch.Tensor:
        """Random perturbations of the GT box (B,4) -> (B, jitter_copies, 4), clamped to the image."""
        B = gt.shape[0]
        size = torch.stack([gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]] * 2, dim=1).clamp(min=1.0)
        noise = (torch.rand(B, self.jitter_copies, 4, device=gt.device) * 2 - 1) * self.jitter_frac
        jit = gt[:, None, :] + noise * size[:, None, :]
        jit = jit.clamp(0, self.image_size)
        x1 = torch.min(jit[..., 0], jit[..., 2] - 1.0)
        y1 = torch.min(jit[..., 1], jit[..., 3] - 1.0)
        return torch.stack([x1, y1, jit[..., 2].clamp(min=1.0), jit[..., 3].clamp(min=1.0)], dim=-1)

    @staticmethod
    def _decode(props: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        pw = (props[..., 2] - props[..., 0]).clamp(min=1.0)
        ph = (props[..., 3] - props[..., 1]).clamp(min=1.0)
        pcx = (props[..., 0] + props[..., 2]) / 2
        pcy = (props[..., 1] + props[..., 3]) / 2
        tx, ty, tw, th = deltas.unbind(-1)
        cx = pcx + tx * pw
        cy = pcy + ty * ph
        w = pw * torch.exp(tw.clamp(-4.0, 4.0))
        h = ph * torch.exp(th.clamp(-4.0, 4.0))
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    def forward(
        self,
        spatial: torch.Tensor,
        bbox_pred_grid: torch.Tensor,
        obj_logits_grid: torch.Tensor,
        gt_pixel: torch.Tensor = None,
    ):
        """
        Args:
            spatial: (B, H, W, D) spatial features from the frame encoder
            bbox_pred_grid: (B, H, W, A, 4) first-stage decoded pixel boxes
            obj_logits_grid: (B, H, W, A) first-stage objectness logits
            gt_pixel: (B, 4) ground-truth boxes in pixels; used only in training
                to add jittered GT proposals
        """
        B, H, W, D = spatial.shape
        S = self.image_size

        cand_boxes = bbox_pred_grid.reshape(B, -1, 4).detach().clamp(0, S)
        cand_scores = torch.sigmoid(obj_logits_grid).reshape(B, -1).detach()

        props = []
        for b in range(B):
            idx = self._select_proposals(cand_boxes[b], cand_scores[b])
            props.append(cand_boxes[b, idx])
        props = torch.stack(props)  # (B, K, 4)

        if self.training and gt_pixel is not None:
            props = torch.cat([props, self._jitter_gt(gt_pixel.detach())], dim=1)

        K2 = props.shape[1]
        centred = spatial - spatial.mean(dim=(1, 2), keepdim=True)
        fmap = self.proj(centred).permute(0, 3, 1, 2)  # (B, hidden, H, W)

        batch_idx = torch.arange(B, device=props.device, dtype=props.dtype).repeat_interleave(K2)
        rois = torch.cat([batch_idx[:, None], props.reshape(-1, 4)], dim=1)
        pooled = roi_align(
            fmap, rois, output_size=self.pool,
            spatial_scale=1.0 / self.stride, sampling_ratio=2, aligned=True,
        )
        h = self.mlp(pooled.flatten(1))
        scores = self.score(h).reshape(B, K2)
        deltas = self.delta(h).reshape(B, K2, 4)

        refined = self._decode(props, deltas).clamp(0, S)
        best = scores.argmax(dim=1)
        chosen = refined[torch.arange(B, device=refined.device), best]

        return {
            "refine_proposals": props,
            "refine_scores": scores,
            "refine_boxes": refined,
            "bbox_refined_pixel": chosen,
        }
