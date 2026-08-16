"""
Loss functions for SENTINEL-Vision.
Multi-task loss: weighted BCE + 5-class CE + GIoU.
Harmful class upweighted for imbalanced dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for extreme class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (any shape)
            targets: Ground truth labels (same shape as inputs), 0 or 1
        """
        # inputs are logits, apply sigmoid
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        # BCE with focal weight
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy with class weights.
    """

    def __init__(
        self,
        pos_weight: float = 5.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, 1) or (B,) logits
            targets: (B, 1) or (B,) labels 0/1
        """
        loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets.float(),
            pos_weight=torch.tensor(self.pos_weight, device=inputs.device),
            reduction=self.reduction,
        )
        return loss


class SentinelLoss(nn.Module):
    """
    Multi-task loss for SENTINEL-Vision.

    Components:
    1. risk_loss: Weighted BCE (harmful class upweighted) or Focal Loss
    2. category_loss: Cross-entropy with class weights
    3. localization_loss: GIoU + L1 on bbox (only for harmful samples)

    Total loss = risk_weight * risk_loss + category_weight * category_loss + localization_weight * loc_loss
    """

    def __init__(
        self,
        risk_weight: float = 2.0,
        category_weight: float = 1.0,
        localization_weight: float = 1.5,
        harmful_class_weight: float = 5.0,
        category_class_weights: Optional[list] = None,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        giou_weight: float = 1.0,
        l1_weight: float = 0.5,
    ):
        super().__init__()

        self.risk_weight = risk_weight
        self.category_weight = category_weight
        self.localization_weight = localization_weight
        self.harmful_class_weight = harmful_class_weight
        self.use_focal_loss = use_focal_loss
        self.giou_weight = giou_weight
        self.l1_weight = l1_weight

        # Risk loss
        if use_focal_loss:
            self.risk_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.risk_loss_fn = WeightedBCELoss(pos_weight=harmful_class_weight)

        # Category loss with class weights
        if category_class_weights is None:
            # Default: upweight harmful categories, downweight benign
            # [destructive, financial, privacy, irreversible_external, benign]
            category_class_weights = [3.0, 3.0, 3.0, 3.0, 1.0]

        self.register_buffer(
            "category_weights",
            torch.tensor(category_class_weights, dtype=torch.float32)
        )

        logger.info(f"SentinelLoss initialized: risk_w={risk_weight}, cat_w={category_weight}, "
                    f"loc_w={localization_weight}, harmful_w={harmful_class_weight}, "
                    f"use_focal={use_focal_loss}")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dict with keys:
                - risk_logits: (B, 1)
                - category_logits: (B, 5)
                - bbox: (B, 4) normalized
                - bbox_pixel: (B, 4) pixel coords
                - objectness: (B,)
            targets: Dict with keys:
                - risk_label: (B,) 0/1
                - category_label: (B,) 0-4
                - bbox: (B, 4) normalized
                - has_bbox: (B,) 0/1

        Returns:
            Dict with individual losses and total
        """
        B = predictions["risk_logits"].shape[0]
        device = predictions["risk_logits"].device

        # 1. Risk loss (binary)
        risk_logits = predictions["risk_logits"].squeeze(-1)  # (B,)
        risk_labels = targets["risk_label"].float()  # (B,)

        risk_loss = self.risk_loss_fn(risk_logits, risk_labels)

        # 2. Category loss (5-class CE with weights)
        category_logits = predictions["category_logits"]  # (B, 5)
        category_labels = targets["category_label"]  # (B,)

        category_loss = F.cross_entropy(
            category_logits,
            category_labels,
            weight=self.category_weights.to(device),
        )

        # 3. Localization loss (only for harmful samples with bbox)
        has_bbox = targets.get("has_bbox", torch.zeros(B, device=device))
        harmful_mask = (targets["risk_label"] == 1) & (has_bbox > 0.5)

        if harmful_mask.any():
            pred_bbox = predictions["bbox"][harmful_mask]  # (N_harm, 4)
            gt_bbox = targets["bbox"][harmful_mask]  # (N_harm, 4)

            # GIoU loss
            giou_loss = self._compute_giou_loss(pred_bbox, gt_bbox)

            # L1 loss
            l1_loss = F.l1_loss(pred_bbox, gt_bbox, reduction="mean")

            loc_loss = self.giou_weight * giou_loss + self.l1_weight * l1_loss
        else:
            loc_loss = torch.tensor(0.0, device=device)
            giou_loss = torch.tensor(0.0, device=device)
            l1_loss = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = (
            self.risk_weight * risk_loss +
            self.category_weight * category_loss +
            self.localization_weight * loc_loss
        )

        return {
            "total_loss": total_loss,
            "risk_loss": risk_loss,
            "category_loss": category_loss,
            "localization_loss": loc_loss,
            "giou_loss": giou_loss,
            "l1_loss": l1_loss,
        }

    def _compute_giou_loss(
        self,
        pred_bbox: torch.Tensor,
        gt_bbox: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GIoU loss.
        Args:
            pred_bbox: (N, 4) predicted bboxes [x1, y1, x2, y2] normalized
            gt_bbox: (N, 4) target bboxes
        """
        # Enforce valid box ordering (x2 >= x1, y2 >= y1). Early in training the
        # regression head can predict degenerate boxes; reordering prevents the
        # area/union terms from going negative and flipping the loss sign.
        pred_x1, pred_x2 = torch.min(pred_bbox[:, [0, 2]], dim=1).values, torch.max(pred_bbox[:, [0, 2]], dim=1).values
        pred_y1, pred_y2 = torch.min(pred_bbox[:, [1, 3]], dim=1).values, torch.max(pred_bbox[:, [1, 3]], dim=1).values
        pred_bbox = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)

        gt_x1, gt_x2 = torch.min(gt_bbox[:, [0, 2]], dim=1).values, torch.max(gt_bbox[:, [0, 2]], dim=1).values
        gt_y1, gt_y2 = torch.min(gt_bbox[:, [1, 3]], dim=1).values, torch.max(gt_bbox[:, [1, 3]], dim=1).values
        gt_bbox = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1)

        # Intersection
        x1_i = torch.max(pred_bbox[:, 0], gt_bbox[:, 0])
        y1_i = torch.max(pred_bbox[:, 1], gt_bbox[:, 1])
        x2_i = torch.min(pred_bbox[:, 2], gt_bbox[:, 2])
        y2_i = torch.min(pred_bbox[:, 3], gt_bbox[:, 3])

        inter_w = torch.clamp(x2_i - x1_i, min=0)
        inter_h = torch.clamp(y2_i - y1_i, min=0)
        inter_area = inter_w * inter_h

        # Union
        area_p = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
        area_t = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])
        union_area = area_p + area_t - inter_area

        iou = inter_area / (union_area + 1e-8)

        # Enclosing box
        x1_c = torch.min(pred_bbox[:, 0], gt_bbox[:, 0])
        y1_c = torch.min(pred_bbox[:, 1], gt_bbox[:, 1])
        x2_c = torch.max(pred_bbox[:, 2], gt_bbox[:, 2])
        y2_c = torch.max(pred_bbox[:, 3], gt_bbox[:, 3])

        c_w = x2_c - x1_c
        c_h = y2_c - y1_c
        c_area = c_w * c_h + 1e-8

        giou = iou - (c_area - union_area) / c_area
        loss = 1.0 - giou

        return loss.mean()


class LocalizationLoss(nn.Module):
    """
    Standalone localization loss combining GIoU, L1, and optional heatmap loss.
    """

    def __init__(
        self,
        giou_weight: float = 1.0,
        l1_weight: float = 0.5,
        heatmap_weight: float = 0.0,
    ):
        super().__init__()
        self.giou_weight = giou_weight
        self.l1_weight = l1_weight
        self.heatmap_weight = heatmap_weight

    def forward(
        self,
        pred_bbox: torch.Tensor,
        gt_bbox: torch.Tensor,
        pred_heatmap: Optional[torch.Tensor] = None,
        gt_heatmap: Optional[torch.Tensor] = None,
        has_bbox: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_bbox: (B, 4) or (B, N, 4)
            gt_bbox: (B, 4) or (B, N, 4)
            pred_heatmap: Optional (B, H, W) predicted heatmap
            gt_heatmap: Optional (B, H, W) target heatmap
            has_bbox: (B,) mask for valid bboxes
        """
        if pred_bbox.dim() == 2:
            pred_bbox = pred_bbox.unsqueeze(1)
            gt_bbox = gt_bbox.unsqueeze(1)

        B, N, _ = pred_bbox.shape

        # Apply mask if provided
        if has_bbox is not None:
            valid_mask = has_bbox.view(-1).bool()
            if not valid_mask.any():
                return {
                    "giou_loss": torch.tensor(0.0, device=pred_bbox.device),
                    "l1_loss": torch.tensor(0.0, device=pred_bbox.device),
                    "heatmap_loss": torch.tensor(0.0, device=pred_bbox.device),
                    "total": torch.tensor(0.0, device=pred_bbox.device),
                }
            pred_bbox = pred_bbox.view(-1, 4)[valid_mask]
            gt_bbox = gt_bbox.view(-1, 4)[valid_mask]

        # GIoU
        giou_loss = self._giou_loss(pred_bbox, gt_bbox)

        # L1
        l1_loss = F.l1_loss(pred_bbox, gt_bbox)

        # Heatmap MSE
        heatmap_loss = torch.tensor(0.0, device=pred_bbox.device)
        if pred_heatmap is not None and gt_heatmap is not None:
            heatmap_loss = F.mse_loss(pred_heatmap, gt_heatmap)

        total = self.giou_weight * giou_loss + self.l1_weight * l1_loss + self.heatmap_weight * heatmap_loss

        return {
            "giou_loss": giou_loss,
            "l1_loss": l1_loss,
            "heatmap_loss": heatmap_loss,
            "total": total,
        }

    def _giou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute GIoU loss."""
        x1_i = torch.max(pred[:, 0], target[:, 0])
        y1_i = torch.max(pred[:, 1], target[:, 1])
        x2_i = torch.min(pred[:, 2], target[:, 2])
        y2_i = torch.min(pred[:, 3], target[:, 3])

        inter_w = torch.clamp(x2_i - x1_i, min=0)
        inter_h = torch.clamp(y2_i - y1_i, min=0)
        inter_area = inter_w * inter_h

        area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = area_p + area_t - inter_area

        iou = inter_area / (union_area + 1e-8)

        x1_c = torch.min(pred[:, 0], target[:, 0])
        y1_c = torch.min(pred[:, 1], target[:, 1])
        x2_c = torch.max(pred[:, 2], target[:, 2])
        y2_c = torch.max(pred[:, 3], target[:, 3])

        c_w = x2_c - x1_c
        c_h = y2_c - y1_c
        c_area = c_w * c_h + 1e-8

        giou = iou - (c_area - union_area) / c_area
        return (1.0 - giou).mean()


class ContrastiveTemporalLoss(nn.Module):
    """
    InfoNCE Contrastive Temporal Loss for pretraining / regularizing temporal frame embeddings.
    Pulls temporally adjacent frames from the same trajectory close in representation space
    while pushing frames from different trajectories or distant steps apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, k, D) sequence of frame embeddings per batch trajectory
        Returns:
            Scalar InfoNCE loss value
        """
        B, k, D = features.shape
        if k < 2 or B < 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Normalize features
        norm_features = F.normalize(features, dim=-1)  # (B, k, D)

        # Positive pairs: consecutive frames within same trajectory (t and t+1)
        z1 = norm_features[:, :-1, :].reshape(-1, D)  # (B*(k-1), D)
        z2 = norm_features[:, 1:, :].reshape(-1, D)   # (B*(k-1), D)

        # Cosine similarity matrix between all positive query and key representations
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature  # (N, N) where N = B*(k-1)

        # Ground truth labels: diagonal elements are positive pairs
        labels = torch.arange(sim_matrix.shape[0], device=features.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) loss wrapper.
    Selects top-k hardest examples with highest unreduced loss to backpropagate gradients.
    """

    def __init__(self, loss_fn: Optional[nn.Module] = None, keep_ratio: float = 0.7):
        super().__init__()
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(reduction="none")
        self.keep_ratio = keep_ratio

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions / logits (B, ...)
            targets: Target labels (B, ...)
        Returns:
            Mean loss over the hardest keep_ratio * B samples
        """
        loss_per_sample = self.loss_fn(inputs, targets)  # (B,) or higher dim

        if loss_per_sample.dim() > 1:
            loss_per_sample = loss_per_sample.view(loss_per_sample.shape[0], -1).mean(dim=1)

        B = loss_per_sample.shape[0]
        k = max(1, int(B * self.keep_ratio))

        topk_loss, _ = torch.topk(loss_per_sample, k=k, sorted=False)
        return topk_loss.mean()


def create_loss_function(config: Dict) -> SentinelLoss:
    """Factory function to create loss from config."""
    loss_config = config.get("loss", {})
    return SentinelLoss(
        risk_weight=loss_config.get("risk_weight", 2.0),
        category_weight=loss_config.get("category_weight", 1.0),
        localization_weight=loss_config.get("localization_weight", 1.5),
        harmful_class_weight=loss_config.get("harmful_class_weight", 5.0),
        category_class_weights=loss_config.get("category_class_weights", None),
        use_focal_loss=loss_config.get("use_focal_loss", True),
        focal_alpha=loss_config.get("focal_alpha", 0.75),
        focal_gamma=loss_config.get("focal_gamma", 2.0),
        giou_weight=loss_config.get("giou_weight", 1.0),
        l1_weight=loss_config.get("l1_weight", 0.5),
    )