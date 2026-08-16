"""
Localization Head for SENTINEL-Vision.
Predicts bounding box of risky UI element using spatial patch embeddings.
YOLO-style multi-scale anchor prediction + Grad-CAM heatmap generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class LocalizationHead(nn.Module):
    """
    Localization head for predicting bounding box of risky UI element.
    Uses spatial patch embeddings (NOT CLS token).
    YOLO-style multi-scale anchor prediction.
    Also generates Grad-CAM heatmap for interpretability.
    """

    def __init__(
        self,
        embed_dim: int,
        anchor_sizes: List[int] = [32, 64, 128, 256],
        num_anchors: int = 9,
        num_classes: int = 1,  # Objectness only
        feature_stride: int = 16,
        image_size: int = 224,
    ):
        """
        Args:
            embed_dim: Input embedding dimension from temporal fusion
            anchor_sizes: Base anchor sizes in pixels
            num_anchors: Number of anchors per spatial location
            num_classes: Number of classes (1 for objectness)
            feature_stride: Stride of feature map relative to input image
            image_size: Input image size
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.anchor_sizes = anchor_sizes
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.feature_stride = feature_stride
        self.image_size = image_size

        # Feature map size
        self.fm_size = image_size // feature_stride  # 14 for 224/16
        self.num_locations = self.fm_size * self.fm_size

        # Generate anchor boxes
        self.register_buffer("anchors", self._generate_anchors())

        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Objectness head (confidence that an object exists at this location)
        self.objectness_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_anchors),
        )

        # Bbox regression head (predicts offsets from anchors)
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_anchors * 4),
        )

        # Class head (optional, for multi-class localization)
        if num_classes > 1:
            self.class_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, num_anchors * num_classes),
            )
        else:
            self.class_head = None

        logger.info(f"LocalizationHead initialized: embed_dim={embed_dim}, "
                    f"fm_size={self.fm_size}, num_anchors={num_anchors}, "
                    f"anchor_sizes={anchor_sizes}")

    def _generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes for each spatial location."""
        anchors = []
        fm_size = self.fm_size
        stride = self.feature_stride

        # Aspect ratios for anchors
        aspect_ratios = [0.5, 1.0, 2.0]  # 3 aspect ratios
        scales = [2**0, 2**(1/3), 2**(2/3)]  # 3 scales

        for base_size in self.anchor_sizes:
            for scale in scales:
                for ar in aspect_ratios:
                    w = base_size * scale * np.sqrt(ar)
                    h = base_size * scale / np.sqrt(ar)
                    anchors.append([w, h])

        # Limit to num_anchors
        anchors = anchors[:self.num_anchors]
        anchors = torch.tensor(anchors, dtype=torch.float32)  # (A, 2) - width, height

        # Create grid of anchor centers
        shift_x = (torch.arange(fm_size) + 0.5) * stride
        shift_y = (torch.arange(fm_size) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

        # (H, W) -> (H*W,)
        shift_x = shift_x.flatten()
        shift_y = shift_y.flatten()

        # Combine shifts with anchors
        # anchors: (A, 2), shifts: (H*W, 2)
        # Result: (H*W, A, 4) - [x1, y1, x2, y2] in pixel coordinates
        all_anchors = []
        for i in range(len(shift_x)):
            cx, cy = shift_x[i], shift_y[i]
            for a in anchors:
                w, h = a
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                all_anchors.append([x1, y1, x2, y2])

        all_anchors = torch.tensor(all_anchors, dtype=torch.float32)  # (H*W*A, 4)
        return all_anchors

    def forward(
        self,
        spatial_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            spatial_embeddings: (B, H, W, D) or (B, N_patches, D) from temporal fusion

        Returns:
            Dict with:
                - bbox: (B, 4) predicted bbox in normalized [0,1] coordinates
                - objectness: (B, H*W*num_anchors) or (B, H, W, num_anchors)
                - anchor_logits: (B, num_anchors, 5) - [obj, x, y, w, h]
        """
        B = spatial_embeddings.shape[0]

        # Handle input format
        if spatial_embeddings.dim() == 4:
            # (B, H, W, D) -> (B, H*W, D)
            B, H, W, D = spatial_embeddings.shape
            spatial_embeddings = spatial_embeddings.view(B, H * W, D)
        elif spatial_embeddings.dim() == 3:
            B, N, D = spatial_embeddings.shape
            H = W = int(N**0.5)
        else:
            raise ValueError(f"Unexpected spatial_embeddings shape: {spatial_embeddings.shape}")

        # Process features
        features = self.feature_processor(spatial_embeddings)  # (B, N, D)

        # Objectness prediction
        obj_logits = self.objectness_head(features)  # (B, N, num_anchors)
        obj_logits = obj_logits.view(B, H, W, self.num_anchors)

        # Bbox regression
        bbox_pred = self.bbox_head(features)  # (B, N, num_anchors*4)
        bbox_pred = bbox_pred.view(B, H, W, self.num_anchors, 4)

        # Get best anchor per spatial location
        obj_probs = torch.sigmoid(obj_logits)  # (B, H, W, A)
        best_anchor_idx = obj_probs.argmax(dim=-1)  # (B, H, W)

        # Gather best bbox predictions
        B_idx = torch.arange(B, device=spatial_embeddings.device).view(B, 1, 1)
        H_idx = torch.arange(H, device=spatial_embeddings.device).view(1, H, 1)
        W_idx = torch.arange(W, device=spatial_embeddings.device).view(1, 1, W)

        best_bbox = bbox_pred[B_idx, H_idx, W_idx, best_anchor_idx]  # (B, H, W, 4)

        # Get best objectness
        best_obj = obj_probs[B_idx, H_idx, W_idx, best_anchor_idx]  # (B, H, W)

        # Find global best location
        best_loc = best_obj.view(B, -1).argmax(dim=-1)  # (B,)
        best_h = best_loc // W
        best_w = best_loc % W

        # Get final bbox prediction
        final_bbox = best_bbox[B_idx.squeeze(), best_h, best_w]  # (B, 4)
        final_obj = best_obj[B_idx.squeeze(), best_h, best_w]  # (B,)

        # Convert to normalized coordinates [0, 1]
        final_bbox_norm = final_bbox / self.image_size
        final_bbox_norm = final_bbox_norm.clamp(0, 1)

        return {
            "bbox": final_bbox_norm,                    # (B, 4) normalized
            "bbox_pixel": final_bbox,                   # (B, 4) pixel coordinates
            "objectness": obj_probs,                    # (B, H, W, A)
            "objectness_max": final_obj,                # (B,)
            "bbox_pred": bbox_pred,                     # (B, H, W, A, 4)
            "anchor_logits": torch.cat([obj_logits.unsqueeze(-1), bbox_pred], dim=-1),  # (B, H, W, A, 5)
        }

    def generate_gradcam(
        self,
        model: nn.Module,
        input_frames: torch.Tensor,
        target_layer: nn.Module,
        target_category: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for interpretability.

        Args:
            model: Full SentinelModel
            input_frames: (B, k, C, H, W) input frames
            target_layer: Layer to compute gradients w.r.t (typically last conv/attention)
            target_category: Target category index for guided Grad-CAM

        Returns:
            Heatmap as numpy array (H, W) in [0, 1]
        """
        model.eval()

        # Forward hook to capture activations
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        # Register hooks
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            output = model(input_frames)

            # Get target score
            if target_category is not None:
                # Use category logits
                score = output["category_logits"][0, target_category]
            else:
                # Use risk score
                score = output["risk_score"][0, 0]

            # Backward pass
            model.zero_grad()
            score.backward(retain_graph=True)

            # Get activations and gradients
            act = activations[0]  # (B, ..., C) or (B, C, H, W)
            grad = gradients[0]  # Same shape

            # Global average pool gradients
            if grad.dim() == 4:
                weights = grad.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
                cam = (weights * act).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            elif grad.dim() == 3:
                # (B, N, C) - spatial tokens
                weights = grad.mean(dim=1, keepdim=True)  # (B, 1, C)
                cam = (weights * act).sum(dim=-1, keepdim=True)  # (B, N, 1)
                # Reshape to spatial
                B, N, _ = cam.shape
                H = W = int(N**0.5)
                cam = cam.view(B, 1, H, W)
            else:
                raise ValueError(f"Unexpected activation shape: {act.shape}")

            # ReLU
            cam = F.relu(cam)

            # Normalize
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            # Resize to input image size
            cam = F.interpolate(cam, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)

            return cam[0, 0].detach().cpu().numpy()

        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    def compute_giou_loss(
        self,
        pred_bbox: torch.Tensor,
        gt_bbox: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute GIoU loss between predicted and ground truth bboxes.

        Args:
            pred_bbox: (B, 4) or (B, N, 4) predicted bboxes [x1, y1, x2, y2] normalized
            gt_bbox: (B, 4) or (B, N, 4) target bboxes
            reduction: "mean", "sum", "none"

        Returns:
            GIoU loss
        """
        if pred_bbox.dim() == 2:
            pred_bbox = pred_bbox.unsqueeze(1)
            gt_bbox = gt_bbox.unsqueeze(1)

        B, N, _ = pred_bbox.shape
        loss = torch.zeros(B, N, device=pred_bbox.device)

        for b in range(B):
            for n in range(N):
                p = pred_bbox[b, n]
                t = gt_bbox[b, n]

                # Intersection
                x1_i = torch.max(p[0], t[0])
                y1_i = torch.max(p[1], t[1])
                x2_i = torch.min(p[2], t[2])
                y2_i = torch.min(p[3], t[3])

                inter_w = torch.clamp(x2_i - x1_i, min=0)
                inter_h = torch.clamp(y2_i - y1_i, min=0)
                inter_area = inter_w * inter_h

                # Union
                area_p = (p[2] - p[0]) * (p[3] - p[1])
                area_t = (t[2] - t[0]) * (t[3] - t[1])
                union_area = area_p + area_t - inter_area

                iou = inter_area / (union_area + 1e-8)

                # Enclosing box
                x1_c = torch.min(p[0], t[0])
                y1_c = torch.min(p[1], t[1])
                x2_c = torch.max(p[2], t[2])
                y2_c = torch.max(p[3], t[3])

                c_w = x2_c - x1_c
                c_h = y2_c - y1_c
                c_area = c_w * c_h + 1e-8

                giou = iou - (c_area - union_area) / c_area
                loss[b, n] = 1.0 - giou

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def decode_predictions(
        self,
        obj_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        top_k: int = 10,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode raw predictions to final bboxes with NMS.

        Args:
            obj_logits: (B, H, W, A) objectness logits
            bbox_pred: (B, H, W, A, 4) bbox predictions in pixel coords
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            top_k: Max detections per image

        Returns:
            List of dicts per batch with keys: bbox, score, anchor_idx
        """
        B, H, W, A = obj_logits.shape
        results = []

        for b in range(B):
            # Flatten
            obj_flat = obj_logits[b].view(-1)  # (H*W*A,)
            bbox_flat = bbox_pred[b].view(-1, 4)  # (H*W*A, 4)

            # Filter by confidence
            scores = torch.sigmoid(obj_flat)
            keep = scores > conf_threshold

            if not keep.any():
                results.append({"bbox": torch.empty(0, 4), "scores": torch.empty(0)})
                continue

            scores = scores[keep]
            bboxes = bbox_flat[keep]

            # Normalize
            bboxes = bboxes / self.image_size
            bboxes = bboxes.clamp(0, 1)

            # NMS (simple implementation)
            indices = torch.argsort(scores, descending=True)
            keep_indices = []

            while len(indices) > 0 and len(keep_indices) < top_k:
                i = indices[0]
                keep_indices.append(i.item())

                if len(indices) == 1:
                    break

                # Compute IoU with remaining
                ious = torch.zeros(len(indices) - 1, device=scores.device)
                for j, idx in enumerate(indices[1:]):
                    ious[j] = self._bbox_iou(bboxes[i], bboxes[idx])

                indices = indices[1:][ious < nms_threshold]

            final_bboxes = bboxes[keep_indices]
            final_scores = scores[keep_indices]

            results.append({
                "bbox": final_bboxes,
                "scores": final_scores,
            })

        return results

    def _bbox_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two normalized boxes."""
        x1_i = max(box1[0].item(), box2[0].item())
        y1_i = max(box1[1].item(), box2[1].item())
        x2_i = min(box1[2].item(), box2[2].item())
        y2_i = min(box1[3].item(), box2[3].item())

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        inter = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
        area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0


def create_localization_head(config: Dict) -> LocalizationHead:
    """Factory function to create localization head from config."""
    loc_config = config.get("localization", {})
    return LocalizationHead(
        embed_dim=config.get("embed_dim", 384),
        anchor_sizes=loc_config.get("anchor_sizes", [32, 64, 128, 256]),
        num_anchors=loc_config.get("num_anchors", 9),
        feature_stride=loc_config.get("feature_stride", 16),
        image_size=config.get("image_size", 224),
    )