"""
Heatmap and bounding box label generation for SENTINEL-Vision.
Generates ground truth localization targets from DOM at injection time.
DOM is used ONLY for label generation, then discarded.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional, Dict, Any
import cv2


def extract_bbox_from_dom(page, element_selector: str) -> Optional[List[float]]:
    """
    Extract bounding box from DOM element using Playwright page.
    Returns normalized [x1, y1, x2, y2] in range [0, 1].

    This is used ONLY at synthetic injection time to generate ground truth.
    The DOM is NOT available to the model during training/inference.
    """
    try:
        element = page.query_selector(element_selector)
        if not element:
            return None

        box = element.bounding_box()
        if not box:
            return None

        viewport_size = page.viewport_size
        if not viewport_size:
            return None

        x1 = box["x"] / viewport_size["width"]
        y1 = box["y"] / viewport_size["height"]
        x2 = (box["x"] + box["width"]) / viewport_size["width"]
        y2 = (box["y"] + box["height"]) / viewport_size["height"]

        # Clamp to [0, 1]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return [float(x1), float(y1), float(x2), float(y2)]

    except Exception:
        return None


def validate_bbox(bbox: List[float], image_size: Tuple[int, int] = (224, 224)) -> bool:
    """
    Validate bounding box sanity.
    Args:
        bbox: [x1, y1, x2, y2] normalized 0-1
        image_size: (width, height) for pixel validation
    Returns:
        True if valid
    """
    if len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox

    # Check range
    if not all(0.0 <= v <= 1.0 for v in bbox):
        return False

    # Check ordering
    if x2 <= x1 or y2 <= y1:
        return False

    # Check minimum size (at least 1% of image)
    w, h = image_size
    min_w = 0.01 * w
    min_h = 0.01 * h

    if (x2 - x1) * w < min_w or (y2 - y1) * h < min_h:
        return False

    # Check maximum size (not the whole image unless it's a full-screen element)
    if (x2 - x1) > 0.95 and (y2 - y1) > 0.95:
        return False

    return True


def generate_heatmap_gt(
    image_size: Tuple[int, int],
    bbox: List[float],
    sigma_factor: float = 0.02,
) -> torch.Tensor:
    """
    Generate Gaussian heatmap ground truth centered on bbox.

    Args:
        image_size: (H, W) of target heatmap
        bbox: [x1, y1, x2, y2] normalized 0-1
        sigma_factor: Gaussian sigma as fraction of image diagonal

    Returns:
        Heatmap tensor of shape (H, W) with values in [0, 1]
    """
    H, W = image_size
    x1, y1, x2, y2 = bbox

    # Center of bbox in pixel coordinates
    cx = ((x1 + x2) / 2) * W
    cy = ((y1 + y2) / 2) * H

    # Sigma based on bbox size
    bbox_w = (x2 - x1) * W
    bbox_h = (y2 - y1) * H
    sigma = sigma_factor * np.sqrt(W**2 + H**2)
    sigma = max(sigma, max(bbox_w, bbox_h) / 4.0)  # At least 1/4 of bbox size

    # Create coordinate grids
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    # Gaussian
    heatmap = torch.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))

    return heatmap


def generate_multiscale_heatmaps(
    image_size: Tuple[int, int],
    bbox: List[float],
    scales: List[int] = [56, 28, 14, 7],  # Feature map strides
) -> Dict[int, torch.Tensor]:
    """
    Generate heatmaps at multiple scales for multi-scale supervision.
    """
    heatmaps = {}
    for scale in scales:
        heatmaps[scale] = generate_heatmap_gt((scale, scale), bbox)
    return heatmaps


def bbox_to_heatmap_target(
    bbox: List[float],
    feature_map_size: Tuple[int, int],
    stride: int,
) -> torch.Tensor:
    """
    Convert normalized bbox to heatmap target on feature map.
    Used for localization head supervision.
    """
    return generate_heatmap_gt(feature_map_size, bbox)


def heatmap_to_bbox(heatmap: torch.Tensor, threshold: float = 0.5) -> List[float]:
    """
    Extract bbox from predicted heatmap.
    Returns normalized [x1, y1, x2, y2].
    """
    if heatmap.dim() == 3:
        heatmap = heatmap.squeeze(0)

    H, W = heatmap.shape

    # Find connected components above threshold
    binary = (heatmap > threshold).float()

    if binary.sum() == 0:
        return [0.0, 0.0, 0.0, 0.0]

    # Get bounding box of activated region
    y_indices, x_indices = torch.where(binary > 0)
    x1 = x_indices.min().item() / W
    y1 = y_indices.min().item() / H
    x2 = x_indices.max().item() / W
    y2 = y_indices.max().item() / H

    return [float(x1), float(y1), float(x2), float(y2)]


def draw_bbox_on_image(
    image: Image.Image,
    bbox: List[float],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 2,
    label: Optional[str] = None,
) -> Image.Image:
    """Draw bounding box on PIL image for visualization."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    W, H = img.size
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H

    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if label:
        draw.text((x1, max(0, y1 - 15)), label, fill=color)

    return img


def draw_heatmap_on_image(
    image: Image.Image,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """Overlay heatmap on image for visualization."""
    img_np = np.array(image)
    H, W = img_np.shape[:2]

    # Resize heatmap to image size
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)

    return Image.fromarray(overlay)


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute IoU between two normalized bboxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_giou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute Generalized IoU between two normalized bboxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0

    # Enclosing box
    x1_c = min(x1_1, x1_2)
    y1_c = min(y1_1, y1_2)
    x2_c = max(x2_1, x2_2)
    y2_c = max(y2_1, y2_2)

    c_area = (x2_c - x1_c) * (y2_c - y1_c)
    if c_area == 0:
        return iou

    giou = iou - (c_area - union_area) / c_area
    return giou


def bbox_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on normalized bbox coordinates."""
    return F.l1_loss(pred, target, reduction="none").sum(dim=-1)


def bbox_giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    GIoU loss for bbox regression.
    Args:
        pred: (B, 4) or (B, N, 4) predicted bboxes
        target: (B, 4) or (B, N, 4) target bboxes
    Returns:
        Scalar loss or (B,) loss per sample
    """
    # Ensure same shape
    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    B, N, _ = pred.shape
    loss = torch.zeros(B, N, device=pred.device)

    for b in range(B):
        for n in range(N):
            p = pred[b, n]
            t = target[b, n]
            giou = compute_giou(p.tolist(), t.tolist())
            loss[b, n] = 1.0 - giou

    return loss.mean()


class HeatmapLoss(torch.nn.Module):
    """Multi-scale heatmap loss for localization supervision."""

    def __init__(self, scales: List[int] = [56, 28, 14, 7], weights: Optional[List[float]] = None):
        super().__init__()
        self.scales = scales
        self.weights = weights or [1.0] * len(scales)
        self.mse = torch.nn.MSELoss(reduction="mean")

    def forward(
        self,
        pred_heatmaps: Dict[int, torch.Tensor],
        target_bbox: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_heatmaps: Dict mapping scale -> (B, H, W) predicted heatmaps
            target_bbox: (B, 4) normalized bboxes
        """
        total_loss = 0.0
        B = target_bbox.shape[0]

        for i, scale in enumerate(self.scales):
            if scale not in pred_heatmaps:
                continue

            pred = pred_heatmaps[scale]  # (B, H, W)

            # Generate target heatmaps for batch
            target_heatmaps = []
            for b in range(B):
                bbox = target_bbox[b].tolist()
                hm = generate_heatmap_gt((scale, scale), bbox)
                target_heatmaps.append(hm)
            target_heatmaps = torch.stack(target_heatmaps).to(pred.device)  # (B, H, W)

            loss = self.mse(pred, target_heatmaps)
            total_loss += self.weights[i] * loss

        return total_loss