


import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Union

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def draw_bboxes(
    image: Union[Image.Image, np.ndarray],
    bboxes: List[Tuple[float, float, float, float]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    width: int = 3,
    normalized: bool = True,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Args:
        image: PIL Image or numpy array (H, W, 3)
        bboxes: List of (x1, y1, x2, y2) or (x, y, w, h) if normalized
        labels: Optional list of labels for each box
        colors: Optional list of RGB colors for each box
        width: Line width
        normalized: If True, bboxes are in [0, 1] range

    Returns:
        PIL Image with boxes drawn
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    draw = ImageDraw.Draw(image)
    w, h = image.size

    default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    colors = colors or default_colors

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox

        if normalized:
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h

        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        if labels and i < len(labels):
            # Draw label background
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()

            text = labels[i]
            text_w, text_h = draw.textsize(text, font=font)
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=font)

    return image


def overlay_heatmap(
    image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET if HAS_CV2 else None,
) -> Image.Image:
    """
    Overlay a heatmap on an image.

    Args:
        image: PIL Image or numpy array (H, W, 3)
        heatmap: Heatmap array (H, W) or (H, W, 1) in [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap constant

    Returns:
        PIL Image with heatmap overlay
    """
    if not HAS_CV2:
        # Simple fallback without OpenCV
        return _overlay_heatmap_fallback(image, heatmap, alpha)

    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image.copy()

    w, h = pil_image.size
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize heatmap to 0-1
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    image_np = np.array(pil_image)
    blended = cv2.addWeighted(image_np, 1 - alpha, heatmap_rgb, alpha, 0)

    return Image.fromarray(blended)


def _overlay_heatmap_fallback(
    image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Fallback heatmap overlay without OpenCV."""
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image.copy()

    w, h = pil_image.size
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))

    # Simple red heatmap
    heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap_rgb[:, :, 0] = heatmap_resized  # Red channel

    # Blend
    image_np = np.array(pil_image)
    blended = (image_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)

    return Image.fromarray(blended)


def visualize_predictions(
    image: Union[Image.Image, np.ndarray],
    risk_score: float,
    category: str,
    category_conf: float,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    heatmap: Optional[np.ndarray] = None,
    decision: Optional[str] = None,
    normalized: bool = True,
) -> Image.Image:
    """
    Visualize SENTINEL-Vision predictions on an image.

    Args:
        image: Base image
        risk_score: Risk probability [0, 1]
        category: Predicted category name
        category_conf: Category confidence [0, 1]
        bbox: Optional bounding box (x1, y1, x2, y2)
        heatmap: Optional heatmap array
        decision: Optional decision string (ALLOW/PAUSE/HARD_BLOCK)
        normalized: Whether bbox is normalized

    Returns:
        PIL Image with visualizations
    """
    vis = image if isinstance(image, Image.Image) else Image.fromarray(image.astype(np.uint8))
    w, h = vis.size

    # Overlay heatmap if provided
    if heatmap is not None:
        vis = overlay_heatmap(vis, heatmap, alpha=0.4)

    draw = ImageDraw.Draw(vis, "RGBA")

    # Color based on decision/risk
    if decision == "HARD_BLOCK" or risk_score > 0.7:
        risk_color = (255, 0, 0, 255)
    elif decision == "PAUSE" or risk_score > 0.4:
        risk_color = (255, 165, 0, 255)
    else:
        risk_color = (0, 255, 0, 255)

    # Draw risk bar at top
    bar_height = 25
    bar_width = int(w * risk_score)
    draw.rectangle([0, 0, w, bar_height], fill=(0, 0, 0, 180))
    draw.rectangle([0, 0, bar_width, bar_height], fill=risk_color)

    # Risk text
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    risk_text = f"RISK: {risk_score:.3f} | {category} ({category_conf:.2f})"
    if decision:
        risk_text += f" | {decision}"
    draw.text((10, 4), risk_text, fill=(255, 255, 255, 255), font=font)

    # Draw bounding box
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if normalized:
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h

        draw.rectangle([x1, y1, x2, y2], outline=risk_color[:3], width=3)
        label = f"{category}: {category_conf:.2f}"
        text_w, text_h = draw.textsize(label, font=font)
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=risk_color[:3])
        draw.text((x1 + 2, y1 - text_h - 2), label, fill=(255, 255, 255), font=font)

    return vis


def create_comparison_grid(
    images: List[Image.Image],
    titles: List[str],
    cols: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """
    Create a grid of images with titles.

    Args:
        images: List of PIL Images (all same size)
        titles: List of titles
        cols: Number of columns
        font_size: Font size for titles

    Returns:
        PIL Image grid
    """
    if not images:
        raise ValueError("No images provided")

    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Calculate grid size
    grid_w = w * cols
    grid_h = h * rows + font_size * rows

    grid = Image.new("RGB", (grid_w, grid_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid)

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        x = col * w
        y = row * (h + font_size)

        grid.paste(img, (x, y))
        draw.text((x + 5, y + h + 2), title, fill=(0, 0, 0), font=font)

    return grid


def tensor_to_pil(tensor: "torch.Tensor") -> Image.Image:
    """Convert a [C, H, W] or [H, W, C] tensor to PIL Image."""
    import torch

    if tensor.dim() == 3:
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        if tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)

    if tensor.max() <= 1.0:
        tensor = tensor * 255

    return Image.fromarray(tensor.cpu().byte().numpy())
