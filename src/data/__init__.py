"""Data loading and preprocessing modules for SENTINEL-Vision."""

from .loaders import (
    load_multimodal_mind2web,
    load_screenspot,
    SentinelDataset,
    load_real_dataset,
    parse_target_bounding_box,
)
from .frame_windowing import (
    extract_frame_window,
    FrameWindowDataset,
)
from .heatmap_labels import (
    generate_heatmap_gt,
    validate_bbox,
    compute_iou,
    compute_giou,
)
from .augmentation import ScreenshotAugmentation

__all__ = [
    "load_multimodal_mind2web",
    "load_screenspot",
    "SentinelDataset",
    "load_real_dataset",
    "extract_frame_window",
    "FrameWindowDataset",
    "generate_heatmap_gt",
    "validate_bbox",
    "compute_iou",
    "compute_giou",
    "ScreenshotAugmentation",
]