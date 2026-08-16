"""
Screenshot augmentations for SENTINEL-Vision.
Applies consistent augmentations across all frames in a temporal window.
"""

import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class ScreenshotAugmentation:
    """
    Applies consistent augmentations across all frames in a window.
    Same spatial transform applied to all frames; temporal dropout randomly drops frames.
    """

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (224, 224),
        random_crop: bool = True,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        brightness_jitter: float = 0.2,
        contrast_jitter: float = 0.2,
        saturation_jitter: float = 0.2,
        hue_jitter: float = 0.1,
        jpeg_noise: bool = True,
        jpeg_quality_range: Tuple[int, int] = (70, 95),
        temporal_dropout: float = 0.1,
        gaussian_blur_prob: float = 0.1,
        gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0),
        horizontal_flip: float = 0.0,  # Usually 0 for screenshots (text would be mirrored)
        seed: Optional[int] = None,
    ):
        self.target_resolution = target_resolution
        self.random_crop = random_crop
        self.crop_scale = crop_scale
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.saturation_jitter = saturation_jitter
        self.hue_jitter = hue_jitter
        self.jpeg_noise = jpeg_noise
        self.jpeg_quality_range = jpeg_quality_range
        self.temporal_dropout = temporal_dropout
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.horizontal_flip = horizontal_flip
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

    def _get_crop_params(self, img: Image.Image) -> Tuple[int, int, int, int]:
        """Get random crop parameters consistent across frames."""
        width, height = img.size
        scale = random.uniform(*self.crop_scale)
        new_w = int(width * scale)
        new_h = int(height * scale)

        if new_w >= width and new_h >= height:
            return 0, 0, width, height

        i = random.randint(0, height - new_h)
        j = random.randint(0, width - new_w)
        return i, j, new_h, new_w

    def _get_color_jitter_params(self) -> Tuple[float, float, float, float]:
        """Get color jitter parameters consistent across frames."""
        brightness = random.uniform(1 - self.brightness_jitter, 1 + self.brightness_jitter)
        contrast = random.uniform(1 - self.contrast_jitter, 1 + self.contrast_jitter)
        saturation = random.uniform(1 - self.saturation_jitter, 1 + self.saturation_jitter)
        hue = random.uniform(-self.hue_jitter, self.hue_jitter)
        return brightness, contrast, saturation, hue

    def _apply_jpeg_compression(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression noise."""
        quality = random.randint(*self.jpeg_quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _apply_gaussian_blur(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian blur."""
        sigma = random.uniform(*self.gaussian_blur_sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        """
        Apply augmentations to a list of frames.

        Args:
            frames: List of PIL Images (length k)

        Returns:
            List of tensors (C, H, W) in range [0, 1]
        """
        if not frames:
            return [torch.zeros(3, *self.target_resolution) for _ in range(6)]

        k = len(frames)
        output_frames = []

        # Sample augmentation parameters ONCE per window (consistent across frames)
        crop_params = None
        if self.random_crop and random.random() < 0.5:
            crop_params = self._get_crop_params(frames[0])

        color_params = self._get_color_jitter_params()
        do_jpeg = self.jpeg_noise and random.random() < 0.3
        do_blur = random.random() < self.gaussian_blur_prob
        do_flip = self.horizontal_flip > 0 and random.random() < self.horizontal_flip

        # Temporal dropout: randomly drop frames
        keep_mask = [True] * k
        if self.temporal_dropout > 0 and k > 1:
            n_drop = max(1, int(k * self.temporal_dropout))
            drop_indices = random.sample(range(k), min(n_drop, k - 1))
            for idx in drop_indices:
                keep_mask[idx] = False

        for i, frame in enumerate(frames):
            # Skip dropped frames (replace with neighbor)
            if not keep_mask[i]:
                # Find nearest kept frame
                left = next((j for j in range(i - 1, -1, -1) if keep_mask[j]), None)
                right = next((j for j in range(i + 1, k) if keep_mask[j]), None)

                if left is not None and right is not None:
                    src_idx = left if (i - left) <= (right - i) else right
                elif left is not None:
                    src_idx = left
                elif right is not None:
                    src_idx = right
                else:
                    src_idx = i

                frame = frames[src_idx]

            # Apply spatial transforms
            img = frame.convert("RGB")

            # Random crop (consistent)
            if crop_params is not None:
                i_crop, j_crop, h_crop, w_crop = crop_params
                img = TF.crop(img, i_crop, j_crop, h_crop, w_crop)

            # Resize to target
            img = TF.resize(img, self.target_resolution, interpolation=TF.InterpolationMode.LANCZOS)

            # Horizontal flip (consistent)
            if do_flip:
                img = TF.hflip(img)

            # Color jitter (consistent)
            brightness, contrast, saturation, hue = color_params
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            img = TF.adjust_hue(img, hue)

            # JPEG compression noise
            if do_jpeg:
                img = self._apply_jpeg_compression(img)

            # Gaussian blur
            if do_blur:
                img = self._apply_gaussian_blur(img)

            # To tensor [0, 1]
            img_tensor = TF.to_tensor(img)
            output_frames.append(img_tensor)

        return output_frames


class ValidationAugmentation:
    """Minimal augmentation for validation - only resize and normalize."""

    def __init__(self, target_resolution: Tuple[int, int] = (224, 224)):
        self.target_resolution = target_resolution

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        output = []
        for frame in frames:
            img = frame.convert("RGB")
            img = TF.resize(img, self.target_resolution, interpolation=TF.InterpolationMode.LANCZOS)
            img_tensor = TF.to_tensor(img)
            output.append(img_tensor)
        return output


import io  # For JPEG compression


def create_train_transform(config: Dict[str, Any]) -> ScreenshotAugmentation:
    """Create training augmentation from config."""
    aug_config = config.get("augmentation", {})
    return ScreenshotAugmentation(
        target_resolution=tuple(config.get("frame_window", {}).get("resolution", [224, 224])),
        random_crop=aug_config.get("random_crop", True),
        brightness_jitter=aug_config.get("brightness_jitter", 0.2),
        jpeg_noise=aug_config.get("jpeg_noise", True),
        temporal_dropout=aug_config.get("temporal_dropout", 0.1),
    )


def create_val_transform(config: Dict[str, Any]) -> ValidationAugmentation:
    """Create validation transform from config."""
    return ValidationAugmentation(
        target_resolution=tuple(config.get("frame_window", {}).get("resolution", [224, 224])),
    )


class FrameWindowAugmentation:
    """
    Higher-level augmentation that handles frame window creation + augmentation.
    """

    def __init__(
        self,
        k: int = 6,
        target_resolution: Tuple[int, int] = (224, 224),
        train: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.k = k
        self.target_resolution = target_resolution
        self.train = train

        if train:
            self.augmentation = create_train_transform(config or {})
        else:
            self.augmentation = create_val_transform(config or {})

    def __call__(
        self,
        trajectory_frames: List[Image.Image],
        action_idx: int,
    ) -> torch.Tensor:
        """
        Extract window and apply augmentation.

        Args:
            trajectory_frames: Full trajectory frames
            action_idx: Index of action frame

        Returns:
            Tensor of shape (k, C, H, W)
        """
        # Extract window
        half_k = self.k // 2
        start_idx = max(0, action_idx - half_k)
        end_idx = min(len(trajectory_frames), start_idx + self.k)

        if end_idx - start_idx < self.k:
            start_idx = max(0, end_idx - self.k)

        window_frames = trajectory_frames[start_idx:end_idx]

        # Pad if needed
        while len(window_frames) < self.k:
            window_frames.append(window_frames[-1] if window_frames else Image.new("RGB", self.target_resolution, (128, 128, 128)))

        # Apply augmentation
        augmented = self.augmentation(window_frames)

        # Stack to (k, C, H, W)
        return torch.stack(augmented)