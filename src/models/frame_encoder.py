"""
Frame Encoder for SENTINEL-Vision.
Wraps timm backbones (ViT-S, ConvNeXt-Tiny, DINOv2) to return patch-level spatial embeddings.
CRITICAL: Input is ONLY pixels (frames). No agent logs, tool calls, or text.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple, Optional, List
import logging

from ..utils.constants import DEFAULT_IMAGE_SIZE

logger = logging.getLogger(__name__)


class FrameEncoder(nn.Module):
    """
    Frame encoder that extracts patch-level spatial embeddings from video frames.

    Supports multiple backbones:
    - vit_small_patch16_224 (ViT-S/16)
    - convnext_tiny (ConvNeXt-Tiny)
    - dino_vits14 (DINOv2 ViT-S/14) - for stronger few-shot localization

    Returns patch embeddings preserving spatial layout for localization head.
    """

    def __init__(
        self,
        backbone: str = "vit_small_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0,
        output_dim: Optional[int] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        """
        Args:
            backbone: timm model name
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Initially freeze backbone weights
            freeze_epochs: Number of epochs to keep frozen (0 = never freeze)
            output_dim: Project embeddings to this dimension (None = keep backbone dim)
            image_size: Expected square input resolution (was hardcoded to
                224 inside _init_backbone_info regardless of what the caller
                actually passed in).
        """
        super().__init__()

        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.freeze_epochs = freeze_epochs
        self.current_epoch = 0
        self.image_size = image_size

        # Create backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=False,
            num_classes=0,  # Remove classification head
            global_pool="",  # No global pooling - we want patch tokens
        )

        # Determine embedding dimension and patch info
        self._init_backbone_info()

        # Optional projection layer
        if output_dim is not None and output_dim != self.embed_dim:
            self.projection = nn.Linear(self.embed_dim, output_dim)
            self.embed_dim = output_dim
        else:
            self.projection = None

        # Freeze if requested
        if freeze_backbone and freeze_epochs > 0:
            self._freeze_backbone()

        logger.info(f"FrameEncoder initialized with {backbone}, embed_dim={self.embed_dim}, "
                    f"patch_grid={self.patch_grid}, num_patches={self.num_patches}")

    def _init_backbone_info(self):
        """Determine backbone-specific properties."""
        if "vit" in self.backbone_name.lower() or "dino" in self.backbone_name.lower():
            # ViT-based models
            self.is_vit = True
            # Get patch size from model
            if hasattr(self.backbone, "patch_embed"):
                patch_size = self.backbone.patch_embed.patch_size
                if isinstance(patch_size, tuple):
                    patch_size = patch_size[0]
            else:
                patch_size = 16  # default

            self.patch_size = patch_size

            self.patch_grid = (self.image_size // patch_size, self.image_size // patch_size)
            self.num_patches = self.patch_grid[0] * self.patch_grid[1]

            # Embedding dimension. If neither attribute is present, this is
            # an unrecognized/unsupported backbone -- fail loudly instead of
            # silently guessing 384, which would produce a model whose
            # dimensions don't match its actual output and fail (or worse,
            # subtly misbehave) much later at a confusing call site.
            if hasattr(self.backbone, "embed_dim"):
                self.embed_dim = self.backbone.embed_dim
            elif hasattr(self.backbone, "num_features"):
                self.embed_dim = self.backbone.num_features
            else:
                raise ValueError(
                    f"Cannot determine embed_dim for ViT backbone '{self.backbone_name}': "
                    "it exposes neither .embed_dim nor .num_features. Add explicit "
                    "handling for this backbone instead of silently guessing."
                )

        elif "convnext" in self.backbone_name.lower():
            # ConvNeXt
            self.is_vit = False
            # ConvNeXt feature map is 1/32 of input
            self.patch_size = 32
            grid = self.image_size // self.patch_size
            self.patch_grid = (grid, grid)
            self.num_patches = grid * grid
            self.embed_dim = self.backbone.num_features  # 768 for tiny

        else:
            raise ValueError(
                f"Unsupported backbone '{self.backbone_name}': FrameEncoder only "
                "recognizes ViT-family ('vit'/'dino' in the name) and 'convnext' "
                "backbones. A silent generic fallback here previously guessed "
                "patch_grid=(14,14)/embed_dim=384 for ANY unrecognized backbone, "
                "which would produce wrong dimensions for anything else."
            )

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info(f"Backbone frozen for {self.freeze_epochs} epochs")

    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def set_epoch(self, epoch: int):
        """Call at start of each epoch to handle freeze schedule."""
        self.current_epoch = epoch
        if self.freeze_backbone and self.freeze_epochs > 0:
            if epoch == self.freeze_epochs:
                self._unfreeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input frames tensor of shape (B, k, C, H, W) or (B, C, H, W)

        Returns:
            Patch embeddings of shape (B, k, N_patches, D) or (B, N_patches, D)
        """
        # Handle both (B, k, C, H, W) and (B, C, H, W)
        has_time_dim = x.dim() == 5

        if has_time_dim:
            B, k, C, H, W = x.shape
            x = x.view(B * k, C, H, W)
        else:
            B = x.shape[0]
            k = 1
            C, H, W = x.shape[1:]

        if C != 3:
            raise ValueError(
                f"FrameEncoder expects 3-channel RGB input, got {C} channels. "
                "This is a pixels-only oversight system -- a wrong channel "
                "count usually means the caller passed a non-image tensor."
            )
        if (H, W) != (self.image_size, self.image_size):
            raise ValueError(
                f"FrameEncoder was built for {self.image_size}x{self.image_size} "
                f"input but got {H}x{W}. Resize frames to the configured "
                "image_size before calling forward()."
            )

        # Extract features
        if self.is_vit:
            features = self._forward_vit(x)  # (B*k, N_patches, D)
        else:
            features = self._forward_convnext(x)  # (B*k, N_patches, D)

        # Apply projection if needed
        if self.projection is not None:
            features = self.projection(features)

        # Reshape back to include time dimension
        if has_time_dim:
            features = features.view(B, k, self.num_patches, -1)

        return features

    def _forward_vit(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ViT-based models."""
        # timm ViT forward_features returns (B, N_patches+1, D) with CLS token
        # We need to remove CLS token and return patch tokens only
        features = self.backbone.forward_features(x)

        # Handle different return formats
        if isinstance(features, dict):
            features = features.get("x", features.get("pre_logits", features))
        if features.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = features.shape
            features = features.flatten(2).transpose(1, 2)
        elif features.dim() == 3:
            # (B, N+1, D) - remove CLS token
            if features.shape[1] == self.num_patches + 1:
                features = features[:, 1:, :]
            # If already (B, N, D), keep as is
        return features

    def _forward_convnext(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ConvNeXt."""
        # ConvNeXt forward_features returns (B, C, H, W)
        features = self.backbone.forward_features(x)

        # (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, N_patches, C)

        # Verify patch count
        if features.shape[1] != self.num_patches:
            logger.warning(f"Expected {self.num_patches} patches, got {features.shape[1]}")
            self.num_patches = features.shape[1]
            self.patch_grid = (H, W)

        return features

    def get_spatial_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatial embeddings in grid format for localization head.

        Args:
            x: Input tensor (B, k, C, H, W) or (B, C, H, W)

        Returns:
            Spatial embeddings (B, k, H, W, D) or (B, H, W, D)
        """
        has_time_dim = x.dim() == 5

        if has_time_dim:
            B, k, C, H, W = x.shape
            x = x.view(B * k, C, H, W)
        else:
            B = x.shape[0]
            k = 1

        # Get patch embeddings
        if self.is_vit:
            features = self._forward_vit(x)
        else:
            features = self._forward_convnext(x)

        if self.projection is not None:
            features = self.projection(features)

        # Reshape to spatial grid
        H_grid, W_grid = self.patch_grid
        features = features.view(B * k, H_grid, W_grid, -1)

        if has_time_dim:
            features = features.view(B, k, H_grid, W_grid, -1)
        else:
            features = features.view(B, H_grid, W_grid, -1)

        return features

    def get_cls_token(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract CLS token for classification (ViT only)."""
        if not self.is_vit:
            return None

        has_time_dim = x.dim() == 5
        if has_time_dim:
            B, k, C, H, W = x.shape
            x = x.view(B * k, C, H, W)

        features = self.backbone.forward_features(x)

        if isinstance(features, dict):
            features = features.get("x", features.get("pre_logits", features))

        if features.dim() == 3 and features.shape[1] == self.num_patches + 1:
            cls_token = features[:, 0, :]
            if has_time_dim:
                cls_token = cls_token.view(B, k, -1)
            return cls_token

        return None


class MultiScaleFrameEncoder(nn.Module):
    """
    Frame encoder that returns multi-scale features for better localization.
    Combines features from multiple layers of the backbone.
    """

    def __init__(
        self,
        backbone: str = "vit_small_patch16_224",
        pretrained: bool = True,
        output_dim: int = 256,
        feature_layers: Optional[List[int]] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,  # Return multi-scale features
            out_indices=feature_layers or [2, 5, 8, 11],  # Default for ViT-S (12 layers)
        )

        # Get feature info
        feature_info = self.backbone.feature_info
        self.feature_channels = [info["num_chs"] for info in feature_info]
        self.feature_strides = [info["reduction"] for info in feature_info]

        # Project all to same dimension
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, output_dim, 1) for ch in self.feature_channels
        ])

        self.output_dim = output_dim
        self.num_scales = len(self.feature_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns list of feature maps at different scales.
        Each: (B, k, D, H, W) or (B, D, H, W)
        """
        has_time_dim = x.dim() == 5
        if has_time_dim:
            B, k, C, H, W = x.shape
            x = x.view(B * k, C, H, W)

        features = self.backbone(x)  # List of (B*k, C, H, W)

        # Project each scale
        projected = []
        for i, feat in enumerate(features):
            feat = self.projections[i](feat)  # (B*k, D, H, W)
            if has_time_dim:
                _, D, H, W = feat.shape
                feat = feat.view(B, k, D, H, W)
            projected.append(feat)

        return projected

    def get_spatial_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get finest scale spatial embeddings for localization."""
        features = self.forward(x)
        finest = features[-1]  # Highest resolution

        if finest.dim() == 5:
            B, k, D, H, W = finest.shape
            return finest.permute(0, 1, 3, 4, 2)  # (B, k, H, W, D)
        else:
            B, D, H, W = finest.shape
            return finest.permute(0, 2, 3, 1)  # (B, H, W, D)


def create_frame_encoder(config: Dict) -> FrameEncoder:
    """Factory function to create frame encoder from config."""
    return FrameEncoder(
        backbone=config.get("backbone", "vit_small_patch16_224"),
        pretrained=config.get("pretrained", True),
        freeze_backbone=config.get("freeze_backbone", False),
        freeze_epochs=config.get("freeze_backbone_epochs", 3),
        output_dim=config.get("output_dim", None),
        image_size=config.get("image_size", DEFAULT_IMAGE_SIZE),
    )