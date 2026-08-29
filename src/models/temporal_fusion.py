"""
Temporal Fusion Module for SENTINEL-Vision.
Processes k frame embeddings through temporal self-attention with delta-sensitivity.
Key design: Joint spatiotemporal attention across time AND space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import logging

logger = logging.getLogger(__name__)


class TemporalPositionalEncoding(nn.Module):
    """Learnable temporal positional encoding for frame positions in window."""

    def __init__(self, max_frames: int, embed_dim: int):
        super().__init__()
        self.max_frames = max_frames
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames, 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, k, N_patches, D)
        Returns:
            x with temporal positional encoding added
        """
        k = x.shape[1]
        if k > self.max_frames:
            raise ValueError(
                f"TemporalPositionalEncoding was built for at most "
                f"max_frames={self.max_frames} but received a window of "
                f"k={k} frames. This used to silently index past the end of "
                "a fixed-size positional-embedding buffer (undefined slicing "
                "behavior, not a clean error) -- pass a frame window no "
                "longer than max_frames, or rebuild with a larger max_frames."
            )
        return x + self.pos_embedding[:, :k, :, :]


class DeltaFeatureExtractor(nn.Module):
    """
    Explicitly computes frame-to-frame differences (delta features)
    to enhance sensitivity to state changes.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.delta_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute delta features between consecutive frames.

        Args:
            frame_embeddings: (B, k, N_patches, D)

        Returns:
            Delta features: (B, k-1, N_patches, D)
        """
        B, k, N, D = frame_embeddings.shape

        # Compute differences: frame_t - frame_{t-1}
        frame_t = frame_embeddings[:, 1:, :, :]    # (B, k-1, N, D)
        frame_tm1 = frame_embeddings[:, :-1, :, :]  # (B, k-1, N, D)

        # Concatenate and project
        delta_input = torch.cat([frame_t, frame_tm1], dim=-1)  # (B, k-1, N, 2D)
        delta_features = self.delta_proj(delta_input)  # (B, k-1, N, D)

        return delta_features


class SpatiotemporalAttention(nn.Module):
    """
    Joint spatiotemporal self-attention.
    Attends across both time and space dimensions simultaneously.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn and hasattr(F, "scaled_dot_product_attention")

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, k, N, D) - batch, frames, patches, embed_dim
            mask: Optional attention mask (B, k, N) or (B, k*N)

        Returns:
            Output: (B, k, N, D)
        """
        B, k, N, D = x.shape

        # Flatten spatiotemporal dimensions
        x_flat = x.view(B, k * N, D)  # (B, k*N, D)

        # QKV projection
        qkv = self.qkv(x_flat).reshape(B, k * N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, k*N, head_dim)
        q, k_attn, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash_attn:
            # Use PyTorch's flash attention
            attn_mask = None
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.view(B, k * N)
                attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k*N)

            out = F.scaled_dot_product_attention(
                q, k_attn, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Standard attention
            attn = (q @ k_attn.transpose(-2, -1)) * self.scale  # (B, num_heads, k*N, k*N)

            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.view(B, k * N)
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k*N)
                attn = attn.masked_fill(mask == 0, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ v  # (B, num_heads, k*N, head_dim)

        out = out.transpose(1, 2).reshape(B, k * N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reshape back to spatiotemporal
        out = out.view(B, k, N, D)
        return out


class TemporalFusionBlock(nn.Module):
    """
    Single temporal fusion block with spatiotemporal attention and FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        use_flash_attn: bool = False,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SpatiotemporalAttention(
            embed_dim, num_heads, dropout, use_flash_attn
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x), mask))

        # FFN with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TemporalFusion(nn.Module):
    """
    Temporal Fusion Module for SENTINEL-Vision.

    Processes k frame embeddings through temporal self-attention.
    Delta-sensitive: explicitly models frame-to-frame change.
    Key design: attends across time AND space (joint spatiotemporal attention).

    Input: (B, k, N_patches, D) - batch, frames, patches, embed_dim
    Output: (B, N_patches, D) - fused representation (last frame or pooled)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        max_frames: int = 8,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_delta_features: bool = True,
        use_flash_attn: bool = True,
        fusion_mode: str = "last",  # "last", "mean", "attn_pool"
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_frames = max_frames
        self.use_delta_features = use_delta_features
        self.fusion_mode = fusion_mode

        # Temporal positional encoding
        self.temporal_pos_enc = TemporalPositionalEncoding(max_frames, embed_dim)

        # Delta feature extractor
        if use_delta_features:
            self.delta_extractor = DeltaFeatureExtractor(embed_dim)
            self.delta_proj = nn.Linear(embed_dim * 2, embed_dim)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TemporalFusionBlock(
                embed_dim, num_heads,
                dropout=dropout,
                drop_path=dpr[i],
                use_flash_attn=use_flash_attn,
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output fusion
        if fusion_mode == "attn_pool":
            self.attn_pool = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, 1),
            )

        logger.info(f"TemporalFusion initialized: embed_dim={embed_dim}, "
                    f"num_layers={num_layers}, max_frames={max_frames}, "
                    f"use_delta={use_delta_features}, fusion_mode={fusion_mode}")

    def forward(
        self,
        frame_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: (B, k, N_patches, D)
            mask: Optional mask (B, k) or (B, k, N) for padding

        Returns:
            Fused representation: (B, N_patches, D)
        """
        B, k, N, D = frame_embeddings.shape

        # Add temporal positional encoding
        x = self.temporal_pos_enc(frame_embeddings)

        # Compute and inject delta features
        if self.use_delta_features and k > 1:
            delta_features = self.delta_extractor(frame_embeddings)  # (B, k-1, N, D)

            # Pad delta to match k frames (prepend zero for first frame)
            delta_padded = F.pad(delta_features, (0, 0, 0, 0, 1, 0))  # (B, k, N, D)

            # Concatenate and project
            x = torch.cat([x, delta_padded], dim=-1)  # (B, k, N, 2D)
            x = self.delta_proj(x)  # (B, k, N, D)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)  # (B, k, N, D)

        # Fuse across time dimension
        if self.fusion_mode == "last":
            fused = x[:, -1, :, :]  # (B, N, D) - last frame
        elif self.fusion_mode == "mean":
            fused = x.mean(dim=1)  # (B, N, D)
        elif self.fusion_mode == "attn_pool":
            # Attention pooling across time
            attn_weights = self.attn_pool(x).squeeze(-1)  # (B, k, N)
            attn_weights = F.softmax(attn_weights, dim=1)  # (B, k, N)
            fused = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, N, D)
        else:
            fused = x[:, -1, :, :]

        return fused

    def compute_delta_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Explicitly compute frame difference features.
        Args:
            frames: (B, k, N, D)
        Returns:
            Delta features: (B, k-1, N, D)
        """
        if not self.use_delta_features:
            raise RuntimeError("Delta features not enabled. Set use_delta_features=True")

        return self.delta_extractor(frames)

    def positional_encoding_temporal(self, k: int) -> torch.Tensor:
        """
        Get temporal positional encoding for k frames.
        Returns: (1, k, 1, D)
        """
        return self.temporal_pos_enc.pos_embedding[:, :k, :, :]


class CrossFrameAttention(nn.Module):
    """
    Alternative: Cross-frame attention where each frame attends to all others.
    More efficient for large k.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, k, N, D)
        Returns:
            Fused: (B, N, D)
        """
        B, k, N, D = x.shape

        # Reshape to (B*N, k, D) - each patch attends across time
        x = x.permute(0, 2, 1, 3).reshape(B * N, k, D)

        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x, _ = attn(x, x, x)
            x = x + residual

        # Pool across time (mean)
        x = x.mean(dim=1)  # (B*N, D)
        x = x.view(B, N, D)

        return x


def create_temporal_fusion(config: Dict) -> TemporalFusion:
    """Factory function to create temporal fusion from config."""
    return TemporalFusion(
        embed_dim=config.get("embed_dim", 384),
        num_heads=config.get("temporal", {}).get("num_heads", 8),
        num_layers=config.get("temporal", {}).get("num_layers", 3),
        max_frames=config.get("temporal", {}).get("max_frames", 8),
        dropout=config.get("temporal", {}).get("dropout", 0.1),
        use_delta_features=config.get("temporal", {}).get("use_delta", True),
        fusion_mode=config.get("temporal", {}).get("fusion_mode", "last"),
    )