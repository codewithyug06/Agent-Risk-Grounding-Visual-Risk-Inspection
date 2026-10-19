"""
Frozen CLIP vision backbone for the detection path.

Rationale: four from-scratch-trained architectures all landed at chance-level precision on the
MisActBench gold test, which is the signature of a too-small labeled set (~1,800 real
harmful training examples) rather than an architecture problem. Freezing a web-scale
pretrained encoder and training only a small head on top is the standard remedy for
this regime (linear probes on frozen CLIP features are a strong few-shot baseline).

This backbone is used only for the risk/category path. Localization keeps its existing
FrameEncoder (src/models/frame_encoder.py), which is not data-starved the same way.
"""

import torch
import torch.nn as nn

try:
    import open_clip
except ImportError as e:  # pragma: no cover - surfaced at config time, not import time
    open_clip = None
    _IMPORT_ERROR = e


class ClipFrozenBackbone(nn.Module):
    """
    Wraps an open_clip vision tower, fully frozen, and pools a k-frame window into
    a single embedding per sample.

    Default pretrained="openai": the laion2b_s34b_b88k weights stalled twice in this
    environment's open_clip downloader (confirmed via persistent .incomplete blobs);
    the openai weights downloaded cleanly and have been verified to load + run forward.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16-quickgelu",
        pretrained: str = "openai",
        temporal_pool: str = "mean",  # "mean" or "attn"
        with_text_tower: bool = False,
    ):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open_clip_torch is required for risk_backbone=clip_frozen "
                f"(pip install open_clip_torch). Original error: {_IMPORT_ERROR}"
            )
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = model.visual
        self.embed_dim = model.visual.output_dim
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.visual.eval()

        # Optional paired text tower (see plan: and-once-again-go-velvet-fairy.md, Phase 3).
        # `create_model_and_transforms` already builds this internally; previously it was
        # discarded. Kept frozen, same as the vision tower, since there's no labeled text
        # data to fine-tune it on -- only the fusion head downstream is trainable.
        self.with_text_tower = with_text_tower
        if with_text_tower:
            self.text_model = model  # keep the full CLIP model for encode_text()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            for p in self.text_model.parameters():
                p.requires_grad_(False)
            self.text_model.eval()
        else:
            self.text_model = None
            self.tokenizer = None

        # CLIP's own normalization (differs from the rest of the pipeline's ImageNet stats).
        mean = torch.tensor(preprocess.transforms[-1].mean).view(1, 3, 1, 1)
        std = torch.tensor(preprocess.transforms[-1].std).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std", std)
        self.clip_input_size = model.visual.image_size[0] if hasattr(model.visual.image_size, "__len__") else model.visual.image_size

        self.temporal_pool = temporal_pool
        if temporal_pool == "attn":
            self.temporal_query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
            self.temporal_attn = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)

    def _clip_normalize(self, frame_window: torch.Tensor) -> torch.Tensor:
        """frame_window is already in [0,1] or ImageNet-normalized space upstream; re-normalize for CLIP."""
        x = frame_window
        if x.shape[-1] != self.clip_input_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.clip_input_size, self.clip_input_size), mode="bilinear", align_corners=False
            )
        return (x - self.clip_mean) / self.clip_std

    @torch.no_grad()
    def _encode_frames(self, frames_flat: torch.Tensor) -> torch.Tensor:
        """frames_flat: (B*k, 3, H, W) already CLIP-normalized -> (B*k, D)."""
        self.visual.eval()
        return self.visual(frames_flat)

    def forward(self, frame_window: torch.Tensor) -> torch.Tensor:
        """
        frame_window: (B, k, 3, H, W) in the pipeline's standard (already-resized) space.
        Returns: (B, embed_dim) pooled, frozen CLIP embedding over the k-frame window.
        """
        B, k, C, H, W = frame_window.shape
        flat = frame_window.reshape(B * k, C, H, W)
        flat = self._clip_normalize(flat)
        feats = self._encode_frames(flat).float()  # (B*k, D), no grad (frozen + no_grad context)
        feats = feats.reshape(B, k, self.embed_dim)

        if self.temporal_pool == "mean":
            return feats.mean(dim=1)
        query = self.temporal_query.expand(B, -1, -1)
        pooled, _ = self.temporal_attn(query, feats, feats)
        return pooled.squeeze(1)

    @torch.no_grad()
    def encode_text(self, texts: list) -> torch.Tensor:
        """
        texts: list of B raw strings (e.g. the agent's action/instruction text).
        Returns: (B, embed_dim) frozen CLIP text embedding. Empty strings get a
        zero vector (not run through the tower) so missing text is a well-defined
        all-zero signal to the fusion head rather than an arbitrary embedding of "".
        """
        if not self.with_text_tower:
            raise RuntimeError("encode_text called but with_text_tower=False")
        device = self.visual.conv1.weight.device if hasattr(self.visual, "conv1") else next(self.visual.parameters()).device
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        out = torch.zeros(len(texts), self.embed_dim, device=device)
        if non_empty:
            idxs, strs = zip(*non_empty)
            tokens = self.tokenizer(list(strs)).to(device)
            feats = self.text_model.encode_text(tokens).float()
            out[list(idxs)] = feats
        return out


class GatedTextVisionFusion(nn.Module):
    """
    Learned gated fusion of frozen CLIP image + text embeddings.

    Naive averaging/concatenation of frozen CLIP embeddings is a known-weak baseline
    for this kind of task (GatedCLIP on hateful-memes detection reports only 0.49 AUROC --
    chance level -- from simple averaging; see plan and-once-again-go-velvet-fairy.md,
    Phase 3 research notes). This learns a per-sample gate instead of a fixed mixing weight.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([image_emb, text_emb], dim=-1))
        fused = g * image_emb + (1 - g) * text_emb
        return self.out_norm(fused)


class ClipLinearRiskHead(nn.Module):
    """
    The only trainable risk-path weights when risk_backbone=clip_frozen: a small head
    on top of the frozen pooled embedding. Starts as a pure linear probe (num_layers=0),
    matching the strongest few-shot baseline in the literature; num_layers>0 adds depth.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, num_categories: int = 5,
                 num_layers: int = 0, dropout: float = 0.1):
        super().__init__()
        if num_layers <= 0:
            self.trunk = nn.LayerNorm(embed_dim)
            trunk_out = embed_dim
        else:
            layers = [nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(num_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            self.trunk = nn.Sequential(*layers)
            trunk_out = hidden_dim
        self.risk_logit = nn.Linear(trunk_out, 1)
        self.category_logits = nn.Linear(trunk_out, num_categories)

    def forward(self, pooled: torch.Tensor):
        h = self.trunk(pooled)
        return {
            "risk_logits": self.risk_logit(h),
            "risk_score": torch.sigmoid(self.risk_logit(h)),
            "category_logits": self.category_logits(h),
            "category_probs": torch.softmax(self.category_logits(h), dim=-1),
        }
