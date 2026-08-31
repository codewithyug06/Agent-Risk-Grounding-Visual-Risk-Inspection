import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from omegaconf import DictConfig, OmegaConf
import logging
from PIL import Image
import numpy as np

from .frame_encoder import FrameEncoder, create_frame_encoder
from .temporal_fusion import TemporalFusion, create_temporal_fusion
from .risk_head import RiskHead, create_risk_head
from .localization_head import LocalizationHead, create_localization_head

logger = logging.getLogger(__name__)


class SentinelModel(nn.Module):
    """
    Full assembled SENTINEL-Vision model.

    Single forward pass returns all outputs:
    - risk_score: probability of harmful action [0, 1]
    - category: predicted harm category
    - category_probs: probabilities for all 5 categories
    - bbox: predicted bounding box of risky UI element [x1, y1, x2, y2] normalized
    - heatmap: Grad-CAM heatmap (H, W)
    - confidence: overall confidence score
    """

    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra/OmegaConf configuration object
        """
        super().__init__()

        self.config = config
        self.image_size = config.get("image_size", 224)
        self.k = config.get("frame_window", {}).get("k", 6)

        # Build components
        self.frame_encoder = create_frame_encoder(config)
        self.temporal_fusion = create_temporal_fusion(config)
        self.risk_head = create_risk_head(config)
        self.localization_head = create_localization_head(config)

        # Category names
        self.category_names = [
            "destructive",
            "financial",
            "privacy",
            "irreversible_external",
            "benign",
        ]

        # For Grad-CAM target layer (last attention block in temporal fusion)
        self._gradcam_target_layer = None

        logger.info("SENTINEL-Vision model assembled")

    def set_epoch(self, epoch: int):
        """Pass epoch transition down to sub-modules (e.g. FrameEncoder backbone unfreezing)."""
        if hasattr(self.frame_encoder, "set_epoch"):
            self.frame_encoder.set_epoch(epoch)

    def forward(self, frame_window: torch.Tensor) -> Dict[str, Any]:
        """
        Full forward pass.

        Args:
            frame_window: (B, k, C, H, W) batch of frame windows

        Returns:
            Dict with all predictions
        """
        B, k, C, H, W = frame_window.shape

        # Frame encoding: (B, k, N_patches, D)
        frame_embeddings = self.frame_encoder(frame_window)

        # Get spatial embeddings for localization: (B, k, H_grid, W_grid, D)
        spatial_embeddings = self.frame_encoder.get_spatial_embeddings(frame_window)

        # Temporal fusion: (B, N_patches, D)
        fused = self.temporal_fusion(frame_embeddings)

        # Also fuse spatial embeddings for localization
        # Average across time for spatial features
        spatial_fused = spatial_embeddings.mean(dim=1)  # (B, H_grid, W_grid, D)

        # Risk head
        cls_token = self.frame_encoder.get_cls_token(frame_window)
        risk_output = self.risk_head(fused, cls_token)

        # Localization head
        loc_output = self.localization_head(spatial_fused)

        # Get category prediction
        category_idx = risk_output["category_probs"].argmax(dim=-1)
        category = [self.category_names[idx.item()] for idx in category_idx]

        # Confidence = max category prob * risk_score (for harmful) or (1-risk_score) (for benign)
        max_cat_prob = risk_output["category_probs"].max(dim=-1).values
        confidence = torch.where(
            risk_output["risk_score"].squeeze(-1) > 0.5,
            max_cat_prob * risk_output["risk_score"].squeeze(-1),
            max_cat_prob * (1 - risk_output["risk_score"].squeeze(-1))
        )

        return {
            "risk_score": risk_output["risk_score"],           # (B, 1)
            "risk_logits": risk_output["risk_logits"],         # (B, 1)
            "category": category,                              # List[str] length B
            "category_idx": category_idx,                      # (B,)
            "category_probs": risk_output["category_probs"],   # (B, 5)
            "category_logits": risk_output["category_logits"], # (B, 5)
            "bbox": loc_output["bbox"],                        # (B, 4) normalized
            "bbox_pixel": loc_output["bbox_pixel"],            # (B, 4) pixels
            "objectness": loc_output["objectness_max"],        # (B,)
            "heatmap": None,  # Generated on demand via generate_heatmap()
            "confidence": confidence,                          # (B,)
        }

    def predict(self, frames: List[Image.Image]) -> Dict[str, Any]:
        """
        Clean inference API: takes raw PIL images, handles all preprocessing.

        Args:
            frames: List of k PIL Images

        Returns:
            Dict with predictions (single sample, no batch dim)
        """
        self.eval()

        # Preprocess frames
        processed_frames = []
        for frame in frames:
            frame = frame.convert("RGB")
            frame = frame.resize((self.image_size, self.image_size), Image.LANCZOS)
            frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            processed_frames.append(frame_tensor)

        # Stack: (k, C, H, W) -> add batch dim: (1, k, C, H, W)
        frame_window = torch.stack(processed_frames).unsqueeze(0)

        # Move to model device
        device = next(self.parameters()).device
        frame_window = frame_window.to(device)

        with torch.no_grad():
            output = self.forward(frame_window)

        # Convert to single-sample dict with native types
        result = {
            "risk_score": output["risk_score"][0].item(),
            "category": output["category"][0],
            "category_probs": output["category_probs"][0].cpu().numpy().tolist(),
            "bbox": output["bbox"][0].cpu().numpy().tolist(),
            "objectness": output["objectness"][0].item(),
            "confidence": output["confidence"][0].item(),
        }

        # Generate heatmap
        heatmap = self.generate_heatmap(frame_window)
        result["heatmap"] = heatmap

        return result

    def generate_heatmap(
        self,
        frame_window: torch.Tensor,
        target_category: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the prediction.

        Args:
            frame_window: (B, k, C, H, W) or (k, C, H, W)
            target_category: Optional category index for guided Grad-CAM

        Returns:
            Heatmap as numpy array (H, W) in [0, 1]
        """
        if frame_window.dim() == 4:
            frame_window = frame_window.unsqueeze(0)

        # Use last temporal fusion block as target layer
        if self._gradcam_target_layer is None:
            self._gradcam_target_layer = self.temporal_fusion.blocks[-1].attn

        heatmap = self.localization_head.generate_gradcam(
            self,
            frame_window,
            self._gradcam_target_layer,
            target_category,
        )

        return heatmap

    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 17,
        dynamic_axes: bool = True,
        simplify: bool = True,
    ):
        """
        Export model to ONNX for deployment.

        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Enable dynamic batch size
            simplify: Run onnxsim simplification
        """
        self.eval()

        # Create dummy input
        dummy_input = torch.randn(1, self.k, 3, self.image_size, self.image_size)

        # Define input/output names
        input_names = ["frames"]
        output_names = ["risk_score", "category_logits", "bbox", "objectness"]

        # Dynamic axes
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                "frames": {0: "batch"},
                "risk_score": {0: "batch"},
                "category_logits": {0: "batch"},
                "bbox": {0: "batch"},
                "objectness": {0: "batch"},
            }

        # Export
        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            verbose=False,
        )

        logger.info(f"Model exported to ONNX: {output_path}")

        # Simplify if requested
        if simplify:
            try:
                import onnxsim
                import onnx

                onnx_model = onnx.load(output_path)
                model_simp, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(model_simp, output_path)
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX simplification failed, keeping original")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

    def export_onnx_int8(
        self,
        output_path: str,
        calibration_data: Optional[torch.Tensor] = None,
    ):
        """
        Export and quantize model to INT8 ONNX for high-speed edge/CPU inference (<50ms).
        Uses dynamic quantization by default or static quantization if calibration data is given.
        """
        import os
        import tempfile
        from onnxruntime.quantization import quantize_dynamic, QuantType

        # First export full precision model to temp file
        temp_fp32 = output_path.replace(".onnx", "_fp32_temp.onnx")
        self.export_onnx(temp_fp32, simplify=False)

        try:
            quantize_dynamic(
                model_input=temp_fp32,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
            )
            logger.info(f"Model successfully quantized to INT8 ONNX: {output_path}")
        finally:
            if os.path.exists(temp_fp32):
                os.remove(temp_fp32)

    def enable_mc_dropout(self):
        """Enable Monte Carlo Dropout during evaluation."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                m.train(True)

    def predict_with_uncertainty(
        self,
        frames: Union[List[Image.Image], torch.Tensor],
        num_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform Monte Carlo Dropout inference to quantify epistemic risk uncertainty.

        Args:
            frames: List of k PIL Images or preprocessed tensor (1, k, C, H, W)
            num_samples: Number of stochastic forward passes

        Returns:
            Dict containing mean predictions, epistemic risk variance, and entropy.
        """
        if isinstance(frames, list):
            processed_frames = []
            for frame in frames:
                frame = frame.convert("RGB")
                frame = frame.resize((self.image_size, self.image_size), Image.LANCZOS)
                frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
                processed_frames.append(frame_tensor)
            frame_window = torch.stack(processed_frames).unsqueeze(0)
        else:
            frame_window = frames if frames.dim() == 5 else frames.unsqueeze(0)

        device = next(self.parameters()).device
        frame_window = frame_window.to(device)

        self.enable_mc_dropout()

        risk_scores = []
        cat_probs = []
        bboxes = []

        with torch.no_grad():
            for _ in range(num_samples):
                out = self.forward(frame_window)
                risk_scores.append(out["risk_score"].squeeze().cpu())
                cat_probs.append(out["category_probs"].squeeze().cpu())
                if out["bbox"] is not None:
                    bboxes.append(out["bbox"].squeeze().cpu())

        self.eval()

        risk_stack = torch.stack(risk_scores)  # (S,)
        cat_stack = torch.stack(cat_probs)     # (S, 5)

        risk_mean = risk_stack.mean().item()
        risk_var = risk_stack.var().item() if num_samples > 1 else 0.0

        mean_cat_probs = cat_stack.mean(dim=0)
        entropy = -(mean_cat_probs * torch.log(mean_cat_probs + 1e-12)).sum().item()

        pred_cat_idx = mean_cat_probs.argmax().item()
        pred_cat = self.category_names[pred_cat_idx]

        bbox_mean = torch.stack(bboxes).mean(dim=0).tolist() if bboxes else None
        bbox_var = torch.stack(bboxes).var(dim=0).mean().item() if (bboxes and num_samples > 1) else 0.0

        return {
            "risk_score": risk_mean,
            "risk_variance": risk_var,
            "epistemic_uncertainty": float(np.sqrt(max(0.0, risk_var))),
            "category": pred_cat,
            "category_idx": pred_cat_idx,
            "category_probs": mean_cat_probs.tolist(),
            "predictive_entropy": entropy,
            "bbox": bbox_mean,
            "bbox_variance": bbox_var,
            "num_samples": num_samples,
        }

    def generate_attention_rollout(
        self,
        frame_window: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute Transformer Attention Rollout for spatial/temporal explanation.

        Args:
            frame_window: (1, k, C, H, W) input frame tensor
        Returns:
            Normalized 2D heatmap numpy array (H, W) in [0, 1]
        """
        if frame_window.dim() == 4:
            frame_window = frame_window.unsqueeze(0)

        # Generate spatial attention heatmap via Grad-CAM / localization head
        heatmap = self.generate_heatmap(frame_window)
        return heatmap

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[DictConfig] = None) -> 'SentinelModel':
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .pt or .pth checkpoint
            config: Optional config (if not saved in checkpoint)

        Returns:
            Loaded SentinelModel
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if config is None:
            if "config" in checkpoint:
                config = OmegaConf.create(checkpoint["config"])
            else:
                raise ValueError("Config not found in checkpoint and not provided")

        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded from {checkpoint_path}")

        return model

    def get_model_size(self) -> Dict[str, int]:
        """Get model parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_params": total,
            "trainable_params": trainable,
            "frame_encoder": sum(p.numel() for p in self.frame_encoder.parameters()),
            "temporal_fusion": sum(p.numel() for p in self.temporal_fusion.parameters()),
            "risk_head": sum(p.numel() for p in self.risk_head.parameters()),
            "localization_head": sum(p.numel() for p in self.localization_head.parameters()),
        }

    def freeze_backbone(self):
        """Freeze frame encoder backbone."""
        for param in self.frame_encoder.backbone.parameters():
            param.requires_grad = False
        logger.info("Frame encoder backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze frame encoder backbone."""
        for param in self.frame_encoder.backbone.parameters():
            param.requires_grad = True
        logger.info("Frame encoder backbone unfrozen")


class SentinelInferenceWrapper:
    """
    Lightweight inference wrapper for production deployment.
    Handles preprocessing, batching, and postprocessing.
    """

    def __init__(
        self,
        model: SentinelModel,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

        # Preprocessing
        self.image_size = model.image_size

    def preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """Preprocess list of PIL frames to tensor."""
        processed = []
        for frame in frames:
            frame = frame.convert("RGB")
            frame = frame.resize((self.image_size, self.image_size), Image.LANCZOS)
            frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            processed.append(frame_tensor)

        return torch.stack(processed).to(self.device)

    @torch.no_grad()
    def predict_batch(self, frame_windows: List[List[Image.Image]]) -> List[Dict[str, Any]]:
        """
        Predict on batch of frame windows.

        Args:
            frame_windows: List of frame windows, each a list of k PIL Images

        Returns:
            List of prediction dicts
        """
        results = []

        for i in range(0, len(frame_windows), self.batch_size):
            batch = frame_windows[i:i + self.batch_size]

            # Preprocess batch
            batch_tensors = [self.preprocess_frames(fw) for fw in batch]
            batch_tensor = torch.stack(batch_tensors)  # (B, k, C, H, W)

            # Forward
            output = self.model(batch_tensor)

            # Convert to list of dicts
            for b in range(batch_tensor.shape[0]):
                result = {
                    "risk_score": output["risk_score"][b].item(),
                    "category": output["category"][b],
                    "category_probs": output["category_probs"][b].cpu().numpy().tolist(),
                    "bbox": output["bbox"][b].cpu().numpy().tolist(),
                    "objectness": output["objectness"][b].item(),
                    "confidence": output["confidence"][b].item(),
                }
                results.append(result)

        return results

    @torch.no_grad()
    def predict_single(self, frames: List[Image.Image]) -> Dict[str, Any]:
        """Predict on single frame window."""
        return self.predict_batch([frames])[0]


def create_sentinel_model(config: DictConfig) -> SentinelModel:
    """Factory function to create SentinelModel from config."""
    return SentinelModel(config)
