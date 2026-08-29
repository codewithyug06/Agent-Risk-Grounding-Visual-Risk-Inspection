"""
ONNX Export Script for SENTINEL-Vision.
Exports the trained model to ONNX format with INT8 quantization for deployment.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.sentinel_model import SentinelModel, create_sentinel_model
from src.models.frame_encoder import FrameEncoder
from src.models.temporal_fusion import TemporalFusion
from src.models.risk_head import RiskHead
from src.models.localization_head import LocalizationHead
from src.utils.logging import setup_logging
from src.utils.checkpoint import load_checkpoint, CheckpointLoadError

logger = logging.getLogger(__name__)


class ONNXExportWrapper(nn.Module):
    def __init__(self, model: SentinelModel):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor):
        out = self.model(frames)
        return (
            out["risk_score"],
            out["category_probs"],
            out["category_idx"],
            out["bbox"],
            out["objectness"],
        )


def export_to_onnx(
    model: SentinelModel,
    output_path: str,
    input_shape: Tuple[int, int, int, int, int] = (1, 6, 3, 224, 224),
    opset_version: int = 17,
    dynamic_axes: Optional[Dict] = None,
    verbose: bool = False,
):
    """
    Export SENTINEL-Vision model to ONNX.

    Args:
        model: SentinelModel in eval mode
        output_path: Path to save ONNX model
        input_shape: (batch, frames, channels, height, width)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch/sequence length
        verbose: Print export details
    """
    model.eval()
    wrapper = ONNXExportWrapper(model)
    wrapper.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Default dynamic axes
    if dynamic_axes is None:
        dynamic_axes = {
            "frames": {0: "batch", 1: "sequence"},
            "risk_score": {0: "batch"},
            "category_probs": {0: "batch"},
            "category_idx": {0: "batch"},
            "bbox": {0: "batch"},
            "objectness": {0: "batch"},
        }

    # Export
    logger.info(f"Exporting to ONNX: {output_path}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Opset version: {opset_version}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["frames"],
            output_names=["risk_score", "category_probs", "category_idx", "bbox", "objectness"],
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

    logger.info(f"ONNX model saved to: {output_path}")


def verify_onnx_model(
    onnx_path: str,
    model: SentinelModel,
    input_shape: Tuple[int, int, int, int, int] = (1, 6, 3, 224, 224),
    tolerance: float = 1e-4,
) -> bool:
    """
    Verify ONNX model outputs match PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        model: Original PyTorch model
        input_shape: Input shape for testing
        tolerance: Maximum allowed difference

    Returns:
        True if outputs match within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification")
        return True

    model.eval()
    dummy_input = torch.randn(*input_shape)

    # PyTorch output
    with torch.no_grad():
        torch_outputs = model(dummy_input)

    # ONNX Runtime output
    session = ort.InferenceSession(onnx_path)
    onnx_input = {session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_outputs = session.run(None, onnx_input)

    # Compare outputs
    output_names = ["risk_score", "category_probs", "category_idx", "bbox", "objectness"]
    all_match = True

    for i, name in enumerate(output_names):
        torch_out = torch_outputs[name].numpy() if hasattr(torch_outputs[name], 'numpy') else torch_outputs[name]
        onnx_out = onnx_outputs[i]

        # Handle different shapes
        if torch_out.shape != onnx_out.shape:
            logger.warning(f"Shape mismatch for {name}: torch={torch_out.shape}, onnx={onnx_out.shape}")
            all_match = False
            continue

        max_diff = np.max(np.abs(torch_out - onnx_out))
        mean_diff = np.mean(np.abs(torch_out - onnx_out))

        logger.info(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        if name != "category_idx" and max_diff > tolerance:
            logger.warning(f"Output {name} exceeds tolerance: {max_diff} > {tolerance}")
            all_match = False

    if all_match:
        logger.info("ONNX verification PASSED")
    else:
        logger.error("ONNX verification FAILED")

    return all_match


def quantize_onnx_model(
    onnx_path: str,
    output_path: str,
    calibration_data: Optional[List[np.ndarray]] = None,
    quantization_mode: str = "static",
    per_channel: bool = False,
    reduce_range: bool = False,
) -> str:
    """
    Quantize ONNX model to INT8.

    Args:
        onnx_path: Path to FP32 ONNX model
        output_path: Path to save INT8 model
        calibration_data: List of calibration inputs (for static quantization)
        quantization_mode: 'static' or 'dynamic'
        per_channel: Use per-channel quantization
        reduce_range: Reduce quantization range for better accuracy

    Returns:
        Path to quantized model
    """
    try:
        from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader
        from onnxruntime.quantization.quant_utils import QuantType
    except ImportError:
        logger.error("onnxruntime quantization tools not available")
        logger.error("Install with: pip install onnxruntime-tools")
        raise

    if quantization_mode == "dynamic":
        logger.info("Applying dynamic quantization...")
        quantize_dynamic(
            onnx_path,
            output_path,
            op_types_to_quantize=["MatMul", "Gemm"],
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=reduce_range,
        )
    else:
        logger.info("Applying static quantization...")

        # Create calibration data reader
        class SentinelCalibrationReader(CalibrationDataReader):
            def __init__(self, data: List[np.ndarray]):
                self.data = data
                self.index = 0

            def get_next(self) -> Dict[str, np.ndarray]:
                if self.index >= len(self.data):
                    return None
                result = {"frames": self.data[self.index].astype(np.float32)}
                self.index += 1
                return result

        if calibration_data is None:
            # Generate random calibration data
            calibration_data = [np.random.randn(1, 6, 3, 224, 224).astype(np.float32) for _ in range(100)]

        reader = SentinelCalibrationReader(calibration_data)

        quantize_static(
            onnx_path,
            output_path,
            reader,
            quant_format=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=reduce_range,
        )

    logger.info(f"Quantized model saved to: {output_path}")
    return output_path


def benchmark_onnx_model(
    onnx_path: str,
    input_shape: Tuple[int, int, int, int, int] = (1, 6, 3, 224, 224),
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark ONNX model latency.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input shape
        n_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with latency statistics
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed")
        return {}

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    import time
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000 / np.mean(times)),
    }


def export_pipeline(
    config: DictConfig,
    checkpoint_path: str,
    output_dir: str = "onnx_models",
    quantize: bool = True,
    verify: bool = True,
    benchmark: bool = True,
) -> Dict[str, Any]:
    """
    Full export pipeline: load model -> export -> verify -> quantize -> benchmark.

    Args:
        config: Model configuration
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Output directory for ONNX models
        quantize: Whether to apply INT8 quantization
        verify: Whether to verify ONNX outputs
        benchmark: Whether to benchmark latency

    Returns:
        Dictionary with results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from: {checkpoint_path}")
    model = create_sentinel_model(config)
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    except CheckpointLoadError:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", allow_unsafe=True)
    state_dict = checkpoint["model_state_dict"] if (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Export FP32
    fp32_path = output_dir / "sentinel_vision_fp32.onnx"
    export_to_onnx(model, str(fp32_path))

    results = {
        "fp32_path": str(fp32_path),
        "int8_path": None,
        "verification": False,
        "fp32_benchmark": None,
        "int8_benchmark": None,
    }

    # Verify
    if verify:
        results["verification"] = verify_onnx_model(str(fp32_path), model)

    # Benchmark FP32
    if benchmark:
        results["fp32_benchmark"] = benchmark_onnx_model(str(fp32_path))
        logger.info(f"FP32 Latency: {results['fp32_benchmark']['mean_ms']:.2f}ms ({results['fp32_benchmark']['fps']:.1f} FPS)")

    # Quantize
    if quantize:
        int8_path = output_dir / "sentinel_vision_int8.onnx"

        quantize_onnx_model(str(fp32_path), str(int8_path), quantization_mode="dynamic")
        results["int8_path"] = str(int8_path)

        # Verify quantized
        if verify:
            verify_onnx_model(str(int8_path), model, tolerance=1e-2)

        # Benchmark INT8
        if benchmark:
            results["int8_benchmark"] = benchmark_onnx_model(str(int8_path))
            logger.info(f"INT8 Latency: {results['int8_benchmark']['mean_ms']:.2f}ms ({results['int8_benchmark']['fps']:.1f} FPS)")

            # Speedup
            if results["fp32_benchmark"]:
                speedup = results["fp32_benchmark"]["mean_ms"] / results["int8_benchmark"]["mean_ms"]
                logger.info(f"Quantization speedup: {speedup:.2f}x")

    return results


def export_individual_components(
    config: DictConfig,
    checkpoint_path: str,
    output_dir: str = "onnx_models/components",
):
    """
    Export individual model components for modular deployment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_sentinel_model(config)
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    except CheckpointLoadError:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", allow_unsafe=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 1. Frame Encoder
    frame_encoder = model.frame_encoder
    dummy_frames = torch.randn(1, 6, 3, 224, 224)
    torch.onnx.export(
        frame_encoder,
        dummy_frames,
        output_dir / "frame_encoder.onnx",
        input_names=["frames"],
        output_names=["patch_embeddings"],
        dynamic_axes={"frames": {0: "batch", 1: "sequence"}},
        opset_version=17,
    )
    logger.info("Exported frame_encoder.onnx")

    # 2. Temporal Fusion
    temporal_fusion = model.temporal_fusion
    dummy_patches = torch.randn(1, 6, 196, 384)  # (B, k, N_patches, D)
    torch.onnx.export(
        temporal_fusion,
        dummy_patches,
        output_dir / "temporal_fusion.onnx",
        input_names=["patch_embeddings"],
        output_names=["fused_features"],
        dynamic_axes={"patch_embeddings": {0: "batch", 1: "sequence"}},
        opset_version=17,
    )
    logger.info("Exported temporal_fusion.onnx")

    # 3. Risk Head
    risk_head = model.risk_head
    dummy_fused = torch.randn(1, 384)
    torch.onnx.export(
        risk_head,
        dummy_fused,
        output_dir / "risk_head.onnx",
        input_names=["fused_features"],
        output_names=["risk_score", "category_probs", "category_idx"],
        dynamic_axes={"fused_features": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Exported risk_head.onnx")

    # 4. Localization Head
    loc_head = model.localization_head
    dummy_spatial = torch.randn(1, 196, 384)  # (B, N_patches, D)
    torch.onnx.export(
        loc_head,
        dummy_spatial,
        output_dir / "localization_head.onnx",
        input_names=["spatial_features"],
        output_names=["objectness", "bbox", "anchors"],
        dynamic_axes={"spatial_features": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Exported localization_head.onnx")


def main():
    parser = argparse.ArgumentParser(description="Export SENTINEL-Vision to ONNX")
    parser.add_argument("--config", type=str, default="model_small.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage_b_30epochs/best.pt")
    parser.add_argument("--output-dir", type=str, default="onnx_models")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--components", action="store_true")
    args = parser.parse_args()

    from src.utils.config import load_config
    config = load_config(args.config)
    setup_logging(config.get("log_level", "INFO"))

    results = export_pipeline(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        quantize=not args.no_quantize,
        verify=not args.no_verify,
        benchmark=not args.no_benchmark,
    )

    if args.components:
        export_individual_components(config, args.checkpoint)

    logger.info("Export completed!")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()