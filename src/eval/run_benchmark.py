"""
Full benchmark runner for SENTINEL-Vision.
Compares against baselines: single-frame, random threshold, rule-based OCR.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig, OmegaConf
import hydra
from datetime import datetime

from ..models.sentinel_model import SentinelModel, create_sentinel_model
from ..data.loaders import SentinelDataset, _weak_label_from_action
from ..data.augmentation import create_val_transform
from ..data.frame_windowing import collate_frame_windows
from ..gate.decision_gate import DecisionGate
from ..gate.reward import AsymmetricReward
from ..utils.checkpoint import load_checkpoint, load_model_state_dict, CheckpointLoadError
from ..utils.constants import DEFAULT_SENTINEL_CHECKPOINT, DEFAULT_GATE_CHECKPOINT
from .metrics import (
    compute_safety_metrics,
    compute_localization_iou,
    compute_latency,
    compute_precision_recall_curve,
    compute_confusion_matrix,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs full benchmark suite comparing SENTINEL-Vision against baselines.
    """

    def __init__(
        self,
        config: DictConfig,
        model: SentinelModel,
        gate: Optional[DecisionGate] = None,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device).eval()
        self.gate = gate.to(device).eval() if gate else None
        self.device = device

        # Create validation dataset
        val_transform = create_val_transform(OmegaConf.to_container(config))
        self.val_dataset = SentinelDataset(
            data_config=OmegaConf.to_container(config),
            split="val",
            transform=val_transform,
            frame_window_k=config.frame_window.k,
            target_resolution=tuple(config.frame_window.resolution),
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.get("batch_size", 16),
            shuffle=False,
            num_workers=config.training.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_frame_windows,
        )

        logger.info(f"BenchmarkRunner initialized: {len(self.val_dataset)} validation samples")

    def run_all(self, quick: bool = False) -> Dict[str, Any]:
        """Run all benchmarks."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": OmegaConf.to_container(self.config, resolve=True),
            "models": {},
        }

        # 1. Full SENTINEL-Vision (with gate if available)
        logger.info("Running SENTINEL-Vision (full)...")
        results["models"]["sentinel_vision"] = self._evaluate_model(
            "SENTINEL-Vision",
            use_gate=self.gate is not None,
        )

        # 2. Single-frame baseline (k=1)
        logger.info("Running single-frame baseline...")
        results["models"]["single_frame"] = self._evaluate_single_frame()

        # 3. Random threshold baseline
        logger.info("Running random threshold baseline...")
        results["models"]["random_threshold"] = self._evaluate_random_threshold()

        # 4. Rule-based OCR baseline (if available)
        logger.info("Running rule-based baseline...")
        results["models"]["rule_based"] = self._evaluate_rule_based()

        # Latency benchmark
        logger.info("Running latency benchmark...")
        results["latency"] = self._benchmark_latency()

        # Summary table
        results["summary"] = self._create_summary_table(results["models"])

        return results

    def _evaluate_model(
        self,
        name: str,
        use_gate: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a model configuration."""
        all_preds = {"risk_score": [], "category_probs": [], "bbox": [], "objectness": []}
        all_targets = {"risk_label": [], "category_label": [], "bbox": [], "has_bbox": []}

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                frames = batch["frames"]

                # Model forward
                output = self.model(frames)

                # Gate decision if available
                if use_gate and self.gate:
                    gate_decisions = []
                    for i in range(frames.shape[0]):
                        risk = output["risk_score"][i].item()
                        cat = output["category_idx"][i].item()
                        conf = output["objectness"][i].item()
                        decision = self.gate.get_action(risk, cat, conf, deterministic=True)
                        gate_decisions.append(decision)

                    output["gate_decision"] = gate_decisions

                # Collect
                all_preds["risk_score"].append(output["risk_score"].cpu())
                all_preds["category_probs"].append(output["category_probs"].cpu())
                all_preds["bbox"].append(output["bbox"].cpu())
                all_preds["objectness"].append(output["objectness"].cpu())

                all_targets["risk_label"].append(batch["risk_label"].cpu())
                all_targets["category_label"].append(batch["category_label"].cpu())
                all_targets["bbox"].append(batch["bbox"].cpu())
                all_targets["has_bbox"].append(batch["has_bbox"].cpu())

        # Concatenate
        for k in all_preds:
            all_preds[k] = torch.cat(all_preds[k], dim=0)
        for k in all_targets:
            all_targets[k] = torch.cat(all_targets[k], dim=0)

        # Compute metrics
        risk_scores = all_preds["risk_score"].squeeze(-1).numpy()
        risk_labels = all_targets["risk_label"].numpy()

        safety_metrics = compute_safety_metrics(risk_scores, risk_labels)

        # Localization (only harmful with bbox)
        has_bbox = all_targets["has_bbox"] > 0.5
        harmful_mask = (all_targets["risk_label"] == 1) & has_bbox

        loc_iou = 0.0
        if harmful_mask.any():
            loc_metrics = compute_localization_iou(
                all_preds["bbox"][harmful_mask].numpy(),
                all_targets["bbox"][harmful_mask].numpy(),
            )
            loc_iou = loc_metrics["mean_iou"]

        # PR curve
        pr_curve = compute_precision_recall_curve(risk_scores, risk_labels)

        # Confusion matrix for categories
        cat_probs = all_preds["category_probs"].numpy()
        cat_labels = all_targets["category_label"].numpy()
        cm = compute_confusion_matrix(cat_probs, cat_labels, num_classes=5)

        return {
            "safety_metrics": safety_metrics,
            "localization_iou": float(loc_iou),
            "pr_auc": pr_curve["auc"],
            "confusion_matrix": cm.tolist(),
            "category_names": ["destructive", "financial", "privacy", "irreversible_external", "benign"],
        }

    def _evaluate_single_frame(self) -> Dict[str, Any]:
        """Evaluate single-frame baseline (k=1)."""
        # Create dataset with k=1
        val_transform = create_val_transform(OmegaConf.to_container(self.config))
        val_dataset_k1 = SentinelDataset(
            data_config=OmegaConf.to_container(self.config),
            split="val",
            transform=val_transform,
            frame_window_k=1,
            target_resolution=tuple(self.config.frame_window.resolution),
        )

        val_loader_k1 = DataLoader(
            val_dataset_k1,
            batch_size=self.config.training.get("batch_size", 16),
            shuffle=False,
            num_workers=self.config.training.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_frame_windows,
        )

        # We need a model that accepts k=1, or we just take first frame
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader_k1:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                frames = batch["frames"]  # (B, 1, C, H, W)
                frames = frames[:, 0:1, :, :, :]  # Ensure k=1

                output = self.model(frames)
                all_scores.append(output["risk_score"].cpu())
                all_labels.append(batch["risk_label"].cpu())

        all_scores = torch.cat(all_scores).squeeze(-1).numpy()
        all_labels = torch.cat(all_labels).numpy()

        return {"safety_metrics": compute_safety_metrics(all_scores, all_labels)}

    def _evaluate_random_threshold(self) -> Dict[str, Any]:
        """Evaluate random threshold baseline (random predictions)."""
        np.random.seed(42)
        all_labels = []

        for batch in self.val_loader:
            all_labels.append(batch["risk_label"].numpy())

        all_labels = np.concatenate(all_labels)
        random_scores = np.random.rand(len(all_labels))

        return {"safety_metrics": compute_safety_metrics(random_scores, all_labels)}

    def _evaluate_rule_based(self) -> Dict[str, Any]:
        """
        Evaluate a real (not simulated) rule-based baseline: run the same
        keyword-cue matcher used for weak-labeling actions
        (`_weak_label_from_action`, src/data/loaders.py) against each
        sample's action text and score it as harmful (1.0) or benign (0.0).

        IMPORTANT CAVEAT: this dataset's own risk_label ground truth is
        *itself* generated by `_weak_label_from_action` (see
        configs/data.yaml -- synthetic Playwright injection was removed and
        weak-labels come from keyword matching). That makes this baseline
        partially circular: it will score close to the ground truth by
        construction, not because keyword matching is actually a strong
        real-world detector. Report this number as "keyword-cue baseline
        (uses the same heuristic as the dataset's own weak labels)", not as
        an independent OCR system -- a true OCR-based baseline would need to
        run OCR on `batch["frames"]` pixels and is not implemented here.
        """
        all_labels = []
        rule_scores = []

        for batch in self.val_loader:
            labels = batch["risk_label"].numpy()
            all_labels.append(labels)

            actions = batch.get("actions", [""] * len(labels))
            for action_text in actions:
                pred_label, _ = _weak_label_from_action(action_text or "")
                rule_scores.append(float(pred_label))

        all_labels = np.concatenate(all_labels)
        rule_scores = np.array(rule_scores[: len(all_labels)])

        result = compute_safety_metrics(rule_scores, all_labels)
        return {
            "safety_metrics": result,
            "caveat": (
                "Keyword-cue baseline using the same heuristic that generated "
                "this dataset's weak labels -- not an independent OCR system. "
                "Treat as a sanity check, not a fair external comparison."
            ),
        }

    def _benchmark_latency(self) -> Dict[str, Any]:
        """Benchmark inference latency."""
        # Create sample input
        sample = torch.randn(1, self.config.frame_window.k, 3,
                           self.config.frame_window.resolution[0],
                           self.config.frame_window.resolution[1])

        latency = compute_latency(self.model, sample, n_runs=100, device=self.device)

        return latency

    def _create_summary_table(self, model_results: Dict) -> str:
        """Create formatted summary table."""
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("BENCHMARK SUMMARY")
        lines.append("=" * 100)
        lines.append(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FNR':>10} {'FPR':>10} {'Loc IoU':>10}")
        lines.append("-" * 100)

        for name, results in model_results.items():
            sm = results.get("safety_metrics", {})
            loc = results.get("localization_iou", 0.0)
            lines.append(
                f"{name:<25} "
                f"{sm.get('precision', 0):>10.4f} "
                f"{sm.get('recall', 0):>10.4f} "
                f"{sm.get('f1', 0):>10.4f} "
                f"{sm.get('false_negative_rate', 0):>10.4f} "
                f"{sm.get('false_positive_rate', 0):>10.4f} "
                f"{loc:>10.4f}"
            )

        lines.append("=" * 100)
        return "\n".join(lines)

    def save_results(self, results: Dict, output_dir: str = "results"):
        """Save benchmark results to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"benchmark_{timestamp}.json"

        # Convert tensors/arrays to lists for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        serializable_results = make_serializable(results)

        with open(file_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Benchmark results saved to: {file_path}")
        return str(file_path)


def run_benchmark(config: DictConfig, quick: bool = False) -> Dict[str, Any]:
    """Main entry point for benchmark."""
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Benchmark")
    logger.info("=" * 60)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_path = config.get("checkpoint", DEFAULT_SENTINEL_CHECKPOINT)
    logger.info(f"Loading model from: {checkpoint_path}")

    model = create_sentinel_model(config)
    try:
        state_dict = load_model_state_dict(checkpoint_path, map_location=device)
    except CheckpointLoadError:
        state_dict = load_model_state_dict(checkpoint_path, map_location=device, allow_unsafe=True)
    model.load_state_dict(state_dict)

    # Load gate if available
    gate = None
    gate_path = config.get("gate_checkpoint", DEFAULT_GATE_CHECKPOINT)
    if Path(gate_path).exists():
        logger.info(f"Loading gate from: {gate_path}")
        gate = DecisionGate()
        try:
            gate_state = load_model_state_dict(gate_path, map_location=device)
        except CheckpointLoadError:
            gate_state = load_model_state_dict(gate_path, map_location=device, allow_unsafe=True)
        gate.load_state_dict(gate_state)

    # Run benchmark
    runner = BenchmarkRunner(config, model, gate, device)
    results = runner.run_all(quick=quick)

    # Print summary
    print(results["summary"])

    # Save
    runner.save_results(results)

    return results


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
    def main(config: DictConfig):
        run_benchmark(config)

    main()