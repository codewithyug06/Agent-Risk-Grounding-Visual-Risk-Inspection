"""
Ablation experiments for SENTINEL-Vision.
1. Temporal window size: k ∈ {1, 2, 4, 6, 8}
2. With gate vs. raw threshold
3. Real-only vs. real+synthetic training data
4. Backbone: ViT-S vs. ConvNeXt-Tiny vs. DINOv2
5. Cross-agent: train on Mind2Web → test on ScreenSpot/AgentTrek
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import json
from datetime import datetime

from ..models.sentinel_model import SentinelModel, create_sentinel_model
from ..models.frame_encoder import create_frame_encoder
from ..data.loaders import SentinelDataset
from ..data.augmentation import create_val_transform
from ..data.frame_windowing import collate_frame_windows
from .metrics import compute_safety_metrics, compute_localization_iou, compute_latency
from ..gate.decision_gate import DecisionGate
from ..utils.checkpoint import load_model_state_dict, CheckpointLoadError
from ..utils.constants import DEFAULT_SENTINEL_CHECKPOINT, DEFAULT_GATE_CHECKPOINT

logger = logging.getLogger(__name__)


def _load_state_dict_safe(path, device):
    try:
        return load_model_state_dict(path, map_location=device)
    except CheckpointLoadError:
        return load_model_state_dict(path, map_location=device, allow_unsafe=True)


class AblationRunner:
    """Runs all ablation experiments."""

    def __init__(self, config: DictConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.results = {}

    def run_all(self) -> Dict[str, Any]:
        """Run all 5 ablations."""
        logger.info("Starting ablation experiments...")

        # 1. Temporal window size ablation
        logger.info("Ablation 1: Temporal window size")
        self.results["temporal_window"] = self._ablation_temporal_window()

        # 2. Gate vs threshold
        logger.info("Ablation 2: Gate vs raw threshold")
        self.results["gate_vs_threshold"] = self._ablation_gate_vs_threshold()

        # 3. Real vs Real+Synthetic
        logger.info("Ablation 3: Real vs Real+Synthetic data")
        self.results["real_vs_synthetic"] = self._ablation_real_vs_synthetic()

        # 4. Backbone comparison
        logger.info("Ablation 4: Backbone comparison")
        self.results["backbone"] = self._ablation_backbone()

        # 5. Cross-agent generalization
        logger.info("Ablation 5: Cross-agent generalization")
        self.results["cross_agent"] = self._ablation_cross_agent()

        return self.results

    def _ablation_temporal_window(self) -> Dict[str, Any]:
        """Ablation 1: Vary temporal window size k ∈ {1, 2, 4, 6, 8}."""
        k_values = [1, 2, 4, 6, 8]
        results = {}

        for k in k_values:
            logger.info(f"  Testing k={k}")

            # Create dataset with this k
            val_transform = create_val_transform(OmegaConf.to_container(self.config))
            val_dataset = SentinelDataset(
                data_config=OmegaConf.to_container(self.config),
                split="val",
                transform=val_transform,
                frame_window_k=k,
                target_resolution=tuple(self.config.frame_window.resolution),
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.get("batch_size", 16),
                shuffle=False,
                num_workers=self.config.training.get("num_workers", 4),
                pin_memory=True,
                collate_fn=collate_frame_windows,
            )

            # NOTE: this is an approximation, not a true per-k ablation. A
            # rigorous version would retrain a separate model for each k
            # (temporal positional embeddings and the temporal-fusion
            # module were fit to k=6). Here we evaluate the *same* k=6
            # checkpoint fed truncated/padded windows of length k, which
            # measures "how much the trained model degrades outside its
            # training window," not "what a model trained natively at that
            # k would achieve." Both numbers are useful but not the same
            # claim -- label results accordingly.
            model = create_sentinel_model(self.config)
            state_dict = _load_state_dict_safe(
                self.config.get("checkpoint", DEFAULT_SENTINEL_CHECKPOINT), self.device
            )
            model.load_state_dict(state_dict)
            model = model.to(self.device).eval()

            # Evaluate
            all_scores, all_labels = self._evaluate_loader(model, val_loader)
            safety = compute_safety_metrics(all_scores, all_labels)

            results[f"k_{k}"] = {
                "k": k,
                "recall": safety["recall"],
                "precision": safety["precision"],
                "f1": safety["f1"],
                "fnr": safety["false_negative_rate"],
                "fpr": safety["false_positive_rate"],
                "caveat": "evaluated with the k=6-trained checkpoint fed k-length windows, not a model natively trained at this k",
            }

        return results

    def _ablation_gate_vs_threshold(self) -> Dict[str, Any]:
        """Ablation 2: Compare PPO gate vs raw threshold at various operating points."""
        from ..gate.decision_gate import DecisionGate
        from ..gate.reward import AsymmetricReward

        # Load model
        model = create_sentinel_model(self.config)
        state_dict = _load_state_dict_safe(
            self.config.get("checkpoint", DEFAULT_SENTINEL_CHECKPOINT), self.device
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device).eval()

        # Create validation loader
        val_transform = create_val_transform(OmegaConf.to_container(self.config))
        val_dataset = SentinelDataset(
            data_config=OmegaConf.to_container(self.config),
            split="val",
            transform=val_transform,
            frame_window_k=self.config.frame_window.k,
            target_resolution=tuple(self.config.frame_window.resolution),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.get("batch_size", 16),
            shuffle=False,
            num_workers=self.config.training.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_frame_windows,
        )

        # Evaluate with different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold_results = {}

        for thresh in thresholds:
            all_scores, all_labels = self._evaluate_loader(model, val_loader)
            preds = (all_scores > thresh).astype(int)
            safety = compute_safety_metrics(preds, all_labels)
            threshold_results[f"thresh_{thresh}"] = safety

        # Evaluate with PPO gate if available
        gate_results = {}
        gate_path = self.config.get("gate_checkpoint", DEFAULT_GATE_CHECKPOINT)
        if Path(gate_path).exists():
            logger.info("  Evaluating with PPO gate")
            gate = DecisionGate()
            gate_state = _load_state_dict_safe(gate_path, self.device)
            gate.load_state_dict(gate_state)
            gate = gate.to(self.device).eval()

            gate_safety = self._evaluate_with_gate(model, gate, val_loader)
            gate_results["ppo_gate"] = gate_safety

        return {
            "threshold_sweep": threshold_results,
            "gate_results": gate_results,
        }

    def _ablation_real_vs_synthetic(self) -> Dict[str, Any]:
        """Ablation 3: Train on real-only vs real+synthetic data."""
        # This would require training separate models
        # For now, evaluate existing model on real vs synthetic test splits
        logger.info("  Evaluating on real vs synthetic test subsets")

        val_transform = create_val_transform(OmegaConf.to_container(self.config))

        # Real test set (from original datasets)
        real_dataset = SentinelDataset(
            data_config=OmegaConf.to_container(self.config),
            split="val",
            transform=val_transform,
            frame_window_k=self.config.frame_window.k,
            target_resolution=tuple(self.config.frame_window.resolution),
            # Filter to real-only (this would need dataset modification)
        )

        # Synthetic test set
        # synthetic_dataset = ...

        # NOT IMPLEMENTED. This requires training two separate models (one
        # real-only, one real+synthetic) and comparing their validation
        # metrics -- out of scope for a single-checkpoint evaluation run.
        # Reporting is deliberately explicit rather than a silent stub, so a
        # results table built from this JSON can't accidentally present
        # placeholder text as a numeric result (see README/system_report
        # truth-pass note: earlier versions of this repo's docs presented
        # numbers here that this code cannot actually produce).
        return {
            "status": "not_implemented",
            "real_only": None,
            "real_plus_synthetic": None,
            "reason": "Requires training two separate models (real-only vs real+synthetic) and comparing validation metrics.",
        }

    def _ablation_backbone(self) -> Dict[str, Any]:
        """Ablation 4: Compare backbones - ViT-S, ConvNeXt-Tiny, DINOv2."""
        backbones = [
            ("vit_small_patch16_224", "ViT-S/16"),
            ("convnext_tiny", "ConvNeXt-Tiny"),
            ("dino_vits14", "DINOv2 ViT-S/14"),
        ]

        results = {}

        for backbone_name, display_name in backbones:
            logger.info(f"  Testing {display_name}")

            # Create model with this backbone
            config = OmegaConf.to_container(self.config)
            config["backbone"] = backbone_name

            model = create_sentinel_model(OmegaConf.create(config))
            model = model.to(self.device)

            # Check if checkpoint exists for this backbone
            ckpt_path = Path(f"checkpoints/{backbone_name}/best.pt")
            if ckpt_path.exists():
                state_dict = _load_state_dict_safe(ckpt_path, self.device)
                model.load_state_dict(state_dict)
                model.eval()

                # Evaluate
                val_transform = create_val_transform(config)
                val_dataset = SentinelDataset(
                    data_config=config,
                    split="val",
                    transform=val_transform,
                    frame_window_k=self.config.frame_window.k,
                    target_resolution=tuple(self.config.frame_window.resolution),
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.training.get("batch_size", 16),
                    shuffle=False,
                    num_workers=self.config.training.get("num_workers", 4),
                    pin_memory=True,
                    collate_fn=collate_frame_windows,
                )

                all_scores, all_labels = self._evaluate_loader(model, val_loader)
                safety = compute_safety_metrics(all_scores, all_labels)

                # Latency
                sample = torch.randn(1, self.config.frame_window.k, 3,
                                   self.config.frame_window.resolution[0],
                                   self.config.frame_window.resolution[1])
                latency = compute_latency(model, sample, n_runs=50, device=self.device)

                results[display_name] = {
                    "backbone": backbone_name,
                    "recall": safety["recall"],
                    "precision": safety["precision"],
                    "f1": safety["f1"],
                    "fnr": safety["false_negative_rate"],
                    "fpr": safety["false_positive_rate"],
                    "latency_ms": latency["mean_ms"],
                    "params": sum(p.numel() for p in model.parameters()),
                }
            else:
                logger.warning(f"  No checkpoint found for {display_name}")
                results[display_name] = {
                    "status": "not_run",
                    "reason": f"No checkpoint found at {ckpt_path} -- this backbone was never trained in this repo.",
                }

        return results

    def _ablation_cross_agent(self) -> Dict[str, Any]:
        """Ablation 5: Cross-agent generalization."""
        # Train on Mind2Web, test on ScreenSpot/AgentTrek
        domains = ["multimodal_mind2web", "screenspot", "agenttrek"]
        results = {}

        for train_domain in domains:
            for test_domain in domains:
                if train_domain == test_domain:
                    continue

                key = f"{train_domain}_to_{test_domain}"
                logger.info(f"  {key}")

                # NOT IMPLEMENTED. This requires per-domain dataset splits
                # and (for the "train on domain A" half) training a model
                # exclusively on that domain, then evaluating on the held-out
                # domain. Neither exists in this repo yet -- reported
                # explicitly as not_implemented rather than a silent
                # placeholder dict, since earlier docs in this repo
                # presented specific accuracy numbers for exactly this
                # ablation that this code was never able to produce.
                results[key] = {
                    "status": "not_implemented",
                    "reason": "Requires domain-specific dataset splits and a model trained exclusively on train_domain.",
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                }

        return results

    def _evaluate_loader(self, model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate model on loader, return scores and labels."""
        all_scores = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                frames = batch["frames"]
                output = model(frames)

                scores = output["risk_score"].squeeze(-1).cpu().numpy()
                labels = batch["risk_label"].cpu().numpy()

                all_scores.append(scores)
                all_labels.append(labels)

        return np.concatenate(all_scores), np.concatenate(all_labels)

    def _evaluate_with_gate(
        self,
        model: nn.Module,
        gate: DecisionGate,
        loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate with PPO gate."""
        all_preds = []
        all_labels = []

        model.eval()
        gate.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                frames = batch["frames"]
                output = model(frames)

                for i in range(frames.shape[0]):
                    risk = output["risk_score"][i].item()
                    cat = output["category_idx"][i].item()
                    conf = output["objectness"][i].item()

                    decision = gate.get_action(risk, cat, conf, deterministic=True)
                    pred = 1 if decision in ["PAUSE", "HARD_BLOCK"] else 0
                    all_preds.append(pred)

                all_labels.extend(batch["risk_label"].cpu().numpy())

        return compute_safety_metrics(np.array(all_preds), np.array(all_labels))

    def save_results(self, output_dir: str = "results/ablations"):
        """Save ablation results."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(output_dir) / f"ablations_{timestamp}.json"

        with open(file_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Ablation results saved to: {file_path}")

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print formatted ablation summary."""
        print("\n" + "=" * 80)
        print("ABLATION SUMMARY")
        print("=" * 80)

        # Temporal window
        if "temporal_window" in self.results:
            print("\n1. TEMPORAL WINDOW SIZE:")
            for k, v in self.results["temporal_window"].items():
                print(f"   {k}: Recall={v.get('recall', 0):.4f}, F1={v.get('f1', 0):.4f}, "
                      f"FNR={v.get('fnr', 0):.4f}, FPR={v.get('fpr', 0):.4f}")

        # Backbone
        if "backbone" in self.results:
            print("\n2. BACKBONE COMPARISON:")
            for name, v in self.results["backbone"].items():
                if "recall" in v:
                    print(f"   {name}: Recall={v['recall']:.4f}, F1={v['f1']:.4f}, "
                          f"Latency={v.get('latency_ms', 0):.1f}ms, "
                          f"Params={v.get('params', 0)/1e6:.1f}M")


def run_ablations(config: DictConfig) -> Dict[str, Any]:
    """Main entry point for ablations."""
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Ablation Experiments")
    logger.info("=" * 60)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    runner = AblationRunner(config, device)
    results = runner.run_all()
    runner.save_results()

    return results


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
    def main(config: DictConfig):
        run_ablations(config)

    main()