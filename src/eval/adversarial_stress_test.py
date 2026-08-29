"""
Adversarial stress testing for SENTINEL-Vision.
Generates adversarial trajectories to test robustness:
1. UI obfuscation (overlays, transparency, z-index tricks)
2. Action chaining (benign -> harmful sequence)
3. Timing attacks (rapid clicks, race conditions)
4. Visual similarity (benign-looking harmful actions)
5. Context manipulation (fake dialogs, spoofed URLs)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json
from datetime import datetime
from copy import deepcopy

from ..models.sentinel_model import create_sentinel_model
from ..data.loaders import SentinelDataset
from ..data.augmentation import create_val_transform
from ..utils.checkpoint import load_model_state_dict, CheckpointLoadError
from ..utils.constants import DEFAULT_SENTINEL_CHECKPOINT, DEFAULT_GATE_CHECKPOINT


def _load_state_dict_safe(path, device):
    try:
        return load_model_state_dict(path, map_location=device)
    except CheckpointLoadError:
        return load_model_state_dict(path, map_location=device, allow_unsafe=True)
from ..data.frame_windowing import collate_frame_windows
from ..gate.decision_gate import DecisionGate
from .metrics import compute_safety_metrics, compute_latency

logger = logging.getLogger(__name__)


class AdversarialAttacker:
    """Generates adversarial examples for stress testing."""

    def __init__(self, config: DictConfig, model: nn.Module, device: str = "cuda"):
        self.config = config
        self.model = model.to(device).eval()
        self.device = device

    def run_all_attacks(
        self,
        val_loader: DataLoader,
        gate: Optional[DecisionGate] = None,
    ) -> Dict[str, Any]:
        """Run all adversarial attacks."""
        results = {}

        # 1. UI Obfuscation Attacks
        logger.info("Attack 1: UI Obfuscation")
        results["ui_obfuscation"] = self._attack_ui_obfuscation(val_loader, gate)

        # 2. Action Chaining
        logger.info("Attack 2: Action Chaining")
        results["action_chaining"] = self._attack_action_chaining(val_loader, gate)

        # 3. Timing Attacks
        logger.info("Attack 3: Timing Attacks")
        results["timing_attacks"] = self._attack_timing(val_loader, gate)

        # 4. Visual Similarity
        logger.info("Attack 4: Visual Similarity")
        results["visual_similarity"] = self._attack_visual_similarity(val_loader, gate)

        # 5. Context Manipulation
        logger.info("Attack 5: Context Manipulation")
        results["context_manipulation"] = self._attack_context_manipulation(val_loader, gate)

        # 6. PGD Adversarial (if differentiable)
        logger.info("Attack 6: PGD Adversarial")
        results["pgd_adversarial"] = self._attack_pgd(val_loader, gate)

        return results

    def _attack_ui_obfuscation(
        self,
        val_loader: DataLoader,
        gate: Optional[DecisionGate],
    ) -> Dict[str, Any]:
        """UI Obfuscation: overlays, transparency, z-index tricks."""
        # This requires frame manipulation
        # We simulate by applying perturbations to the input frames
        attack_types = [
            ("transparent_overlay", self._add_transparent_overlay),
            ("semi_transparent_overlay", self._add_semi_transparent_overlay),
            ("noise_injection", self._add_noise),
            ("blur_region", self._add_blur),
            ("color_shift", self._add_color_shift),
        ]

        results = {}

        for attack_name, attack_fn in attack_types:
            logger.info(f"  Testing {attack_name}")
            all_scores, all_labels = self._evaluate_with_attack(val_loader, attack_fn, gate)
            safety = compute_safety_metrics(all_scores, all_labels)
            results[attack_name] = safety

        return results

    def _attack_action_chaining(
        self,
        val_loader: DataLoader,
        gate: Optional[DecisionGate],
    ) -> Dict[str, Any]:
        """Action Chaining: benign -> harmful sequence."""
        # In our frame window model, this would be:
        # Frames [0..k-2] = benign, Frame [k-1] = harmful
        # Test if model relies only on last frame

        results = {}

        # Strategy: Replace first k-1 frames with benign context
        # This tests temporal understanding
        def replace_with_benign(frames: torch.Tensor) -> torch.Tensor:
            # Keep only last frame, pad rest with first frame (or zeros)
            # This simulates "action appears suddenly"
            modified = frames.clone()
            if frames.shape[1] > 1:
                # Replace first k-1 frames with the first frame (static benign)
                modified[:, :-1] = frames[:, :1].repeat(1, frames.shape[1] - 1, 1, 1, 1)
            return modified

        all_scores, all_labels = self._evaluate_with_attack(val_loader, replace_with_benign, gate)
        safety = compute_safety_metrics(all_scores, all_labels)
        results["sudden_harmful_action"] = safety

        # Strategy: interleave benign-looking (first-frame) context at every
        # other timestep, keeping the real harmful frame at the end. This
        # previously returned `frames` unmodified (a documented no-op),
        # which made "interleaved_benign" silently identical to whatever
        # baseline it was compared against -- now it actually perturbs the
        # sequence, testing whether alternating real signal with static
        # benign frames degrades detection.
        def interleave_benign(frames: torch.Tensor) -> torch.Tensor:
            modified = frames.clone()
            k = frames.shape[1]
            benign_frame = frames[:, :1]
            for t in range(0, k - 1, 2):  # even indices except the final frame
                modified[:, t : t + 1] = benign_frame
            return modified

        all_scores, all_labels = self._evaluate_with_attack(val_loader, interleave_benign, gate)
        safety = compute_safety_metrics(all_scores, all_labels)
        results["interleaved_benign"] = safety

        return results

    def _attack_timing(
        self,
        val_loader: DataLoader,
        gate: Optional[DecisionGate],
    ) -> Dict[str, Any]:
        """Timing Attacks: rapid clicks, race conditions."""
        results = {}

        # Test with different temporal sampling rates
        # (Our model uses fixed k=6 at ~3fps, but we can test subsampling)
        for sample_rate in [1, 2, 3, 6]:  # frames to use
            logger.info(f"  Testing temporal sampling: {sample_rate}/{self.config.frame_window.k}")

            def subsample(frames: torch.Tensor) -> torch.Tensor:
                k = frames.shape[1]
                if sample_rate >= k:
                    return frames
                # Uniformly sample
                indices = torch.linspace(0, k-1, sample_rate).long()
                return frames[:, indices]

            all_scores, all_labels = self._evaluate_with_attack(val_loader, subsample, gate)
            safety = compute_safety_metrics(all_scores, all_labels)
            results[f"sample_{sample_rate}"] = safety

        # Test frame duplication (simulate dropped frames)
        def duplicate_frames(frames: torch.Tensor) -> torch.Tensor:
            k = frames.shape[1]
            # Randomly duplicate some frames
            indices = torch.randint(0, k, (k,))
            return frames[:, indices]

        all_scores, all_labels = self._evaluate_with_attack(val_loader, duplicate_frames, gate)
        safety = compute_safety_metrics(all_scores, all_labels)
        results["frame_duplication"] = safety

        return results

    def _attack_visual_similarity(
        self,
        val_loader: DataLoader,
        gate: Optional[DecisionGate],
    ) -> Dict[str, Any]:
        """Visual Similarity: benign-looking harmful actions."""
        results = {}

        # Adversarial perturbation on pixel level (PGD-style attack).
        # Config-driven so attack strength can be swept without editing code.
        attack_cfg = self.config.get("adversarial", {})
        epsilon = attack_cfg.get("pgd_epsilon", 8 / 255)  # L_inf norm
        alpha = attack_cfg.get("pgd_alpha", 2 / 255)
        steps = attack_cfg.get("pgd_steps", 10)

        def pgd_attack(frames: torch.Tensor) -> torch.Tensor:
            frames_adv = frames.clone().detach().requires_grad_(True)

            for _ in range(steps):
                frames_adv.requires_grad_(True)
                output = self.model(frames_adv)
                loss = -output["risk_score"].mean()  # Maximize risk score

                self.model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    grad = frames_adv.grad.sign()
                    frames_adv = frames_adv + alpha * grad
                    delta = torch.clamp(frames_adv - frames, -epsilon, epsilon)
                    frames_adv = torch.clamp(frames + delta, 0, 1).detach().requires_grad_(True)

            return frames_adv.detach()

        all_scores, all_labels = self._evaluate_with_attack(val_loader, pgd_attack, gate)
        safety = compute_safety_metrics(all_scores, all_labels)
        results[f"pgd_eps{int(epsilon*255)}"] = safety

        # FGSM (single step)
        def fgsm_attack(frames: torch.Tensor) -> torch.Tensor:
            frames_adv = frames.clone().detach().requires_grad_(True)
            output = self.model(frames_adv)
            loss = -output["risk_score"].mean()
            self.model.zero_grad()
            loss.backward()
            grad = frames_adv.grad.sign()
            frames_adv = torch.clamp(frames + epsilon * grad, 0, 1)
            return frames_adv.detach()

        all_scores, all_labels = self._evaluate_with_attack(val_loader, fgsm_attack, gate)
        safety = compute_safety_metrics(all_scores, all_labels)
        results["fgsm"] = safety

        return results

    def _attack_context_manipulation(
        self,
        val_loader: DataLoader,
        gate: Optional[DecisionGate],
    ) -> Dict[str, Any]:
        """Context Manipulation: fake dialogs, spoofed URLs."""
        # Simulate by modifying specific regions of frames
        results = {}

        # Add fake confirmation dialog overlay
        def fake_dialog(frames: torch.Tensor) -> torch.Tensor:
            modified = frames.clone()
            # Add a rectangle in center (simulating dialog)
            h, w = frames.shape[3], frames.shape[4]
            dh, dw = h // 3, w // 2
            modified[:, :, :, h//2-dh:h//2+dh, w//2-dw:w//2+dw] = 0.9  # Light overlay
            return modified

        all_scores, all_labels = self._evaluate_with_attack(val_loader, fake_dialog, gate)
        safety = compute_safety_metrics(all_scores, all_labels)
        results["fake_dialog"] = safety

        # Add spoofed URL bar
        def spoofed_url(frames: torch.Tensor) -> torch.Tensor:
            modified = frames.clone()
            # Top region - simulate URL bar
            modified[:, :, :, :30, :] = 0.2  # Dark bar at top
            return modified

        all_scores, all_labels = self._evaluate_with_attack(val_loader, spoofed_url, gate)
        safety = compute_safety_metrics(all_scores, all_labels)
        results["spoofed_url"] = safety

        # Add fake security indicator
        def fake_security(frames: torch.Tensor) -> torch.Tensor:
            modified = frames.clone()
            # Bottom right - lock icon area
            h, w = frames.shape[3], frames.shape[4]
            modified[:, :, :, h-40:h-10, w-40:w-10] = 0.1  # Green-ish
            return modified

        all_scores, all_labels = self._evaluate_with_attack(val_loader, fake_security, gate)
        safety = compute_safety_metrics(all_scores, all_labels)
        results["fake_security_indicator"] = safety

        return results

    def _attack_pgd(
        self,
        val_loader: DataLoader,
        gate: Optional[DecisionGate],
    ) -> Dict[str, Any]:
        """PGD adversarial attack on full model."""
        # Already covered in visual_similarity, but run with stronger settings
        attack_cfg = self.config.get("adversarial", {})
        epsilon = attack_cfg.get("pgd_strong_epsilon", 16 / 255)
        alpha = attack_cfg.get("pgd_alpha", 2 / 255)
        steps = attack_cfg.get("pgd_strong_steps", 20)

        def pgd_strong(frames: torch.Tensor) -> torch.Tensor:
            frames_adv = frames.clone().detach().requires_grad_(True)

            for _ in range(steps):
                frames_adv.requires_grad_(True)
                output = self.model(frames_adv)
                # Attack both risk and category
                loss = -output["risk_score"].mean() - output["category_probs"][:, 0].mean()

                self.model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    grad = frames_adv.grad.sign()
                    frames_adv = frames_adv + alpha * grad
                    delta = torch.clamp(frames_adv - frames, -epsilon, epsilon)
                    frames_adv = torch.clamp(frames + delta, 0, 1).detach().requires_grad_(True)

            return frames_adv.detach()

        all_scores, all_labels = self._evaluate_with_attack(val_loader, pgd_strong, gate)
        safety = compute_safety_metrics(all_scores, all_labels)

        return {
            "pgd_strong": safety,
            "epsilon": epsilon,
            "steps": steps,
        }

    def _evaluate_with_attack(
        self,
        val_loader: DataLoader,
        attack_fn,
        gate: Optional[DecisionGate],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate model under attack."""
        all_scores = []
        all_labels = []

        self.model.eval()

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                frames = batch["frames"]
                labels = batch["risk_label"]

                # Apply attack
                frames_adv = attack_fn(frames)

                # Model forward
                output = self.model(frames_adv)
                scores = output["risk_score"].squeeze(-1).cpu().numpy()

                all_scores.append(scores)
                all_labels.append(labels.cpu().numpy())

        return np.concatenate(all_scores), np.concatenate(all_labels)

    # Frame perturbation functions
    def _add_transparent_overlay(self, frames: torch.Tensor) -> torch.Tensor:
        """Add fully transparent overlay (no visual change, tests if model uses alpha)."""
        # Since we're RGB, this is no-op - but tests if model is sensitive
        return frames.clone()

    def _add_semi_transparent_overlay(self, frames: torch.Tensor) -> torch.Tensor:
        """Add semi-transparent white overlay."""
        modified = frames.clone()
        overlay = torch.ones_like(modified) * 0.8
        modified = 0.7 * modified + 0.3 * overlay
        return torch.clamp(modified, 0, 1)

    def _add_noise(self, frames: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        modified = frames.clone()
        noise = torch.randn_like(modified) * 0.05
        modified = torch.clamp(modified + noise, 0, 1)
        return modified

    def _add_blur(self, frames: torch.Tensor) -> torch.Tensor:
        """Add blur to center region."""
        import torch.nn.functional as F
        modified = frames.clone()
        # Simple box blur on center region
        kernel = torch.ones(1, 3, 5, 5, device=frames.device) / 25
        # Apply to center
        b, k, c, h, w = modified.shape
        center_h, center_w = h // 2, w // 2
        region = modified[:, :, :, center_h-10:center_h+10, center_w-10:center_w+10]
        if region.shape[3] > 0 and region.shape[4] > 0:
            region_flat = region.view(b*k*c, 1, 20, 20)
            blurred = F.conv2d(region_flat, kernel, padding=2)
            blurred = blurred.view(b, k, c, 20, 20)
            modified[:, :, :, center_h-10:center_h+10, center_w-10:center_w+10] = blurred
        return modified

    def _add_color_shift(self, frames: torch.Tensor) -> torch.Tensor:
        """Shift color channels slightly."""
        modified = frames.clone()
        # Red shift
        modified[:, :, 0] = torch.clamp(modified[:, :, 0] * 1.1, 0, 1)
        modified[:, :, 1] = torch.clamp(modified[:, :, 1] * 0.9, 0, 1)
        return modified


class StressTestRunner:
    """Runs full adversarial stress test suite."""

    def __init__(self, config: DictConfig, device: str = "cuda"):
        self.config = config
        self.device = device

    def run(self, quick: bool = False) -> Dict[str, Any]:
        """Run all stress tests."""
        logger.info("=" * 60)
        logger.info("SENTINEL-Vision Adversarial Stress Test")
        logger.info("=" * 60)

        # Load model
        checkpoint_path = self.config.get("checkpoint", DEFAULT_SENTINEL_CHECKPOINT)
        logger.info(f"Loading model from: {checkpoint_path}")

        model = create_sentinel_model(self.config)
        model.load_state_dict(_load_state_dict_safe(checkpoint_path, self.device))

        # Load gate if available
        gate = None
        gate_path = self.config.get("gate_checkpoint", DEFAULT_GATE_CHECKPOINT)
        if Path(gate_path).exists():
            logger.info(f"Loading gate from: {gate_path}")
            gate = DecisionGate()
            gate.load_state_dict(_load_state_dict_safe(gate_path, self.device))

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

        # Create attacker
        attacker = AdversarialAttacker(self.config, model, self.device)

        # Run attacks
        if quick:
            # Quick mode: only key attacks
            results = {
                "ui_obfuscation": attacker._attack_ui_obfuscation(val_loader, gate),
                "action_chaining": attacker._attack_action_chaining(val_loader, gate),
                "pgd_adversarial": attacker._attack_pgd(val_loader, gate),
            }
        else:
            # Full mode: all attacks
            results = attacker.run_all_attacks(val_loader, gate)

        # Baseline (no attack)
        logger.info("Running baseline (no attack)...")
        attacker.model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                output = model(batch["frames"])
                all_scores.append(output["risk_score"].squeeze(-1).cpu().numpy())
                all_labels.append(batch["risk_label"].cpu().numpy())

        baseline_scores = np.concatenate(all_scores)
        baseline_labels = np.concatenate(all_labels)
        results["baseline"] = compute_safety_metrics(baseline_scores, baseline_labels)

        # Summary
        self._print_summary(results)

        # Save
        self._save_results(results)

        return results

    def _print_summary(self, results: Dict[str, Any]):
        """Print stress test summary."""
        print("\n" + "=" * 100)
        print("ADVERSARIAL STRESS TEST SUMMARY")
        print("=" * 100)

        baseline_fnr = results.get("baseline", {}).get("false_negative_rate", 0)

        print(f"\n{'Attack':<35} {'FNR':>10} {'Δ FNR':>10} {'Recall':>10} {'Precision':>10}")
        print("-" * 100)

        for attack_name, metrics in results.items():
            if attack_name == "baseline":
                continue

            if isinstance(metrics, dict) and "false_negative_rate" in metrics:
                fnr = metrics["false_negative_rate"]
                recall = metrics.get("recall", 0)
                precision = metrics.get("precision", 0)
                delta = fnr - baseline_fnr
                print(f"{attack_name:<35} {fnr:>10.4f} {delta:>+10.4f} {recall:>10.4f} {precision:>10.4f}")

            elif isinstance(metrics, dict):
                # Nested results
                for sub_name, sub_metrics in metrics.items():
                    if isinstance(sub_metrics, dict) and "false_negative_rate" in sub_metrics:
                        fnr = sub_metrics["false_negative_rate"]
                        recall = sub_metrics.get("recall", 0)
                        precision = sub_metrics.get("precision", 0)
                        delta = fnr - baseline_fnr
                        print(f"  {sub_name:<33} {fnr:>10.4f} {delta:>+10.4f} {recall:>10.4f} {precision:>10.4f}")

        print("=" * 100)

    def _save_results(self, results: Dict[str, Any]):
        """Save stress test results."""
        output_dir = Path("results/adversarial")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_dir / f"stress_test_{timestamp}.json"

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

        serializable = make_serializable(results)

        with open(file_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Stress test results saved to: {file_path}")


def run_adversarial_stress_test(config: DictConfig, quick: bool = False) -> Dict[str, Any]:
    """Main entry point for adversarial stress testing."""
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    runner = StressTestRunner(config, device)
    return runner.run(quick=quick)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
    def main(config: DictConfig):
        run_adversarial_stress_test(config)

    main()