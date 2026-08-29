"""
PPO Training for SENTINEL-Vision Decision Gate.
Trains the gate policy using simulated episodes from the dataset.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import deque

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

from src.models.sentinel_model import SentinelModel, create_sentinel_model
from src.data.loaders import SentinelDataset
from src.data.augmentation import create_val_transform
from src.data.frame_windowing import collate_frame_windows
from src.gate.decision_gate import DecisionGate, create_decision_gate
from src.gate.reward import AsymmetricReward, create_reward_function
from src.utils.logging import setup_logging
from src.utils.checkpoint import load_checkpoint, save_checkpoint, CheckpointLoadError
from src.utils.constants import DEFAULT_SENTINEL_CHECKPOINT

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO Trainer for Decision Gate.
    Simulates episodes: sample from dataset -> run sentinel -> gate decides -> compute reward.
    """

    def __init__(
        self,
        config: DictConfig,
        sentinel_model: SentinelModel,
        gate: DecisionGate,
        reward_fn: AsymmetricReward,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
    ):
        self.config = config
        self.sentinel = sentinel_model.to(device).eval()
        self.gate = gate.to(device).train()
        self.reward_fn = reward_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # PPO hyperparameters
        ppo_config = config.get("ppo", {})
        self.learning_rate = ppo_config.get("learning_rate", 1e-5)
        self.batch_size = ppo_config.get("batch_size", 32)
        self.n_epochs = ppo_config.get("n_epochs", 4)
        self.clip_epsilon = ppo_config.get("clip_epsilon", 0.2)
        self.gamma = ppo_config.get("gamma", 0.99)
        self.gae_lambda = ppo_config.get("gae_lambda", 0.95)
        self.ent_coef = ppo_config.get("ent_coef", 0.01)
        self.vf_coef = ppo_config.get("vf_coef", 0.5)
        self.max_grad_norm = ppo_config.get("max_grad_norm", 0.5)

        # Training config
        train_config = config.get("training", {})
        self.total_timesteps = train_config.get("total_timesteps", 10000)
        self.eval_freq = train_config.get("eval_freq", 2000)
        self.save_freq = train_config.get("save_freq", 5000)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.gate.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
        )

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/gate"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.timesteps = 0
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=100)
        self.action_counts = {0: 0, 1: 0, 2: 0}  # ALLOW, PAUSE, HARD_BLOCK
        self.fn_count = 0  # False negatives (missed harm)
        self.fp_count = 0  # False positives (false blocks)

        logger.info(f"PPOTrainer initialized: lr={self.learning_rate}, "
                    f"clip_eps={self.clip_epsilon}, gamma={self.gamma}")

    def train(self) -> Dict[str, float]:
        """Main training loop."""
        logger.info("Starting PPO training for Decision Gate")

        while self.timesteps < self.total_timesteps:
            # Collect trajectories
            trajectories = self._collect_trajectories()
            if not trajectories:
                break

            # Compute returns and advantages
            self._compute_returns(trajectories)

            # PPO update
            self._ppo_update(trajectories)

            # Log metrics
            if self.timesteps % self.eval_freq == 0:
                self._log_metrics()

            # Save checkpoint
            if self.timesteps % self.save_freq == 0:
                self._save_checkpoint()

        # Final save
        self._save_checkpoint(suffix="_final")
        self._log_metrics()
        logger.info("PPO training completed successfully!")
        return {"final_timesteps": self.timesteps}

    def _collect_trajectories(self) -> List[Dict]:
        """Collect trajectories by running sentinel+gate on dataset."""
        trajectories = []
        steps_collected = 0

        self.gate.eval()
        self.sentinel.eval()

        with torch.no_grad():
            for batch in self.train_loader:
                if steps_collected >= self.batch_size:
                    break

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                frames = batch["frames"]  # (B, k, C, H, W)
                risk_labels = batch["risk_label"]  # (B,)

                # Sentinel forward
                sentinel_out = self.sentinel(frames)

                # Get gate actions for each sample
                for i in range(frames.shape[0]):
                    risk_score = sentinel_out["risk_score"][i].item()
                    category = sentinel_out["category_idx"][i].item()
                    heatmap_conf = sentinel_out["objectness"][i].item()
                    true_label = int(risk_labels[i].item())

                    # Gate decision
                    state_vec = self.gate._build_state(risk_score, category, heatmap_conf, 0)
                    gate_out = self.gate(torch.tensor(
                        state_vec,
                        dtype=torch.float32, device=self.device
                    ).unsqueeze(0))

                    action = int(gate_out["action"].item())
                    log_prob = gate_out["log_prob"].item()
                    value = gate_out["value"].item()

                    # Compute reward
                    reward = self.reward_fn.compute(action, true_label)

                    # Track metrics
                    self.action_counts[action] += 1
                    if true_label == 1 and action == 0:  # Missed harm
                        self.fn_count += 1
                    if true_label == 0 and action == 2:  # False block
                        self.fp_count += 1

                    trajectories.append({
                        "state": state_vec,
                        "action": action,
                        "log_prob": log_prob,
                        "value": value,
                        "reward": reward,
                        "true_label": true_label,
                        "risk_score": risk_score,
                        "category": category,
                        "heatmap_conf": heatmap_conf,
                    })

                    steps_collected += 1
                    self.timesteps += 1
                    self.episode_count += 1

                    if steps_collected >= self.batch_size:
                        break

        self.gate.train()
        return trajectories

    def _compute_returns(self, trajectories: List[Dict]):
        """Compute GAE returns and advantages."""
        rewards = [t["reward"] for t in trajectories]
        values = [t["value"] for t in trajectories]

        # Compute returns (discounted sum)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # Compute GAE advantages
        advantages = []
        gae = 0
        for i in reversed(range(len(trajectories))):
            if i == len(trajectories) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if len(advantages) > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        for i, t in enumerate(trajectories):
            t["return"] = returns[i].item()
            t["advantage"] = advantages[i].item()

    def _ppo_update(self, trajectories: List[Dict]):
        """Perform PPO update on collected trajectories."""
        # Prepare batch tensors
        states = torch.stack([
            torch.tensor(t["state"], dtype=torch.float32, device=self.device)
            for t in trajectories
        ])

        actions = torch.tensor([t["action"] for t in trajectories], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t["log_prob"] for t in trajectories], dtype=torch.float32, device=self.device)
        returns = torch.tensor([t["return"] for t in trajectories], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([t["advantage"] for t in trajectories], dtype=torch.float32, device=self.device)

        # PPO epochs
        for _ in range(self.n_epochs):
            # Forward pass
            gate_out = self.gate(states)

            # Policy loss
            dist = torch.distributions.Categorical(gate_out["action_probs"])
            log_probs = dist.log_prob(actions)
            ratio = (log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(gate_out["value"].squeeze(-1), returns)

            # Entropy bonus
            entropy_loss = -dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gate.parameters(), self.max_grad_norm)
            self.optimizer.step()

    def _log_metrics(self):
        """Log training metrics."""
        total_actions = sum(self.action_counts.values())
        action_dist = {k: f"{v / max(1, total_actions):.2%}" for k, v in self.action_counts.items()}
        fn_rate = self.fn_count / max(1, self.episode_count)
        fp_rate = self.fp_count / max(1, self.episode_count)

        logger.info(
            f"Timestep {self.timesteps}/{self.total_timesteps} | "
            f"Actions: {action_dist} | "
            f"FNR (Missed Harm): {fn_rate:.2%} | "
            f"FPR (False Block): {fp_rate:.2%}"
        )

    def _save_checkpoint(self, suffix: str = ""):
        """Save gate checkpoint."""
        filename = f"gate_checkpoint_{self.timesteps}{suffix}.pt" if suffix == "" else f"gate_best{suffix}.pt"
        path = self.checkpoint_dir / filename
        # self.optimizer is always a plain torch.optim.AdamW (constructed
        # directly above) -- the `hasattr(self.optimizer, "optimizer")`
        # branch was dead defensive code for an LR-scheduler-wrapper type
        # that is never used here.
        save_checkpoint(
            {
                "model_state_dict": self.gate.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "timestep": self.timesteps,
                "config": OmegaConf.to_container(self.config) if isinstance(self.config, DictConfig) else self.config,
            },
            path,
        )


def train_gate_rl(config: Optional[Union[DictConfig, dict]] = None) -> dict:
    """Main entry point for gate RL training."""
    if config is None:
        config = OmegaConf.load("configs/gate_rl.yaml")

    setup_logging(config.get("log_level", "INFO"))
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Gate RL Training")
    logger.info("=" * 60)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build default full configuration
    default_config = OmegaConf.create({
        "model": {
            "backbone": "vit_small_patch16_224",
            "embed_dim": 384,
            "pretrained": False,
            "temporal": {
                "embed_dim": 384,
                "num_layers": 3,
                "num_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
                "max_frames": 8,
                "use_delta": True,
                "fusion_mode": "last",
            },
            "risk_head": {
                "embed_dim": 384,
                "hidden_dim": 64,
                "num_categories": 5,
                "dropout": 0.1,
            },
            "localization_head": {
                "embed_dim": 384,
                "fm_size": 14,
                "num_anchors": 9,
                "anchor_sizes": [32, 64, 128, 256],
            },
        },
        "data": {
            "data_dir": "data/processed",
            "frame_window_k": 6,
            "target_resolution": [224, 224],
        },
        "frame_window": {
            "k": 6,
            "resolution": [224, 224],
        },
        "training": {
            "batch_size": 16,
            "num_workers": 0 if os.name == "nt" else 4,
            "total_timesteps": 10000,
            "eval_freq": 2000,
            "save_freq": 5000,
        },
        "gate": {
            "state_dim": 8,
            "hidden_dim": 128,
            "num_actions": 3,
        },
        "reward": {
            "correct_allow": 1.0,
            "false_block": -1.0,
            "missed_harm": -10.0,
            "correct_pause": 0.5,
            "correct_block": 2.0,
        },
        "ppo": {
            "learning_rate": 1.0e-5,
            "batch_size": 32,
            "n_epochs": 4,
            "clip_epsilon": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    })

    full_config = OmegaConf.merge(default_config, config)

    # Locate pretrained SENTINEL model checkpoint
    sentinel_checkpoint = full_config.get("sentinel_checkpoint", None)
    if not sentinel_checkpoint or not os.path.exists(sentinel_checkpoint):
        for candidate in [
            DEFAULT_SENTINEL_CHECKPOINT,
            "checkpoints/stage_b_30epochs/best.pt",
        ]:
            if os.path.exists(candidate):
                sentinel_checkpoint = candidate
                break

    if not sentinel_checkpoint or not os.path.exists(sentinel_checkpoint):
        logger.warning(
            "No SENTINEL checkpoint found -- training the gate against an "
            "UNTRAINED sentinel model. The gate would learn a policy over "
            "meaningless random risk scores. Refusing to proceed silently; "
            "pass sentinel_checkpoint explicitly if this is intentional "
            "(e.g. an architecture smoke test)."
        )
        raise FileNotFoundError(
            "No sentinel checkpoint found under checkpoints/. Set "
            "config.sentinel_checkpoint or train a stage model first."
        )
    else:
        logger.info(f"Loading SENTINEL from: {sentinel_checkpoint}")
        try:
            checkpoint = load_checkpoint(sentinel_checkpoint, map_location=device)
        except CheckpointLoadError:
            logger.warning(
                "'%s' failed restricted (weights_only) load; retrying with "
                "allow_unsafe=True because it is a locally-trained checkpoint.",
                sentinel_checkpoint,
            )
            checkpoint = load_checkpoint(sentinel_checkpoint, map_location=device, allow_unsafe=True)
        ckpt_config = checkpoint.get("config", {})
        if ckpt_config:
            if isinstance(ckpt_config, dict):
                model_cfg = OmegaConf.create(ckpt_config)
            else:
                model_cfg = ckpt_config
            model_cfg = OmegaConf.merge(full_config, model_cfg)
        else:
            model_cfg = full_config
        sentinel = create_sentinel_model(model_cfg)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        sentinel.load_state_dict(state_dict)
        sentinel = sentinel.to(device).eval()

    # Create gate
    gate_config_dict = OmegaConf.to_container(full_config.get("gate", default_config.gate), resolve=True)
    gate = create_decision_gate(gate_config_dict).to(device)

    # Create reward
    reward_config_dict = OmegaConf.to_container(full_config, resolve=True)
    reward_fn = create_reward_function(reward_config_dict)

    # Create data loader
    val_transform = create_val_transform(reward_config_dict)

    val_dataset = SentinelDataset(
        data_config=reward_config_dict,
        split="val",
        transform=val_transform,
        frame_window_k=full_config.frame_window.k,
        target_resolution=tuple(full_config.frame_window.resolution),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=full_config.training.get("batch_size", 16),
        shuffle=True,
        num_workers=full_config.training.get("num_workers", 0),
        pin_memory=True if device == "cuda" else False,
        collate_fn=collate_frame_windows,
    )

    trainer = PPOTrainer(
        config=full_config,
        sentinel_model=sentinel,
        gate=gate,
        reward_fn=reward_fn,
        train_loader=val_loader,
        val_loader=val_loader,
        device=device,
    )

    results = trainer.train()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SENTINEL-Vision Decision Gate via PPO")
    parser.add_argument("--config", type=str, default="configs/gate_rl.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config) if os.path.exists(args.config) else OmegaConf.create({})
    cfg.device = args.device
    train_gate_rl(cfg)