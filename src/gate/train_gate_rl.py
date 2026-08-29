"""
PPO Training for SENTINEL-Vision Decision Gate.
Trains the gate policy using simulated episodes from the dataset.
"""

import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
from collections import deque
from omegaconf import DictConfig
import hydra

from ..models.sentinel_model import SentinelModel, create_sentinel_model
from ..data.loaders import SentinelDataset
from ..data.augmentation import create_val_transform
from ..data.frame_windowing import collate_frame_windows
from .decision_gate import DecisionGate, create_decision_gate
from .reward import AsymmetricReward, create_reward_function
from ..utils.logging import setup_logging

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
        self.total_timesteps = train_config.get("total_timesteps", 100000)
        self.eval_freq = train_config.get("eval_freq", 10000)
        self.save_freq = train_config.get("save_freq", 20000)

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
        logger.info("PPO training completed")
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
                category_labels = batch["category_label"]  # (B,)

                # Sentinel forward
                sentinel_out = self.sentinel(frames)

                # Get gate actions for each sample
                for i in range(frames.shape[0]):
                    risk_score = sentinel_out["risk_score"][i].item()
                    category = sentinel_out["category_idx"][i].item()
                    heatmap_conf = sentinel_out["objectness"][i].item()
                    true_label = risk_labels[i].item()

                    # Gate decision
                    gate_out = self.gate(torch.tensor(
                        self.gate._build_state(risk_score, category, heatmap_conf, 0),
                        dtype=torch.float32, device=self.device
                    ).unsqueeze(0))

                    action = gate_out["action"].item()
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
                        "state": gate_out.get("state", None),  # Not stored in forward, rebuild if needed
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        for i, t in enumerate(trajectories):
            t["return"] = returns[i].item()
            t["advantage"] = advantages[i].item()

    def _ppo_update(self, trajectories: List[Dict]):
        """Perform PPO update on collected trajectories."""
        # Prepare batch tensors
        states = torch.stack([
            torch.tensor(
                self.gate._build_state(
                    t["risk_score"], t["category"], t["heatmap_conf"], 0
                ),
                dtype=torch.float32, device=self.device
            ) for t in trajectories
        ])

        actions = torch.tensor([t["action"] for t in trajectories], device=self.device)
        old_log_probs = torch.tensor([t["log_prob"] for t in trajectories], device=self.device)
        returns = torch.tensor([t["return"] for t in trajectories], device=self.device)
        advantages = torch.tensor([t["advantage"] for t in trajectories], device=self.device)

        # PPO epochs
        for _ in range(self.n_epochs):
            # Forward pass
            gate_out = self.gate(states)

            # Policy loss
            log_probs = gate_out["log_prob"]
            ratio = (log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(gate_out["value"].squeeze(), returns)

            # Entropy bonus
            entropy_loss = -gate_out["entropy"].mean()

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
        action_dist = {k: v / max(1, total_actions) for k, v in self.action_counts.items()}
        fn_rate = self.fn_count / max(1, self.episode_count)
        fp_rate = self.fp_count / max(1, self.episode_count)

        logger.info(
            f"Timestep {self.timesteps} | "
            f"Actions: ALLOW={action_dist[0]:.3f} PAUSE={action_dist[1]:.3f} BLOCK={action_dist[2]:.3f} | "
            f"FN Rate: {fn_rate:.4f} FP Rate: {fp_rate:.4f}"
        )

        # Wandb logging if available
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "gate/timestep": self.timesteps,
                    "gate/allow_ratio": action_dist[0],
                    "gate/pause_ratio": action_dist[1],
                    "gate/block_ratio": action_dist[2],
                    "gate/fn_rate": fn_rate,
                    "gate/fp_rate": fp_rate,
                    "gate/fn_fp_ratio": fn_rate / max(fp_rate, 1e-8),
                })
        except ImportError:
            pass

    def _save_checkpoint(self, suffix: str = ""):
        """Save gate checkpoint."""
        checkpoint = {
            "timestep": self.timesteps,
            "model_state_dict": self.gate.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "action_counts": self.action_counts,
            "fn_count": self.fn_count,
            "fp_count": self.fp_count,
        }

        path = self.checkpoint_dir / f"gate_{self.timesteps}{suffix}.pt"
        torch.save(checkpoint, path)

        # Latest checkpoint
        latest = self.checkpoint_dir / "latest.pt"
        if latest.exists() or latest.is_symlink():
            try:
                latest.unlink()
            except Exception:
                pass
        try:
            latest.symlink_to(path.name)
        except (OSError, NotImplementedError, Exception):
            shutil.copy2(path, latest)

        logger.info(f"Gate checkpoint saved: {path}")

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: DictConfig, device: str = "cuda"):
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Recreate models
        sentinel = create_sentinel_model(config)
        gate = create_decision_gate(OmegaConf.to_container(config))

        trainer = cls(config, sentinel, gate, None, None, None, device)

        trainer.gate.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.timesteps = checkpoint["timestep"]

        return trainer


def train_gate_rl(config: DictConfig) -> DictConfig:
    """Main entry point for gate RL training."""
    setup_logging(config.get("log_level", "INFO"))
    logger.info("=" * 60)
    logger.info("SENTINEL-Vision Gate RL Training")
    logger.info("=" * 60)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained SENTINEL model
    sentinel_checkpoint = config.get("sentinel_checkpoint", "checkpoints/stage_c/best.pt")
    logger.info(f"Loading SENTINEL from: {sentinel_checkpoint}")

    sentinel = create_sentinel_model(config)
    checkpoint = torch.load(sentinel_checkpoint, map_location=device)
    sentinel.load_state_dict(checkpoint["model_state_dict"])
    sentinel = sentinel.to(device).eval()

    # Create gate
    gate = create_decision_gate(OmegaConf.to_container(config)).to(device)

    # Create reward
    reward_fn = create_reward_function(OmegaConf.to_container(config))

    # Create data loaders (use validation set for RL to avoid overfitting)
    from ..data.augmentation import create_val_transform
    val_transform = create_val_transform(OmegaConf.to_container(config))

    val_dataset = SentinelDataset(
        data_config=OmegaConf.to_container(config),
        split="val",
        transform=val_transform,
        frame_window_k=config.frame_window.k,
        target_resolution=tuple(config.frame_window.resolution),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.get("batch_size", 16),
        shuffle=True,
        num_workers=config.training.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_frame_windows,
    )

    # Use same loader for train (we're training gate, not sentinel)
    train_loader = val_loader

    # Create trainer
    trainer = PPOTrainer(
        config=config,
        sentinel_model=sentinel,
        gate=gate,
        reward_fn=reward_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Train
    results = trainer.train()

    return results


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../configs", config_name="gate_rl")
    def main(config: DictConfig):
        train_gate_rl(config)

    main()