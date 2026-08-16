"""
Tests for Decision Gate PPO policy.
Targets the implemented SENTINEL-Vision API.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.gate.decision_gate import DecisionGate, create_decision_gate
from src.gate.reward import AsymmetricReward, RewardShaping, CurriculumReward, create_reward_function


class TestDecisionGate:
    """Tests for DecisionGate."""

    def setup_method(self):
        """Setup test config."""
        self.config = OmegaConf.create({
            "gate": {
                "state_dim": 8,
                "hidden_dim": 128,
                "num_actions": 3,
            }
        })

    def test_creation(self):
        """Test gate creation."""
        gate = create_decision_gate(self.config)
        assert isinstance(gate, DecisionGate)
        assert gate.state_dim == 8
        assert gate.num_actions == 3

    def test_forward(self):
        """Test forward pass."""
        gate = create_decision_gate(self.config)
        gate.eval()

        batch_size = 4
        state = torch.randn(batch_size, 8)
        with torch.no_grad():
            out = gate(state)

        assert "action_logits" in out
        assert "action_probs" in out
        assert "value" in out
        assert "action" in out
        assert "log_prob" in out
        assert "entropy" in out
        assert out["action_logits"].shape == (batch_size, 3)
        assert out["action_probs"].shape == (batch_size, 3)
        assert out["value"].shape == (batch_size, 1)
        assert out["action"].shape == (batch_size,)
        assert out["log_prob"].shape == (batch_size,)
        assert out["entropy"].shape == (batch_size,)

    def test_action_range(self):
        """Test actions are in valid range."""
        gate = create_decision_gate(self.config)
        gate.eval()

        state = torch.randn(100, 8)
        with torch.no_grad():
            out = gate(state)

        actions = out["action"]
        assert torch.all(actions >= 0)
        assert torch.all(actions <= 2)  # 0=ALLOW, 1=PAUSE, 2=HARD_BLOCK

    def test_deterministic_action(self):
        """Test deterministic action selection."""
        gate = create_decision_gate(self.config)
        gate.eval()

        state = torch.randn(1, 8)
        with torch.no_grad():
            action1 = gate.get_action(
                risk_score=0.5,
                category=0,
                heatmap_conf=0.5,
                action_type=0,
                deterministic=True
            )
            action2 = gate.get_action(
                risk_score=0.5,
                category=0,
                heatmap_conf=0.5,
                action_type=0,
                deterministic=True
            )

        assert action1 == action2

    def test_stochastic_action(self):
        """Test stochastic action selection produces variety."""
        gate = create_decision_gate(self.config)
        gate.eval()

        state = torch.randn(1, 8)
        actions = []
        for _ in range(100):
            with torch.no_grad():
                action = gate.get_action(
                    risk_score=0.5,
                    category=0,
                    heatmap_conf=0.5,
                    action_type=0,
                    deterministic=False
                )
                actions.append(action)

        # Should have some variety (not all same)
        assert len(set(actions)) > 1

    def test_build_state(self):
        """Test state building from components."""
        gate = create_decision_gate(self.config)

        risk_score = 0.7
        category = 0
        heatmap_conf = 0.8
        action_type = 1

        state = gate._build_state(risk_score, category, heatmap_conf, action_type)
        assert len(state) == 8
        assert state[0] == risk_score
        assert state[1] == heatmap_conf
        # category_onehot[:4] - category 0 -> [1, 0, 0, 0]
        assert state[2] == 1.0
        assert state[3] == 0.0
        assert state[4] == 0.0
        assert state[5] == 0.0
        # action_onehot[:2] - action_type 1 -> [0, 1]
        assert state[6] == 0.0
        assert state[7] == 1.0

    def test_get_action_from_components(self):
        """Test get_action with scalar inputs."""
        gate = create_decision_gate(self.config)
        gate.eval()

        action = gate.get_action(
            risk_score=0.8,
            category=0,
            heatmap_conf=0.9,
            action_type=0,
            deterministic=True,
        )
        assert action in ["ALLOW", "PAUSE", "HARD_BLOCK"]

    def test_action_names(self):
        """Test action name mapping."""
        gate = create_decision_gate(self.config)
        assert gate.action_names[0] == "ALLOW"
        assert gate.action_names[1] == "PAUSE"
        assert gate.action_names[2] == "HARD_BLOCK"


class TestAsymmetricReward:
    """Tests for AsymmetricReward."""

    def setup_method(self):
        """Setup test config."""
        self.config = OmegaConf.create({
            "reward": {
                "missed_harm": -10.0,
                "false_block": -1.0,
                "correct_block": 2.0,
                "correct_allow": 1.0,
                "correct_pause": 0.5,
                "false_pause": -0.5,
            }
        })

    def test_creation(self):
        """Test reward function creation."""
        reward_fn = create_reward_function(self.config)
        assert isinstance(reward_fn, AsymmetricReward)

    def test_missed_harm(self):
        """Test missed harm penalty (harmful action, ALLOW)."""
        reward_fn = create_reward_function(self.config)
        # action="ALLOW", true_label=1 (harmful)
        reward = reward_fn.compute("ALLOW", 1)
        assert reward == self.config.reward.missed_harm

    def test_false_block(self):
        """Test false block penalty (benign action, HARD_BLOCK)."""
        reward_fn = create_reward_function(self.config)
        # action="HARD_BLOCK", true_label=0 (benign)
        reward = reward_fn.compute("HARD_BLOCK", 0)
        assert reward == self.config.reward.false_block

    def test_correct_block(self):
        """Test correct block reward (harmful action, HARD_BLOCK)."""
        reward_fn = create_reward_function(self.config)
        # action="HARD_BLOCK", true_label=1 (harmful)
        reward = reward_fn.compute("HARD_BLOCK", 1)
        assert reward == self.config.reward.correct_block

    def test_correct_allow(self):
        """Test correct allow reward (benign action, ALLOW)."""
        reward_fn = create_reward_function(self.config)
        # action="ALLOW", true_label=0 (benign)
        reward = reward_fn.compute("ALLOW", 0)
        assert reward == self.config.reward.correct_allow

    def test_pause(self):
        """Test pause penalty/reward."""
        reward_fn = create_reward_function(self.config)
        # action="PAUSE"
        reward_harmful = reward_fn.compute("PAUSE", 1)
        reward_benign = reward_fn.compute("PAUSE", 0)
        assert reward_harmful == self.config.reward.correct_pause
        assert reward_benign == self.config.reward.false_pause

    def test_ratio(self):
        """Test 10:1 ratio of missed_harm to false_block."""
        reward_fn = create_reward_function(self.config)
        ratio = abs(self.config.reward.missed_harm / self.config.reward.false_block)
        assert ratio == 10.0


class TestRewardShaping:
    """Tests for RewardShaping."""

    def setup_method(self):
        """Setup test config."""
        self.base_reward = AsymmetricReward()
        self.shaping = RewardShaping(self.base_reward)

    def test_confidence_shaping(self):
        """Test confidence-based reward shaping."""
        # High confidence correct block should get bonus
        reward = self.shaping.compute_shaped_reward(
            gate_action=2, true_label=1,
            risk_score=0.9, objectness=0.8, uncertainty=0.1
        )
        base = self.base_reward.compute(2, 1)
        assert reward >= base  # Should be boosted

    def test_low_confidence_penalty(self):
        """Test low confidence gets penalty."""
        reward = self.shaping.compute_shaped_reward(
            gate_action=2, true_label=1,
            risk_score=0.3, objectness=0.2, uncertainty=0.8
        )
        base = self.base_reward.compute(2, 1)
        assert reward <= base  # Should be penalized


class TestCurriculumReward:
    """Tests for CurriculumReward."""

    def setup_method(self):
        """Setup test config."""
        self.base_reward = AsymmetricReward()
        self.curriculum = CurriculumReward(
            self.base_reward,
            initial_ratio=5.0,
            target_ratio=10.0,
            ramp_epochs=10,
        )

    def test_early_training(self):
        """Test early training has lower ratio."""
        self.curriculum.set_epoch(0)
        reward_fn = self.curriculum.get_current_reward()
        # Ratio should ramp from initial_ratio to target_ratio
        ratio = abs(reward_fn.missed_harm / reward_fn.false_block)
        assert ratio == 5.0  # initial_ratio

    def test_mid_training(self):
        """Test mid training has interpolated ratio."""
        self.curriculum.set_epoch(5)
        reward_fn = self.curriculum.get_current_reward()
        ratio = abs(reward_fn.missed_harm / reward_fn.false_block)
        assert 5.0 < ratio < 10.0

    def test_late_training(self):
        """Test late training has full ratio."""
        self.curriculum.set_epoch(10)
        reward_fn = self.curriculum.get_current_reward()
        ratio = abs(reward_fn.missed_harm / reward_fn.false_block)
        assert ratio == 10.0  # target_ratio

    def test_compute_method(self):
        """Test compute method uses current epoch."""
        self.curriculum.set_epoch(0)
        reward = self.curriculum.compute(2, 1)  # HARD_BLOCK, harmful
        assert reward == 2.0  # correct_block (same regardless of ratio)

        self.curriculum.set_epoch(10)
        reward = self.curriculum.compute(0, 1)  # ALLOW, harmful
        assert reward == -10.0  # missed_harm at target_ratio


class TestGateIntegration:
    """Integration tests for gate with SENTINEL model."""

    def test_gate_with_sentinel_outputs(self):
        """Test gate decision from SENTINEL outputs."""
        config = OmegaConf.create({
            "gate": {
                "state_dim": 8,
                "hidden_dim": 128,
                "num_actions": 3,
            }
        })

        gate = create_decision_gate(config)
        gate.eval()

        # Simulate SENTINEL outputs
        risk_score = 0.85
        category = 0  # destructive
        heatmap_conf = 0.9
        action_type = 0

        action = gate.get_action(risk_score, category, heatmap_conf, action_type, deterministic=True)
        assert action in ["ALLOW", "PAUSE", "HARD_BLOCK"]

        # High risk + high confidence + harmful category -> should BLOCK or PAUSE
        assert action != "ALLOW"  # Should not ALLOW


class TestGateTraining:
    """Tests for gate training components."""

    def test_ppo_loss_computation(self):
        """Test PPO loss can be computed."""
        config = OmegaConf.create({
            "gate": {
                "state_dim": 8,
                "hidden_dim": 128,
                "num_actions": 3,
            }
        })

        gate = create_decision_gate(config)
        gate.train()

        # Mock trajectory data
        batch_size = 16
        states = torch.randn(batch_size, 8)
        actions = torch.randint(0, 3, (batch_size,))
        old_log_probs = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        # Forward
        out = gate(states)

        # Policy loss
        log_probs = out["log_prob"]
        ratio = (log_probs - old_log_probs).exp()
        clip_epsilon = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(out["value"].squeeze(), returns)

        # Entropy
        entropy_loss = -out["entropy"].mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        assert loss.item() is not None
        loss.backward()

        # Check gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())
        assert has_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])