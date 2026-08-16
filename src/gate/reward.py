"""
Asymmetric Reward Function for SENTINEL-Vision Decision Gate.

Core safety principle: Missed harm (false negative) costs 10x more than false block (false positive).
This is the primary safety design decision.
"""

import torch
from typing import Dict, List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


class AsymmetricReward:
    """
    Asymmetric reward function for safety-critical decision making.

    Reward matrix:
    - Correct ALLOW (benign, ALLOW): +1.0
    - False BLOCK (benign, HARD_BLOCK): -1.0
    - False PAUSE (benign, PAUSE): -0.5
    - Missed HARM (harmful, ALLOW): -10.0  <-- CRITICAL: 10x penalty
    - Correct PAUSE (harmful, PAUSE): +0.5
    - Correct BLOCK (harmful, HARD_BLOCK): +2.0

    Actions: 0=ALLOW, 1=PAUSE, 2=HARD_BLOCK
    Labels: 0=benign, 1=harmful
    """

    def __init__(
        self,
        correct_allow: float = 1.0,
        false_block: float = -1.0,
        false_pause: float = -0.5,
        missed_harm: float = -10.0,
        correct_pause: float = 0.5,
        correct_block: float = 2.0,
    ):
        """
        Args:
            correct_allow: Reward for correctly allowing benign action
            false_block: Penalty for blocking benign action (false positive)
            false_pause: Penalty for pausing on benign action
            missed_harm: Penalty for allowing harmful action (FALSE NEGATIVE - CRITICAL)
            correct_pause: Reward for pausing on harmful action
            correct_block: Reward for blocking harmful action
        """
        self.correct_allow = correct_allow
        self.false_block = false_block
        self.false_pause = false_pause
        self.missed_harm = missed_harm
        self.correct_pause = correct_pause
        self.correct_block = correct_block

        # Build reward matrix: [label][action] -> reward
        # label: 0=benign, 1=harmful
        # action: 0=ALLOW, 1=PAUSE, 2=HARD_BLOCK
        self.reward_matrix = {
            0: {  # Benign
                0: correct_allow,      # ALLOW -> correct
                1: false_pause,        # PAUSE -> false pause
                2: false_block,        # HARD_BLOCK -> false block
            },
            1: {  # Harmful
                0: missed_harm,        # ALLOW -> MISSED HARM (CRITICAL)
                1: correct_pause,      # PAUSE -> correct pause
                2: correct_block,      # HARD_BLOCK -> correct block
            },
        }

        # Log the critical ratio
        ratio = abs(missed_harm / false_block) if false_block != 0 else float('inf')
        logger.info(f"AsymmetricReward initialized: missed_harm/false_block ratio = {ratio:.1f}:1")
        logger.info(f"Reward matrix: {self.reward_matrix}")

    def compute(self, gate_action: Union[int, str], true_label: int) -> float:
        """
        Compute single-step reward.

        Args:
            gate_action: 0=ALLOW, 1=PAUSE, 2=HARD_BLOCK (or string)
            true_label: 0=benign, 1=harmful

        Returns:
            Scalar reward
        """
        if isinstance(gate_action, str):
            action_map = {"ALLOW": 0, "PAUSE": 1, "HARD_BLOCK": 2}
            gate_action = action_map.get(gate_action.upper(), 0)

        return self.reward_matrix[true_label][gate_action]

    def compute_batch(
        self,
        gate_actions: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rewards for a batch.

        Args:
            gate_actions: (B,) action indices
            true_labels: (B,) label indices (0 or 1)

        Returns:
            (B,) rewards
        """
        rewards = torch.zeros_like(gate_actions, dtype=torch.float32)

        for label in [0, 1]:
            for action in [0, 1, 2]:
                mask = (true_labels == label) & (gate_actions == action)
                rewards[mask] = self.reward_matrix[label][action]

        return rewards

    def compute_episode_return(self, trajectory: List[Dict]) -> float:
        """
        Compute total return for an episode.

        Args:
            trajectory: List of dicts with keys 'gate_action', 'true_label'

        Returns:
            Total episode return
        """
        total = 0.0
        for step in trajectory:
            total += self.compute(step["gate_action"], step["true_label"])
        return total


class RewardShaping:
    """
    Additional reward shaping for better learning.
    Adds intermediate rewards for confidence, localization quality, etc.
    """

    def __init__(
        self,
        base_reward: AsymmetricReward,
        confidence_weight: float = 0.1,
        localization_weight: float = 0.2,
        uncertainty_penalty: float = 0.1,
    ):
        self.base_reward = base_reward
        self.confidence_weight = confidence_weight
        self.localization_weight = localization_weight
        self.uncertainty_penalty = uncertainty_penalty

    def compute_shaped_reward(
        self,
        gate_action: int,
        true_label: int,
        risk_score: float,
        objectness: float = 0.0,
        uncertainty: float = 0.0,
    ) -> float:
        """
        Compute shaped reward with auxiliary signals.

        Args:
            gate_action: 0=ALLOW, 1=PAUSE, 2=HARD_BLOCK
            true_label: 0=benign, 1=harmful
            risk_score: Model risk score [0, 1]
            objectness: Localization confidence [0, 1]
            uncertainty: Epistemic uncertainty [0, 1]

        Returns:
            Shaped reward
        """
        base = self.base_reward.compute(gate_action, true_label)

        # Confidence shaping: reward calibrated confidence
        if true_label == 1:  # Harmful
            # Should have high risk_score
            confidence_bonus = self.confidence_weight * risk_score
        else:  # Benign
            # Should have low risk_score
            confidence_bonus = self.confidence_weight * (1 - risk_score)

        # Localization quality: reward good localization on harmful
        loc_bonus = 0.0
        if true_label == 1 and gate_action in [1, 2]:  # PAUSE or BLOCK on harmful
            loc_bonus = self.localization_weight * objectness

        # Uncertainty penalty: penalize high uncertainty decisions
        uncertainty_penalty = -self.uncertainty_penalty * uncertainty

        return base + confidence_bonus + loc_bonus + uncertainty_penalty


class CurriculumReward:
    """
    Curriculum reward that changes asymmetry ratio during training.
    Starts with lower ratio, increases to target ratio.
    """

    def __init__(
        self,
        base_reward: Optional[AsymmetricReward] = None,
        initial_ratio: float = 5.0,
        target_ratio: float = 10.0,
        ramp_epochs: int = 10,
    ):
        if isinstance(base_reward, (int, float)):
            initial_ratio = float(base_reward)
            base_reward = None

        self.initial_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.ramp_epochs = ramp_epochs

        if base_reward is not None:
            self.base_reward = base_reward
        else:
            self.base_reward = AsymmetricReward(
                correct_allow=1.0,
                false_block=-1.0,
                missed_harm=-initial_ratio,
                correct_pause=0.5,
                correct_block=2.0,
            )

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum."""
        self.current_epoch = epoch

    def get_current_reward(self) -> AsymmetricReward:
        """Get reward with current asymmetry ratio."""
        if self.current_epoch >= self.ramp_epochs:
            ratio = self.target_ratio
        else:
            # Linear interpolation
            progress = self.current_epoch / self.ramp_epochs
            ratio = self.initial_ratio + progress * (self.target_ratio - self.initial_ratio)

        return AsymmetricReward(
            correct_allow=self.base_reward.correct_allow,
            false_block=self.base_reward.false_block,
            missed_harm=-ratio,
            correct_pause=self.base_reward.correct_pause,
            correct_block=self.base_reward.correct_block,
        )

    def compute(self, gate_action: int, true_label: int) -> float:
        return self.get_current_reward().compute(gate_action, true_label)


def create_reward_function(config: Dict) -> AsymmetricReward:
    """Factory function to create reward from config."""
    reward_config = config.get("reward", {})
    return AsymmetricReward(
        correct_allow=reward_config.get("correct_allow", 1.0),
        false_block=reward_config.get("false_block", -1.0),
        missed_harm=reward_config.get("missed_harm", -10.0),
        correct_pause=reward_config.get("correct_pause", 0.5),
        correct_block=reward_config.get("correct_block", 2.0),
    )