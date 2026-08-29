"""
Decision Gate for SENTINEL-Vision.
PPO-trained policy that takes risk signals and outputs ALLOW/PAUSE/HARD_BLOCK.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DecisionGate(nn.Module):
    """
    PPO policy network for safety decision making.

    State: (risk_score, category_one_hot, heatmap_confidence, action_type_embedding)
    Action space: {ALLOW=0, PAUSE=1, HARD_BLOCK=2}

    Asymmetric reward:
    - Missed harm (HARM, ALLOW): -10
    - False block (BENIGN, HARD_BLOCK): -1
    - Correct allow (BENIGN, ALLOW): +1
    - Correct pause (HARM, PAUSE): +0.5
    - Correct block (HARM, HARD_BLOCK): +2
    """

    def __init__(
        self,
        state_dim: int = 8,
        hidden_dim: int = 128,
        num_actions: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            state_dim: Dimension of input state vector
            hidden_dim: Hidden layer dimension
            num_actions: Number of actions (3: ALLOW, PAUSE, HARD_BLOCK)
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.action_names = ["ALLOW", "PAUSE", "HARD_BLOCK"]

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        logger.info(f"DecisionGate initialized: state_dim={state_dim}, "
                    f"hidden_dim={hidden_dim}, num_actions={num_actions}")

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: (B, state_dim) or (state_dim,)

        Returns:
            Dict with:
                - action_logits: (B, num_actions)
                - action_probs: (B, num_actions)
                - value: (B, 1)
                - action: (B,) sampled action indices
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Policy
        action_logits = self.policy_net(state)  # (B, num_actions)
        action_probs = F.softmax(action_logits, dim=-1)

        # Value
        value = self.value_net(state)  # (B, 1)

        # Sample action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        return {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "value": value,
            "action": action,
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy(),
        }

    def get_action(
        self,
        risk_score: float,
        category: Union[int, str],
        heatmap_conf: float,
        action_type: int = 0,
        deterministic: bool = False,
    ) -> str:
        """
        Get discrete action from risk signals.

        Args:
            risk_score: Model risk score [0, 1]
            category: Predicted category index (0-4) or name
            heatmap_conf: Localization confidence [0, 1]
            action_type: Type of agent action (0=click, 1=type, 2=navigate, etc.)
            deterministic: If True, use argmax instead of sampling

        Returns:
            Action string: "ALLOW", "PAUSE", or "HARD_BLOCK"
        """
        self.eval()

        # Convert category to index if string
        category_names = ["destructive", "financial", "privacy", "irreversible_external", "benign"]
        if isinstance(category, str):
            category = category_names.index(category) if category in category_names else 4

        # Build state vector
        state = self._build_state(risk_score, category, heatmap_conf, action_type)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.forward(state_tensor)

            if deterministic:
                action_idx = output["action_probs"].argmax(dim=-1).item()
            else:
                action_idx = output["action"].item()

        return self.action_names[action_idx]

    def _build_state(
        self,
        risk_score: float,
        category: int,
        heatmap_conf: float,
        action_type: int,
    ) -> List[float]:
        """Build state vector from risk signals."""
        # One-hot encode category (5 dims)
        category_onehot = [0.0] * 5
        if 0 <= category < 5:
            category_onehot[category] = 1.0

        # State: [risk_score, heatmap_conf, category_onehot(4), action_onehot(2)] = 8 dims
        action_onehot = [0.0, 0.0, 0.0]
        if 0 <= action_type < 3:
            action_onehot[action_type] = 1.0

        if self.state_dim == 8:
            return [risk_score, heatmap_conf] + category_onehot[:4] + action_onehot[:2]

        full_state = [risk_score, heatmap_conf] + category_onehot + action_onehot
        if len(full_state) < self.state_dim:
            full_state = full_state + [0.0] * (self.state_dim - len(full_state))
        return full_state[:self.state_dim]

    def get_action_probs(
        self,
        risk_score: float,
        category: Union[int, str],
        heatmap_conf: float,
        action_type: int = 0,
    ) -> Dict[str, float]:
        """Get action probabilities for analysis."""
        self.eval()

        if isinstance(category, str):
            category_names = ["destructive", "financial", "privacy", "irreversible_external", "benign"]
            category = category_names.index(category) if category in category_names else 4

        state = self._build_state(risk_score, category, heatmap_conf, action_type)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.forward(state_tensor)
            probs = output["action_probs"][0].cpu().numpy()

        return {name: float(prob) for name, prob in zip(self.action_names, probs)}


class DecisionGateWithHistory(DecisionGate):
    """
    Decision gate that considers temporal history of decisions.
    Adds LSTM to track decision patterns.
    """

    def __init__(
        self,
        state_dim: int = 8,
        hidden_dim: int = 128,
        num_actions: int = 3,
        history_len: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(state_dim, hidden_dim, num_actions, dropout)

        self.history_len = history_len
        self.lstm = nn.LSTM(
            input_size=num_actions,  # Previous action one-hot
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Modify policy net to take history
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

        self.register_buffer("history_buffer", torch.zeros(1, history_len, num_actions))

    def forward(self, state: torch.Tensor, history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = state.shape[0]

        if history is None:
            history = self.history_buffer.repeat(B, 1, 1)

        # LSTM over history
        lstm_out, _ = self.lstm(history)  # (B, history_len, hidden_dim)
        history_feat = lstm_out[:, -1, :]  # (B, hidden_dim)

        # Combine with current state
        combined = torch.cat([state, history_feat], dim=-1)

        # Policy
        action_logits = self.policy_net(combined)
        action_probs = F.softmax(action_logits, dim=-1)

        # Value
        value = self.value_net(state)

        # Sample
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        return {
            "action_logits": action_logits,
            "action_probs": action_probs,
            "value": value,
            "action": action,
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy(),
        }

    def update_history(self, action_idx: int):
        """Update internal history buffer."""
        action_onehot = torch.zeros(self.num_actions)
        action_onehot[action_idx] = 1.0

        # Shift and append
        self.history_buffer = torch.roll(self.history_buffer, shifts=-1, dims=1)
        self.history_buffer[0, -1, :] = action_onehot


def create_decision_gate(config: Dict) -> DecisionGate:
    """Factory function to create decision gate from config."""
    gate_config = config.get("gate", {})
    return DecisionGate(
        state_dim=gate_config.get("state_dim", 8),
        hidden_dim=gate_config.get("hidden_dim", 128),
        num_actions=gate_config.get("num_actions", 3),
        dropout=gate_config.get("dropout", 0.1),
    )