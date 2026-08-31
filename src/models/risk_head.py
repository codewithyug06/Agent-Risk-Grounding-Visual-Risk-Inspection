import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskHead(nn.Module):
    """
    Risk classification head.
    Two outputs from fused representation:
    1. Binary risk score (0-1) via sigmoid
    2. 5-class category (destructive/financial/privacy/irreversible_external/benign) via softmax

    Uses global average pool of fused spatial embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_categories: int = 5,
        dropout: float = 0.1,
        use_cls_token: bool = False,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_categories: Number of harm categories (5)
            dropout: Dropout rate
            use_cls_token: If True, use CLS token; else global average pool
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        self.use_cls_token = use_cls_token

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Binary risk head (harmful vs benign)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Category classification head
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories),
        )

        logger.info(f"RiskHead initialized: embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
                    f"num_categories={num_categories}")

    def forward(
        self,
        fused: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: Fused spatial embeddings (B, N_patches, D) or (B, D) if already pooled
            cls_token: Optional CLS token (B, D) - used if use_cls_token=True

        Returns:
            Dict with:
                - risk_score: (B, 1) sigmoid output [0, 1]
                - category_logits: (B, num_categories) raw logits
                - category_probs: (B, num_categories) softmax probabilities
        """
        # Pool spatial embeddings if needed
        if fused.dim() == 3:
            # (B, N, D) -> global average pool
            if self.use_cls_token and cls_token is not None:
                pooled = cls_token
            else:
                pooled = fused.mean(dim=1)  # (B, D)
        elif fused.dim() == 2:
            pooled = fused
        else:
            raise ValueError(f"Unexpected fused shape: {fused.shape}")

        # Extract features
        features = self.feature_extractor(pooled)  # (B, hidden_dim)

        # Binary risk score
        risk_logits = self.risk_head(features)  # (B, 1)
        risk_score = torch.sigmoid(risk_logits)

        # Category classification
        category_logits = self.category_head(features)  # (B, num_categories)
        category_probs = F.softmax(category_logits, dim=-1)

        return {
            "risk_score": risk_score,          # (B, 1)
            "risk_logits": risk_logits,        # (B, 1)
            "category_logits": category_logits,  # (B, 5)
            "category_probs": category_probs,  # (B, 5)
        }

    def predict_category(self, category_probs: torch.Tensor) -> torch.Tensor:
        """Get predicted category indices."""
        return category_probs.argmax(dim=-1)

    def get_risk_prediction(self, risk_score: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary risk prediction."""
        return (risk_score > threshold).long()


class RiskHeadWithUncertainty(RiskHead):
    """
    Risk head with Monte Carlo Dropout for uncertainty quantification.
    Run multiple forward passes with dropout enabled to estimate epistemic uncertainty.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_categories: int = 5,
        dropout: float = 0.1,
        use_cls_token: bool = False,
        mc_dropout_samples: int = 10,
    ):
        super().__init__(embed_dim, hidden_dim, num_categories, dropout, use_cls_token)
        self.mc_dropout_samples = mc_dropout_samples

    def forward_with_uncertainty(
        self,
        fused: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MC Dropout uncertainty estimation.

        Returns additional keys:
            - risk_uncertainty: (B, 1) epistemic uncertainty (variance)
            - category_uncertainty: (B, num_categories) predictive entropy
        """
        n_samples = n_samples or self.mc_dropout_samples

        # Enable dropout at inference time
        self.train()

        risk_scores = []
        category_probs_list = []

        with torch.no_grad():
            for _ in range(n_samples):
                out = super().forward(fused, cls_token)
                risk_scores.append(out["risk_score"])
                category_probs_list.append(out["category_probs"])

        self.eval()

        # Stack samples
        risk_scores = torch.stack(risk_scores, dim=0)  # (n_samples, B, 1)
        category_probs = torch.stack(category_probs_list, dim=0)  # (n_samples, B, C)

        # Mean predictions
        risk_mean = risk_scores.mean(dim=0)
        category_mean = category_probs.mean(dim=0)

        # Uncertainty estimates
        # Epistemic uncertainty: variance of predictions
        risk_uncertainty = risk_scores.var(dim=0)  # (B, 1)

        # Predictive entropy for categories
        eps = 1e-8
        category_entropy = -(category_mean * (category_mean + eps).log()).sum(dim=-1, keepdim=True)
        category_uncertainty = category_probs.var(dim=0).mean(dim=-1, keepdim=True)  # (B, 1)

        return {
            "risk_score": risk_mean,
            "risk_logits": None,  # Not meaningful with MC
            "category_logits": None,
            "category_probs": category_mean,
            "risk_uncertainty": risk_uncertainty,
            "category_uncertainty": category_uncertainty,
            "risk_samples": risk_scores,
            "category_samples": category_probs,
        }


def create_risk_head(config: Dict) -> RiskHead:
    """Factory function to create risk head from config."""
    risk_config = config.get("risk_head", {})
    return RiskHead(
        embed_dim=config.get("embed_dim", 384),
        hidden_dim=risk_config.get("hidden_dim", 256),
        num_categories=risk_config.get("num_categories", 5),
        dropout=risk_config.get("dropout", 0.1),
        use_cls_token=risk_config.get("use_cls_token", False),
    )
