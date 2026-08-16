"""
Desktop Security Wall Service for SENTINEL-Vision.
Continuously watches agent previews / browser windows and blocks harmful malpractice.
"""

import asyncio
import logging
import time
from typing import Optional, Tuple, Dict, Any, Callable
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf

from ..models.sentinel_model import SentinelModel, create_sentinel_model
from ..gate.decision_gate import DecisionGate, create_decision_gate
from ..integration.agent_wrapper import SentinelWrapper, AgentAction, SentinelDecision
from .incident_reporter import IncidentReporter

logger = logging.getLogger(__name__)


class SentinelSecurityWall:
    """
    Personal Security Firewall for Computer-Use AI Agents.
    Watches screen pixels, intercepts dangerous clicks/actions, and reports malpractice.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        gate_checkpoint: Optional[str] = None,
        device: str = "cpu",
        risk_threshold: float = 0.5,
        on_incident_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.device = device
        self.risk_threshold = risk_threshold
        self.on_incident_callback = on_incident_callback

        # Load minimal configuration
        self.config = OmegaConf.create({
            "backbone": "vit_small_patch16_224",
            "pretrained": False,
            "freeze_backbone": False,
            "freeze_epochs": 0,
            "temporal_fusion": {
                "num_layers": 2,
                "num_heads": 4,
                "embed_dim": 384,
                "dropout": 0.1,
                "use_delta_features": True,
            },
            "risk_head": {
                "dropout": 0.1,
                "hidden_dim": 64,
            },
            "localization_head": {
                "heatmap_size": 14,
                "dropout": 0.1,
                "use_fpn": False,
            },
            "gate": {
                "state_dim": 8,
                "hidden_dim": 128,
                "num_actions": 3,
            },
            "frame_window": {
                "k": 6,
                "resolution": [224, 224],
            },
            "image_size": 224,
        })

        # Initialize core wrapper
        self.wrapper = SentinelWrapper(
            config=self.config,
            sentinel_checkpoint=model_checkpoint,
            gate_checkpoint=gate_checkpoint,
            device=device,
        )

        self.reporter = IncidentReporter()
        self.is_running = False
        logger.info("🛡️ SentinelSecurityWall initialized and armed.")

    def monitor_action(
        self,
        current_frame: Image.Image,
        action_type: str = "click",
        selector: Optional[str] = None,
        coordinates: Optional[Tuple[int, int]] = None,
        agent_name: str = "Claude Computer Use / Browser Agent",
    ) -> Tuple[bool, SentinelDecision, Optional[Dict[str, Any]]]:
        """
        Inspect an upcoming action against current screen pixels before it executes.

        Returns:
            should_proceed (bool): True if safe to execute, False if blocked.
            decision (SentinelDecision): Visual risk decision object.
            incident_report (Optional[Dict]): Incident report dict if blocked.
        """
        # Add frame to temporal buffer
        self.wrapper.add_frame(current_frame)

        action = AgentAction(
            action_type=action_type,
            selector=selector,
            coordinates=coordinates,
        )

        decision, should_proceed = self.wrapper.intercept_action(action)

        incident_data = None
        if not should_proceed or decision.action in ["HARD_BLOCK", "PAUSE"]:
            # Report malpractice
            action_str = f"{action_type} on {selector or coordinates or 'target UI element'}"
            incident_data = self.reporter.report_incident(
                frame=current_frame,
                decision=decision,
                action_desc=action_str,
                agent_name=agent_name,
            )

            if self.on_incident_callback:
                self.on_incident_callback(incident_data)

        return should_proceed, decision, incident_data

    def get_audit_dashboard(self) -> str:
        """Returns file path to the generated HTML security audit dashboard."""
        return self.reporter.generate_html_summary()
