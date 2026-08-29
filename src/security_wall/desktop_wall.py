"""
Desktop Security Wall Service for SENTINEL-Vision.
Continuously watches agent previews / browser windows and blocks harmful malpractice.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf

from ..models.sentinel_model import SentinelModel, create_sentinel_model
from ..gate.decision_gate import DecisionGate, create_decision_gate
from ..integration.agent_wrapper import SentinelWrapper, AgentAction, SentinelDecision
from ..utils.constants import DEFAULT_SENTINEL_CHECKPOINT, DEFAULT_GATE_CHECKPOINT
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
        model_checkpoint: Optional[str] = DEFAULT_SENTINEL_CHECKPOINT,
        gate_checkpoint: Optional[str] = DEFAULT_GATE_CHECKPOINT,
        device: str = "cpu",
        risk_threshold: float = 0.5,
        on_incident_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.device = device
        self.risk_threshold = risk_threshold
        self.on_incident_callback = on_incident_callback

        if not (model_checkpoint and Path(model_checkpoint).exists()):
            logger.warning(
                "SentinelSecurityWall starting WITHOUT a trained checkpoint (%s not found). "
                "Risk scores will be meaningless random-init output, not real detections. "
                "This is only acceptable for architecture smoke tests.",
                model_checkpoint,
            )

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

        # auto_capture=False: the frame was already pushed above via
        # add_frame(current_frame) -- intercept_action must not pull a
        # second, unrelated frame from self.wrapper.capture (which has no
        # real screen source wired up for this desktop-wall usage pattern).
        decision, should_proceed = self.wrapper.intercept_action(action, auto_capture=False)

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

    def run_forever(
        self,
        capture_interval_sec: float = 1.0 / 3.0,
        on_tick: Optional[Callable[[SentinelDecision], None]] = None,
    ):
        """
        Continuous background monitoring loop: grabs the desktop screen at
        `capture_interval_sec`, runs a risk decision on every tick, and
        reports/blocks on PAUSE/HARD_BLOCK.

        This is the real implementation backing `sentinel-wall start` --
        previously that CLI command just re-ran the one-shot demo and never
        actually watched the screen.
        """
        import mss

        self.is_running = True
        logger.info("SentinelSecurityWall.run_forever: continuous background monitoring started.")
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                while self.is_running:
                    tick_start = time.time()
                    shot = sct.grab(monitor)
                    frame = Image.frombytes("RGB", shot.size, shot.rgb)

                    should_proceed, decision, incident = self.monitor_action(
                        current_frame=frame,
                        action_type="monitor",
                        agent_name="Desktop Screen Monitor",
                    )

                    if on_tick:
                        on_tick(decision)

                    elapsed = time.time() - tick_start
                    time.sleep(max(0.0, capture_interval_sec - elapsed))
        finally:
            self.is_running = False
            logger.info("SentinelSecurityWall.run_forever: stopped.")

    def stop(self):
        """Signal run_forever's loop to exit after its current tick."""
        self.is_running = False
