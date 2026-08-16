"""
Unit tests for SentinelSecurityWall and IncidentReporter desktop security modules.
"""

import pytest
import os
import tempfile
from PIL import Image
from src.security_wall.incident_reporter import IncidentReporter
from src.security_wall.desktop_wall import SentinelSecurityWall
from src.integration.agent_wrapper import SentinelDecision


def test_incident_reporter():
    """Test incident reporting and HTML dashboard generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = IncidentReporter(output_dir=temp_dir)
        dummy_img = Image.new("RGB", (200, 200), color="blue")
        decision = SentinelDecision(
            action="HARD_BLOCK",
            risk_score=0.92,
            category="destructive",
            category_conf=0.95,
            bbox=(0.1, 0.1, 0.5, 0.5),
            reasoning="Attempted deletion of sensitive database cluster.",
        )

        record = reporter.report_incident(
            frame=dummy_img,
            decision=decision,
            action_desc="click on button#drop-db",
            agent_name="TestAgent",
        )

        assert record["status"] == "INTERCEPTED_AND_BLOCKED"
        assert record["risk_score"] == 0.92
        assert record["category"] == "destructive"
        assert os.path.exists(record["screenshot_path"])

        # Test HTML summary dashboard generation
        html_path = reporter.generate_html_summary()
        assert os.path.exists(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "SENTINEL-Vision Visual Security Wall" in content
            assert "INC-" in content


def test_security_wall_monitor_action():
    """Test SentinelSecurityWall monitor_action interception flow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        wall = SentinelSecurityWall(device="cpu")
        wall.reporter = IncidentReporter(output_dir=temp_dir)

        test_frame = Image.new("RGB", (224, 224), color=(200, 50, 50))
        should_proceed, decision, incident = wall.monitor_action(
            current_frame=test_frame,
            action_type="click",
            selector="button#transfer-funds",
            agent_name="Autonomous Financial Bot",
        )

        assert isinstance(should_proceed, bool)
        assert isinstance(decision, SentinelDecision)
        assert decision.action in ["ALLOW", "PAUSE", "HARD_BLOCK"]
