"""
Incident Reporting and Visual Alerting Module for SENTINEL-Vision Security Wall.
Generates structured security alerts, visual audit screenshots, and malpractice reports.
"""

import html as html_lib
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IncidentReporter:
    """
    Logs and formats malpractice incidents when an AI agent attempts a harmful action.
    """

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self.output_dir = Path.home() / ".sentinel_vision" / "incidents"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "incident_history.jsonl"

    def report_incident(
        self,
        frame: Image.Image,
        decision: Any,
        action_desc: Optional[str] = None,
        agent_name: str = "Autonomous Agent",
    ) -> Dict[str, Any]:
        """
        Record a security incident, annotate the risky screenshot, and generate a malpractice report.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        incident_id = f"INC-{timestamp_str}"

        # 1. Create annotated screenshot with visual bounding box & heatmap overlay
        annotated_image = frame.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated_image)

        # Draw red warning border
        w, h = annotated_image.size
        draw.rectangle([0, 0, w - 1, h - 1], outline="red", width=6)

        # Draw localized UI bounding box if available
        bbox = getattr(decision, "bbox", None)
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            if max(abs(x2), abs(y2)) <= 1.0:
                bx0, by0, bx1, by1 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
                bx0, by0, bx1, by1 = x1, y1, x2, y2

            x_min, x_max = max(0, min(bx0, bx1)), min(w - 1, max(bx0, bx1))
            y_min, y_max = max(0, min(by0, by1)), min(h - 1, max(by0, by1))

            if x_max > x_min and y_max > y_min:
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=4)
                draw.rectangle([x_min, max(0, y_min - 25), min(w, x_min + 180), y_min], fill="red")
                draw.text((x_min + 5, max(0, y_min - 22)), f"RISK UI: {getattr(decision, 'category', 'harmful').upper()}", fill="white")

        # Save annotated image
        image_path = self.output_dir / f"{incident_id}.png"
        annotated_image.save(image_path)

        risk_score = float(getattr(decision, "risk_score", 0.0))
        category = str(getattr(decision, "category", "unknown"))
        reasoning = str(getattr(decision, "reasoning", f"Agent attempted high-risk {category} operation."))

        incident_record = {
            "incident_id": incident_id,
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "decision": str(getattr(decision, "action", "HARD_BLOCK")),
            "risk_score": risk_score,
            "category": category,
            "action_attempted": action_desc or "Direct UI interaction / click",
            "reasoning": reasoning,
            "screenshot_path": str(image_path.resolve()),
            "status": "INTERCEPTED_AND_BLOCKED",
        }

        # Append to JSONL history
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(incident_record) + "\n")

        logger.warning(
            f"[SENTINEL SECURITY WALL] Action Blocked! Incident ID: {incident_id} | "
            f"Category: {category} | Risk: {risk_score:.2f} | Evidence: {image_path}"
        )

        return incident_record

    def generate_html_summary(self) -> str:
        """Generate a clean visual dashboard HTML page of all intercepted incidents."""
        if not self.log_file.exists():
            return "<html><body><h2>No security incidents recorded.</h2></body></html>"

        records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        records.reverse()  # Newest first

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SENTINEL-Vision Security Wall - Incident Audit Log</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .header {{ border-bottom: 2px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border-radius: 8px; border-left: 6px solid #ef4444; padding: 18px; margin-bottom: 18px; display: flex; gap: 20px; }}
        .badge {{ background: #ef4444; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .thumb {{ max-width: 320px; border-radius: 6px; border: 1px solid #475569; }}
        .info {{ flex: 1; }}
        h3 {{ margin-top: 0; color: #f87171; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ SENTINEL-Vision Visual Security Wall</h1>
        <p>Real-Time Audit & Malpractice Incident History | Total Interceptions: <b>{len(records)}</b></p>
    </div>
"""
        for r in records:
            # All fields below can be influenced by attacker-controlled page
            # content (an agent's action selector/text, "agent_name", etc.),
            # so every value must be HTML-escaped before embedding -- this
            # dashboard is opened as a local file in a real browser.
            esc = html_lib.escape
            screenshot_path = str(r.get("screenshot_path", ""))
            decision = esc(str(r.get("decision", "")))
            incident_id = esc(str(r.get("incident_id", "")))
            category = esc(str(r.get("category", "unknown")).upper())
            risk_score = float(r.get("risk_score", 0) or 0)
            agent_name = esc(str(r.get("agent_name", "")))
            timestamp = esc(str(r.get("timestamp", "")))
            action_attempted = esc(str(r.get("action_attempted", "")))
            reasoning = esc(str(r.get("reasoning", "")))
            # Screenshot paths are generated by this module itself (never
            # attacker-controlled), but escape the attribute value too since
            # it is still user-visible file-system data.
            img_src = "file:///" + esc(screenshot_path.replace("\\", "/"))

            html += f"""
    <div class="card">
        <img class="thumb" src="{img_src}" alt="Evidence Screenshot">
        <div class="info">
            <span class="badge">{decision}</span>
            <h3>{incident_id} — {category} (Risk: {risk_score:.2%})</h3>
            <p><b>Agent:</b> {agent_name} | <b>Time:</b> {timestamp}</p>
            <p><b>Attempted Action:</b> <code>{action_attempted}</code></p>
            <p><b>Visual Oversight Analysis:</b> {reasoning}</p>
        </div>
    </div>
"""
        html += "</body></html>"
        dashboard_path = self.output_dir / "security_dashboard.html"
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html)

        return str(dashboard_path.resolve())
