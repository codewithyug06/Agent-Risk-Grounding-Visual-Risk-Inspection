"""
OpticWall Lightweight Python SDK
================================
Drop-in client for Python agents, PyAutoGUI scripts, LangChain tools, and OpenAI Operator.
Connects to local OpticWall daemon (http://localhost:3000) or runs in-process.
"""

import io
import base64
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Tuple
from PIL import Image


class OpticWallClient:
    """Lightweight client connecting to OpticWall visual security firewall."""

    def __init__(self, endpoint: str = "http://localhost:3000", threshold: float = 0.50):
        self.endpoint = endpoint.rstrip("/")
        self.threshold = threshold

    def is_online(self) -> bool:
        """Check if local OpticWall engine is running."""
        try:
            req = urllib.request.Request(f"{self.endpoint}/api/health", headers={"User-Agent": "OpticWall-SDK"})
            with urllib.request.urlopen(req, timeout=1.0) as res:
                return res.status == 200
        except Exception:
            return False

    def inspect(
        self,
        image: Image.Image,
        action_type: str = "click",
        selector: str = "unspecified",
        agent_name: str = "Autonomous Agent"
    ) -> Dict[str, Any]:
        """
        Inspect visual screen frame before executing an action.
        Returns evaluation dict with:
        - `decision`: "ALLOW", "PAUSE", or "HARD_BLOCK"
        - `should_proceed`: bool
        - `risk_score`: float
        - `category`: str
        - `reasoning`: str
        - `bbox`: [y1, x1, y2, x2]
        """
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "image_b64": b64_data,
            "action_type": action_type,
            "selector": selector,
            "threshold": self.threshold,
            "agent_name": agent_name
        }

        # Encode form data
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/api/inspect",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5.0) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.URLError as e:
            # Fallback heuristic if local API isn't running
            is_harm = any(w in selector.lower() for w in ["drop", "delete", "wipe", "pay", "order", "export", "rm"])
            return {
                "decision": "HARD_BLOCK" if is_harm else "ALLOW",
                "should_proceed": not is_harm,
                "risk_score": 0.95 if is_harm else 0.05,
                "category": "DESTRUCTIVE" if is_harm else "BENIGN",
                "reasoning": "Heuristic fallback evaluation (local server offline)",
                "bbox": [0.0, 0.0, 1.0, 1.0]
            }


# Backwards compatibility alias
SentinelClient = OpticWallClient
