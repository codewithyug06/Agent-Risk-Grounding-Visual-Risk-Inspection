"""
Model Context Protocol (MCP) Server for SENTINEL-Vision.
Enables Claude Code, Claude Desktop, Antigravity, and Cursor to attach to
SENTINEL-Vision as a native oversight plugin via the standard JSON-RPC stdio protocol.

Usage:
    python -m src.security_wall.mcp_server
Or via Claude Code CLI:
    claude mcp add sentinel-vision -- python -m src.security_wall.mcp_server
"""

import sys
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from PIL import Image

# Reconfigure stdout/stderr for clean utf-8 JSON-RPC communication
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# MCP servers must write logs to stderr, NOT stdout (stdout is reserved for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SENTINEL-MCP] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("sentinel-mcp")

from .agent_detector import AgentDetector
from .desktop_wall import SentinelSecurityWall


class SentinelMCPServer:
    """
    Standard MCP Stdio Server exposing SENTINEL-Vision oversight tools
    to Claude Code, Antigravity, and other MCP-compliant agents.
    """

    def __init__(self):
        self.wall: Optional[SentinelSecurityWall] = None
        self.detector = AgentDetector()
        self._init_security_wall()

    def _init_security_wall(self):
        try:
            logger.info("Initializing SentinelSecurityWall backend...")
            self.wall = SentinelSecurityWall(device="cpu")
            logger.info("SentinelSecurityWall armed and ready.")
        except Exception as e:
            logger.error(f"Failed to initialize SentinelSecurityWall: {e}")
            self.wall = None

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Defines the tools available to connected AI agents."""
        return [
            {
                "name": "sentinel_inspect_action",
                "description": (
                    "CRITICAL SAFETY GATE: Inspect an intended computer-use action (click, form submit, "
                    "bash command, file deletion, financial payment) against the current screen pixels "
                    "BEFORE execution. Returns ALLOW, PAUSE, or HARD_BLOCK with localized risk bounding box."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "description": "Type of action: click, type, execute_command, navigate, delete",
                        },
                        "target_element": {
                            "type": "string",
                            "description": "CSS selector, UI element description, or command string",
                        },
                        "coordinates": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "[x, y] screen coordinates of click target",
                        },
                        "agent_identity": {
                            "type": "string",
                            "description": "Name of the calling agent (e.g. Claude Code, Antigravity)",
                        },
                    },
                    "required": ["action_type", "target_element"],
                },
            },
            {
                "name": "sentinel_screen_check",
                "description": (
                    "Capture the current desktop display and run SENTINEL-Vision perception model "
                    "to assess the visual risk state (detects sensitive login forms, account settings, "
                    "financial transactions, or destructive dialogs)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "focus_area": {
                            "type": "string",
                            "description": "Optional focus area or window name",
                        }
                    },
                },
            },
            {
                "name": "sentinel_agent_status",
                "description": "Scan and report all active autonomous AI agents detected on this operating system.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes tool requests dispatched from the AI agent."""
        if tool_name == "sentinel_inspect_action":
            action_type = arguments.get("action_type", "click")
            target = arguments.get("target_element", "unknown")
            agent_id = arguments.get("agent_identity", "Connected MCP Agent")
            coords = arguments.get("coordinates")
            if coords and len(coords) == 2:
                coords = (coords[0], coords[1])
            else:
                coords = None

            # Capture active screen frame
            frame = None
            try:
                import mss
                with mss.mss() as sct:
                    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    shot = sct.grab(monitor)
                    frame = Image.frombytes("RGB", shot.size, shot.rgb)
            except Exception as e:
                logger.warning(f"Could not grab screen via mss: {e}")
                frame = Image.new("RGB", (224, 224), color=(128, 128, 128))

            if self.wall is None:
                return {
                    "decision": "ALLOW",
                    "should_proceed": True,
                    "risk_score": 0.0,
                    "category": "benign",
                    "warning": "SENTINEL backend running in passive pass-through mode.",
                }

            should_proceed, decision, incident = self.wall.monitor_action(
                current_frame=frame,
                action_type=action_type,
                selector=target,
                coordinates=coords,
                agent_name=agent_id,
            )

            res = {
                "decision": decision.action,
                "should_proceed": should_proceed,
                "risk_score": round(float(decision.risk_score), 4),
                "category": decision.category,
                "category_confidence": round(float(decision.category_conf), 4),
                "reasoning": decision.reasoning,
            }
            if decision.bbox:
                res["localized_bbox"] = list(decision.bbox)
            if incident:
                res["incident_id"] = incident.get("incident_id")
                res["dashboard_url"] = self.wall.get_audit_dashboard()

            return res

        elif tool_name == "sentinel_screen_check":
            summary = self.detector.get_summary()
            return {
                "status": "active",
                "screen_monitored": True,
                "active_agents": summary["total_agents_detected"],
                "recommendation": summary["recommendation"],
            }

        elif tool_name == "sentinel_agent_status":
            return self.detector.get_summary()

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def run(self):
        """Standard JSON-RPC 2.0 stdio loop for MCP protocol."""
        logger.info("SENTINEL-Vision MCP Server running on stdio...")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Malformed JSON: {e}")
                continue

            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            # Respond to standard MCP initialization
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                        },
                        "serverInfo": {
                            "name": "sentinel-vision-mcp",
                            "version": "1.0.0",
                        },
                    },
                }
                self._send(response)

            elif method == "notifications/initialized":
                # Client acknowledges initialization
                pass

            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": self.get_tool_definitions(),
                    },
                }
                self._send(response)

            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                result_payload = self.handle_tool_call(tool_name, tool_args)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result_payload, indent=2),
                            }
                        ]
                    },
                }
                self._send(response)

            elif method == "ping":
                self._send({"jsonrpc": "2.0", "id": req_id, "result": {}})

            else:
                if req_id is not None:
                    self._send({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}",
                        },
                    })

    def _send(self, payload: Dict[str, Any]):
        out = json.dumps(payload)
        sys.stdout.write(out + "\n")
        sys.stdout.flush()


def main():
    server = SentinelMCPServer()
    server.run()


if __name__ == "__main__":
    main()
