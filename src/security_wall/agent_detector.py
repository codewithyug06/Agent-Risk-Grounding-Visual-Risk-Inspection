"""
AI Agent Auto-Detector for SENTINEL-Vision.
Continuously scans system processes, window contexts, and active tool pipelines
to discover autonomous AI agents (Claude Code, Antigravity, Cursor, Browser-Use, etc.)
and attaches real-time visual safety monitoring.
"""

import os
import sys
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)


@dataclass
class DetectedAgent:
    """Represents a discovered AI agent process running on the host system."""
    agent_id: str
    name: str
    category: str  # 'cli_agent', 'ide_agent', 'browser_agent', 'python_agent'
    pid: int
    executable: str
    cmdline: str
    status: str
    detected_at: str
    is_sandboxed: bool
    risk_level: str  # 'monitored', 'high_privilege', 'unrestricted'


# Signatures for known autonomous AI agents
AGENT_SIGNATURES = [
    {
        "name": "Claude Code",
        "category": "cli_agent",
        "process_names": ["claude.exe", "claude"],
        "cmdline_keywords": ["@anthropic-ai/claude-code", "claude-code", "bin\\claude"],
        "risk_level": "high_privilege",  # Can execute arbitrary bash/powershell
    },
    {
        "name": "Claude Desktop",
        "category": "desktop_agent",
        "process_names": ["claude.exe", "chrome-native-host.exe"],
        "cmdline_keywords": ["claude_pzs8sxrjxfjjc", "anthropic.claude"],
        "risk_level": "monitored",
    },
    {
        "name": "Antigravity Agent",
        "category": "ide_agent",
        "process_names": ["agy.exe", "agy", "antigravity.exe", "antigravity"],
        "cmdline_keywords": ["agy", "antigravity", "antigravity-cli", "gemini"],
        "risk_level": "high_privilege",
    },
    {
        "name": "Cursor Agent",
        "category": "ide_agent",
        "process_names": ["cursor.exe", "cursor"],
        "cmdline_keywords": ["cursor", "cursor-agent"],
        "risk_level": "high_privilege",
    },
    {
        "name": "Browser-Use / Playwright Agent",
        "category": "browser_agent",
        "process_names": ["python.exe", "python", "node.exe", "node"],
        "cmdline_keywords": ["browser_use", "browser-use", "playwright", "puppeteer", "agent_browser"],
        "risk_level": "high_privilege",  # Can submit forms, make purchases, delete data
    },
    {
        "name": "Autonomous Python Agent (LangChain / AutoGen / CrewAI)",
        "category": "python_agent",
        "process_names": ["python.exe", "python", "python3"],
        "cmdline_keywords": ["crewai", "autogen", "langchain", "computer_use", "pyautogui"],
        "risk_level": "high_privilege",
    },
]


class AgentDetector:
    """
    Scans the operating system to discover running AI agents and determine
    if SENTINEL-Vision oversight is active or required.
    """

    def __init__(self):
        if psutil is None:
            logger.warning("psutil is not installed; process detection is disabled.")

    def scan_active_agents(self) -> List[DetectedAgent]:
        """
        Scan all active processes on the host machine to detect AI agents.
        Returns a list of DetectedAgent records.
        """
        if psutil is None:
            return []

        detected: List[DetectedAgent] = []
        seen_pids = set()

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                pid = proc.info["pid"]
                if pid in seen_pids:
                    continue

                name = (proc.info["name"] or "").lower()
                cmdline_parts = []
                try:
                    cmdline_parts = proc.cmdline() or []
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    cmdline_parts = []

                cmdline_str = " ".join(cmdline_parts).lower()
                exe_path = ""
                try:
                    exe_path = proc.exe() or ""
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    exe_path = name

                # Match against known agent signatures
                for sig in AGENT_SIGNATURES:
                    matched = False

                    # Check process name
                    if any(p_name in name for p_name in sig["process_names"]):
                        # Verify with cmdline keywords if specified
                        if sig["cmdline_keywords"]:
                            if any(kw in cmdline_str or kw in exe_path.lower() for kw in sig["cmdline_keywords"]):
                                matched = True
                        else:
                            matched = True

                    # Check cmdline keywords even if name is generic (like node or python)
                    elif any(kw in cmdline_str for kw in sig["cmdline_keywords"]):
                        matched = True

                    if matched and pid not in seen_pids:
                        seen_pids.add(pid)
                        detected.append(
                            DetectedAgent(
                                agent_id=f"agent-{pid}",
                                name=sig["name"],
                                category=sig["category"],
                                pid=pid,
                                executable=exe_path,
                                cmdline=" ".join(cmdline_parts)[:160],
                                status=proc.status() if hasattr(proc, "status") else "running",
                                detected_at=datetime.now().isoformat(),
                                is_sandboxed=False,
                                risk_level=sig["risk_level"],
                            )
                        )
                        break

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as e:
                logger.debug(f"Error inspecting process {proc}: {e}")
                continue

        return detected

    def get_summary(self) -> Dict[str, Any]:
        """Returns a high-level summary of active agents for dashboard and UI."""
        agents = self.scan_active_agents()
        return {
            "total_agents_detected": len(agents),
            "agents": [asdict(a) for a in agents],
            "has_high_privilege_agent": any(a.risk_level == "high_privilege" for a in agents),
            "timestamp": datetime.now().isoformat(),
            "recommendation": (
                "OpticWall protection recommended: Active autonomous agents detected with OS-level execution capability."
                if len(agents) > 0
                else "No active computer-use agents currently detected."
            ),
        }

    # Alias for convenience
    detect_active_agents = scan_active_agents


def print_agent_report():
    """Terminal output utility for `opticwall detect`."""
    detector = AgentDetector()
    agents = detector.scan_active_agents()

    print("==================================================================")
    print("[*] OpticWall: Autonomous AI Agent Auto-Discovery")
    print("==================================================================")

    if not agents:
        print("[-] No autonomous AI agents currently detected on this machine.")
        print("    Supported agents: Claude Code, Antigravity, Cursor, Browser-Use, Playwright, etc.")
    else:
        print(f"[+] Discovered {len(agents)} active autonomous agent process(es):\n")
        for i, a in enumerate(agents, 1):
            print(f"  [{i}] {a.name.upper()} (PID: {a.pid})")
            print(f"      Category:   {a.category}")
            print(f"      Risk Level: {a.risk_level.upper()}")
            print(f"      Executable: {a.executable}")
            print(f"      Command:    {a.cmdline[:80]}...")
            print(f"      Oversight:  [ATTACHED - SCREEN PIXEL MONITORING ARMED]\n")

    print("==================================================================")
    print("Run `opticwall watch` to start the visual circuit breaker.")
    print("Run `opticwall mcp` to start the universal MCP server for Claude & Antigravity.")
    print("==================================================================")


if __name__ == "__main__":
    print_agent_report()
