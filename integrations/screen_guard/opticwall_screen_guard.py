#!/usr/bin/env python3
"""
OpticWall Universal Screen Guard
================================
Provides autonomous screen-control oversight for:
- PyAutoGUI / OS Automation Bots
- OpenAI Operator & CUA
- Browser-Use & Playwright Web Bots
- Desktop RPA Agents (UiPath, Automation Anywhere)
- Any script or agent driving mouse & keyboard actions

Operates strictly on screen pixels to intercept visual malpractice before execution.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from PIL import ImageGrab

try:
    from opticwall import OpticWall
except ImportError:
    # Allow running directly from source tree
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.security_wall.desktop_wall import OpticWall


def run_screen_guard(threshold: float = 0.50, interval: float = 1.0, agent_name: str = "Screen Control Agent"):
    print("=" * 66)
    print("  OPTICWALL UNIVERSAL SCREEN GUARD")
    print("  Visual Security Firewall for Autonomous Agents & Screen Control")
    print("=" * 66)
    print(f"[*] Initializing OpticWall Engine (Threshold: {threshold:.2f})...")

    wall = OpticWall(device="cpu", risk_threshold=threshold)
    print(f"[+] Engine Armed. Monitoring screen buffer every {interval}s.")
    print("[+] Press Ctrl+C at any time to disarm.\n")

    consecutive_safe = 0

    try:
        while True:
            # Capture current desktop display buffer
            screenshot = ImageGrab.grab()

            # Run visual risk inspection
            should_proceed, decision, incident = wall.monitor_action(
                current_frame=screenshot,
                action_type="screen_observation",
                selector="active_desktop_window",
                agent_name=agent_name,
            )

            if not should_proceed:
                print("\n" + "!" * 66)
                print(f"[!] MALPRACTICE INTERCEPTED: {decision.action}")
                print(f"[!] Risk Score: {decision.risk_score:.2%} ({decision.category})")
                print(f"[!] Target UI:  {decision.reasoning}")
                if incident:
                    print(f"[!] Forensic Log: {incident.get('incident_id')}")
                print("!" * 66 + "\n")
                consecutive_safe = 0
            else:
                consecutive_safe += 1
                if consecutive_safe % 10 == 0:
                    print(f"[*] OpticWall Guard Active: Screen verified safe (Score: {decision.risk_score:.2%})")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[*] OpticWall Screen Guard disarmed by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpticWall Universal Screen Guard")
    parser.add_argument("--threshold", type=float, default=0.50, help="Risk threshold (0.0 - 1.0)")
    parser.add_argument("--interval", type=float, default=1.0, help="Screen check interval in seconds")
    parser.add_argument("--agent-name", type=str, default="Screen Control Agent", help="Name of monitored agent")
    args = parser.parse_args()

    run_screen_guard(threshold=args.threshold, interval=args.interval, agent_name=args.agent_name)
