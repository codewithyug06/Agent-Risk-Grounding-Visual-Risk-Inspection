"""
CLI Entrypoint for SENTINEL-Vision Desktop Security Wall.
Usage:
    sentinel-wall start       # Start live visual security monitoring
    sentinel-wall dashboard   # Open visual malpractice incident log
    sentinel-wall demo        # Run simulated agent interception demo
"""

import signal
import sys
import os
import argparse
import webbrowser
from PIL import Image
import numpy as np
import logging

from .desktop_wall import SentinelSecurityWall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SentinelWall")


def _notify(title: str, message: str) -> None:
    """Best-effort desktop toast notification. Never fatal if unavailable
    (e.g. plyer not installed, or running headless in CI)."""
    try:
        from plyer import notification

        notification.notify(title=title, message=message, app_name="SENTINEL-Vision", timeout=6)
    except Exception as exc:
        logger.debug("Desktop notification unavailable: %s", exc)


def run_start(device: str = "cpu"):
    """Real always-on background monitoring loop (system-tray-less console
    mode). Watches the desktop screen continuously and blocks/reports
    HARD_BLOCK/PAUSE decisions until interrupted (Ctrl+C)."""
    print("=================================================================")
    print("[*] SENTINEL-Vision: Background Security Wall -- ACTIVE")
    print("    Watching desktop screen. Press Ctrl+C to stop.")
    print("=================================================================")

    wall = SentinelSecurityWall(device=device)
    stop_requested = {"flag": False}

    def _handle_sigint(signum, frame):
        stop_requested["flag"] = True
        wall.stop()

    signal.signal(signal.SIGINT, _handle_sigint)

    def on_tick(decision):
        if decision.action in ("HARD_BLOCK", "PAUSE"):
            print(
                f"[!] {decision.action} risk={decision.risk_score:.2%} "
                f"category={decision.category} -- {decision.reasoning}"
            )
            _notify(
                f"SENTINEL-Vision: {decision.action}",
                f"{decision.category.upper()} risk detected ({decision.risk_score:.0%})",
            )

    try:
        wall.run_forever(on_tick=on_tick)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[+] Security Wall stopped. Incident log:")
        print(f"    {wall.get_audit_dashboard()}")


def run_demo():
    """Simulates an AI agent attempting a destructive financial operation and verifies the wall intercepts it."""
    print("=================================================================")
    print("[*] SENTINEL-Vision: Desktop Visual Security Firewall Simulation")
    print("=================================================================")
    print("[1] Initializing Security Wall on local CPU...")
    wall = SentinelSecurityWall(device="cpu")

    print("[2] Simulating AI Agent browsing to account settings...")
    for i in range(5):
        dummy_screen = Image.new("RGB", (224, 224), color=(100 + i * 20, 100, 100))
        wall.wrapper.add_frame(dummy_screen)

    print("[3] Agent initiates potentially harmful action: 'Confirm $499.00 Payment' / 'Delete Account'...")
    suspicious_frame = Image.new("RGB", (224, 224), color=(220, 30, 30))

    should_proceed, decision, incident = wall.monitor_action(
        current_frame=suspicious_frame,
        action_type="click",
        selector="button#confirm-purchase",
        agent_name="Autonomous Browser Assistant",
    )

    print("\n------------------ SECURITY WALL VERDICT ------------------")
    print(f"Decision:         {decision.action}")
    print(f"Risk Score:       {decision.risk_score:.2%}")
    print(f"Category:         {decision.category.upper()}")
    print(f"Should Proceed:   {should_proceed}")
    print(f"Action Intercept: {'[BLOCKED & REPORTED]' if not should_proceed else '[ALLOWED]'}")
    print("-----------------------------------------------------------")

    dashboard_url = wall.get_audit_dashboard()
    print(f"\n[+] Incident Report & Evidence Log saved to:")
    print(f"   {dashboard_url}")
    print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="SENTINEL-Vision Visual Security Wall CLI")
    parser.add_argument("command", choices=["start", "dashboard", "demo"], help="Command to execute")
    parser.add_argument("--device", default="cpu", help="Inference device for 'start' (cpu or cuda)")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "dashboard":
        wall = SentinelSecurityWall(device="cpu")
        dashboard_path = wall.get_audit_dashboard()
        print(f"Opening Security Audit Dashboard: {dashboard_path}")
        webbrowser.open(f"file:///{dashboard_path}")
    elif args.command == "start":
        run_start(device=args.device)


if __name__ == "__main__":
    main()
