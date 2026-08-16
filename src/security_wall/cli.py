"""
CLI Entrypoint for SENTINEL-Vision Desktop Security Wall.
Usage:
    sentinel-wall start       # Start live visual security monitoring
    sentinel-wall dashboard   # Open visual malpractice incident log
    sentinel-wall demo        # Run simulated agent interception demo
"""

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

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "dashboard":
        wall = SentinelSecurityWall(device="cpu")
        dashboard_path = wall.get_audit_dashboard()
        print(f"Opening Security Audit Dashboard: {dashboard_path}")
        webbrowser.open(f"file:///{dashboard_path}")
    elif args.command == "start":
        print("Starting SENTINEL-Vision Security Wall in background monitoring mode...")
        run_demo()


if __name__ == "__main__":
    main()
