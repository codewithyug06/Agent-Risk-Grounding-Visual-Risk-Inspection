"""
CLI Entrypoint for OpticWall.
The Personal Visual Security Firewall for Autonomous AI Agents.

Usage:
    opticwall watch --checkpoint <path> [--threshold 0.6]
    opticwall serve --checkpoint <path> [--port 8000]
    opticwall detect          # Scan and discover active AI agents (Claude, Antigravity, etc.)
    opticwall mcp             # Run universal MCP server for Claude Code & Antigravity
    opticwall web             # Serve global landing page & visual risk sandbox
    opticwall demo            # Simulated interception demo (no checkpoint needed)
    opticwall dashboard       # Open incident audit dashboard
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image

from .desktop_wall import OpticWall, SentinelSecurityWall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("opticwall")


# ---------------------------------------------------------------------------
# demo / dashboard
# ---------------------------------------------------------------------------

def run_demo():
    """Simulates an AI agent attempting a destructive financial operation and verifies OpticWall intercepts it."""
    print("=================================================================")
    print("[*] OpticWall: Desktop Visual Security Firewall Simulation")
    print("=================================================================")
    print("[1] Initializing OpticWall on local CPU...")
    wall = OpticWall(device="cpu")

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


def run_dashboard():
    import webbrowser
    wall = SentinelSecurityWall(device="cpu")
    dashboard_path = wall.get_audit_dashboard()
    print(f"Opening Security Audit Dashboard: {dashboard_path}")
    webbrowser.open(f"file:///{dashboard_path}")


# ---------------------------------------------------------------------------
# watch — real continuous monitoring loop
# ---------------------------------------------------------------------------

def run_watch(checkpoint: str, threshold: float, gate_checkpoint: str = None):
    """
    Runs SENTINEL-Vision's real continuous screen-monitoring loop (LiveMonitor),
    not a scripted demo. Requires a live display for screen capture (mss) — if
    no display is attached, this fails loudly rather than silently monitoring
    a blank frame.
    """
    import asyncio
    from ..integration.live_monitor import LiveMonitor, MonitorConfig
    from ..integration.agent_wrapper import SentinelWrapper
    from ..models.sentinel_model import SentinelModel

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = MonitorConfig(
        sentinel_checkpoint=checkpoint,
        gate_checkpoint=gate_checkpoint,
        device="cpu",
    )

    model_config = SentinelModel.from_pretrained(checkpoint).config

    wrapper = SentinelWrapper(
        config=model_config,
        sentinel_checkpoint=checkpoint,
        gate_checkpoint=gate_checkpoint,
        device="cpu",
        frame_buffer_size=config.frame_window_size,
        target_resolution=config.target_resolution,
    )

    monitor = LiveMonitor(config, sentinel_wrapper=wrapper)
    monitor.on_high_risk = lambda decision: logger.warning(
        f"HIGH RISK ({decision.risk_score:.2%}, {decision.category}): action={decision.action}"
    )

    print(f"Watching screen for risky agent actions (threshold={threshold}). Press Ctrl+C to stop.")
    try:
        asyncio.run(monitor.start())
    except KeyboardInterrupt:
        monitor.stop()
        print(f"\nStopped. Stats: {monitor.get_stats()}")


# ---------------------------------------------------------------------------
# serve — FastAPI intercept API
# ---------------------------------------------------------------------------

def run_serve(checkpoint: str, port: int, gate_checkpoint: str = None):
    from ..integration.intercept_api import initialize_api, app
    from ..models.sentinel_model import SentinelModel
    import uvicorn

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model_config = SentinelModel.from_pretrained(checkpoint).config

    initialize_api(
        config=model_config,
        sentinel_checkpoint=checkpoint,
        gate_checkpoint=gate_checkpoint,
        device="cpu",
    )
    print(f"Serving SENTINEL-Vision intercept API on http://0.0.0.0:{port} (health: /health)")
    uvicorn.run(app, host="0.0.0.0", port=port)


# ---------------------------------------------------------------------------
# benchmark — real evaluation against a labeled test set
# ---------------------------------------------------------------------------

def run_benchmark_cmd(checkpoint: str, test_dir: str):
    import torch
    from ..models.sentinel_model import SentinelModel
    from ..data.loaders import SyntheticInjectionDataset

    model = SentinelModel.from_pretrained(checkpoint)
    model.eval()

    test_dir_path = Path(test_dir)
    if not (test_dir_path / "test").exists():
        raise FileNotFoundError(
            f"No 'test' split found under {test_dir}. Expected the "
            "synthetic-injection directory layout (train/val/test/<sample>/frames)."
        )

    dataset = SyntheticInjectionDataset(str(test_dir_path), split="test", k=model.k,
                                         target_resolution=(model.image_size, model.image_size))
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            frames = item["frames"].unsqueeze(0)
            out = model(frames)
            pred = out["risk_score"].item() > 0.5
            true = bool(item["risk_label"].item())
            if pred and true:
                tp += 1
            elif pred and not true:
                fp += 1
            elif not pred and true:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"n_samples={len(dataset)} tp={tp} fp={fp} tn={tn} fn={fn}")
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")


# ---------------------------------------------------------------------------
# inject — synthetic injection dataset generation
# ---------------------------------------------------------------------------

def run_inject(output_dir: str, n_per_category: int, n_benign: int):
    from ..data.synthetic_injection import run_injection_pipeline

    stats = run_injection_pipeline(output_dir=output_dir, n_per_category=n_per_category, n_benign=n_benign)
    print(f"Done. total={stats['total']} harmful={stats['n_harmful']} benign={stats['n_benign']}")
    print(f"by_category={stats['by_category']}")


# ---------------------------------------------------------------------------
# export — ONNX export
# ---------------------------------------------------------------------------

def run_export(checkpoint: str, output: str):
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from scripts.export_onnx import export_pipeline

    output_path = Path(output)
    results = export_pipeline(checkpoint_path=checkpoint, output_dir=str(output_path.parent))
    print(f"Exported: {results['fp32_path']}")
    if results.get("int8_path"):
        print(f"Quantized: {results['int8_path']}")


# ---------------------------------------------------------------------------
# info — model/checkpoint introspection
# ---------------------------------------------------------------------------

def run_info(checkpoint: str):
    from ..models.sentinel_model import SentinelModel

    model = SentinelModel.from_pretrained(checkpoint)
    sizes = model.get_model_size()
    print(f"Checkpoint: {checkpoint}")
    print(f"k={model.k} image_size={model.image_size}")
    for key, val in sizes.items():
        print(f"  {key}: {val:,}")


def run_detect():
    """Discover all active autonomous AI agents on the system."""
    from .agent_detector import print_agent_report
    print_agent_report()


def run_mcp():
    """Run Model Context Protocol (MCP) server for Claude Code & Antigravity."""
    from .mcp_server import main as mcp_main
    mcp_main()


def run_web(port: int = 3000):
    """Serve the global landing page & interactive API platform locally."""
    from .web_server import run_server
    run_server(port=port)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpticWall: Visual Security Firewall for Autonomous AI Agents")
    sub = parser.add_subparsers(dest="command", required=True)

    p_watch = sub.add_parser("watch", help="Run the real continuous screen-monitoring loop")
    p_watch.add_argument("--checkpoint", required=True)
    p_watch.add_argument("--gate-checkpoint", default=None)
    p_watch.add_argument("--threshold", type=float, default=0.6)

    p_serve = sub.add_parser("serve", help="Run the FastAPI intercept server")
    p_serve.add_argument("--checkpoint", required=True)
    p_serve.add_argument("--gate-checkpoint", default=None)
    p_serve.add_argument("--port", type=int, default=8000)

    p_bench = sub.add_parser("benchmark", help="Evaluate a checkpoint on a labeled test directory")
    p_bench.add_argument("--checkpoint", required=True)
    p_bench.add_argument("--test-dir", required=True)

    p_inject = sub.add_parser("inject", help="Generate the synthetic injection dataset")
    p_inject.add_argument("--output-dir", default="data/synthetic_injections")
    p_inject.add_argument("--n-per-category", type=int, default=600)
    p_inject.add_argument("--n-benign", type=int, default=2000)

    p_export = sub.add_parser("export", help="Export a checkpoint to ONNX")
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--output", default="onnx_models/sentinel.onnx")

    p_info = sub.add_parser("info", help="Show model architecture/parameter info for a checkpoint")
    p_info.add_argument("--checkpoint", required=True)

    sub.add_parser("demo", help="Run the simulated interception demo (no checkpoint needed)")
    sub.add_parser("dashboard", help="Open the incident audit dashboard")
    sub.add_parser("detect", help="Scan and detect active autonomous AI agents on the system")
    sub.add_parser("mcp", help="Run the universal MCP server for Claude Code and Antigravity")
    
    p_web = sub.add_parser("web", help="Serve the global landing page & interactive demo")
    p_web.add_argument("--port", type=int, default=3000)

    args = parser.parse_args()

    if args.command == "watch":
        run_watch(args.checkpoint, args.threshold, args.gate_checkpoint)
    elif args.command == "serve":
        run_serve(args.checkpoint, args.port, args.gate_checkpoint)
    elif args.command == "benchmark":
        run_benchmark_cmd(args.checkpoint, args.test_dir)
    elif args.command == "inject":
        run_inject(args.output_dir, args.n_per_category, args.n_benign)
    elif args.command == "export":
        run_export(args.checkpoint, args.output)
    elif args.command == "info":
        run_info(args.checkpoint)
    elif args.command == "demo":
        run_demo()
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "detect":
        run_detect()
    elif args.command == "mcp":
        run_mcp()
    elif args.command == "web":
        run_web(args.port)


if __name__ == "__main__":
    main()
