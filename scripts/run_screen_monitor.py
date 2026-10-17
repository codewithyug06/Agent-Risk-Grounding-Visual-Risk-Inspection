"""
Live screen monitor for SENTINEL-Vision.

Watches whatever is on the screen (including a browser driven by any agent,
for example Antigravity) and shows the detector's risk score, category and
bounding box in a live window. It logs every decision to logs/screen_monitor.

MONITOR-ONLY: decisions are reported as MONITOR and never block or pause an
action. The detector is not validated for enforcement: on clean human labels
(results/misactbench_gold_eval.json) its precision is at chance level.

Usage:
    python scripts/run_screen_monitor.py            # live window
    python scripts/run_screen_monitor.py --smoke    # load model and capture one frame, then exit
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKPOINT = "checkpoints/stage_c/best_loc.pt"
GOLD_RESULTS = ROOT / "results" / "misactbench_gold_eval.json"


def banner_text() -> str:
    if GOLD_RESULTS.exists():
        g = json.loads(GOLD_RESULTS.read_text(encoding="utf-8"))["test"]
        return (f"MONITOR ONLY | clean-label test: precision {g['precision']:.2f} "
                f"(base rate ~0.24), FPR {g['fpr']:.2f}, AUPRC {g['auprc']:.2f}. "
                f"Not a safety gate.")
    return "MONITOR ONLY | no gold evaluation found. Not a safety gate."


def run_tk_window(monitor, fps: float):
    import time
    import tkinter as tk

    from src.integration.agent_wrapper import AgentAction

    root = tk.Tk()
    root.title("SENTINEL-Vision Screen Monitor (monitor only)")
    root.configure(bg="#111")
    banner = tk.Label(root, text=banner_text(), fg="#ffb347", bg="#111", wraplength=900, justify="left")
    banner.pack(padx=10, pady=(10, 4), anchor="w")
    status = tk.Label(root, text="starting...", fg="#eee", bg="#111", font=("Consolas", 14), justify="left")
    status.pack(padx=10, pady=10, anchor="w")
    counts = {"n": 0}

    def tick():
        t0 = time.perf_counter()
        try:
            frame = monitor.capture.capture()
            monitor.sentinel.add_frame(frame)
            prediction = monitor.sentinel.predict()
            action = AgentAction(action_type="monitor", timestamp=time.time())
            decision = monitor.sentinel.decide(prediction, action)
            if monitor.logger:
                monitor.logger.log(decision, action, frame)
            counts["n"] += 1
            cat = decision.category or "-"
            lines = [
                f"action: {decision.action}   (monitor only, nothing is blocked)",
                f"risk score: {decision.risk_score:.3f}",
                f"category:   {cat}  (conf {float(decision.category_conf):.2f})",
                f"box:        {decision.bbox}",
                f"decisions logged: {counts['n']}   latency: {(time.perf_counter()-t0)*1000:.0f} ms",
            ]
            status.config(text="\n".join(lines))
        except Exception as exc:
            status.config(text=f"error: {exc}")
        root.after(max(1, int(1000 / fps)), tick)

    root.after(10, tick)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="load model and capture one frame, then exit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args()

    from src.integration.live_monitor import LiveMonitor, MonitorConfig

    print(banner_text())
    config = MonitorConfig(
        sentinel_checkpoint=CHECKPOINT,
        gate_checkpoint=None,
        device=args.device,
        capture_fps=args.fps,
        save_dir="logs/screen_monitor",
        monitor_only=True,
    )
    monitor = LiveMonitor(config)

    if args.smoke:
        frame = monitor.capture.capture()
        print(f"captured screen frame: {frame.size[0]}x{frame.size[1]}")
        print("smoke test OK: model loaded, screen capture working, monitor-only mode set")
        return

    run_tk_window(monitor, args.fps)


if __name__ == "__main__":
    main()
