import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..models.sentinel_model import create_sentinel_model
from ..gate.decision_gate import DecisionGate, create_decision_gate
from ..data.augmentation import create_val_transform
from ..utils.constants import DEFAULT_SENTINEL_CHECKPOINT, DEFAULT_GATE_CHECKPOINT
from .agent_wrapper import SentinelWrapper, AgentAction, SentinelDecision

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for live monitor."""
    sentinel_checkpoint: str = DEFAULT_SENTINEL_CHECKPOINT
    gate_checkpoint: Optional[str] = DEFAULT_GATE_CHECKPOINT
    device: str = "cuda"
    frame_window_size: int = 6
    target_resolution: Tuple[int, int] = (224, 224)
    capture_fps: float = 3.0
    display_fps: float = 30.0
    overlay_alpha: float = 0.5
    show_heatmap: bool = True
    show_bbox: bool = True
    show_risk_bar: bool = True
    log_decisions: bool = True
    save_dir: Optional[str] = None


class FrameCapture:
    """Screen capture for live monitoring."""

    def __init__(self, monitor_index: int = 0):
        self.monitor_index = monitor_index
        self._setup_capture()

    def _setup_capture(self):
        """Setup screen capture."""
        import mss
        self.sct = mss.mss()
        self.monitors = self.sct.monitors

    def capture(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """Capture screen region."""
        if region:
            monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
        else:
            monitor = self.monitors[self.monitor_index + 1] if self.monitor_index + 1 < len(self.monitors) else self.monitors[0]
        screenshot = self.sct.grab(monitor)
        return Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    def capture_numpy(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture as numpy array (BGR for OpenCV)."""
        pil_img = self.capture(region)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class LiveOverlay:
    """Draws SENTINEL-Vision decisions on screen."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """Load font for text overlay."""
        try:
            return ImageFont.truetype("arial.ttf", 16)
        except:
            return ImageFont.load_default()

    def draw_decision(
        self,
        frame: Image.Image,
        decision: SentinelDecision,
        heatmap: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Image.Image:
        """Draw decision overlay on frame."""
        overlay = frame.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")

        w, h = frame.size

        # Color based on decision
        if decision.action == "HARD_BLOCK":
            color = (255, 0, 0, int(255 * self.config.overlay_alpha))
            border_color = (255, 0, 0, 255)
        elif decision.action == "PAUSE":
            color = (255, 165, 0, int(255 * self.config.overlay_alpha))
            border_color = (255, 165, 0, 255)
        else:
            color = (0, 255, 0, int(255 * self.config.overlay_alpha))
            border_color = (0, 255, 0, 255)

        # Draw risk bar at top
        if self.config.show_risk_bar:
            bar_height = 30
            bar_width = int(w * decision.risk_score)
            draw.rectangle([0, 0, w, bar_height], fill=(0, 0, 0, 180))
            draw.rectangle([0, 0, bar_width, bar_height], fill=color)
            # Risk text
            draw.text(
                (10, 5),
                f"RISK: {decision.risk_score:.3f} | {decision.action} | {decision.category} ({decision.category_conf:.2f})",
                fill=(255, 255, 255, 255),
                font=self.font,
            )

        # Draw heatmap overlay
        if self.config.show_heatmap and heatmap is not None:
            heatmap_resized = self._resize_heatmap(heatmap, (w, h))
            heatmap_img = self._heatmap_to_image(heatmap_resized)
            overlay = Image.alpha_composite(overlay.convert("RGBA"), heatmap_img).convert("RGB")
            draw = ImageDraw.Draw(overlay, "RGBA")

        # Draw bounding box
        if self.config.show_bbox and bbox is not None:
            x1, y1, x2, y2 = bbox
            # Convert normalized to pixel
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            draw.rectangle([px1, py1, px2, py2], outline=border_color, width=3)
            # Label
            draw.text(
                (px1, py1 - 20),
                f"{decision.category}: {decision.category_conf:.2f}",
                fill=border_color,
                font=self.font,
            )

        # Draw decision banner at bottom
        banner_height = 60
        draw.rectangle(
            [0, h - banner_height, w, h],
            fill=(0, 0, 0, 200)
        )
        draw.text(
            (10, h - banner_height + 10),
            f"DECISION: {decision.action} | Risk: {decision.risk_score:.3f} | {decision.reasoning[:100]}",
            fill=(255, 255, 255, 255),
            font=self.font,
        )

        return overlay.convert("RGB")

    def _resize_heatmap(self, heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize heatmap to target size."""
        w, h = target_size
        return cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    def _heatmap_to_image(self, heatmap: np.ndarray) -> Image.Image:
        """Convert heatmap to RGBA image."""
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        # Apply colormap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        # Create alpha channel based on intensity
        alpha = (heatmap * 128).astype(np.uint8)
        heatmap_rgba = np.dstack([heatmap_rgb, alpha])
        return Image.fromarray(heatmap_rgba, "RGBA")


class DecisionLogger:
    """Logs decisions for audit trail."""

    def __init__(self, save_dir: Optional[str] = None):
        import threading

        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_buffer = deque(maxlen=1000)
        self._flush_lock = threading.Lock()

    def log(self, decision: SentinelDecision, action: Optional[AgentAction] = None, frame: Optional[Image.Image] = None):
        """Log a decision."""
        entry = {
            "timestamp": decision.timestamp,
            "decision": decision.action,
            "risk_score": decision.risk_score,
            "category": decision.category,
            "category_conf": decision.category_conf,
            "bbox": decision.bbox,
            "reasoning": decision.reasoning,
            "action": {
                "type": action.action_type if action else None,
                "selector": action.selector if action else None,
                "coordinates": action.coordinates if action else None,
                "text": action.text if action else None,
                "url": action.url if action else None,
            } if action else None,
        }
        self.log_buffer.append(entry)

        # Write to file periodically
        if len(self.log_buffer) % 10 == 0 and self.save_dir:
            self._flush()

    def _flush(self):
        """Flush buffer to disk. Locked so concurrent callers (multiple
        capture ticks racing on a background thread) never interleave
        partial lines in the same file."""
        if not self.save_dir:
            return
        import json

        log_file = self.save_dir / f"decisions_{time.strftime('%Y%m%d')}.jsonl"
        with self._flush_lock:
            with open(log_file, "a") as f:
                for entry in list(self.log_buffer)[-10:]:
                    f.write(json.dumps(entry) + "\n")

    def get_recent(self, n: int = 100) -> List[Dict]:
        """Get recent decisions."""
        return list(self.log_buffer)[-n:]


class LiveMonitor:
    """
    Real-time SENTINEL-Vision monitor.
    Continuously captures screen, runs inference, displays overlay.
    """

    def __init__(self, config: MonitorConfig, sentinel_wrapper: Optional[SentinelWrapper] = None):
        self.config = config
        self.running = False

        # Components
        self.capture = FrameCapture()
        self.overlay = LiveOverlay(config)
        self.logger = DecisionLogger(config.save_dir) if config.log_decisions else None

        # SENTINEL wrapper
        if sentinel_wrapper:
            self.sentinel = sentinel_wrapper
        else:
            self.sentinel = SentinelWrapper(
                config=None,  # Will be set via hydra
                sentinel_checkpoint=config.sentinel_checkpoint,
                gate_checkpoint=config.gate_checkpoint,
                device=config.device,
                frame_buffer_size=config.frame_window_size,
                target_resolution=config.target_resolution,
                capture=self.capture,
            )

        # Stats
        self.stats = {
            "frames_processed": 0,
            "decisions_made": 0,
            "avg_latency_ms": 0.0,
            "fps": 0.0,
        }
        self.latencies = deque(maxlen=100)

        # Callbacks
        self.on_decision: Optional[Callable[[SentinelDecision, Optional[AgentAction]], None]] = None
        self.on_high_risk: Optional[Callable[[SentinelDecision], None]] = None
        # Called on the display tick with the latest (frame, decision) so a
        # consumer (tray app, OpenCV GUI, web dashboard) can render it.
        # Previously this was hardwired to a no-op, silently dropping the
        # display tick entirely.
        self.on_display_tick: Optional[Callable[[Optional[Image.Image], Optional[SentinelDecision]], None]] = None
        self._last_frame: Optional[Image.Image] = None
        self._last_decision: Optional[SentinelDecision] = None

    async def start(self):
        """Start live monitoring."""
        self.running = True
        logger.info("Starting LiveMonitor...")

        capture_interval = 1.0 / self.config.capture_fps
        display_interval = 1.0 / self.config.display_fps

        last_capture = 0
        last_display = 0

        while self.running:
            now = time.time()

            # Capture and inference
            if now - last_capture >= capture_interval:
                await self._capture_and_infer()
                last_capture = now

            # Display update (if using GUI)
            if now - last_display >= display_interval:
                self._update_display()
                last_display = now

            await asyncio.sleep(0.01)  # Small sleep to prevent busy loop

    async def _capture_and_infer(self):
        """Capture frame and run inference."""
        start = time.perf_counter()

        # Capture
        frame = self.capture.capture()

        # Add to buffer and predict
        self.sentinel.add_frame(frame)
        prediction = self.sentinel.predict()

        # Create dummy action for decision (in real use, this comes from agent)
        action = AgentAction(action_type="monitor", timestamp=time.time())

        # Decide
        decision = self.sentinel.decide(prediction, action)

        self._last_frame = frame
        self._last_decision = decision

        # Log
        if self.logger:
            self.logger.log(decision, action, frame)

        # Callbacks
        if self.on_decision:
            self.on_decision(decision, action)

        if self.on_high_risk and decision.risk_score > 0.7:
            self.on_high_risk(decision)

        # Update stats
        latency = (time.perf_counter() - start) * 1000
        self.latencies.append(latency)
        self.stats["frames_processed"] += 1
        self.stats["decisions_made"] += 1
        self.stats["avg_latency_ms"] = np.mean(self.latencies)
        self.stats["fps"] = 1000 / np.mean(self.latencies) if self.latencies else 0

    def _update_display(self):
        """Push the latest frame/decision to whatever is rendering it."""
        if self.on_display_tick:
            self.on_display_tick(self._last_frame, self._last_decision)

    def stop(self):
        """Stop monitoring."""
        self.running = False
        logger.info("LiveMonitor stopped")
        if self.logger:
            self.logger._flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        return {**self.stats, **self.sentinel.get_stats()}

    def get_recent_decisions(self, n: int = 10) -> List[Dict]:
        """Get recent decisions."""
        if self.logger:
            return self.logger.get_recent(n)
        return []


class LiveMonitorGUI:
    """OpenCV-based GUI for live monitoring."""

    def __init__(self, monitor: LiveMonitor, window_name: str = "SENTINEL-Vision Live Monitor"):
        self.monitor = monitor
        self.window_name = window_name
        self.current_frame = None
        self.current_decision = None

    def run(self):
        """Run GUI loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        # Set callback
        self.monitor.on_decision = self._on_decision

        # Start monitor in background
        import threading
        monitor_thread = threading.Thread(target=asyncio.run, args=(self.monitor.start(),), daemon=True)
        monitor_thread.start()

        while True:
            if self.current_frame is not None and self.current_decision is not None:
                # Draw overlay
                display_frame = self.monitor.overlay.draw_decision(
                    self.current_frame,
                    self.current_decision,
                    heatmap=getattr(self.current_decision, 'heatmap', None),
                    bbox=self.current_decision.bbox,
                )
                # Convert to BGR for OpenCV
                display_bgr = cv2.cvtColor(np.array(display_frame), cv2.COLOR_RGB2BGR)
                cv2.imshow(self.window_name, display_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                break

        self.monitor.stop()
        cv2.destroyAllWindows()

    def _on_decision(self, decision: SentinelDecision, action: Optional[AgentAction]):
        """Callback when decision is made."""
        self.current_decision = decision
        # Capture current frame for display
        self.current_frame = self.monitor.capture.capture()


def create_live_monitor(
    config,
    sentinel_checkpoint: str = DEFAULT_SENTINEL_CHECKPOINT,
    gate_checkpoint: str = DEFAULT_GATE_CHECKPOINT,
    device: str = "cuda",
    capture_fps: float = 3.0,
    save_dir: Optional[str] = "logs/live_monitor",
) -> LiveMonitor:
    """Factory to create LiveMonitor."""
    monitor_config = MonitorConfig(
        sentinel_checkpoint=sentinel_checkpoint,
        gate_checkpoint=gate_checkpoint,
        device=device,
        capture_fps=capture_fps,
        save_dir=save_dir,
    )
    return LiveMonitor(monitor_config)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
    def main(config: DictConfig):
        monitor = create_live_monitor(config)
        gui = LiveMonitorGUI(monitor)
        gui.run()

    main()
