
import asyncio
import base64
import io
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf

from ..models.sentinel_model import SentinelModel, create_sentinel_model
from ..gate.decision_gate import DecisionGate, create_decision_gate
from ..data.frame_windowing import extract_frame_window
from ..data.augmentation import create_val_transform
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Represents an action from the AI agent."""
    action_type: str  
    selector: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    url: Optional[str] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class SentinelDecision:
    """Decision from SENTINEL-Vision."""
    action: str  # ALLOW, PAUSE, HARD_BLOCK
    risk_score: float
    category: str
    category_conf: float
    heatmap: Optional[np.ndarray] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    reasoning: str = ""
    timestamp: float = 0.0


class ScreenshotCapture(ABC):
    """Abstract base for screenshot capture."""

    @abstractmethod
    async def capture(self) -> Image.Image:
        """Capture current screen as PIL Image."""
        pass

    @abstractmethod
    async def get_size(self) -> Tuple[int, int]:
        """Get screen size."""
        pass


class PlaywrightCapture(ScreenshotCapture):
    """Screenshot capture using Playwright."""

    def __init__(self, page=None):
        self.page = page

    async def capture(self) -> Image.Image:
        if self.page:
            screenshot_bytes = await self.page.screenshot(type="png")
            return Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        return Image.new("RGB", (1920, 1080), color=(128, 128, 128))
        

    async def get_size(self) -> Tuple[int, int]:
        if self.page:
            size = await self.page.evaluate("() => ({w: window.innerWidth, h: window.innerHeight})")
            return (size["w"], size["h"])
        return (1920, 1080)


class FrameBuffer:
    """Rolling buffer of frames for temporal context."""

    def __init__(self, max_frames: int = 6, target_size: Tuple[int, int] = (224, 224)):
        self.max_frames = max_frames
        self.target_size = target_size
        self.frames: List[Image.Image] = []
        self.timestamps: List[float] = []

    def add_frame(self, frame: Image.Image):
        """Add frame to buffer."""
        # Resize
        frame_resized = frame.resize(self.target_size, Image.Resampling.LANCZOS)
        self.frames.append(frame_resized)
        self.timestamps.append(time.time())

        # Maintain max size
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
            self.timestamps.pop(0)

    def get_window(self) -> List[Image.Image]:
        """Get current frame window."""
        return self.frames.copy()

    def get_tensor(self, transform) -> torch.Tensor:
        """Get frames as tensor ready for model."""
        if len(self.frames) == 0:
            return None

        # Apply transform to all frames
        tensors = transform(self.frames)
        window = torch.stack(tensors)  # (k, C, H, W)

        # Pad if needed
        if len(self.frames) < self.max_frames:
            pad_count = self.max_frames - len(self.frames)
            last_frame = tensors[-1] if tensors else torch.zeros(3, *self.target_size)
            pad = last_frame.unsqueeze(0).repeat(pad_count, 1, 1, 1)
            window = torch.cat([pad, window], dim=0)

        return window  # (k, C, H, W)

    def clear(self):
        """Clear buffer."""
        self.frames.clear()
        self.timestamps.clear()

    def __len__(self):
        return len(self.frames)


class SentinelWrapper:
    """
    Main wrapper that integrates SENTINEL-Vision with an AI agent.
    Provides real-time oversight of agent actions.
    """

    def __init__(
        self,
        config,
        sentinel_checkpoint: str,
        gate_checkpoint: Optional[str] = None,
        device: str = "cuda",
        frame_buffer_size: int = 6,
        target_resolution: Tuple[int, int] = (224, 224),
        capture: Optional[ScreenshotCapture] = None,
    ):
        self.config = config
        self.device = device
        self.frame_buffer = FrameBuffer(frame_buffer_size, target_resolution)
        self.capture = capture or PlaywrightCapture()

        # Load SENTINEL model
        self.sentinel = create_sentinel_model(config)
        if sentinel_checkpoint and Path(sentinel_checkpoint).exists():
            logger.info(f"Loading SENTINEL model from {sentinel_checkpoint}")
            checkpoint = torch.load(sentinel_checkpoint, map_location=device)
            self.sentinel.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        else:
            logger.warning(f"No sentinel checkpoint provided or found at '{sentinel_checkpoint}', using initialized model weights.")
        self.sentinel = self.sentinel.to(device).eval()

        # Load gate if provided
        self.gate = None
        if gate_checkpoint and Path(gate_checkpoint).exists():
            logger.info(f"Loading Decision Gate from {gate_checkpoint}")
            self.gate = create_decision_gate(OmegaConf.to_container(config))
            gate_ckpt = torch.load(gate_checkpoint, map_location=device)
            self.gate.load_state_dict(gate_ckpt["model_state_dict"])
            self.gate = self.gate.to(device).eval()
        else:
            logger.warning("No gate checkpoint provided, using threshold-based decisions")

        # Transform
        self.transform = create_val_transform(OmegaConf.to_container(config))

        # Category names
        self.categories = ["destructive", "financial", "privacy", "irreversible_external", "benign"]

        # Statistics
        self.stats = {
            "total_actions": 0,
            "allowed": 0,
            "paused": 0,
            "blocked": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        # Callbacks
        self.on_decision: Optional[Callable[[SentinelDecision, AgentAction], None]] = None
        self.on_block: Optional[Callable[[SentinelDecision, AgentAction], bool]] = None  # Return True to override block

        logger.info("SentinelWrapper initialized")

    def add_frame(self, frame: Optional[Image.Image] = None):
        """Add frame to buffer. Captures screenshot if frame not provided."""
        if frame is None:
            frame = asyncio.run(self.capture.capture())
        self.frame_buffer.add_frame(frame)

    async def add_frame_async(self, frame: Optional[Image.Image] = None):
        """Async version of add_frame."""
        if frame is None:
            frame = await self.capture.capture()
        self.frame_buffer.add_frame(frame)

    def predict(self, frames: Optional[List[Image.Image]] = None) -> Dict[str, Any]:
        """Run SENTINEL prediction on current or provided frames."""
        if frames is not None:
            # Use provided frames
            tensors = [self.transform(f.resize(self.frame_buffer.target_size, Image.Resampling.LANCZOS)) for f in frames]
            window = torch.stack(tensors)
        else:
            # Use frame buffer
            window = self.frame_buffer.get_tensor(self.transform)

        if window is None:
            return {"risk_score": 0.0, "category_idx": 4, "category_probs": torch.zeros(5), "bbox": None, "objectness": 0.0}

        # Add batch dimension
        window = window.unsqueeze(0).to(self.device)  # (1, k, C, H, W)

        with torch.no_grad():
            output = self.sentinel(window)

        # Process outputs
        risk_score = output["risk_score"].squeeze().item()
        category_idx = output["category_idx"].squeeze().item()
        category_probs = output["category_probs"].squeeze().cpu().numpy()
        bbox = output["bbox"].squeeze().cpu().numpy() if output["bbox"] is not None else None
        objectness = output["objectness"].squeeze().item() if output["objectness"] is not None else 0.0
        heatmap = output.get("heatmap", None)

        return {
            "risk_score": risk_score,
            "category_idx": category_idx,
            "category_probs": category_probs,
            "bbox": bbox,
            "objectness": objectness,
            "heatmap": heatmap,
        }

    def decide(self, prediction: Dict[str, Any], action: AgentAction) -> SentinelDecision:
        """Make oversight decision based on prediction and action."""
        risk_score = prediction["risk_score"]
        category_idx = prediction["category_idx"]
        category_probs = prediction["category_probs"]
        objectness = prediction["objectness"]
        bbox = prediction["bbox"]
        heatmap = prediction["heatmap"]

        category = self.categories[category_idx]
        category_conf = category_probs[category_idx]

        # Get gate decision
        if self.gate:
            gate_output = self.gate.get_action(risk_score, category_idx, objectness, deterministic=True)
            decision_action = gate_output  # ALLOW, PAUSE, HARD_BLOCK
        else:
            # Threshold-based fallback
            if risk_score > 0.7:
                decision_action = "HARD_BLOCK"
            elif risk_score > 0.4:
                decision_action = "PAUSE"
            else:
                decision_action = "ALLOW"

        # Generate reasoning
        reasoning = self._generate_reasoning(risk_score, category, category_conf, decision_action, action)

        decision = SentinelDecision(
            action=decision_action,
            risk_score=risk_score,
            category=category,
            category_conf=category_conf,
            heatmap=heatmap,
            bbox=tuple(bbox) if bbox is not None else None,
            reasoning=reasoning,
            timestamp=time.time(),
        )

        # Update stats
        self.stats["total_actions"] += 1
        if decision_action == "ALLOW":
            self.stats["allowed"] += 1
        elif decision_action == "PAUSE":
            self.stats["paused"] += 1
        elif decision_action == "HARD_BLOCK":
            self.stats["blocked"] += 1

        return decision

    def _generate_reasoning(
        self,
        risk_score: float,
        category: str,
        category_conf: float,
        decision: str,
        action: AgentAction,
    ) -> str:
        """Generate human-readable reasoning."""
        parts = [
            f"Risk score: {risk_score:.3f}",
            f"Category: {category} ({category_conf:.3f})",
            f"Decision: {decision}",
            f"Action: {action.action_type}",
        ]
        if action.selector:
            parts.append(f"Selector: {action.selector}")
        if action.coordinates:
            parts.append(f"Coordinates: {action.coordinates}")
        return " | ".join(parts)

    def intercept_action(self, action: AgentAction) -> Tuple[SentinelDecision, bool]:
        """
        Intercept an agent action. Returns (decision, should_proceed).
        should_proceed=False means action should be blocked.
        """
        # Capture current screen
        self.add_frame()

        # Predict
        prediction = self.predict()

        # Decide
        decision = self.decide(prediction, action)

        # Callback
        if self.on_decision:
            self.on_decision(decision, action)

        # Determine if action should proceed
        should_proceed = True
        if decision.action == "HARD_BLOCK":
            should_proceed = False
            if self.on_block:
                # Allow override
                should_proceed = self.on_block(decision, action)
        elif decision.action == "PAUSE":
            # In PAUSE mode, wait for human confirmation
            # For now, proceed but log warning
            logger.warning(f"PAUSE: {decision.reasoning}")

        return decision, should_proceed

    async def intercept_action_async(self, action: AgentAction) -> Tuple[SentinelDecision, bool]:
        """Async version of intercept_action."""
        await self.add_frame_async()
        prediction = self.predict()
        decision = self.decide(prediction, action)

        if self.on_decision:
            self.on_decision(decision, action)

        should_proceed = True
        if decision.action == "HARD_BLOCK":
            should_proceed = False
            if self.on_block:
                should_proceed = self.on_block(decision, action)
        elif decision.action == "PAUSE":
            logger.warning(f"PAUSE: {decision.reasoning}")

        return decision, should_proceed

    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        total = max(1, self.stats["total_actions"])
        return {
            **self.stats,
            "allow_rate": self.stats["allowed"] / total,
            "pause_rate": self.stats["paused"] / total,
            "block_rate": self.stats["blocked"] / total,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {k: 0 for k in self.stats}


class AgentWrapper:
    """
    High-level wrapper that integrates with specific agent frameworks.
    """

    def __init__(self, sentinel_wrapper: SentinelWrapper):
        self.sentinel = sentinel_wrapper

    def wrap_click(self, selector: str, coordinates: Optional[Tuple[int, int]] = None) -> bool:
        """Wrap a click action."""
        action = AgentAction(
            action_type="click",
            selector=selector,
            coordinates=coordinates,
            timestamp=time.time(),
        )
        decision, should_proceed = self.sentinel.intercept_action(action)
        return should_proceed

    def wrap_type(self, selector: str, text: str) -> bool:
        """Wrap a type action."""
        action = AgentAction(
            action_type="type",
            selector=selector,
            text=text,
            timestamp=time.time(),
        )
        decision, should_proceed = self.sentinel.intercept_action(action)
        return should_proceed

    def wrap_navigate(self, url: str) -> bool:
        """Wrap a navigation action."""
        action = AgentAction(
            action_type="navigate",
            url=url,
            timestamp=time.time(),
        )
        decision, should_proceed = self.sentinel.intercept_action(action)
        return should_proceed

    def wrap_scroll(self, direction: str, amount: int) -> bool:
        """Wrap a scroll action."""
        action = AgentAction(
            action_type="scroll",
            metadata={"direction": direction, "amount": amount},
            timestamp=time.time(),
        )
        decision, should_proceed = self.sentinel.intercept_action(action)
        return should_proceed


def create_sentinel_wrapper(
    config,
    sentinel_checkpoint: str = "checkpoints/stage_c/best.pt",
    gate_checkpoint: Optional[str] = "checkpoints/gate/latest.pt",
    device: str = "cuda",
    frame_buffer_size: int = 6,
    target_resolution: Tuple[int, int] = (224, 224),
    capture: Optional[ScreenshotCapture] = None,
) -> SentinelWrapper:
    """Factory function to create SentinelWrapper."""
    return SentinelWrapper(
        config=config,
        sentinel_checkpoint=sentinel_checkpoint,
        gate_checkpoint=gate_checkpoint,
        device=device,
        frame_buffer_size=frame_buffer_size,
        target_resolution=target_resolution,
        capture=capture,
    )


# Example usage with Playwright
class PlaywrightAgentWrapper:
    """Example: Wrap Playwright-based agent."""

    def __init__(self, page, sentinel_wrapper: SentinelWrapper):
        self.page = page
        self.sentinel = sentinel_wrapper
        self.sentinel.capture = PlaywrightCapture(page)

    async def click(self, selector: str, **kwargs) -> bool:
        """Click with SENTINEL oversight."""
        # Check before clicking
        should_proceed = self.sentinel.wrap_click(selector)
        if not should_proceed:
            logger.warning(f"BLOCKED click on {selector}")
            return False

        await self.page.click(selector, **kwargs)
        return True

    async def fill(self, selector: str, value: str, **kwargs) -> bool:
        """Fill with SENTINEL oversight."""
        should_proceed = self.sentinel.wrap_type(selector, value)
        if not should_proceed:
            logger.warning(f"BLOCKED fill on {selector}")
            return False

        await self.page.fill(selector, value, **kwargs)
        return True

    async def goto(self, url: str, **kwargs) -> bool:
        """Navigate with SENTINEL oversight."""
        should_proceed = self.sentinel.wrap_navigate(url)
        if not should_proceed:
            logger.warning(f"BLOCKED navigation to {url}")
            return False

        await self.page.goto(url, **kwargs)
        return True


if __name__ == "__main__":
    # Demo
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(version_base=None, config_path="../../configs", config_name="model_small")
    def main(config: DictConfig):
        wrapper = create_sentinel_wrapper(config)
        print("SentinelWrapper created successfully")
        print(f"Stats: {wrapper.get_stats()}")

    main()
