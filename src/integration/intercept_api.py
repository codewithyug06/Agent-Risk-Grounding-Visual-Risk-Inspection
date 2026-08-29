import asyncio
import base64
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from ..models.sentinel_model import create_sentinel_model
from ..gate.decision_gate import DecisionGate, create_decision_gate
from ..data.augmentation import create_val_transform
from .agent_wrapper import SentinelWrapper, AgentAction, SentinelDecision, FrameBuffer

logger = logging.getLogger(__name__)


# ============================================================
# Pydantic Models for API
# ============================================================

class FrameRequest(BaseModel):
    """Request to submit a frame."""
    frame_b64: str = Field(..., description="Base64 encoded frame (RGB)")
    timestamp: Optional[float] = Field(default_factory=time.time)
    frame_id: Optional[str] = None


class ActionRequest(BaseModel):
    """Request to intercept an action."""
    action_type: str = Field(..., description="Type of action: click, type, navigate, scroll, etc.")
    selector: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[float] = Field(default_factory=time.time)
    frame_b64: Optional[str] = Field(None, description="Optional current frame")


class DecisionResponse(BaseModel):
    """Response with SENTINEL decision."""
    request_id: str
    decision: str  # ALLOW, PAUSE, HARD_BLOCK
    risk_score: float
    category: str
    category_conf: float
    bbox: Optional[List[float]] = None
    heatmap_b64: Optional[str] = None
    reasoning: str
    timestamp: float
    should_proceed: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gate_loaded: bool
    uptime_seconds: float
    stats: Dict[str, Any]


class StatsResponse(BaseModel):
    """Statistics response."""
    total_requests: int
    allowed: int
    paused: int
    blocked: int
    avg_latency_ms: float
    fps: float


# ============================================================
# Global State
# ============================================================

class APIState:
    """Global API state."""
    def __init__(self):
        self.sentinel_wrapper: Optional[SentinelWrapper] = None
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.latencies: List[float] = []
        self.config = None

    def record_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        total = max(1, self.request_count)
        sentinel_stats = self.sentinel_wrapper.get_stats() if self.sentinel_wrapper else {}
        return {
            "total_requests": self.request_count,
            "allowed": sentinel_stats.get("allowed", 0),
            "paused": sentinel_stats.get("paused", 0),
            "blocked": sentinel_stats.get("blocked", 0),
            "avg_latency_ms": np.mean(self.latencies) if self.latencies else 0,
            "fps": 1000 / np.mean(self.latencies) if self.latencies else 0,
            "uptime_seconds": time.time() - self.start_time,
        }


api_state = APIState()


# ============================================================
# Helper Functions
# ============================================================

def decode_frame(frame_b64: str) -> Image.Image:
    """Decode base64 frame to PIL Image."""
    try:
        img_data = base64.b64decode(frame_b64)
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {str(e)}")


def encode_heatmap(heatmap: np.ndarray) -> str:
    """Encode heatmap as base64 PNG."""
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".png", heatmap_rgb)
    return base64.b64encode(buffer).decode("utf-8")


def create_decision_response(
    request_id: str,
    decision: SentinelDecision,
    should_proceed: bool,
    heatmap: Optional[np.ndarray] = None,
) -> DecisionResponse:
    """Create API response from decision."""
    heatmap_b64 = encode_heatmap(heatmap) if heatmap is not None else None
    return DecisionResponse(
        request_id=request_id,
        decision=decision.action,
        risk_score=decision.risk_score,
        category=decision.category,
        category_conf=decision.category_conf,
        bbox=list(decision.bbox) if decision.bbox else None,
        heatmap_b64=heatmap_b64,
        reasoning=decision.reasoning,
        timestamp=decision.timestamp,
        should_proceed=should_proceed,
    )


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    logger.info("Starting SENTINEL-Vision Intercept API")
    yield
    logger.info("Shutting down SENTINEL-Vision Intercept API")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="SENTINEL-Vision Intercept API",
    description="Real-time AI agent oversight via screen pixel monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if api_state.sentinel_wrapper else "initializing",
        model_loaded=api_state.sentinel_wrapper is not None and api_state.sentinel_wrapper.sentinel is not None,
        gate_loaded=api_state.sentinel_wrapper is not None and api_state.sentinel_wrapper.gate is not None,
        uptime_seconds=time.time() - api_state.start_time,
        stats=api_state.get_stats(),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API statistics."""
    stats = api_state.get_stats()
    return StatsResponse(**stats)


@app.post("/frame", response_model=Dict[str, Any])
async def submit_frame(request: FrameRequest):
    """
    Submit a frame to the rolling buffer.
    Returns current buffer status.
    """
    start = time.perf_counter()

    if not api_state.sentinel_wrapper:
        raise HTTPException(status_code=503, detail="Sentinel not initialized")

    frame = decode_frame(request.frame_b64)
    api_state.sentinel_wrapper.add_frame(frame)

    latency = (time.perf_counter() - start) * 1000
    api_state.record_latency(latency)
    api_state.request_count += 1

    return {
        "status": "ok",
        "buffer_size": len(api_state.sentinel_wrapper.frame_buffer),
        "latency_ms": latency,
    }


@app.post("/intercept", response_model=DecisionResponse)
async def intercept_action(request: ActionRequest, background_tasks: BackgroundTasks):
    """
    Intercept an agent action.
    Returns SENTINEL-Vision decision: ALLOW, PAUSE, or HARD_BLOCK.
    """
    start = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    if not api_state.sentinel_wrapper:
        raise HTTPException(status_code=503, detail="Sentinel not initialized")

    # Add frame if provided
    if request.frame_b64:
        frame = decode_frame(request.frame_b64)
        api_state.sentinel_wrapper.add_frame(frame)
    else:
        # Capture from screen if no frame provided
        api_state.sentinel_wrapper.add_frame()

    # Create action
    action = AgentAction(
        action_type=request.action_type,
        selector=request.selector,
        coordinates=request.coordinates,
        text=request.text,
        url=request.url,
        timestamp=request.timestamp,
        metadata=request.metadata,
    )

    # Get decision
    decision, should_proceed = api_state.sentinel_wrapper.intercept_action(action)

    # Include heatmap in response
    prediction = api_state.sentinel_wrapper.predict()
    heatmap = prediction.get("heatmap")

    latency = (time.perf_counter() - start) * 1000
    api_state.record_latency(latency)
    api_state.request_count += 1

    # Log in background
    background_tasks.add_task(log_decision, request_id, decision, action, latency)

    return create_decision_response(request_id, decision, should_proceed, heatmap)


@app.post("/predict", response_model=Dict[str, Any])
async def predict_only(request: FrameRequest):
    """
    Run prediction on provided frames without action interception.
    Useful for monitoring without blocking.
    """
    start = time.perf_counter()

    if not api_state.sentinel_wrapper:
        raise HTTPException(status_code=503, detail="Sentinel not initialized")

    frame = decode_frame(request.frame_b64)
    api_state.sentinel_wrapper.add_frame(frame)

    prediction = api_state.sentinel_wrapper.predict()

    latency = (time.perf_counter() - start) * 1000
    api_state.record_latency(latency)
    api_state.request_count += 1

    # Convert numpy arrays to lists
    result = {}
    for k, v in prediction.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            result[k] = v.cpu().numpy().tolist()
        else:
            result[k] = v

    result["latency_ms"] = latency
    return result


@app.get("/buffer", response_model=Dict[str, Any])
async def get_buffer_status():
    """Get current frame buffer status."""
    if not api_state.sentinel_wrapper:
        raise HTTPException(status_code=503, detail="Sentinel not initialized")

    buffer = api_state.sentinel_wrapper.frame_buffer
    return {
        "size": len(buffer),
        "max_size": buffer.max_frames,
        "timestamps": buffer.timestamps,
    }


@app.post("/buffer/clear", response_model=Dict[str, str])
async def clear_buffer():
    """Clear the frame buffer."""
    if not api_state.sentinel_wrapper:
        raise HTTPException(status_code=503, detail="Sentinel not initialized")

    api_state.sentinel_wrapper.frame_buffer.clear()
    return {"status": "cleared"}


@app.get("/decision/{request_id}", response_model=DecisionResponse)
async def get_decision(request_id: str):
    """Get a previous decision by ID (if logged)."""
    # In production, this would query a database
    raise HTTPException(status_code=404, detail="Decision not found")


# ============================================================
# WebSocket for Real-time Streaming
# ============================================================

from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    """Manages WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time decisions."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Handle incoming frames/actions
            if data.get("type") == "frame":
                frame_b64 = data.get("frame")
                if frame_b64 and api_state.sentinel_wrapper:
                    frame = decode_frame(frame_b64)
                    api_state.sentinel_wrapper.add_frame(frame)
                    prediction = api_state.sentinel_wrapper.predict()

                    await websocket.send_json({
                        "type": "prediction",
                        "risk_score": prediction["risk_score"],
                        "category": prediction["category_idx"],
                        "category_probs": prediction["category_probs"].tolist() if isinstance(prediction["category_probs"], np.ndarray) else prediction["category_probs"],
                    })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================
# Background Tasks
# ============================================================

async def log_decision(request_id: str, decision: SentinelDecision, action: AgentAction, latency: float):
    """Background task to log decisions."""
    logger.info(
        f"Decision [{request_id}]: {decision.action} | "
        f"Risk: {decision.risk_score:.3f} | "
        f"Category: {decision.category} | "
        f"Action: {action.action_type} | "
        f"Latency: {latency:.1f}ms"
    )


# ============================================================
# Initialization
# ============================================================

def initialize_api(
    config,
    sentinel_checkpoint: str = "checkpoints/stage_c/best.pt",
    gate_checkpoint: Optional[str] = "checkpoints/gate/latest.pt",
    device: str = "cuda",
    frame_buffer_size: int = 6,
    target_resolution: Tuple[int, int] = (224, 224),
):
    """Initialize the API with SENTINEL-Vision models."""
    api_state.config = config
    api_state.sentinel_wrapper = SentinelWrapper(
        config=config,
        sentinel_checkpoint=sentinel_checkpoint,
        gate_checkpoint=gate_checkpoint,
        device=device,
        frame_buffer_size=frame_buffer_size,
        target_resolution=target_resolution,
    )
    logger.info("API initialized with SENTINEL-Vision")


def create_app(
    config,
    sentinel_checkpoint: str = "checkpoints/stage_c/best.pt",
    gate_checkpoint: Optional[str] = "checkpoints/gate/latest.pt",
    device: str = "cuda",
    frame_buffer_size: int = 6,
    target_resolution: Tuple[int, int] = (224, 224),
) -> FastAPI:
    """Create and configure FastAPI app."""
    initialize_api(
        config=config,
        sentinel_checkpoint=sentinel_checkpoint,
        gate_checkpoint=gate_checkpoint,
        device=device,
        frame_buffer_size=frame_buffer_size,
        target_resolution=target_resolution,
    )
    return app


# ============================================================
# CLI Entry Point
# ============================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_path: str = "../../configs",
    config_name: str = "model_small",
    sentinel_checkpoint: str = "checkpoints/stage_c/best.pt",
    gate_checkpoint: str = "checkpoints/gate/latest.pt",
    device: str = "cuda",
):
    """Run the API server."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def main(config: DictConfig):
        initialize_api(
            config=config,
            sentinel_checkpoint=sentinel_checkpoint,
            gate_checkpoint=gate_checkpoint,
            device=device,
        )

        import uvicorn
        uvicorn.run(app, host=host, port=port)

    main()


if __name__ == "__main__":
    run_server()
