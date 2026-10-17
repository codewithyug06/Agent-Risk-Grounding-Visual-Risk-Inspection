# OpticWall: Public Deployment & Integration Guide

This guide provides end-to-end instructions for deploying **OpticWall (ARG-VRI: Agent Risk Grounding & Visual Risk Inspection)** across developer workstations, enterprise environments, and autonomous agent sandboxes.

---

## 1. Architectural Overview

OpticWall runs as a **non-collusive, vision-only security firewall** that observes raw screen pixels before agent actions (clicks, keystrokes, form submissions) are executed.

```
┌─────────────────────────────────┐
│     Agent Environment           │  (OSWorld, WebArena, Browser, Claude Computer Use)
└────────────────┬────────────────┘
                 │ Raw Screen Frames (k=6 @ 3-5 FPS)
                 ▼
┌─────────────────────────────────┐
│      OpticWall Core Engine      │
│  - Vision Encoder (ViT/CLIP)    │
│  - Temporal Attention Fusion    │
│  - UI Localization Head         │
│  - 3-Tier Decision Gate Policy  │
└────────────────┬────────────────┘
                 │ Verdict & Visual Grounding
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  Tier 1 (ALLOW): Action Executes Immediately                 │
│  Tier 2 (PAUSE): Suspends Action, Alerts Human Reviewer       │
│  Tier 3 (HARD_BLOCK): Instantly Aborts Action & Logs Alert   │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Quickstart Installation

### Prerequisites
- **Python**: 3.10 or higher
- **PyTorch**: 2.0+ with CUDA or CPU support
- **OS**: Windows 10/11, macOS (Apple Silicon or Intel), or Linux (Ubuntu 20.04+)

```bash
# Clone the repository
git clone https://github.com/codewithyug06/Agent-Risk-Grounding-Visual-Risk-Inspection.git
cd Agent-Risk-Grounding-Visual-Risk-Inspection

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies and opticwall in editable mode
pip install -e .
```

---

## 3. Deployment Modes

### Mode A: Background Desktop Firewall (CLI)
Run OpticWall directly in the terminal to monitor active computer-use sessions:

```bash
# 1. Run live screen monitor daemon
python scripts/run_screen_monitor.py --interval 0.5 --threshold 0.75

# 2. Launch interactive cyber cockpit web platform
opticwall web --port 3000

# 3. Trigger simulated malpractice test (verifies interception)
opticwall demo

# 4. Open visual incident audit dashboard
opticwall dashboard
```

### Mode B: 3-Line Python SDK Integration
Integrate OpticWall directly into custom agent frameworks (e.g., LangChain, AutoGen, CrewAI, Browser-Use):

```python
from PIL import Image
from src.security_wall.desktop_wall import OpticWall

# Initialize security firewall
firewall = OpticWall(device="cpu", checkpoint_path="checkpoints/stage_c/best_loc.pt")

# Intercept action before execution
screenshot = Image.open("current_screen.png")
should_proceed, decision, incident = firewall.monitor_action(
    current_frame=screenshot,
    action_type="click",
    selector="#confirm-delete-db",
    agent_name="Autonomous-RPA-Worker"
)

if not should_proceed:
    print(f"Action Aborted! Category: {decision.category}, Risk: {decision.risk_score:.2%}")
    # Inspect highlighted UI coordinates
    print(f"Target Bounding Box: {decision.bounding_box}")
else:
    print("Action validated as safe. Proceeding.")
```

### Mode C: FastAPI Microservice (Remote Sandboxes & Multi-Agent Clusters)
For containerized or distributed agent testbeds (OSWorld / WebArena Docker containers):

```bash
# Launch FastAPI interception service on port 8000
python -m uvicorn src.integration.intercept_api:app --host 0.0.0.0 --port 8000 --workers 2
```

**Client Interception Request:**
```bash
curl -X POST http://localhost:8000/intercept \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "claude-computer-use-sandbox-1",
    "action_type": "click",
    "coordinates": [840, 520],
    "frame_base64": "<base64-encoded-screenshot>"
  }'
```

**API Response:**
```json
{
  "verdict": "PAUSE",
  "risk_score": 0.842,
  "category": "destructive",
  "requires_human_confirmation": true,
  "bounding_box": [0.45, 0.48, 0.58, 0.54],
  "message": "Potential destructive action detected on delete button."
}
```

### Mode D: Model Context Protocol (MCP) Server
OpticWall provides a native MCP server for integration with modern agentic IDEs and tools (Claude Desktop, Cursor, Antigravity):

```json
{
  "mcpServers": {
    "opticwall": {
      "command": "python",
      "args": ["-m", "src.security_wall.mcp_server"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

Available MCP Tools:
- `inspect_screen`: Captures active display and calculates risk score.
- `verify_action`: Evaluates target UI element coordinates and intent against visual risk bounds.
- `get_security_status`: Returns active threat level and recent incidents.

---

## 4. Hardware Optimization & ONNX Runtime

For production environments requiring sub-200ms latency without dedicated GPUs:

```bash
# Export trained PyTorch weights to FP32 and INT8 quantized ONNX graphs
python scripts/export_onnx.py --checkpoint checkpoints/stage_c/best_loc.pt --output onnx_models/
```

- **FP32 ONNX Runtime (CPU)**: 198.2 ms mean latency (5.0 FPS) — Recommended for general CPU deployments.
- **INT8 Dynamic Quantization**: 366.7 ms mean latency — Recommended for CPUs with AVX-512 VNNI support.

---

## 5. Security & Incident Audit Logging

All intercepted threats and paused actions generate cryptographic visual incident logs stored in:
`~/.opticwall/incidents/`

Each incident record contains:
1. Timestamp & Agent ID
2. Full resolution screenshot with red target bounding box overlay
3. Gaussian attention rollout heatmap
4. Predicted risk score & categorical classification
5. Decision gate verdict (`ALLOW`, `PAUSE`, `HARD_BLOCK`)

Open the full interactive dashboard anytime:
```bash
opticwall dashboard
```
