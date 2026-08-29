# 🛡️ SENTINEL-Vision / ARG-VRI: Public Deployment & User Security Wall Guide

This guide explains how any end user or organization can download, install, and run **SENTINEL-Vision (Agent Risk Grounding & Visual Risk Inspection)** on their computer as a **Real-Time Visual Security Firewall** to protect against autonomous AI agent malpractice.

---

## 🌟 How the Visual Security Firewall Works

When an AI agent (such as **Claude Computer Use**, **Antigravity Browser Preview**, **OpenAI Operator**, or custom Playwright/Selenium bots) runs on your desktop or browser:

1. **Pixel-Only Watchdog**: SENTINEL-Vision captures the agent's screen buffer in real time ($k=6$ frames at 3 FPS). It **never reads or trusts agent logs or reasoning traces**.
2. **Pre-Action Risk Grounding**: *Before* the agent clicks or submits a form, SENTINEL-Vision analyzes the visual context for harmful operations:
   - 💥 **Destructive Operations** (deleting databases, wiping folders, terminating cloud instances)
   - 💳 **Unauthorized Financial Actions** (confirming payments, checkout, subscription charges)
   - 🔒 **Privacy Exfiltration** (exporting private keys, sharing sensitive documents, granting public access)
   - ⚠️ **Irreversible External Changes** (deploying code, publishing unauthorized posts)
3. **Instant Interception**: If risk exceeds threshold, the action is **immediately blocked/paused**.
4. **Malpractice Incident Reporting**: The user receives an automated visual evidence alert highlighting the exact UI element targeted, risk category, and audit reasoning.

---

## 🚀 Quick Start Installation

### Step 1: Clone & Install Package
```bash
git clone https://github.com/codewithyug06/Agent-Risk-Grounding-Visual-Risk-Inspection.git
cd Agent-Risk-Grounding-Visual-Risk-Inspection

# Install dependencies and CLI
pip install -e .
```

### Step 2: Run the Security Wall Demo
Test the firewall against a simulated agent attempting a dangerous financial operation:
```bash
sentinel-wall demo
```

### Step 2b: Run Continuous Background Monitoring
`sentinel-wall start` now actually watches your desktop screen continuously
(previously this command silently re-ran the one-shot demo instead of
monitoring anything -- fixed). It captures the primary monitor at ~3 FPS,
runs a risk decision on every frame, and shows a desktop toast plus an
incident-log entry on any PAUSE/HARD_BLOCK. Stop it with Ctrl+C.
```bash
sentinel-wall start
```
If no trained checkpoint is found, this now prints an explicit warning
instead of silently running an untrained model and reporting meaningless
risk scores.

### Step 3: Open the Security Audit Dashboard
View the visual incident history and evidence screenshots:
```bash
sentinel-wall dashboard
```

---

## 💻 Integrating With AI Agents

### Option A: 3-Line Python Agent Wrapper
Wrap any computer-use agent or browser automation script:

```python
from src.security_wall import SentinelSecurityWall
from PIL import Image

# 1. Initialize the security wall (runs in real-time on CPU or GPU)
wall = SentinelSecurityWall(device="cpu")

# 2. Before executing any click or action:
screenshot = take_current_screen()  # PIL Image
action_type = "click"
selector = "button#confirm-checkout"

# 3. Inspect action against screen pixels
should_proceed, decision, incident = wall.monitor_action(
    current_frame=screenshot,
    action_type=action_type,
    selector=selector,
    agent_name="Claude Computer Use",
)

if not should_proceed:
    print(f"🚨 Action BLOCKED by Sentinel Security Wall! Risk: {decision.category}")
    # Abort action and notify user
else:
    # Safe to execute
    execute_action(action_type, selector)
```

---

### Option B: FastAPI Local Interceptor Gateway
Run the local REST gateway on `localhost:8000`:
```bash
python -m uvicorn src.integration.intercept_api:app --host 127.0.0.1 --port 8000
```

Send action requests before execution:
```bash
curl -X POST http://localhost:8000/intercept \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "click",
    "selector": "#delete-account-btn"
  }'
```

Response:
```json
{
  "decision": "HARD_BLOCK",
  "risk_score": 0.94,
  "category": "destructive",
  "should_proceed": false,
  "reasoning": "Detected high-risk destructive deletion UI element targeted."
}
```

---

## 📊 Incident Audit Logs & Storage
- **Incident Reports Directory**: `~/.sentinel_vision/incidents/`
- **Dashboard**: `~/.sentinel_vision/incidents/security_dashboard.html`
- **Audit Format**: Every blocked event saves high-resolution annotated screenshots with red bounding boxes and Grad-CAM risk heatmaps.

## ⚠️ Known Limitations (accurate as of this hardening pass)
- **The checkpoints in `checkpoints/` do not match the current model code
  and will not load.** Verified directly: `checkpoints/stageC_final.pt`'s
  state_dict has top-level module names like `encoder.backbone.*` and a
  simple conv-based localization head (`shared_conv`/`heatmap_conv`/
  `bbox_conv`), while the current `SentinelModel` uses `frame_encoder.*` and
  an anchor-based localization head (`feature_processor`/`objectness_head`/
  `bbox_head`). Same story for `checkpoints/gate_rl.pt` (a bare 2-layer
  `fc1`/`fc2` MLP) vs. the current `DecisionGate`'s `policy_net`/
  `value_net`. **This means `sentinel-wall demo`/`dashboard`/`start` all
  currently run on a randomly-initialized, untrained model** -- they log a
  clear error and degrade gracefully (fixed: this used to be an unhandled
  crash) rather than silently pretending the checkpoint loaded, but the
  risk scores you'll see are not meaningful until either a compatible
  checkpoint is trained against the current architecture, or the
  architecture is reverted/adapted to match these older checkpoints.
- **PAUSE now actually pauses.** Previously the wrapper logged a PAUSE
  decision and let the action proceed anyway. It now blocks by default
  unless you register an `on_pause` confirmation callback on
  `SentinelWrapper` (see `src/integration/agent_wrapper.py`).
- **Config schema drift**: `src/security_wall/desktop_wall.py` builds an
  inline model config that does not match the key layout in
  `configs/model_small.yaml`/`model_base.yaml` (e.g. `temporal_fusion` vs.
  `temporal`, `frame_window.k` vs. no such key). `SentinelModel` expects the
  inline schema; the YAML configs need to be migrated to match before they
  can be used interchangeably with the desktop wall. This was found during
  the hardening pass and is not yet fixed -- do not assume
  `model_small.yaml` and `SentinelSecurityWall`'s default config are
  equivalent.
- **Extension requires the local gateway.** The browser extension (see
  `extension/`) is a thin client of `src/integration/intercept_api.py` -- it
  cannot detect risk on its own and does nothing useful unless the gateway
  is running locally.
- **No code-signed installer.** The desktop `.exe` build (via PyInstaller)
  is unsigned; Windows SmartScreen may warn on first run.
