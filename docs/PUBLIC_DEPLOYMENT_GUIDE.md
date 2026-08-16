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
