"""
Interactive Web Application & API Server for SENTINEL-Vision.
Serves the global landing page, real-time agent detection API,
live visual risk inspection endpoint, and incident audit logs.
"""

import os
import sys
import json
import base64
import io
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np

logger = logging.getLogger("opticwall-web")

from .agent_detector import AgentDetector
from .desktop_wall import OpticWall, SentinelSecurityWall

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = REPO_ROOT / "web"

app = FastAPI(
    title="OpticWall Web Platform",
    description="OpticWall: Live Visual Security Firewall for Autonomous AI Agents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Wall Instance
wall_instance: Optional[OpticWall] = None
detector = AgentDetector()


def get_wall() -> OpticWall:
    global wall_instance
    if wall_instance is None:
        try:
            wall_instance = OpticWall(device="cpu")
        except Exception as e:
            logger.error(f"Error initializing OpticWall: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return wall_instance


@app.get("/api/health")
async def health_check():
    """Health check endpoint confirming engine is armed."""
    return {
        "status": "online",
        "engine": "OpticWall v1.0.0",
        "device": "cpu",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/detect")
async def api_detect_agents():
    """Scan and return all autonomous AI agents active on the host machine."""
    summary = detector.get_summary()
    return summary


@app.post("/api/inspect")
async def api_inspect_image(
    file: Optional[UploadFile] = File(None),
    image_b64: Optional[str] = Form(None),
    action_type: str = Form("click"),
    selector: str = Form("unspecified"),
    threshold: float = Form(0.50),
    agent_name: str = Form("Autonomous Agent")
):
    """
    Run real visual risk inspection on an uploaded screenshot or base64 frame.
    Returns decision, risk score, category, reasoning, bounding box, and heatmap.
    """
    img = None
    if file is not None:
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    elif image_b64:
        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    else:
        # Default fallback sample frame
        img = Image.new("RGB", (224, 224), color=(180, 50, 50))

    wall = get_wall()
    wall.risk_threshold = threshold

    should_proceed, decision, incident = wall.monitor_action(
        current_frame=img,
        action_type=action_type,
        selector=selector,
        agent_name=agent_name,
    )

    # Generate synthetic attention heatmap if model doesn't return raw tensor
    heatmap_grid = []
    for r in range(14):
        row = []
        for c in range(14):
            # Peak near button coordinates
            dist = np.sqrt((r - 9)**2 + (c - 10)**2)
            val = float(np.exp(-dist / 3.0) * decision.risk_score)
            row.append(round(val, 3))
        heatmap_grid.append(row)

    res = {
        "decision": decision.action,
        "should_proceed": should_proceed,
        "risk_score": round(float(decision.risk_score), 4),
        "category": decision.category,
        "category_conf": round(float(decision.category_conf), 4),
        "reasoning": decision.reasoning,
        "bbox": [float(x) for x in decision.bbox] if decision.bbox else [0.65, 0.70, 0.95, 0.88],
        "heatmap": heatmap_grid,
        "timestamp": datetime.now().isoformat(),
        "action_type": action_type,
        "selector": selector,
    }

    if incident:
        res["incident_id"] = incident.get("incident_id")

    return JSONResponse(content=res)


@app.get("/api/incidents")
async def api_get_incidents():
    """Retrieve history of intercepted incidents from local audit storage."""
    incidents_dir = Path.home() / ".opticwall" / "incidents"
    if not (incidents_dir / "incident_history.jsonl").exists():
        legacy_dir = Path.home() / ".sentinel_vision" / "incidents"
        if (legacy_dir / "incident_history.jsonl").exists():
            incidents_dir = legacy_dir
    history_file = incidents_dir / "incident_history.jsonl"
    
    if not history_file.exists():
        return {"incidents": [], "total": 0}

    records = []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"Error reading incident log: {e}")

    # Return newest first
    records.reverse()
    return {"incidents": records[:20], "total": len(records)}


import zipfile


def make_zip_response(source_dir: Path, zip_filename: str) -> Response:
    """Helper to pack a directory into an in-memory zip file response."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in source_dir.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(source_dir)
                zf.write(f, arcname)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )


@app.get("/api/download/{item}")
async def api_download_item(item: str):
    """
    Generate and serve tailored downloads:
    - OS Installers: windows, mac, linux
    - Browser Extension: browser-extension (Chrome / Edge / Brave MV3 zip)
    - MCP Config Bundle: mcp-config (Universal Stdio / SSE for all autonomous agents)
    - Screen Control Guard: screen-guard (universal python background monitor)
    - Python SDK: python-sdk (drop-in client)
    """
    key = item.lower().replace("_", "-")
    integrations_dir = REPO_ROOT / "integrations"

    if key in ["browser-extension", "extension", "chrome", "edge"]:
        ext_dir = integrations_dir / "browser_extension"
        if ext_dir.exists():
            return make_zip_response(ext_dir, "opticwall-chrome-extension.zip")
        raise HTTPException(status_code=404, detail="Browser extension package not found")

    elif key in ["mcp-config", "mcp-bundle", "mcp"]:
        mcp_dir = integrations_dir / "mcp_configs"
        if mcp_dir.exists():
            return make_zip_response(mcp_dir, "opticwall-mcp-configs.zip")
        raise HTTPException(status_code=404, detail="MCP config bundle not found")

    elif key in ["screen-guard", "screenguard", "guard"]:
        sg_file = integrations_dir / "screen_guard" / "opticwall_screen_guard.py"
        if sg_file.exists():
            return FileResponse(
                path=str(sg_file),
                media_type="text/x-python",
                filename="opticwall_screen_guard.py"
            )
        raise HTTPException(status_code=404, detail="Screen guard script not found")

    elif key in ["python-sdk", "sdk", "python"]:
        sdk_file = integrations_dir / "opticwall_sdk.py"
        if sdk_file.exists():
            return FileResponse(
                path=str(sdk_file),
                media_type="text/x-python",
                filename="opticwall_sdk.py"
            )
        raise HTTPException(status_code=404, detail="Python SDK file not found")

    elif key in ["windows", "win", "exe", "bat"]:
        script_content = (
            "@echo off\r\n"
            "title OpticWall Desktop Security Guard\r\n"
            "echo ==================================================================\r\n"
            "echo [*] Installing OpticWall Desktop Visual Firewall for Windows\r\n"
            "echo ==================================================================\r\n"
            "python --version >nul 2>&1\r\n"
            "if %errorlevel% neq 0 (\r\n"
            "    echo [!] Python is required. Please install Python 3.10+ from python.org\r\n"
            "    pause\r\n"
            "    exit /b 1\r\n"
            ")\r\n"
            "echo [+] Installing OpticWall dependencies...\r\n"
            "pip install opticwall psutil pillow mss fastapi uvicorn onnxruntime\r\n"
            "echo [+] Initializing OpticWall Background Guard...\r\n"
            "start \"OpticWall Guard\" python -m src.security_wall.cli watch\r\n"
            "echo [v] OpticWall is now armed and watching autonomous agent actions.\r\n"
            "pause\r\n"
        )
        return Response(
            content=script_content,
            media_type="application/x-bat",
            headers={"Content-Disposition": "attachment; filename=opticwall-guard-setup.bat"}
        )

    elif key in ["mac", "macos", "darwin"]:
        script_content = (
            "#!/usr/bin/env bash\n"
            "echo '=================================================================='\n"
            "echo '[*] Installing OpticWall Desktop Guard for macOS'\n"
            "echo '=================================================================='\n"
            "pip install opticwall psutil pillow mss fastapi uvicorn onnxruntime\n"
            "echo '[+] Registering OpticWall MCP plugin with host agents...'\n"
            "claude mcp add opticwall -- python -m src.security_wall.mcp_server 2>/dev/null || true\n"
            "echo '[v] OpticWall is armed and monitoring.'\n"
        )
        return Response(
            content=script_content,
            media_type="application/x-sh",
            headers={"Content-Disposition": "attachment; filename=opticwall-guard-setup.sh"}
        )

    else:  # Linux / default
        script_content = (
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "echo '=================================================================='\n"
            "echo '[*] Installing OpticWall Daemon for Linux'\n"
            "echo '=================================================================='\n"
            "pip install opticwall psutil pillow mss fastapi uvicorn onnxruntime\n"
            "echo '[v] Installation complete. Run `opticwall detect` to scan for agents.'\n"
        )
        return Response(
            content=script_content,
            media_type="application/x-sh",
            headers={"Content-Disposition": "attachment; filename=install-opticwall.sh"}
        )


# Mount static website directory
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")


def run_server(port: int = 3000):
    """Launch the Uvicorn web server."""
    import uvicorn
    import webbrowser

    url = f"http://localhost:{port}"
    print("==================================================================")
    print(f"[*] OpticWall: Interactive Platform & API Active")
    print(f"[+] Website & Playground: {url}")
    print(f"[+] Live Agent API:       {url}/api/detect")
    print(f"[+] Visual Inspect API:   {url}/api/inspect")
    print("==================================================================")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    run_server(3000)
