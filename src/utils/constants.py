"""
Single source of truth for values that were previously duplicated
(and drifting) across sentinel_model.py, decision_gate.py, agent_wrapper.py,
intercept_api.py, live_monitor.py, and the eval/ scripts.
"""

from pathlib import Path

# 5-way harm category taxonomy, in the fixed index order the model heads were
# trained with. Do not reorder without retraining the category classifier.
CATEGORY_NAMES = ["destructive", "financial", "privacy", "irreversible_external", "benign"]
NUM_CATEGORIES = len(CATEGORY_NAMES)
BENIGN_CATEGORY_IDX = CATEGORY_NAMES.index("benign")

# Agent action types the gate's state vector one-hot encodes.
ACTION_TYPES = ["click", "type", "navigate", "scroll", "other"]
NUM_ACTION_TYPES = len(ACTION_TYPES)

# Gate decision labels.
GATE_ACTIONS = ["ALLOW", "PAUSE", "HARD_BLOCK"]

# Real, current checkpoint layout on disk (see checkpoints/). Older code
# pointed at checkpoints/stage_c/best.pt, checkpoints/gate/latest.pt, etc.,
# which do not exist and silently produced an untrained model.
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
DEFAULT_SENTINEL_CHECKPOINT = str(CHECKPOINTS_DIR / "stageC_final.pt")
DEFAULT_GATE_CHECKPOINT = str(CHECKPOINTS_DIR / "gate_rl.pt")

# Default image resolution the frame encoder / temporal fusion were built for.
DEFAULT_IMAGE_SIZE = 224
DEFAULT_FRAME_WINDOW = 6

# Local-only deployment defaults for the FastAPI gateway.
DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 8000
