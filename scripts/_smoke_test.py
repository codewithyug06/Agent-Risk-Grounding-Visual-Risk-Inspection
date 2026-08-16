"""Smoke test for SENTINEL-Vision model forward + gate + losses."""
import sys
sys.path.insert(0, ".")
import torch
from omegaconf import OmegaConf

from src.models.sentinel_model import SentinelModel
from src.gate.decision_gate import DecisionGate
from src.training.losses import create_loss_function
from src.data.frame_windowing import collate_frame_windows

torch.manual_seed(42)

# --- Model forward ---
cfg = OmegaConf.load("configs/model_small.yaml")
model = SentinelModel(cfg)
model.eval()
frames = torch.randn(2, 6, 3, 224, 224)
with torch.no_grad():
    out = model(frames)
print("MODEL_FWD_OK", sorted(out.keys()))
for k, v in out.items():
    if hasattr(v, "shape"):
        print("  ", k, tuple(v.shape))

# --- Gradients ---
model.train()
out2 = model(frames)
print("MODEL_TRAIN_OK")

# --- Loss ---
loss_fn = create_loss_function(cfg)
loss_dict = loss_fn(out2, {
    "risk_label": torch.tensor([1, 0]),
    "category_label": torch.tensor([2, 4]),
    "bbox": torch.randn(2, 4),
    "has_bbox": torch.tensor([1.0, 0.0]),
})
total = loss_dict["total_loss"]
total.backward()
print("LOSS_OK", {k: round(float(v), 4) for k, v in loss_dict.items()})

# --- Gate ---
gate = DecisionGate()
state = torch.tensor([[0.8, 1, 0, 0, 0, 0, 1, 0.5]])
with torch.no_grad():
    g = gate(state)
print("GATE_OK", sorted(g.keys()))
action = gate.get_action(0.8, 2, 0.5, deterministic=True)
print("GATE_ACTION", action)

# --- Collate ---
batch = [{
    "frames": torch.randn(6, 3, 224, 224),
    "risk_label": torch.tensor(1),
    "category_label": torch.tensor(3),
    "bbox": torch.randn(4),
    "has_bbox": torch.tensor(1.0),
    "action": "delete",
    "trajectory_idx": 0,
    "action_idx": 0,
} for _ in range(2)]
collated = collate_frame_windows(batch)
print("COLLATE_OK", tuple(collated["frames"].shape))

print("ALL_SMOKE_OK")
