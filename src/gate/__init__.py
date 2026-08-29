"""Decision Gate (RL) modules for SENTINEL-Vision."""

from .decision_gate import DecisionGate
from .reward import AsymmetricReward

__all__ = [
    "DecisionGate",
    "AsymmetricReward",
    "train_gate_rl",
]


def __getattr__(name):
    # train_gate_rl pulls in the full training/data stack (datasets, hydra,
    # SentinelDataset, ...) via src.gate.train_gate_rl -> src.data.loaders.
    # Eagerly importing it here meant `import src.gate` for something as
    # lightweight as DecisionGate (e.g. from the desktop security wall)
    # unconditionally required every training dependency to be installed.
    # Lazily resolve it on first access instead.
    if name == "train_gate_rl":
        from .train_gate_rl import train_gate_rl

        return train_gate_rl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")