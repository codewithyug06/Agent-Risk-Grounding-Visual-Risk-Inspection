"""
Centralized, audited checkpoint I/O for SENTINEL-Vision.

Every module that previously called `torch.load(path, map_location=...)`
directly used pickle-based deserialization with no restriction on what
object graph a ".pt" file is allowed to reconstruct. A checkpoint from an
untrusted source (e.g. downloaded from a model-sharing site) could execute
arbitrary code on load. This module is the single place that loads
checkpoints, using `weights_only=True` (restricts unpickling to tensors,
plain Python containers/scalars, and numpy scalars) as the default and
requiring callers to explicitly opt in if a checkpoint truly needs full
unpickling (e.g. legacy files containing an OmegaConf node).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be safely loaded."""


def load_checkpoint(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    allow_unsafe: bool = False,
) -> Dict[str, Any]:
    """
    Safely load a checkpoint dict.

    Args:
        path: Path to the .pt checkpoint file.
        map_location: Device to map tensors onto.
        allow_unsafe: If True, falls back to `weights_only=False` when the
            restricted loader rejects the file. Only set this for checkpoints
            you trust (e.g. ones you trained locally) — never for a
            checkpoint whose provenance you cannot verify.

    Returns:
        The deserialized checkpoint dict.

    Raises:
        FileNotFoundError: if the checkpoint file does not exist.
        CheckpointLoadError: if the file cannot be safely deserialized and
            `allow_unsafe` was not set.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # pickle opcode rejected, unsupported global, etc.
        if not allow_unsafe:
            raise CheckpointLoadError(
                f"Refusing to load '{path}' with full unpickling. "
                f"Restricted (weights_only=True) load failed: {exc}. "
                "If you trust this file's origin, call with allow_unsafe=True."
            ) from exc
        logger.warning(
            "Loading '%s' with weights_only=False (unsafe deserialization). "
            "Only do this for checkpoints you trust.",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def load_model_state_dict(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    allow_unsafe: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint and return just the model state dict (handles both
    raw state-dict checkpoints and {"model_state_dict": ...} wrapper dicts)."""
    checkpoint = load_checkpoint(path, map_location=map_location, allow_unsafe=allow_unsafe)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def save_checkpoint(checkpoint: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a checkpoint dict, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to: %s", path)
