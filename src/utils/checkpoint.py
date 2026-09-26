"""
Centralized, safe checkpoint loading for SENTINEL-Vision.

torch.load with weights_only=False can execute arbitrary code embedded in a
pickle payload. Every checkpoint load in this project should go through
load_checkpoint() so that untrusted/third-party .pt files can't achieve code
execution just by being loaded.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch

logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    weights_only: bool = True,
) -> Dict[str, Any]:
    """
    Safely load a SENTINEL-Vision checkpoint dict.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        map_location: Device to map tensors to.
        weights_only: If True (default), restricts unpickling to tensors/basic
            types only, preventing arbitrary code execution from a malicious
            or corrupted checkpoint file. Only set False for checkpoints you
            trust AND that require non-tensor objects the safe unpickler
            rejects — prefer re-saving those checkpoints in a safe format
            instead.

    Returns:
        The loaded checkpoint dict.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
