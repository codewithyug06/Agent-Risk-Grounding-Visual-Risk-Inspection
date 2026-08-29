
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf, ListConfig

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_config(
    config_name: str,
    config_dir: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Load a YAML config file as a DictConfig.

    Args:
        config_name: Name of config file (with or without .yaml extension)
        config_dir: Directory containing configs (default: project configs/)
        overrides: Optional dict of values to override after loading

    Returns:
        DictConfig with merged values
    """
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR

    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    path = config_dir / config_name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = OmegaConf.create(yaml.safe_load(f))

    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))

    return config


def save_config(
    config: DictConfig,
    path: Union[str, Path],
) -> None:
    """Save a DictConfig to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(f"Config saved to: {path}")


def deep_merge(base: DictConfig, override: DictConfig) -> DictConfig:
    """Deep merge two DictConfigs, override wins."""
    return OmegaConf.merge(base, override)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert a DictConfig to a plain dict."""
    return OmegaConf.to_container(config, resolve=True)


def save_config_json(
    config: DictConfig,
    path: Union[str, Path],
) -> None:
    """Save a config to a JSON file (for run artifacts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=2, default=str)
    logger.info(f"Config JSON saved to: {path}")
