

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
    keep_handlers: bool = False,
) -> None:
    """
    Configure root logger for SENTINEL-Vision.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to write logs to
        format_str: Custom log format
        keep_handlers: If True, don't remove existing handlers (idempotent)
    """
    global _CONFIGURED

    if _CONFIGURED and not keep_handlers:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = format_str or (
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers unless asked to keep them
    if not keep_handlers:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Avoid duplicate propagation from nested loggers
    logging.getLogger("hydra").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.INFO)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def log_config_summary(config, logger: logging.Logger):
    """Log a compact summary of a config dict/DictConfig."""
    import yaml
    from omegaconf import DictConfig, OmegaConf

    try:
        if isinstance(config, DictConfig):
            summary = OmegaConf.to_yaml(config)
        else:
            summary = yaml.dump(config, default_flow_style=False)
        logger.info("Config:\n%s", summary)
    except Exception as e:
        logger.warning(f"Could not log config summary: {e}")
