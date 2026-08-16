#!/usr/bin/env python3
"""
Download and prepare datasets for SENTINEL-Vision training.

Supports: Mind2Web, ScreenSpot, AgentTrek, XD-Violence
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def download_mind2web(data_dir: Path, config: dict):
    """Download Mind2Web dataset."""
    logger.info("Downloading Mind2Web dataset...")
    # Mind2Web is typically accessed via HuggingFace datasets
    try:
        from datasets import load_dataset
        dataset = load_dataset("osunlp/Mind2Web", data_dir=str(data_dir / "mind2web"))
        logger.info(f"Mind2Web loaded: {dataset}")
    except Exception as e:
        logger.warning(f"Could not load Mind2Web from HF: {e}")
        logger.info("Please download manually from https://github.com/OSU-NLP-Group/Mind2Web")


def download_screenspot(data_dir: Path, config: dict):
    """Download ScreenSpot dataset."""
    logger.info("Downloading ScreenSpot dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("screenspot/screenspot", data_dir=str(data_dir / "screenspot"))
        logger.info(f"ScreenSpot loaded: {dataset}")
    except Exception as e:
        logger.warning(f"Could not load ScreenSpot from HF: {e}")
        logger.info("Please download manually from https://github.com/X-PLUG/ScreenSpot")


def download_agenttrek(data_dir: Path, config: dict):
    """Download AgentTrek dataset."""
    logger.info("Downloading AgentTrek dataset...")
    logger.warning("AgentTrek download not automated - please download from source")


def download_xd_violence(data_dir: Path, config: dict):
    """Download XD-Violence dataset (for harmful action patterns)."""
    logger.info("Downloading XD-Violence dataset...")
    logger.warning("XD-Violence download not automated - please download from source")


def main():
    parser = argparse.ArgumentParser(description="Download SENTINEL-Vision datasets")
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data config YAML"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mind2web", "screenspot", "agenttrek", "xd_violence", "all"],
        default=["all"],
        help="Which datasets to download"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["mind2web", "screenspot", "agenttrek", "xd_violence"]

    for dataset_name in datasets_to_download:
        logger.info(f"Processing {dataset_name}...")
        if dataset_name == "mind2web":
            download_mind2web(data_dir, config)
        elif dataset_name == "screenspot":
            download_screenspot(data_dir, config)
        elif dataset_name == "agenttrek":
            download_agenttrek(data_dir, config)
        elif dataset_name == "xd_violence":
            download_xd_violence(data_dir, config)

    logger.info("Dataset download complete!")


if __name__ == "__main__":
    main()