#!/usr/bin/env python3
"""Generate synthetic harmful-action dataset using Playwright injection."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_injection import run_injection_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic injection dataset via headless Playwright.")
    parser.add_argument("--output-dir", default="data/synthetic_injections")
    parser.add_argument("--n-per-category", type=int, default=600)
    parser.add_argument("--n-benign", type=int, default=2000)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = run_injection_pipeline(
        output_dir=args.output_dir,
        n_per_category=args.n_per_category,
        n_benign=args.n_benign,
        seed=args.seed,
        k=args.k,
    )
    print(f"Done. total={stats['total']} harmful={stats['n_harmful']} benign={stats['n_benign']}")
    print(f"by_category={stats['by_category']}")
    print(f"by_split={stats['by_split']}")


if __name__ == "__main__":
    main()
