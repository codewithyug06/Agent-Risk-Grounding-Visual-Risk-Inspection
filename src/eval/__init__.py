"""Evaluation modules for SENTINEL-Vision."""

from .metrics import (
    compute_precision_recall_f1,
    compute_localization_iou,
    compute_latency,
    compute_false_positive_rate,
    compute_cross_agent_generalization,
    compute_precision_recall_curve,
    compute_confusion_matrix,
)
from .run_benchmark import run_benchmark
from .ablations import run_ablations
from .adversarial_stress_test import run_adversarial_stress_test

__all__ = [
    "compute_precision_recall_f1",
    "compute_localization_iou",
    "compute_latency",
    "compute_false_positive_rate",
    "compute_cross_agent_generalization",
    "compute_precision_recall_curve",
    "compute_confusion_matrix",
    "run_benchmark",
    "run_ablations",
    "run_adversarial_stress_test",
]