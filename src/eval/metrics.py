"""
Evaluation metrics for SENTINEL-Vision.
Precision, Recall, F1, Localization IoU, Latency, FPR, Cross-agent generalization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
    auc,
    roc_auc_score,
)
import time
import logging

logger = logging.getLogger(__name__)


def compute_precision_recall_f1(
    predictions: Union[torch.Tensor, np.ndarray, List],
    labels: Union[torch.Tensor, np.ndarray, List],
    threshold: float = 0.5,
    average: str = "binary",
    is_logits: bool = False,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for binary or multi-class classification.

    Args:
        predictions: Predicted probabilities (N,) or logits (N,) or class indices (N,)
        labels: Ground truth labels (N,)
        threshold: Threshold for binary classification
        average: 'binary', 'macro', 'micro', 'weighted'
        is_logits: For 1-D `predictions`, whether they are raw logits (apply
            sigmoid before thresholding) or already-probabilities. This used
            to be guessed from `predictions.max() <= 1.0`, which silently
            misclassifies low-confidence logits (e.g. a batch of harmful-risk
            logits near 0) as probabilities -- the caller knows which it
            produced, so require it explicitly instead of guessing.

    Returns:
        Dict with precision, recall, f1, support
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Handle probability/logits input
    if predictions.ndim == 1:
        # Binary probabilities or logits
        if is_logits:
            probs = torch.sigmoid(torch.tensor(predictions)).numpy()
        else:
            probs = predictions
        preds_binary = (probs > threshold).astype(int)
    elif predictions.ndim == 2:
        # Multi-class probabilities or logits
        preds_binary = predictions.argmax(axis=1)
    else:
        preds_binary = predictions.astype(int)

    labels = labels.astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds_binary, average=average, zero_division=0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(support) if np.isscalar(support) else support.tolist(),
    }


def compute_localization_iou(
    pred_bboxes: Union[torch.Tensor, np.ndarray, List],
    gt_bboxes: Union[torch.Tensor, np.ndarray, List],
    format: str = "xyxy",
) -> Dict[str, float]:
    """
    Compute IoU metrics for bounding box localization.

    Args:
        pred_bboxes: (N, 4) predicted bboxes
        gt_bboxes: (N, 4) ground truth bboxes
        format: 'xyxy' (x1,y1,x2,y2) or 'xywh' (x,y,w,h)

    Returns:
        Dict with mean_iou, median_iou, iou_at_05, iou_at_075
    """
    if isinstance(pred_bboxes, torch.Tensor):
        pred_bboxes = pred_bboxes.detach().cpu().numpy()
    if isinstance(gt_bboxes, torch.Tensor):
        gt_bboxes = gt_bboxes.detach().cpu().numpy()

    pred_bboxes = np.asarray(pred_bboxes)
    gt_bboxes = np.asarray(gt_bboxes)

    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return {"mean_iou": 0.0, "median_iou": 0.0, "iou_at_05": 0.0, "iou_at_075": 0.0}

    # Convert to xyxy if needed
    if format == "xywh":
        pred_bboxes = _xywh_to_xyxy(pred_bboxes)
        gt_bboxes = _xywh_to_xyxy(gt_bboxes)

    ious = []
    for pb, gb in zip(pred_bboxes, gt_bboxes):
        iou = _bbox_iou(pb, gb)
        ious.append(iou)

    ious = np.array(ious)

    return {
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "iou_at_05": float(np.mean(ious >= 0.5)),
        "iou_at_075": float(np.mean(ious >= 0.75)),
        "ious": ious.tolist(),
    }


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert xywh to xyxy format."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def _bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_latency(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    n_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute inference latency statistics.

    Args:
        model: PyTorch model in eval mode
        sample_input: Sample input tensor (1, k, C, H, W) or similar
        n_runs: Number of timed runs
        warmup_runs: Number of warmup runs
        device: Device to run on

    Returns:
        Dict with mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms, fps
    """
    model = model.to(device).eval()
    sample_input = sample_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(sample_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    times = np.array(times)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000 / np.mean(times)),
    }


def compute_false_positive_rate(
    predictions: Union[torch.Tensor, np.ndarray, List],
    labels: Union[torch.Tensor, np.ndarray, List],
    threshold: float = 0.5,
) -> float:
    """
    Compute False Positive Rate (FPR) = FP / (FP + TN).
    Critical safety metric: how often benign actions are flagged as harmful.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Binarize
    preds_binary = (predictions > threshold).astype(int) if predictions.max() <= 1.0 else predictions.astype(int)
    labels = labels.astype(int)

    # FPR = FP / (FP + TN)
    fp = np.sum((preds_binary == 1) & (labels == 0))
    tn = np.sum((preds_binary == 0) & (labels == 0))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(fpr)


def compute_false_negative_rate(
    predictions: Union[torch.Tensor, np.ndarray, List],
    labels: Union[torch.Tensor, np.ndarray, List],
    threshold: float = 0.5,
) -> float:
    """
    Compute False Negative Rate (FNR) = FN / (FN + TP).
    PRIMARY SAFETY METRIC: how often harmful actions are missed.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    preds_binary = (predictions > threshold).astype(int) if predictions.max() <= 1.0 else predictions.astype(int)
    labels = labels.astype(int)

    # FNR = FN / (FN + TP)
    fn = np.sum((preds_binary == 0) & (labels == 1))
    tp = np.sum((preds_binary == 1) & (labels == 1))

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return float(fnr)


def compute_cross_agent_generalization(
    train_domain_results: Dict[str, float],
    test_domain_results: Dict[str, float],
    metrics: List[str] = ["recall", "f1", "localization_iou"],
) -> Dict[str, float]:
    """
    Quantify generalization gap between training and test domains.

    Args:
        train_domain_results: Metrics on training domain
        test_domain_results: Metrics on test domain (different agent/dataset)
        metrics: List of metric names to compare

    Returns:
        Dict with gap for each metric and average gap
    """
    gaps = {}
    for metric in metrics:
        train_val = train_domain_results.get(metric, 0.0)
        test_val = test_domain_results.get(metric, 0.0)
        gap = train_val - test_val
        gaps[f"{metric}_gap"] = float(gap)
        gaps[f"{metric}_train"] = float(train_val)
        gaps[f"{metric}_test"] = float(test_val)

    # Select by key name, not position -- a stride-3 slice over dict values
    # silently breaks (picks up train/test values instead of gaps) the
    # moment insertion order changes or `metrics` has a different length.
    gap_values = [v for k, v in gaps.items() if k.endswith("_gap")]
    gaps["average_gap"] = float(np.mean(gap_values)) if gap_values else 0.0
    return gaps


def compute_precision_recall_curve(
    predictions: Union[torch.Tensor, np.ndarray, List],
    labels: Union[torch.Tensor, np.ndarray, List],
) -> Dict[str, np.ndarray]:
    """
    Compute precision-recall curve for plotting.

    Returns:
        Dict with 'precision', 'recall', 'thresholds', 'auc'
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten().astype(int)

    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "auc": float(pr_auc),
    }


def compute_roc_auc(
    predictions: Union[torch.Tensor, np.ndarray, List],
    labels: Union[torch.Tensor, np.ndarray, List],
) -> float:
    """Compute ROC AUC."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    return float(roc_auc_score(labels, predictions))


def compute_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray, List],
    labels: Union[torch.Tensor, np.ndarray, List],
    num_classes: int = 5,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices (N,) or probabilities (N, C)
        labels: Ground truth class indices (N,)
        num_classes: Number of classes
        threshold: For binary case

    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.asarray(predictions)
    labels = np.asarray(labels).astype(int)

    if predictions.ndim == 2:
        # Probabilities -> argmax
        preds = predictions.argmax(axis=1)
    elif predictions.ndim == 1 and predictions.max() <= 1.0:
        # Binary probabilities
        preds = (predictions > threshold).astype(int)
    else:
        preds = predictions.astype(int)

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    return cm


def compute_per_class_metrics(
    predictions: Union[torch.Tensor, np.ndarray, List],
    labels: Union[torch.Tensor, np.ndarray, List],
    num_classes: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.asarray(predictions)
    labels = np.asarray(labels).astype(int)

    if predictions.ndim == 2:
        preds = predictions.argmax(axis=1)
    else:
        preds = predictions.astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0
    )

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    results = {}
    for i, name in enumerate(class_names):
        results[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return results


def compute_safety_metrics(
    predictions: Union[torch.Tensor, np.ndarray, List],
    labels: Union[torch.Tensor, np.ndarray, List],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive safety metrics.
    Primary: False Negative Rate (missed harm)
    Secondary: False Positive Rate (false blocks), Precision, Recall, F1
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.asarray(predictions).flatten()
    labels = np.asarray(labels).flatten().astype(int)

    preds_binary = (predictions > threshold).astype(int)

    tp = np.sum((preds_binary == 1) & (labels == 1))
    fp = np.sum((preds_binary == 1) & (labels == 0))
    fn = np.sum((preds_binary == 0) & (labels == 1))
    tn = np.sum((preds_binary == 0) & (labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),  # PRIMARY SAFETY METRIC
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def compute_latency_detailed(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    n_runs: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Detailed latency breakdown by model component.
    """
    model = model.to(device).eval()
    return compute_latency(model, sample_input, n_runs, device=device)


def compute_cross_agent_metrics(
    agent_predictions: Dict[str, Union[torch.Tensor, np.ndarray, List]],
    agent_labels: Dict[str, Union[torch.Tensor, np.ndarray, List]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute cross-agent generalization metrics across diverse agent architectures
    (e.g., Claude Computer Use, OSWorld, WebArena, Mind2Web).

    Returns:
        Dict containing per-agent metrics, worst-group accuracy, and transfer generalization gap.
    """
    per_agent_results = {}
    f1_scores = []
    accuracies = []
    fnrs = []

    for agent_name, preds in agent_predictions.items():
        if agent_name not in agent_labels:
            continue
        labels = agent_labels[agent_name]
        metrics = compute_safety_metrics(preds, labels, threshold=threshold)
        per_agent_results[agent_name] = metrics
        f1_scores.append(metrics["f1"])
        accuracies.append(metrics["accuracy"])
        fnrs.append(metrics["false_negative_rate"])

    if not accuracies:
        return {
            "per_agent": {},
            "worst_group_accuracy": 0.0,
            "mean_accuracy": 0.0,
            "mean_f1": 0.0,
            "worst_group_fnr": 1.0,
            "generalization_gap": 0.0,
        }

    worst_acc = float(np.min(accuracies))
    mean_acc = float(np.mean(accuracies))
    best_acc = float(np.max(accuracies))
    worst_fnr = float(np.max(fnrs))
    mean_f1 = float(np.mean(f1_scores))
    gen_gap = float(best_acc - worst_acc)

    return {
        "per_agent": per_agent_results,
        "worst_group_accuracy": worst_acc,
        "mean_accuracy": mean_acc,
        "mean_f1": mean_f1,
        "worst_group_fnr": worst_fnr,
        "generalization_gap": gen_gap,
    }