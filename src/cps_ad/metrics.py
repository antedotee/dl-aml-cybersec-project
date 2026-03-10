"""Detection metrics aligned with security operations (FPR vs recall trade-offs)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class DetectionMetrics:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    threshold: float


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float | None = None,
    threshold_quantile_on_scores: float | None = None,
) -> DetectionMetrics:
    """
    Higher `scores` mean *more anomalous*.

    If `threshold` is None, calibrate using a quantile of `scores` (e.g., 0.99).
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y_true, dtype=int).ravel()
    if threshold is None:
        if threshold_quantile_on_scores is None:
            raise ValueError("Provide threshold or threshold_quantile_on_scores.")
        threshold = float(np.quantile(s, threshold_quantile_on_scores))
    y_hat = (s >= threshold).astype(int)

    roc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    pr = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    return DetectionMetrics(
        roc_auc=roc,
        pr_auc=pr,
        f1=float(f1_score(y, y_hat, zero_division=0)),
        precision=float(precision_score(y, y_hat, zero_division=0)),
        recall=float(recall_score(y, y_hat, zero_division=0)),
        threshold=threshold,
    )


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Grid-free F1-optimal threshold on provided scores (validation tuning)."""
    y = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    prec, rec, thr = precision_recall_curve(y, s)
    # precision_recall_curve returns thr length = len(prec)-1
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    j = int(np.nanargmax(f1))
    return float(thr[j]), float(f1[j])


def roc_curve_arrays(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(np.asarray(y_true, dtype=int).ravel(), np.asarray(scores, dtype=float).ravel())
