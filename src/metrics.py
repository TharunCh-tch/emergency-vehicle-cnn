"""
Evaluation metrics: precision, recall, F1, confusion matrix, ROC/AUC.

Kept dependency-light and framework-agnostic (works on plain Python lists or
numpy arrays of labels/scores) so it is easy to unit test with synthetic data
and easy to reuse from both the evaluation script and the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list  # 2x2, rows=true, cols=pred: [[TN, FP], [FN, TP]]
    roc_auc: float | None = None

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
            "roc_auc": self.roc_auc,
        }


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute accuracy/precision/recall/F1/confusion matrix, and ROC-AUC if
    positive-class scores are given. Assumes binary labels {0, 1} where 1 is
    the positive ("emergency") class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = None
    if y_score is not None and len(set(y_true.tolist())) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm.tolist(),
        roc_auc=roc_auc,
    )


def roc_points(y_true, y_score):
    """Return (fpr, tpr, thresholds) arrays for plotting an ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds
