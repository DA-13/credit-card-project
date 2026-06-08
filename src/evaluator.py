"""
Evaluation metrics for imbalanced binary classification.

Why accuracy alone fails
────────────────────────
On this dataset, predicting "legitimate" for every transaction gives:
    Accuracy = |legit| / (|legit| + |fraud|) = 284315 / 284807 = 99.83%

That model catches zero frauds. We therefore evaluate across a suite of
metrics designed for imbalanced settings:

Precision (P)
    P = TP / (TP + FP)
    Of all transactions flagged as fraud, what fraction truly are?
    Low precision => many false alarms, operational burden on fraud teams.

Recall (R)  [aka Sensitivity / True Positive Rate]
    R = TP / (TP + FN)
    Of all actual frauds, what fraction did we catch?
    Low recall => fraudsters escape; direct financial loss.

F1 Score
    F1 = 2*P*R / (P + R)   (harmonic mean of precision and recall)
    Balances the precision-recall trade-off into a single scalar.

AUC-ROC
    Area under the ROC curve (TPR vs FPR across all thresholds).
    Threshold-independent; 0.5 = random, 1.0 = perfect.
    Can be optimistic for extreme imbalance — AUC-PR is preferred.

AUC-PR  (Average Precision)
    Area under the Precision-Recall curve.
    More informative than AUC-ROC when positives are rare: a random
    classifier achieves AUC-PR = prevalence ~ 0.00173, not 0.5.

Matthews Correlation Coefficient (MCC)
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    Equivalent to the Pearson correlation between observed and predicted
    binary labels. Handles imbalance symmetrically; ranges in [-1, 1].
    +1 = perfect, 0 = random, -1 = perfectly inverted predictions.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the full evaluation suite for a binary classifier.

    Args:
        y_true:  Ground truth labels in {0, 1}.
        y_pred:  Hard predictions in {0, 1}.
        y_proba: Predicted probabilities for class 1, in [0, 1].

    Returns:
        Dictionary mapping metric name -> scalar value.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_proba),
        "auc_pr":    average_precision_score(y_true, y_proba),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def format_metrics(metrics: Dict[str, float], title: str = "") -> str:
    """Render a metrics dictionary as a formatted string block."""
    lines = []
    if title:
        lines.append(f"\n{'=' * 52}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 52}")

    scalar_keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "mcc"]
    labels = {
        "accuracy":  "Accuracy ",
        "precision": "Precision",
        "recall":    "Recall   ",
        "f1":        "F1 Score ",
        "auc_roc":   "AUC-ROC  ",
        "auc_pr":    "AUC-PR   ",
        "mcc":       "MCC      ",
    }
    for k in scalar_keys:
        lines.append(f"  {labels[k]}: {metrics[k]:.4f}")

    lines.append(
        f"  Confusion  : TP={metrics['tp']:5d}  TN={metrics['tn']:6d}"
        f"  FP={metrics['fp']:5d}  FN={metrics['fn']:5d}"
    )
    return "\n".join(lines)
