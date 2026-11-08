"""Evaluation metrics computation for multi-class classification."""

from typing import Dict, List, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str]
) -> Dict[str, Union[Dict[str, float], float]]:
    """Compute overall and per-class classification metrics.

    Args:
        y_true: Ground truth integer class indices (N,).
        y_pred: Predicted integer class indices (N,).
        y_probs: Predicted class probability distributions (N, C).
        class_names: List of class label strings.

    Returns:
        Dict with 'overall_metrics' and 'per_class_metrics'.
    """
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    try:
        overall["roc_auc_micro"] = float(
            roc_auc_score(y_true, y_probs, multi_class="ovr", average="micro")
        )
    except Exception:
        overall["roc_auc_micro"] = 0.0

    per_class = {}
    cm = confusion_matrix(y_true, y_pred)

    for idx, class_name in enumerate(class_names):
        support = int(np.sum(y_true == idx))
        prec = float(precision_score(y_true == idx, y_pred == idx, zero_division=0))
        rec = float(recall_score(y_true == idx, y_pred == idx, zero_division=0))
        f1 = float(f1_score(y_true == idx, y_pred == idx, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true == idx, y_probs[:, idx]))
        except Exception:
            auc = 0.0

        per_class[class_name] = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": support,
            "auc": auc
        }

    return {
        "overall_metrics": overall,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist()
    }
