"""Evaluation metrics for multi-class classification.

Provides comprehensive metrics including per-class and weighted
averages for precision, recall, F1, and accuracy.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def classification_metrics(y_true, y_pred, class_names=None):
    """Compute multi-class classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list[str], optional
        Human-readable class names for the report.

    Returns
    -------
    dict
        Contains:
        - 'accuracy': overall accuracy
        - 'precision_weighted': weighted precision
        - 'recall_weighted': weighted recall
        - 'f1_weighted': weighted F1
        - 'precision_macro': macro-averaged precision
        - 'recall_macro': macro-averaged recall
        - 'f1_macro': macro-averaged F1
        - 'per_class': dict mapping class index → {precision, recall, f1, support}
        - 'report_str': formatted classification report string
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Per-class breakdown
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}
    for i, cls in enumerate(classes):
        name = class_names[cls] if class_names else str(cls)
        support = int(np.sum(y_true == cls))
        per_class[name] = {
            "precision": float(per_class_prec[i]) if i < len(per_class_prec) else 0.0,
            "recall": float(per_class_rec[i]) if i < len(per_class_rec) else 0.0,
            "f1": float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
            "support": support,
        }

    results["per_class"] = per_class

    # Formatted report
    results["report_str"] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    return results


def print_metrics(results, model_name="Model"):
    """Pretty-print a metrics dict."""
    print(f"\n{'=' * 60}")
    print(f"METRICS: {model_name}")
    print("=" * 60)
    print(f"  Accuracy:           {results['accuracy']:.4f}")
    print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {results['recall_weighted']:.4f}")
    print(f"  F1 (weighted):        {results['f1_weighted']:.4f}")
    print(f"  F1 (macro):           {results['f1_macro']:.4f}")
    print(f"\n{results['report_str']}")
