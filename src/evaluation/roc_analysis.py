"""ROC curve and AUC analysis utilities.

Computes one-vs-rest ROC curves and AUC scores for multi-class
classification, generates publication-quality plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def compute_roc(y_true, y_prob, class_names=None, save_path=None,
                title="ROC Curves (One-vs-Rest)", figsize=(10, 8)):
    """Compute and plot one-vs-rest ROC curves with AUC.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (integer-encoded).
    y_prob : np.ndarray
        Predicted probabilities, shape [N, num_classes].
    class_names : list[str], optional
        Human-readable class names.
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    dict
        Contains:
        - 'fpr': dict of per-class FPR arrays
        - 'tpr': dict of per-class TPR arrays
        - 'auc': dict of per-class AUC values
        - 'macro_auc': macro-average AUC
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Binarize labels for OVR
    y_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    if class_names is None:
        class_names = [str(c) for c in classes]

    # Compute per-class ROC
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    for i, cls in enumerate(classes):
        name = class_names[i] if i < len(class_names) else str(cls)
        fpr_dict[name], tpr_dict[name], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_dict[name] = auc(fpr_dict[name], tpr_dict[name])

    # Macro-average AUC
    macro_auc = np.mean(list(auc_dict.values()))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, name in enumerate(class_names):
        ax.plot(
            fpr_dict[name], tpr_dict[name],
            color=colors[i],
            lw=2,
            label=f"{name} (AUC = {auc_dict[name]:.4f})",
        )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(f"{title}\n(Macro AUC = {macro_auc:.4f})",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ROC curves saved: {save_path}")

    plt.close(fig)

    return {
        "fpr": fpr_dict,
        "tpr": tpr_dict,
        "auc": auc_dict,
        "macro_auc": macro_auc,
    }
