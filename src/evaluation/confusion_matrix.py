"""Confusion matrix plotting utilities.

Generates and saves heatmap visualizations of confusion matrices
using sklearn and matplotlib/seaborn.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          normalize=False, title="Confusion Matrix",
                          save_path=None, figsize=(10, 8)):
    """Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list[str], optional
        Human-readable class names.
    normalize : bool
        If True, show percentages instead of counts.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    np.ndarray
        The confusion matrix array.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved: {save_path}")

    plt.close(fig)
    return cm
