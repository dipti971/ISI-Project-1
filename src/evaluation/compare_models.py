"""Compare all trained models (Random Forest, XGBoost, GAT).

Loads predictions from each model, computes metrics, generates
comparison tables and visualizations.

Usage:
    python src/evaluation/compare_models.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import classification_metrics, print_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.roc_analysis import compute_roc

# ==========================
# Paths
# ==========================

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PRED_DIR = os.path.join(PROJECT_ROOT, "predictions")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

MODELS = {
    "Random Forest": {"pred": "rf_y_pred.npy", "prob": "rf_y_prob.npy"},
    "XGBoost": {"pred": "xgb_y_pred.npy", "prob": "xgb_y_prob.npy"},
    "GAT": {"pred": "gat_y_pred.npy", "prob": "gat_y_prob.npy"},
}


def main():
    print("\n" + "#" * 60)
    print("#  MODEL COMPARISON")
    print("#" * 60)

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load ground truth
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    class_names = list(le.classes_)

    print(f"  Test samples: {len(y_test)}")
    print(f"  Classes: {class_names}")

    # ---- Compute metrics for each model ----
    all_results = {}

    for model_name, files in MODELS.items():
        pred_path = os.path.join(PRED_DIR, files["pred"])
        prob_path = os.path.join(PRED_DIR, files["prob"])

        if not os.path.exists(pred_path):
            print(f"\n  WARNING: Predictions not found for {model_name}, skipping...")
            continue

        y_pred = np.load(pred_path)
        y_prob = np.load(prob_path) if os.path.exists(prob_path) else None

        # Metrics
        results = classification_metrics(y_test, y_pred, class_names)
        print_metrics(results, model_name)
        all_results[model_name] = results

        # Confusion matrix (counts)
        plot_confusion_matrix(
            y_test, y_pred,
            class_names=class_names,
            normalize=False,
            title=f"{model_name} — Confusion Matrix",
            save_path=os.path.join(REPORT_DIR,
                                   f"cm_{model_name.lower().replace(' ', '_')}.png"),
        )

        # Confusion matrix (normalized)
        plot_confusion_matrix(
            y_test, y_pred,
            class_names=class_names,
            normalize=True,
            title=f"{model_name} — Normalized Confusion Matrix",
            save_path=os.path.join(REPORT_DIR,
                                   f"cm_{model_name.lower().replace(' ', '_')}_norm.png"),
        )

        # ROC curves
        if y_prob is not None:
            roc_results = compute_roc(
                y_test, y_prob,
                class_names=class_names,
                title=f"{model_name} — ROC Curves",
                save_path=os.path.join(REPORT_DIR,
                                       f"roc_{model_name.lower().replace(' ', '_')}.png"),
            )
            results["macro_auc"] = roc_results["macro_auc"]
            results["per_class_auc"] = roc_results["auc"]

    if not all_results:
        print("\n  No model predictions found. Train models first.")
        return

    # ---- Comparison table ----
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)

    rows = []
    for name, res in all_results.items():
        rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision (W)": res["precision_weighted"],
            "Recall (W)": res["recall_weighted"],
            "F1 (W)": res["f1_weighted"],
            "F1 (Macro)": res["f1_macro"],
            "AUC (Macro)": res.get("macro_auc", np.nan),
        })

    df_comparison = pd.DataFrame(rows)
    df_comparison = df_comparison.set_index("Model")

    print(df_comparison.to_string(float_format="%.4f"))

    # Save CSV
    csv_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    df_comparison.to_csv(csv_path)
    print(f"\n  Comparison table saved: {csv_path}")

    # ---- Comparison bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics_to_plot = ["Accuracy", "F1 (W)", "F1 (Macro)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    # Bar chart: main metrics
    x = np.arange(len(df_comparison))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        axes[0].bar(x + i * width, df_comparison[metric].values,
                    width, label=metric, color=colors[i], alpha=0.85)

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(df_comparison.index, fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score", fontsize=12, fontweight="bold")
    axes[0].set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # Bar chart: AUC
    if "AUC (Macro)" in df_comparison.columns:
        auc_vals = df_comparison["AUC (Macro)"].dropna()
        if len(auc_vals) > 0:
            axes[1].bar(range(len(auc_vals)), auc_vals.values,
                        color="#9C27B0", alpha=0.85)
            axes[1].set_xticks(range(len(auc_vals)))
            axes[1].set_xticklabels(auc_vals.index, fontsize=11)
            axes[1].set_ylim(0, 1.05)
            axes[1].set_ylabel("Macro AUC", fontsize=12, fontweight="bold")
            axes[1].set_title("Macro-Average AUC", fontsize=14, fontweight="bold")
            axes[1].grid(axis="y", alpha=0.3)

            for j, v in enumerate(auc_vals.values):
                axes[1].text(j, v + 0.01, f"{v:.4f}", ha="center",
                             fontsize=10, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(REPORT_DIR, "comparison_chart.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison chart saved: {chart_path}")

    print("\n" + "#" * 60)
    print("#  COMPARISON COMPLETE")
    print("#" * 60)
    print(f"  Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
