"""Train traditional ML baselines (Random Forest + XGBoost).

Loads preprocessed .npy data, trains both models, saves models and
predictions, and prints a quick metric summary.

Usage:
    python src/models/train_baselines.py
"""

import os
import sys
import time
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.random_forest import train_rf
from src.models.xgboost_model import train_xgb

# ==========================
# Paths
# ==========================

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PRED_DIR = os.path.join(PROJECT_ROOT, "predictions")


def load_data():
    """Load preprocessed numpy arrays."""
    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # Load label encoder for class names
    le_path = os.path.join(DATA_DIR, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Classes: {list(le.classes_)}")

    return X_train, X_test, y_train, y_test, le


def train_and_evaluate(name, train_fn, X_train, y_train, X_test, y_test, le):
    """Train a model, evaluate, and return predictions + probabilities."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {name}")
    print("=" * 60)

    start = time.time()
    model = train_fn(X_train, y_train)
    elapsed = time.time() - start

    print(f"  Training time: {elapsed:.1f}s")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Quick metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=4,
    ))

    return model, y_pred, y_prob


def save_artifacts(name, model, y_pred, y_prob):
    """Save model and predictions to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved: {model_path}")

    # Save predictions and probabilities
    np.save(os.path.join(PRED_DIR, f"{name}_y_pred.npy"), y_pred)
    np.save(os.path.join(PRED_DIR, f"{name}_y_prob.npy"), y_prob)
    print(f"  Predictions saved to {PRED_DIR}")


def main():
    print("\n" + "#" * 60)
    print("#  BASELINE MODEL TRAINING")
    print("#" * 60)

    X_train, X_test, y_train, y_test, le = load_data()

    # ---- Random Forest ----
    rf_model, rf_pred, rf_prob = train_and_evaluate(
        "Random Forest", train_rf,
        X_train, y_train, X_test, y_test, le,
    )
    save_artifacts("rf", rf_model, rf_pred, rf_prob)

    # ---- XGBoost ----
    xgb_model, xgb_pred, xgb_prob = train_and_evaluate(
        "XGBoost", train_xgb,
        X_train, y_train, X_test, y_test, le,
    )
    save_artifacts("xgb", xgb_model, xgb_pred, xgb_prob)

    # ---- Summary ----
    print("\n" + "#" * 60)
    print("#  TRAINING COMPLETE")
    print("#" * 60)
    print(f"  Random Forest accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(f"  XGBoost accuracy:       {accuracy_score(y_test, xgb_pred):.4f}")
    print(f"  Models saved to:        {MODEL_DIR}")
    print(f"  Predictions saved to:   {PRED_DIR}")


if __name__ == "__main__":
    main()
