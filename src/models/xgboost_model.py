"""XGBoost model wrapper.

Provides a configured XGBoost classifier for multi-class
network intrusion detection on CICIDS2017 data.
"""

import numpy as np
from xgboost import XGBClassifier


def compute_sample_weights(y):
    """Compute per-sample weights inversely proportional to class frequency.

    Parameters
    ----------
    y : np.ndarray
        Integer-encoded label array.

    Returns
    -------
    np.ndarray
        Weight for each sample.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    class_weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([class_weights[label] for label in y])


def train_xgb(X_train, y_train,
              n_estimators=200,
              max_depth=8,
              learning_rate=0.1,
              random_state=42,
              n_jobs=-1,
              **kwargs):
    """Train an XGBoost multi-class classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels (integer-encoded).
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.
    random_state : int
        Seed for reproducibility.
    n_jobs : int
        Number of parallel threads.

    Returns
    -------
    XGBClassifier
        Fitted model.
    """
    num_classes = len(np.unique(y_train))
    sample_weights = compute_sample_weights(y_train)

    import torch
    device_params = {}
    if torch.cuda.is_available():
        device_params["tree_method"] = "hist"
        device_params["device"] = "cuda"

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="mlogloss",
        use_label_encoder=False,
        **device_params,
        **kwargs,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf
