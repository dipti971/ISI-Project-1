"""Random Forest model wrapper.

Provides a configured Random Forest classifier for multi-class
network intrusion detection on CICIDS2017 data.
"""

from sklearn.ensemble import RandomForestClassifier


def train_rf(X_train, y_train,
             n_estimators=200,
             max_depth=20,
             class_weight="balanced",
             random_state=42,
             n_jobs=-1,
             **kwargs):
    """Train a Random Forest classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels (integer-encoded).
    n_estimators : int
        Number of trees.
    max_depth : int
        Maximum tree depth.
    class_weight : str or dict
        Strategy for handling class imbalance.
    random_state : int
        Seed for reproducibility.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    clf.fit(X_train, y_train)
    return clf
