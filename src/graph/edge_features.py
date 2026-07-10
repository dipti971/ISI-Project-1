"""Edge feature engineering.

Each edge (network flow) retains its original flow-level features
from the preprocessed dataset. This module simply repackages the
feature matrix to align with the edge ordering in the graph.
"""

import numpy as np


def compute_edge_features(X_features):
    """Return edge features aligned to edge ordering.

    Since each flow maps 1-to-1 with an edge, and the edge ordering
    matches the row ordering of X_features, this is simply the
    identity operation. We convert to float32 for PyTorch compatibility.

    Parameters
    ----------
    X_features : np.ndarray
        Shape [N, F] — scaled flow-level features.

    Returns
    -------
    np.ndarray
        Edge feature matrix of shape [N, F], dtype float32.
    """
    edge_features = X_features.astype(np.float32)
    print(f"  Edge features: {edge_features.shape}")
    return edge_features
