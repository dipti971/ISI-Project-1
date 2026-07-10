"""Build PyTorch Geometric Data objects.

Orchestrates graph construction by combining:
1. IP-to-node mapping and edge_index (build_graph)
2. Aggregated node features (node_features)
3. Flow-level edge features (edge_features)
4. Edge labels (y)

Produces train_data.pt and test_data.pt files.

Usage:
    python src/graph/build_pyg_data.py
"""

import os
import sys
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.graph.build_graph import build_graph, build_ip_mapping, build_edge_index
from src.graph.node_features import compute_node_features
from src.graph.edge_features import compute_edge_features

# ==========================
# Paths
# ==========================

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graph")


def load_preprocessed():
    """Load preprocessed arrays."""
    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    ip_train = np.load(os.path.join(DATA_DIR, "ip_train.npy"))
    ip_test = np.load(os.path.join(DATA_DIR, "ip_test.npy"))
    feature_names = np.load(os.path.join(DATA_DIR, "feature_names.npy"),
                            allow_pickle=True).tolist()

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  ip_train: {ip_train.shape}, ip_test: {ip_test.shape}")
    print(f"  Features: {len(feature_names)}")

    return X_train, X_test, y_train, y_test, ip_train, ip_test, feature_names


def build_pyg_data_for_split(X, y, ip_pairs, feature_names, split_name):
    """Build a PyG Data object for one split (train or test).

    Parameters
    ----------
    X : np.ndarray
        Scaled features [N, F].
    y : np.ndarray
        Integer labels [N].
    ip_pairs : np.ndarray
        IP pairs [N, 2].
    feature_names : list[str]
        Feature column names.
    split_name : str
        'train' or 'test', for logging.

    Returns
    -------
    torch_geometric.data.Data
        PyG Data object with:
        - x: node features [num_nodes, F_node]
        - edge_index: [2, num_edges]
        - edge_attr: [num_edges, F_edge]
        - edge_y: [num_edges] edge labels
    dict
        ip_to_idx mapping.
    """
    print(f"\n{'=' * 60}")
    print(f"BUILDING {split_name.upper()} GRAPH")
    print("=" * 60)

    # 1. Build graph structure
    graph = build_graph(ip_pairs)
    ip_to_idx = graph["ip_to_idx"]
    num_nodes = graph["num_nodes"]
    edge_index = graph["edge_index"]

    # 2. Compute node features
    node_feat, node_feat_names = compute_node_features(
        ip_pairs, X, feature_names, ip_to_idx, num_nodes
    )

    # 3. Compute edge features (= flow features)
    edge_feat = compute_edge_features(X)

    # 4. Convert to tensors
    data = Data(
        x=torch.tensor(node_feat, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_feat, dtype=torch.float32),
        edge_y=torch.tensor(y, dtype=torch.long),
    )

    # Store metadata
    data.num_classes = len(np.unique(y))
    data.num_node_features = node_feat.shape[1]
    data.num_edge_features = edge_feat.shape[1]

    print(f"\n  PyG Data ({split_name}):")
    print(f"    Nodes:         {data.num_nodes}")
    print(f"    Edges:         {data.num_edges}")
    print(f"    Node features: {data.num_node_features}")
    print(f"    Edge features: {data.num_edge_features}")
    print(f"    Classes:       {data.num_classes}")

    return data, ip_to_idx


def main():
    print("\n" + "#" * 60)
    print("#  GRAPH CONSTRUCTION PIPELINE")
    print("#" * 60)

    X_train, X_test, y_train, y_test, ip_train, ip_test, feature_names = \
        load_preprocessed()

    # Build train graph
    train_data, train_ip_map = build_pyg_data_for_split(
        X_train, y_train, ip_train, feature_names, "train"
    )

    # Build test graph
    test_data, test_ip_map = build_pyg_data_for_split(
        X_test, y_test, ip_test, feature_names, "test"
    )

    # Save
    os.makedirs(GRAPH_DIR, exist_ok=True)

    torch.save(train_data, os.path.join(GRAPH_DIR, "train_data.pt"))
    torch.save(test_data, os.path.join(GRAPH_DIR, "test_data.pt"))

    # Save IP mappings
    with open(os.path.join(GRAPH_DIR, "train_ip_map.pkl"), "wb") as f:
        pickle.dump(train_ip_map, f)
    with open(os.path.join(GRAPH_DIR, "test_ip_map.pkl"), "wb") as f:
        pickle.dump(test_ip_map, f)

    print("\n" + "#" * 60)
    print("#  GRAPH CONSTRUCTION COMPLETE")
    print("#" * 60)
    print(f"  Saved to: {GRAPH_DIR}")
    print(f"    train_data.pt — {train_data.num_nodes} nodes, "
          f"{train_data.num_edges} edges")
    print(f"    test_data.pt  — {test_data.num_nodes} nodes, "
          f"{test_data.num_edges} edges")


if __name__ == "__main__":
    main()
