"""Node feature engineering.

Computes per-IP aggregated features from the raw flow data.
Each node (IP address) gets a feature vector summarizing its
traffic behaviour across all flows involving that IP.
"""

import numpy as np
import pandas as pd


def compute_node_features(ip_pairs, X_features, feature_names, ip_to_idx, num_nodes):
    """Compute aggregated node features for every IP address.

    For each IP, we aggregate statistics over all flows where that IP
    appears as either source or destination.

    Parameters
    ----------
    ip_pairs : np.ndarray
        Shape [N, 2] — (src_ip, dst_ip) per flow.
    X_features : np.ndarray
        Shape [N, F] — scaled flow-level features.
    feature_names : list[str]
        Names of the F flow features.
    ip_to_idx : dict
        IP → node index mapping.
    num_nodes : int
        Total number of unique nodes.

    Returns
    -------
    np.ndarray
        Node feature matrix of shape [num_nodes, num_node_features].
    list[str]
        Names of the node features.
    """
    feature_names = list(feature_names)

    # Build a DataFrame for aggregation
    df = pd.DataFrame(X_features, columns=feature_names)
    df["_src_ip"] = ip_pairs[:, 0]
    df["_dst_ip"] = ip_pairs[:, 1]

    # --- Helper: pick columns by substring matching ---
    def find_col(substrings, cols=feature_names):
        """Return the first column name containing any of the substrings."""
        for s in substrings:
            for c in cols:
                if s.lower() in c.lower():
                    return c
        return None

    # Identify key columns (case-insensitive partial match)
    # These are common CICIDS2017 column name patterns
    col_flow_duration = find_col(["Flow Duration"])
    col_total_fwd = find_col(["Total Fwd Packet", "Total Fwd Pack"])
    col_total_bwd = find_col(["Total Backward Packet", "Total Bwd Packet", "Total Backward Pack"])
    col_flow_bytes = find_col(["Flow Bytes/s"])
    col_flow_packets = find_col(["Flow Packets/s", "Flow Pkts/s"])
    col_pkt_len_mean = find_col(["Average Packet Size", "Avg Packet Size", "Packet Length Mean"])
    col_fwd_pkt_len = find_col(["Fwd Packet Length Mean", "Fwd Pkt Len Mean"])
    col_bwd_pkt_len = find_col(["Bwd Packet Length Mean", "Bwd Pkt Len Mean"])
    col_dst_port = find_col(["Destination Port", "Dst Port"])
    col_protocol = find_col(["Protocol"])

    node_features = []
    node_feature_names = []

    for role, ip_col in [("src", "_src_ip"), ("dst", "_dst_ip")]:
        grouped = df.groupby(ip_col)

        # Flow count (in/out)
        flow_count = grouped.size()
        node_features.append(flow_count)
        node_feature_names.append(f"{role}_flow_count")

        # Mean flow duration
        if col_flow_duration:
            node_features.append(grouped[col_flow_duration].mean())
            node_features.append(grouped[col_flow_duration].std().fillna(0))
            node_feature_names.extend([f"{role}_flow_dur_mean", f"{role}_flow_dur_std"])

        # Mean packet count (fwd/bwd)
        if col_total_fwd:
            node_features.append(grouped[col_total_fwd].mean())
            node_feature_names.append(f"{role}_fwd_pkt_mean")
        if col_total_bwd:
            node_features.append(grouped[col_total_bwd].mean())
            node_feature_names.append(f"{role}_bwd_pkt_mean")

        # Mean flow bytes/s
        if col_flow_bytes:
            node_features.append(grouped[col_flow_bytes].mean())
            node_feature_names.append(f"{role}_flow_bytes_mean")

        # Mean flow packets/s
        if col_flow_packets:
            node_features.append(grouped[col_flow_packets].mean())
            node_feature_names.append(f"{role}_flow_pkts_mean")

        # Mean packet length
        if col_pkt_len_mean:
            node_features.append(grouped[col_pkt_len_mean].mean())
            node_feature_names.append(f"{role}_avg_pkt_size_mean")

        # Unique destination ports contacted (only for src role)
        if role == "src" and col_dst_port:
            node_features.append(grouped[col_dst_port].nunique())
            node_feature_names.append("src_unique_dst_ports")

        # Protocol diversity
        if col_protocol:
            node_features.append(grouped[col_protocol].nunique())
            node_feature_names.append(f"{role}_protocol_diversity")

    # Combine into a matrix aligned to ip_to_idx
    # Each series is indexed by IP value
    all_ips = sorted(ip_to_idx.keys(), key=lambda ip: ip_to_idx[ip])
    node_matrix = np.zeros((num_nodes, len(node_feature_names)), dtype=np.float32)

    for feat_idx, series in enumerate(node_features):
        for ip in all_ips:
            node_idx = ip_to_idx[ip]
            if ip in series.index:
                val = series[ip]
                if np.isfinite(val):
                    node_matrix[node_idx, feat_idx] = val

    # Normalize node features (z-score)
    mean = node_matrix.mean(axis=0, keepdims=True)
    std = node_matrix.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0  # avoid division by zero
    node_matrix = (node_matrix - mean) / std

    print(f"  Node features: {node_matrix.shape[1]} features for {num_nodes} nodes")
    print(f"  Feature names: {node_feature_names}")

    return node_matrix, node_feature_names
