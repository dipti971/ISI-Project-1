"""Graph construction utilities.

Builds a graph representation from network flow data where:
- Nodes = unique IP addresses
- Edges = individual network flows (directed: src → dst)

Multiple flows between the same IP pair create parallel edges.
"""

import numpy as np


def build_ip_mapping(ip_pairs):
    """Create a mapping from unique IP addresses to integer node indices.

    Parameters
    ----------
    ip_pairs : np.ndarray
        Array of shape [N, 2] containing (src_ip, dst_ip) pairs.

    Returns
    -------
    dict
        Mapping from IP (int) → node index (int).
    int
        Total number of unique nodes.
    """
    unique_ips = np.unique(ip_pairs.ravel())
    ip_to_idx = {int(ip): idx for idx, ip in enumerate(unique_ips)}
    return ip_to_idx, len(unique_ips)


def build_edge_index(ip_pairs, ip_to_idx):
    """Convert IP pairs to a PyG-compatible edge_index tensor.

    Parameters
    ----------
    ip_pairs : np.ndarray
        Array of shape [N, 2] with (src_ip, dst_ip) per flow.
    ip_to_idx : dict
        IP address → node index mapping.

    Returns
    -------
    np.ndarray
        Edge index array of shape [2, N] (source row, destination row).
    """
    src_indices = np.array([ip_to_idx[int(ip)] for ip in ip_pairs[:, 0]])
    dst_indices = np.array([ip_to_idx[int(ip)] for ip in ip_pairs[:, 1]])
    edge_index = np.stack([src_indices, dst_indices], axis=0)
    return edge_index


def build_graph(ip_pairs):
    """Build a complete graph structure from IP pair data.

    Parameters
    ----------
    ip_pairs : np.ndarray
        Array of shape [N, 2] with (src_ip, dst_ip) per flow.

    Returns
    -------
    dict
        Contains:
        - 'ip_to_idx': IP → node index mapping
        - 'num_nodes': total number of unique IP nodes
        - 'edge_index': np.ndarray of shape [2, N]
        - 'num_edges': number of edges (flows)
    """
    ip_to_idx, num_nodes = build_ip_mapping(ip_pairs)
    edge_index = build_edge_index(ip_pairs, ip_to_idx)

    print(f"  Graph: {num_nodes} nodes, {edge_index.shape[1]} edges")

    return {
        "ip_to_idx": ip_to_idx,
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "num_edges": edge_index.shape[1],
    }
