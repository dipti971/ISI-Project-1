"""Graph Attention Network (GAT) model for edge-level classification.

Architecture:
  1. Two GATConv layers with multi-head attention process node features
     through the graph structure → learn rich node embeddings.
  2. An edge classification head concatenates source/destination node
     embeddings with edge (flow-level) features and passes through
     an MLP to produce per-edge class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEdgeClassifier(nn.Module):
    """GAT-based edge classifier for network intrusion detection.

    Parameters
    ----------
    num_node_features : int
        Dimensionality of input node features.
    num_edge_features : int
        Dimensionality of input edge (flow) features.
    num_classes : int
        Number of output classes.
    hidden_dim : int
        Hidden dimension per attention head.
    heads_1 : int
        Number of attention heads in GAT layer 1.
    heads_2 : int
        Number of attention heads in GAT layer 2.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_classes,
        hidden_dim=64,
        heads_1=8,
        heads_2=4,
        dropout=0.3,
    ):
        super().__init__()

        self.dropout = dropout

        # ---- GAT backbone (node embedding) ----

        # Layer 1: input → hidden_dim * heads_1
        self.gat1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            heads=heads_1,
            dropout=dropout,
            concat=True,  # output dim = hidden_dim * heads_1
        )
        self.skip1 = nn.Linear(num_node_features, hidden_dim * heads_1)
        self.ln1 = nn.LayerNorm(hidden_dim * heads_1)

        # Layer 2: hidden_dim * heads_1 → hidden_dim * heads_2
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads_1,
            out_channels=hidden_dim,
            heads=heads_2,
            dropout=dropout,
            concat=True,  # output dim = hidden_dim * heads_2
        )
        self.skip2 = nn.Linear(hidden_dim * heads_1, hidden_dim * heads_2)
        self.ln2 = nn.LayerNorm(hidden_dim * heads_2)

        node_embed_dim = hidden_dim * heads_2

        # ---- Edge classification head ----
        # Input: concat(src_embedding, dst_embedding, edge_features)
        edge_input_dim = 2 * node_embed_dim + num_edge_features

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Must have attributes: x, edge_index, edge_attr.

        Returns
        -------
        torch.Tensor
            Logits of shape [num_edges, num_classes].
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # GAT Layer 1
        x_skip = self.skip1(x)
        x = self.gat1(x, edge_index)
        x = x + x_skip
        x = self.ln1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT Layer 2
        x_skip = self.skip2(x)
        x = self.gat2(x, edge_index)
        x = x + x_skip
        x = self.ln2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Node embeddings are now in x of shape [num_nodes, node_embed_dim]

        # Edge classification: gather src and dst embeddings for each edge
        src_idx = edge_index[0]  # [num_edges]
        dst_idx = edge_index[1]  # [num_edges]

        src_embed = x[src_idx]  # [num_edges, node_embed_dim]
        dst_embed = x[dst_idx]  # [num_edges, node_embed_dim]

        # Concatenate: src_embed || dst_embed || edge_features
        edge_repr = torch.cat([src_embed, dst_embed, edge_attr], dim=1)

        # MLP classifier
        logits = self.edge_mlp(edge_repr)

        return logits


def build_gat_model(num_node_features, num_edge_features, num_classes,
                    hidden_dim=64, heads_1=8, heads_2=4, dropout=0.3):
    """Factory function to build a GAT edge classifier.

    Returns
    -------
    GATEdgeClassifier
        Initialized model (not trained).
    """
    model = GATEdgeClassifier(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        heads_1=heads_1,
        heads_2=heads_2,
        dropout=dropout,
    )
    return model
