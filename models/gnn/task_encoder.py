#    task_encoder.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Optional, Tuple

from .attention_layers import MultiHeadGATLayer


class GNNTaskEncoder(nn.Module):
    """
    GNN Encoder for DAG-based dependent tasks.
    Compatible with your custom GATLayer + MultiHeadGATLayer.
    """

    def __init__(
        self,
        node_feature_dim: int = 8,
        edge_feature_dim: int = 3,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = 'mean'
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pooling = pooling

        # ---------------------------------------------------------
        # Input projection for node features
        # ---------------------------------------------------------
        self.input_projection = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # ---------------------------------------------------------
        # Stack of Multi-Head GAT Layers (your version)
        # ---------------------------------------------------------
        self.gat_layers = nn.ModuleList([
            MultiHeadGATLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                edge_dim=edge_feature_dim,
                dropout=dropout,
                residual=True
            )
            for _ in range(num_gat_layers)
        ])

        # ---------------------------------------------------------
        # Final projection to graph-level embedding space
        # ---------------------------------------------------------
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ELU()
        )

        # ---------------------------------------------------------
        # Critical path head
        # ---------------------------------------------------------
        self.critical_path_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )

        # ---------------------------------------------------------
        # Pooling
        # ---------------------------------------------------------
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Invalid pooling type: {pooling}")

    # ===================================================================
    def forward(
        self,
        task_graph: Data,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = task_graph.x
        edge_index = task_graph.edge_index
        edge_attr = task_graph.edge_attr

        # If no edge attributes provided, zero-pad them
        if edge_attr is None and self.edge_feature_dim > 0:
            edge_attr = torch.zeros(
                edge_index.size(1),
                self.edge_feature_dim,
                device=x.device,
                dtype=x.dtype
            )

        # Input projection
        x = self.input_projection(x)

        # Pass through GAT layers
        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_attr)

        # Map node embeddings to embedding_dim
        node_embeddings = self.output_projection(x)

        # Batch vector
        if hasattr(task_graph, "batch") and task_graph.batch is not None:
            batch = task_graph.batch
        else:
            batch = torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )

        # Pooling → graph-level embedding
        graph_embedding = self.pool(node_embeddings, batch)

        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)

        return graph_embedding, node_embeddings

    # ===================================================================
    def get_graph_embedding(self, task_graph: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        g_emb, node_embeddings = self.forward(task_graph)
        if g_emb.dim() == 1:
            g_emb = g_emb.unsqueeze(0)
        return g_emb, node_embeddings

    # ===================================================================
    def get_critical_path(self, task_graph: Data, threshold: float = 0.5):
        _, node_emb = self.forward(task_graph)
        scores = self.critical_path_head(node_emb).squeeze(-1)
        return (scores > threshold).float()

    # ===================================================================
    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )


# ===================================================================
# Helper to construct graph data object
# ===================================================================
def create_task_graph_data(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_features: Optional[torch.Tensor] = None
) -> Data:
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features
    )


def visualize_critical_path(
    task_graph: Data,
    critical_scores: torch.Tensor,
    task_ids: Optional[list] = None
):
    if task_ids is None:
        task_ids = list(range(task_graph.num_nodes))

    print("\n===================== Critical Path =====================")
    scores = critical_scores.detach().cpu().numpy().flatten()
    sorted_ids = scores.argsort()[::-1]

    for idx in sorted_ids:
        s = scores[idx]
        flag = " <== CRITICAL" if s > 0.5 else ""
        print(f"Task {task_ids[idx]}: {s:.4f}{flag}")
