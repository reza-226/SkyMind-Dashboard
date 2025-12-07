"""
GNN-based Task Encoder for DAG Processing
رمزگذار GNN برای پردازش Task DAG
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Tuple, Optional
from .attention_layers import MultiHeadGATLayer


class GNNTaskEncoder(nn.Module):
    """
    رمزگذار GNN برای استخراج embeddings از Task DAG
    
    این مدل از لایه‌های Graph Attention Network برای یادگیری
    نمایش‌های پنهان tasks و محاسبه critical path استفاده می‌کند.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 9,
        edge_feature_dim: int = 3,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_gat_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            node_feature_dim: بعد ویژگی‌های گره (task features)
            edge_feature_dim: بعد ویژگی‌های یال (dependency features)
            hidden_dim: بعد لایه‌های مخفی
            embedding_dim: بعد embedding نهایی
            num_gat_layers: تعداد لایه‌های GAT
            num_heads: تعداد attention heads
            dropout: نرخ dropout
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_gat_layers = num_gat_layers
        
        # لایه ورودی: تبدیل node features به hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # لایه‌های GAT
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            self.gat_layers.append(
                MultiHeadGATLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    edge_dim=edge_feature_dim,
                    dropout=dropout,
                    residual=True
                )
            )
        
        # لایه خروجی: تبدیل به embedding dimension
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ELU()
        )
        
        # پیش‌بینی critical path scores
        self.critical_path_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Global pooling برای graph-level representation
        self.global_pool = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU()
        )
    
    def forward(
        self,
        task_graph: Data,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            task_graph: PyTorch Geometric Data object حاوی:
                - x: node features [num_nodes, node_feature_dim]
                - edge_index: edge indices [2, num_edges]
                - edge_attr: edge features [num_edges, edge_feature_dim]
            return_attention: اگر True باشد، attention weights برگردانده می‌شود
        
        Returns:
            task_embeddings: embeddings هر task [num_nodes, embedding_dim]
            critical_scores: امتیاز critical path هر task [num_nodes, 1]
        """
        x = task_graph.x
        edge_index = task_graph.edge_index
        edge_attr = task_graph.edge_attr
        
        # Input projection
        x = self.input_projection(x)
        
        # GAT layers
        attention_weights = []
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
            
            # ذخیره attention weights برای visualization (اختیاری)
            if return_attention:
                # می‌توانید attention weights را از لایه استخراج کنید
                pass
        
        # Output projection
        task_embeddings = self.output_projection(x)
        
        # محاسبه critical path scores
        critical_scores = self.critical_path_head(task_embeddings)
        
        if return_attention:
            return task_embeddings, critical_scores, attention_weights
        
        return task_embeddings, critical_scores
    
    def get_graph_embedding(self, task_graph: Data) -> torch.Tensor:
        """
        استخراج embedding کل گراف (برای تصمیم‌گیری سطح بالا)
        
        Args:
            task_graph: گراف task
        
        Returns:
            graph_embedding: embedding کل گراف [embedding_dim]
        """
        task_embeddings, _ = self.forward(task_graph)
        
        # Global mean pooling
        graph_embedding = torch.mean(task_embeddings, dim=0)
        
        # اعمال transformation
        graph_embedding = self.global_pool(graph_embedding)
        
        return graph_embedding
    
    def get_critical_path(
        self,
        task_graph: Data,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        شناسایی tasks در critical path
        
        Args:
            task_graph: گراف task
            threshold: آستانه برای تشخیص critical task
        
        Returns:
            critical_mask: ماسک boolean نشان‌دهنده critical tasks [num_nodes]
        """
        _, critical_scores = self.forward(task_graph)
        critical_mask = (critical_scores.squeeze() > threshold)
        return critical_mask
    
    def count_parameters(self) -> int:
        """شمارش پارامترهای قابل آموزش"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========================================
# Utility Functions
# ========================================

def create_task_graph_data(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_features: Optional[torch.Tensor] = None
) -> Data:
    """
    ایجاد PyTorch Geometric Data object از ویژگی‌های task
    
    Args:
        node_features: ویژگی‌های tasks [num_nodes, node_feature_dim]
        edge_index: indices یال‌های DAG [2, num_edges]
        edge_features: ویژگی‌های dependencies [num_edges, edge_feature_dim]
    
    Returns:
        data: PyTorch Geometric Data object
    """
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features
    )
    return data


def visualize_critical_path(
    task_graph: Data,
    critical_scores: torch.Tensor,
    task_ids: Optional[list] = None
):
    """
    نمایش critical path (برای debugging)
    
    Args:
        task_graph: گراف task
        critical_scores: امتیازات critical path
        task_ids: لیست شناسه tasks (اختیاری)
    """
    if task_ids is None:
        task_ids = list(range(task_graph.num_nodes))
    
    print("\n" + "="*50)
    print("🎯 Critical Path Analysis")
    print("="*50)
    
    # مرتب‌سازی tasks بر اساس critical score
    scores = critical_scores.squeeze().detach().cpu().numpy()
    sorted_indices = scores.argsort()[::-1]
    
    for idx in sorted_indices:
        task_id = task_ids[idx]
        score = scores[idx]
        status = "🔴 CRITICAL" if score > 0.5 else "⚪ Normal"
        print(f"Task {task_id}: {score:.4f} {status}")
    
    print("="*50 + "\n")
