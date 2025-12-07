"""
Custom Graph Attention Network (GAT) Layers
لایه‌های توجه گراف سفارشی برای Task DAG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, Tuple


class GATLayer(MessagePassing):
    """
    لایه Graph Attention Network با edge features
    
    این لایه از مکانیزم attention برای یادگیری وزن‌های پویا
    بین گره‌های متصل استفاده می‌کند.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs
    ):
        """
        Args:
            in_channels: تعداد ویژگی‌های ورودی
            out_channels: تعداد ویژگی‌های خروجی (per head)
            heads: تعداد attention heads
            concat: اگر True باشد، خروجی heads concatenate می‌شود
            negative_slope: شیب منفی LeakyReLU
            dropout: نرخ dropout برای attention coefficients
            edge_dim: تعداد ویژگی‌های یال (اختیاری)
            bias: استفاده از bias
        """
        # اضافه کردن node_dim برای سازگاری با PyG
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        # تبدیل خطی ویژگی‌های گره
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # پارامترهای attention
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # تبدیل خطی edge features (اگر وجود داشته باشد)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)
        
        # Bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """مقداردهی اولیه پارامترها"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: ویژگی‌های گره [num_nodes, in_channels]
            edge_index: ایندکس یال‌ها [2, num_edges]
            edge_attr: ویژگی‌های یال [num_edges, edge_dim]
            return_attention_weights: اگر True باشد attention weights برگردانده می‌شود
        
        Returns:
            خروجی لایه [num_nodes, heads * out_channels]
            (اختیاری) attention weights
        """
        # تبدیل خطی ویژگی‌های گره
        H, C = self.heads, self.out_channels
        
        # x: [num_nodes, in_channels] -> [num_nodes, heads * out_channels]
        x = self.lin(x).view(-1, H, C)
        
        # پردازش edge features
        if edge_attr is not None and self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, H, C)
        else:
            edge_attr = None
        
        # ذخیره برای استفاده در message passing
        self._edge_attr = edge_attr
        self._return_attention = return_attention_weights
        
        # Message passing
        out = self.propagate(
            edge_index,
            x=x,
            size=None
        )
        
        # Concatenate یا average کردن heads
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        
        # اضافه کردن bias
        if self.bias is not None:
            out = out + self.bias
        
        # برگرداندن attention weights در صورت درخواست
        if return_attention_weights:
            return out, self._attention_weights
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        index: torch.Tensor,
        size_i: Optional[int]
    ) -> torch.Tensor:
        """
        محاسبه message برای هر یال
        
        Args:
            x_i: ویژگی‌های گره مقصد [num_edges, heads, out_channels]
            x_j: ویژگی‌های گره مبدأ [num_edges, heads, out_channels]
            index: ایندکس گره‌های مقصد
            size_i: تعداد گره‌های مقصد
        
        Returns:
            messages وزن‌دار شده
        """
        # محاسبه attention scores
        # alpha = LeakyReLU(a^T [W*h_i || W*h_j || W*e_ij])
        
        alpha_src = (x_i * self.att_src).sum(dim=-1)  # [num_edges, heads]
        alpha_dst = (x_j * self.att_dst).sum(dim=-1)  # [num_edges, heads]
        
        alpha = alpha_src + alpha_dst
        
        # افزودن edge features به attention
        if self._edge_attr is not None:
            alpha_edge = (self._edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        # LeakyReLU activation
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Softmax normalization
        alpha = softmax(alpha, index, num_nodes=size_i)
        
        # ذخیره attention weights (برای visualization)
        if self._return_attention:
            self._attention_weights = alpha
        
        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # اعمال attention weights به messages
        # [num_edges, heads, 1] * [num_edges, heads, out_channels]
        return x_j * alpha.unsqueeze(-1)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'heads={self.heads})')


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head GAT layer با residual connection
    
    این لایه چند head attention را ترکیب می‌کند و
    یک residual connection اضافه می‌کند.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.0,
        residual: bool = True
    ):
        """
        Args:
            in_channels: تعداد ویژگی‌های ورودی
            out_channels: تعداد ویژگی‌های خروجی
            heads: تعداد attention heads
            edge_dim: تعداد ویژگی‌های یال
            dropout: نرخ dropout
            residual: استفاده از residual connection
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual
        
        # GAT layer
        self.gat = GATLayer(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        # Residual projection (اگر ابعاد متفاوت باشد)
        if residual and in_channels != out_channels:
            self.res_fc = nn.Linear(in_channels, out_channels)
        else:
            self.res_fc = None
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_channels)
        
        # Activation
        self.activation = nn.ELU()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: ویژگی‌های گره
            edge_index: ایندکس یال‌ها
            edge_attr: ویژگی‌های یال
        
        Returns:
            خروجی لایه
        """
        identity = x
        
        # GAT forward
        out = self.gat(x, edge_index, edge_attr)
        
        # Residual connection
        if self.residual:
            if self.res_fc is not None:
                identity = self.res_fc(identity)
            out = out + identity
        
        # Normalization و activation
        out = self.norm(out)
        out = self.activation(out)
        
        return out
