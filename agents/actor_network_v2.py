import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """
    Actor Network for UAV Offloading + Trajectory Control (Version 2).
    
    Input: [graph_emb (256), node_emb (256), flat_state (25)] → total: 537 dims
    Output:
      - offload logits (5)
      - continuous actions: cpu(1), bandwidth(3), movement(2)
    
    Improvements over v1:
    - Better initialization
    - Layer normalization support
    - Dropout for regularization
    """

    def __init__(self, state_dim=537, offload_dim=5, continuous_dim=6, 
                 hidden=512, use_layer_norm=False, dropout=0.0):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # Optional layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden)
            self.ln2 = nn.LayerNorm(hidden)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Output heads
        self.offload_head = nn.Linear(hidden, offload_dim)      # logits
        self.continuous_head = nn.Linear(hidden, continuous_dim)  # continuous ∈ [-1,1]

        self.activation = nn.ELU()
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: tensor (B, state_dim)
            
        Returns:
            offload_logits: (B, 5) - logits for offload destination
            continuous_action: (B, 6) - continuous actions in [-1, 1]
        """
        x = self.activation(self.fc1(state))
        if self.use_layer_norm:
            x = self.ln1(x)
        if self.dropout:
            x = self.dropout(x)
            
        x = self.activation(self.fc2(x))
        if self.use_layer_norm:
            x = self.ln2(x)
        if self.dropout:
            x = self.dropout(x)

        offload_logits = self.offload_head(x)
        cont = torch.tanh(self.continuous_head(x))   # scale to [-1,1]

        return offload_logits, cont
