import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """
    Actor Network for UAV Offloading (TRAINED VERSION).
    Architecture matches the checkpoint structure exactly.
    Input: [graph_emb (256), node_emb (256), flat_state (25)] → total: 537 dims
    Output:
      - offload logits (5)
      - continuous actions: (continuous_dim) - may be 0!
    """

    def __init__(self, state_dim=537, offload_dim=5, continuous_dim=0, hidden_dim=512):
        super().__init__()
        
        self.continuous_dim = continuous_dim

        # 3-layer architecture با LayerNorm (مطابق checkpoint)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # 512 → 256
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        # Output heads
        self.offload_head = nn.Linear(hidden_dim // 2, offload_dim)
        
        # Continuous head - همیشه موجوده (حتی با dim=0)
        self.continuous_head = nn.Linear(hidden_dim // 2, continuous_dim)
        
        self.activation = nn.ELU()

    def forward(self, state):
        """
        state: tensor (B, state_dim)
        return:
            offload_logits: (B, 5)
            continuous_action: (B, continuous_dim) - may be empty (B, 0)
        """
        # 3-layer forward pass with LayerNorm
        x = self.activation(self.ln1(self.fc1(state)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.activation(self.ln3(self.fc3(x)))
        
        # Output heads
        offload_logits = self.offload_head(x)
        
        # Continuous action (با tanh فقط اگر dim>0)
        if self.continuous_dim > 0:
            cont = torch.tanh(self.continuous_head(x))
        else:
            # Empty tensor: shape (B, 0)
            cont = self.continuous_head(x)
        
        return offload_logits, cont
