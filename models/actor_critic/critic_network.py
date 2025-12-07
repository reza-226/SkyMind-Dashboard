import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    """
    Centralized Critic Network for MADDPG
    
    Input:
        - State: (batch_size, 537)
        - Action: (batch_size, 11) - [offload_one_hot(5) + continuous(6)]
    
    Output:
        - Q-value: (batch_size, 1)
    
    The critic uses centralized training with full state-action information
    to provide accurate Q-value estimates for the actor's policy gradient.
    """
    
    def __init__(self, state_dim=537, action_dim=11, hidden_dim=512):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State processing branch
        self.state_fc1 = nn.Linear(state_dim, hidden_dim)
        self.state_ln1 = nn.LayerNorm(hidden_dim)
        
        # Action processing branch  
        self.action_fc1 = nn.Linear(action_dim, hidden_dim // 4)
        self.action_ln1 = nn.LayerNorm(hidden_dim // 4)
        
        # Fusion layers
        fusion_dim = hidden_dim + hidden_dim // 4
        self.fusion_fc1 = nn.Linear(fusion_dim, hidden_dim)
        self.fusion_ln1 = nn.LayerNorm(hidden_dim)
        
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fusion_ln2 = nn.LayerNorm(hidden_dim // 2)
        
        # Q-value output
        self.q_out = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
        # Small initialization for Q output (helps stability)
        nn.init.uniform_(self.q_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_out.bias, -3e-3, 3e-3)
    
    def forward(self, state, action):
        """
        Forward pass
        
        Args:
            state: torch.Tensor (batch_size, 537)
            action: torch.Tensor (batch_size, 11)
        
        Returns:
            q_value: torch.Tensor (batch_size, 1)
        """
        # Process state
        state_features = F.elu(self.state_ln1(self.state_fc1(state)))
        
        # Process action
        action_features = F.elu(self.action_ln1(self.action_fc1(action)))
        
        # Concatenate and fuse
        x = torch.cat([state_features, action_features], dim=-1)
        x = F.elu(self.fusion_ln1(self.fusion_fc1(x)))
        x = F.elu(self.fusion_ln2(self.fusion_fc2(x)))
        
        # Q-value output
        q_value = self.q_out(x)
        
        return q_value
