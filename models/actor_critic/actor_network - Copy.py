import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """
    Actor Network for MADDPG - UAV Offloading Decision Making
    
    Input: State vector (537-dim)
        - Graph embedding (256)
        - Node embeddings (256) 
        - Flat state features (25)
    
    Output:
        - Offload decision logits (5): [local, edge1, edge2, edge3, cloud]
        - Continuous actions (6): [cpu_allocation, bandwidth_x3, movement_x2]
    """
    
    def __init__(self, state_dim=537, offload_dim=5, continuous_dim=6, hidden_dim=512):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.offload_dim = offload_dim
        self.continuous_dim = continuous_dim
        
        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        # Output heads
        self.offload_head = nn.Linear(hidden_dim // 2, offload_dim)
        self.continuous_head = nn.Linear(hidden_dim // 2, continuous_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: torch.Tensor of shape (batch_size, 537)
        
        Returns:
            offload_logits: (batch_size, 5) - raw logits for offloading decision
            continuous_actions: (batch_size, 6) - normalized continuous actions in [-1, 1]
        """
        # Feature extraction with residual connections
        x = F.elu(self.ln1(self.fc1(state)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))
        
        # Offloading decision (discrete)
        offload_logits = self.offload_head(x)
        
        # Continuous actions (normalized to [-1, 1])
        continuous_actions = torch.tanh(self.continuous_head(x))
        
        return offload_logits, continuous_actions
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action with optional exploration noise
        
        Args:
            state: numpy array or torch.Tensor
            epsilon: exploration noise scale
        
        Returns:
            offload_logits, continuous_actions (with noise if epsilon > 0)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            offload_logits, continuous_actions = self.forward(state)
            
            # Add exploration noise
            if epsilon > 0:
                noise = torch.randn_like(continuous_actions) * epsilon
                continuous_actions = torch.clamp(continuous_actions + noise, -1.0, 1.0)
        
        return offload_logits, continuous_actions
