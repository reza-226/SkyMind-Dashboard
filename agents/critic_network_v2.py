import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    """
    Centralized Critic for MADDPG (Version 2).
    
    Input: state (537) + action vector (11)
    Output: Q-value
    
    Improvements over v1:
    - Better initialization
    - Layer normalization support
    - Additional hidden layer option
    - Gradient clipping support
    """

    def __init__(self, state_dim=537, action_dim=11, hidden=512, 
                 use_layer_norm=False, num_hidden_layers=2):
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.use_layer_norm = use_layer_norm

        # Input layer
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden, hidden))
        
        # Output layer
        self.q_out = nn.Linear(hidden, 1)

        # Optional layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList()
            for _ in range(num_hidden_layers):
                self.layer_norms.append(nn.LayerNorm(hidden))

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
        
        # Special initialization for Q output
        nn.init.uniform_(self.q_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_out.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        """
        Forward pass through the network.
        
        Args:
            state: (B, 537) - state vector
            action: (B, 11) - action vector [one-hot offload + continuous 6]
            
        Returns:
            q: (B, 1) - Q-value estimate
        """
        x = torch.cat([state, action], dim=-1)
        
        # First layer
        x = self.activation(self.fc1(x))
        if self.use_layer_norm:
            x = self.layer_norms[0](x)
        
        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            if self.use_layer_norm:
                x = self.layer_norms[i + 1](x)
        
        # Output
        q = self.q_out(x)
        return q
