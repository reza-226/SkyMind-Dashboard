"""
MADDPG Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MADDPGActor(nn.Module):
    """Actor Network برای MADDPG"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128],
    ):
        super().__init__()
        
        # ساخت لایه‌ها
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # اکشن‌ها در [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class MADDPGCritic(nn.Module):
    """Critic Network برای MADDPG"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128],
    ):
        super().__init__()
        
        # ساخت لایه‌ها
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))  # Q-value
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)
