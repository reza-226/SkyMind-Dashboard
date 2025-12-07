import torch
import torch.nn as nn

class IndependentCriticNetwork(nn.Module):
    """
    Decentralized Critic for I-DDPG.
    هر agent فقط state خودش رو می‌بینه.
    
    Input: local_state (268) + local_action (11)
    Output: Q-value
    """

    def __init__(self, local_state_dim=268, action_dim=11, hidden=512):
        super().__init__()

        self.fc1 = nn.Linear(local_state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_out = nn.Linear(hidden, 1)

        self.activation = nn.ELU()

    def forward(self, local_state, action):
        """
        local_state: (B, 268) - فقط state مربوط به خود agent
        action: (B, 11)
        """
        x = torch.cat([local_state, action], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        q = self.q_out(x)
        return q
