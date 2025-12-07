import torch
import torch.nn as nn

class IndependentActorNetwork(nn.Module):
    """
    Actor Network for I-DDPG.
    Input: local_state (268)
    Output: offload logits (5) + continuous actions (6)
    """

    def __init__(self, local_state_dim=268, offload_dim=5, continuous_dim=6, hidden=512):
        super().__init__()

        self.fc1 = nn.Linear(local_state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        self.offload_head = nn.Linear(hidden, offload_dim)
        self.continuous_head = nn.Linear(hidden, continuous_dim)

        self.activation = nn.ELU()

    def forward(self, local_state):
        """
        local_state: (B, local_state_dim)
        """
        x = self.activation(self.fc1(local_state))
        x = self.activation(self.fc2(x))

        offload_logits = self.offload_head(x)
        cont = torch.tanh(self.continuous_head(x))

        return offload_logits, cont
