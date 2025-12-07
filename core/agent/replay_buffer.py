import numpy as np
import random
from collections import deque
import torch


class ReplayBuffer:
    """
    Standard experience replay for MADDPG.
    Stores transitions as:
     (state, action, reward, next_state, done)
    """

    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # تبدیل action از dict به numpy array
        if isinstance(action, dict):
            action = self._dict_to_array(action)
        
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(np.array(action), dtype=torch.float32),
            torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)
    
    def _dict_to_array(self, action_dict):
        """
        تبدیل action dictionary به numpy array.
        
        Format action:
            offload: int (0-4) → one-hot encoded → 5 values
            cpu: float (0-1) → 1 value
            bandwidth: list[3] → 3 values
            move: list[2] → 2 values
        
        Total output: 11 dimensions
        """
        # One-hot encode offload decision
        offload = action_dict.get('offload', 0)
        offload_onehot = np.zeros(5, dtype=np.float32)
        if 0 <= offload < 5:
            offload_onehot[offload] = 1.0
        
        # Continuous values
        cpu = np.array([action_dict.get('cpu', 0.5)], dtype=np.float32)
        bandwidth = np.array(action_dict.get('bandwidth', [0.33, 0.33, 0.34]), dtype=np.float32)
        move = np.array(action_dict.get('move', [0.0, 0.0]), dtype=np.float32)
        
        # Concatenate all components
        action_array = np.concatenate([
            offload_onehot,  # 5 dimensions
            cpu,             # 1 dimension
            bandwidth,       # 3 dimensions
            move             # 2 dimensions
        ])
        
        return action_array
