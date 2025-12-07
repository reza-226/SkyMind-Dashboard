# buffer.py
"""
Replay Buffer for MADDPG
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Experience Replay Buffer for Multi-Agent RL"""
    
    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Maximum size of buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        
        Args:
            state: Concatenated states of all agents
            action: Concatenated actions of all agents
            reward: Dictionary of rewards {agent_id: reward}
            next_state: Concatenated next states
            done: Dictionary of done flags {agent_id: done}
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample random batch from buffer
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            Tuple of batched experiences
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = [exp[2] for exp in batch]  # List of dicts
        next_states = np.array([exp[3] for exp in batch])
        dones = [exp[4] for exp in batch]  # List of dicts
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
