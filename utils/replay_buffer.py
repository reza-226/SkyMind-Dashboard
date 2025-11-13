import numpy as np
from collections import deque
import random


class SimpleReplayBuffer:
    """یک Replay Buffer ساده برای ذخیره تجربیات"""
    
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        """
        اضافه کردن یک تجربه به buffer
        
        Args:
            state: numpy array
            action: numpy array
            reward: float
            next_state: numpy array
            done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        نمونه‌برداری تصادفی از buffer
        
        Args:
            batch_size: تعداد نمونه‌ها
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """پاک کردن تمام buffer"""
        self.buffer.clear()
