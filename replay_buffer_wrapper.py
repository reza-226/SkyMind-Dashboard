"""
Wrapper برای Replay Buffer
"""

import numpy as np
from collections import deque
import random
from utils.replay_buffer import SimpleReplayBuffer


class ReplayBufferWrapper:
    """Wrapper برای استفاده آسان‌تر از Replay Buffer"""
    
    def __init__(self, buffer_size=100000, batch_size=128, **kwargs):
        """
        Args:
            buffer_size: حداکثر تعداد تجربیات در buffer
            batch_size: اندازه batch برای نمونه‌برداری (در اینجا استفاده نمی‌شود)
        """
        # ✅ فقط max_size را به SimpleReplayBuffer ارسال می‌کنیم
        self.buffer = SimpleReplayBuffer(max_size=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """اضافه کردن یک تجربه"""
        self.buffer.add(state, action, reward, next_state, done)
    
    def sample(self, batch_size=None):
        """
        نمونه‌برداری از buffer
        
        Args:
            batch_size: اندازه batch (اگر None باشد از self.batch_size استفاده می‌شود)
        """
        if batch_size is None:
            batch_size = self.batch_size
        return self.buffer.sample(batch_size)
    
    def __len__(self):
        """برگرداندن تعداد تجربیات"""
        return len(self.buffer)
    
    def clear(self):
        """پاک کردن buffer"""
        self.buffer.clear()
