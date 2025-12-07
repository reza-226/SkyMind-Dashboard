"""
utils/ou_noise.py
Ornstein-Uhlenbeck Noise for Exploration
"""
import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process"""
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        """Reset noise state"""
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        """Generate noise sample"""
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
