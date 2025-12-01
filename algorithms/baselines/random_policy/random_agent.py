"""
Random Policy Baseline Agent
Generates random actions within the action space bounds
"""
import numpy as np
import gym

class RandomAgent:
    """Simple random policy baseline"""
    
    def __init__(self, observation_dim, action_dim, action_space=None):
        """
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space  
            action_space: gym.Space object (optional, for proper sampling)
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_space = action_space
        
    def get_action(self, observation, deterministic=False):
        """
        Generate random action
        
        Args:
            observation: Current observation (not used, but kept for interface compatibility)
            deterministic: Not used for random policy
            
        Returns:
            Random action sampled from action space
        """
        if self.action_space is not None:
            return self.action_space.sample()
        else:
            # Fallback: uniform random in [-1, 1]
            return np.random.uniform(-1, 1, size=self.action_dim)
    
    def save(self, filepath):
        """Random policy has no parameters to save"""
        pass
    
    def load(self, filepath):
        """Random policy has no parameters to load"""
        pass
