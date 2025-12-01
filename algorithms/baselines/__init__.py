"""
Baseline algorithms for comparison with MADDPG.
Includes: DDPG, PPO, Independent DDPG, and Random Policy.
"""

from .ddpg_agent import DDPGAgent
from .ppo_agent import PPOAgent
from .independent_ddpg import IndependentDDPGAgent
from .random_policy import RandomAgent

__all__ = [
    'DDPGAgent',
    'PPOAgent', 
    'IndependentDDPGAgent',
    'RandomAgent'
]
