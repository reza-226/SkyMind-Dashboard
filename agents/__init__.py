"""
MADDPG Agent Components (Version 2)
Improved and refactored for better modularity
"""

from .actor_network_v2 import ActorNetwork
from .critic_network_v2 import CriticNetwork
from .action_decoder_v2 import ActionDecoder
from .maddpg_agent_v2 import MADDPGAgent

__all__ = [
    'ActorNetwork',
    'CriticNetwork', 
    'ActionDecoder',
    'MADDPGAgent'
]

__version__ = '2.0.0'
