"""
Multi-Agent Wrapper for UAV Environment
Converts single-agent UAVEnvironment to multi-agent format
"""

import numpy as np
from environment.uav_env import UAVEnvironment


class MultiUAVWrapper:
    """
    Wrapper to convert single-agent UAVEnvironment to multi-agent
    
    Creates n_agents independent UAV environments and manages them
    Returns observations and rewards in multi-agent format
    """
    
    def __init__(self, n_agents=5, **env_kwargs):
        """
        Args:
            n_agents: Number of UAV agents
            **env_kwargs: Arguments passed to UAVEnvironment
        """
        self.n_agents = n_agents
        
        # Create separate environment for each agent
        self.envs = [UAVEnvironment(**env_kwargs) for _ in range(n_agents)]
        
        # Use first env's spaces as reference
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed=None):
        """Reset all agents"""
        observations = {}
        infos = {}
        
        for i in range(self.n_agents):
            obs, info = self.envs[i].reset(seed=seed)
            observations[f'uav_{i}'] = obs
            infos[f'uav_{i}'] = info
        
        return observations, infos
    
    def step(self, actions):
        """
        Step all agents
        
        Args:
            actions: Dict {agent_id: numpy_array(11)}
            
        Returns:
            observations: Dict {agent_id: obs}
            rewards: Dict {agent_id: reward}
            terminated: bool (all done)
            truncated: bool (any truncated)
            infos: Dict {agent_id: info}
        """
        observations = {}
        rewards = {}
        terminateds = []
        truncateds = []
        infos = {}
        
        for i in range(self.n_agents):
            agent_id = f'uav_{i}'
            action = actions[agent_id]
            
            obs, reward, terminated, truncated, info = self.envs[i].step(action)
            
            observations[agent_id] = obs
            rewards[agent_id] = reward
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos[agent_id] = info
        
        # Episode done if all agents done
        all_terminated = all(terminateds)
        any_truncated = any(truncateds)
        
        return observations, rewards, all_terminated, any_truncated, infos
