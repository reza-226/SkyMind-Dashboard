"""
PettingZoo Environment Wrapper for MADDPG
"""
from pettingzoo.mpe import simple_tag_v3
import numpy as np


class MADDPGEnv:
    """
    Wrapper for PettingZoo simple_tag environment compatible with MADDPG
    """
    
    def __init__(self, num_good=1, n_adversaries=3, num_obstacles=2,
                 max_cycles=25, continuous_actions=True, n_agents=None,
                 num_adversaries=None):
        """
        Initialize MADDPG environment
        
        Args:
            num_good: Number of good agents (default: 1)
            n_adversaries: Number of adversary agents (for compatibility)
            num_adversaries: Number of adversary agents (alternative parameter)
            num_obstacles: Number of obstacles in environment
            max_cycles: Maximum number of steps per episode
            continuous_actions: Whether to use continuous action space
            n_agents: Total number of agents (if provided, overrides other settings)
        """
        # Handle different parameter names for adversaries
        if num_adversaries is not None:
            n_adversaries = num_adversaries
        elif n_adversaries is None:
            n_adversaries = 3
            
        # If n_agents is provided, calculate distribution
        if n_agents is not None:
            num_good = 1
            n_adversaries = n_agents - 1
        
        # Create the PettingZoo environment
        self.env = simple_tag_v3.parallel_env(
            num_good=num_good,
            num_adversaries=n_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions
        )
        
        # Store configuration
        self.num_good = num_good
        self.num_adversaries = n_adversaries
        self.n_adversaries = n_adversaries
        self.num_obstacles = num_obstacles
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.n_agents = num_good + n_adversaries
        
        # Initialize to get agent information
        observations, infos = self.env.reset()
        
        # Store agent information
        self.agents = list(self.env.agents)
        self.possible_agents = list(self.env.possible_agents)
        self.observation_spaces = {agent: self.env.observation_space(agent) 
                                   for agent in self.agents}
        self.action_spaces = {agent: self.env.action_space(agent) 
                             for agent in self.agents}
        
        # Get dimensions
        sample_agent = self.agents[0]
        self.observation_dim = self.observation_spaces[sample_agent].shape[0]
        self.action_dim = self.action_spaces[sample_agent].shape[0]
        
        print(f"✅ Environment initialized: {self.n_agents} agents")
        print(f"   - Good agents: {self.num_good}")
        print(f"   - Adversaries: {self.num_adversaries}")
        print(f"   - Observation dim: {self.observation_dim}")
        print(f"   - Action dim: {self.action_dim}")
    
    def reset(self, seed=None):
        """
        Reset the environment
        
        Returns:
            observations: Dictionary of observations for each agent
            infos: Dictionary of info for each agent
        """
        if seed is not None:
            observations, infos = self.env.reset(seed=seed)
        else:
            observations, infos = self.env.reset()
        
        return observations, infos
    
    def step(self, actions):
        """
        Execute actions for all agents
        
        Args:
            actions: Dictionary mapping agent names to actions
            
        Returns:
            observations: Next observations for all agents
            rewards: Rewards for all agents
            dones: Done flags for all agents
            truncations: Truncation flags for all agents
            infos: Additional information
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Combine terminations and truncations into dones
        dones = {agent: terminations.get(agent, False) or truncations.get(agent, False)
                for agent in self.agents}
        
        return observations, rewards, dones, truncations, infos
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def observation_space(self, agent):
        """Get observation space for agent"""
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        """Get action space for agent"""
        return self.action_spaces[agent]


def make_env(config):
    """
    Factory function to create MADDPG environment
    
    Args:
        config: Configuration object with environment parameters
        
    Returns:
        MADDPGEnv instance
    """
    return MADDPGEnv(
        num_good=getattr(config, 'num_good', 1),
        n_adversaries=getattr(config, 'n_adversaries', 3),
        num_obstacles=getattr(config, 'num_obstacles', 2),
        max_cycles=getattr(config, 'max_cycles', 25),
        continuous_actions=True
    )
