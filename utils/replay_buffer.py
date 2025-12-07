"""
Multi-Agent Replay Buffer for MADDPG
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for Multi-Agent MADDPG
    
    Stores transitions: (states, actions, rewards, next_states, dones)
    where each is a list/array of length n_agents
    """
    
    def __init__(self, capacity=100000, n_agents=5):
        """
        Args:
            capacity: Maximum buffer size
            n_agents: Number of agents
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states, actions, rewards, next_states, done):
        """
        Store a transition
        
        Args:
            states: List of states for each agent (n_agents, state_dim)
            actions: List of action vectors for each agent (n_agents, action_dim)
            rewards: List of rewards for each agent (n_agents,)
            next_states: List of next states (n_agents, state_dim)
            done: Boolean (same for all agents in this implementation)
        """
        # Convert to numpy arrays if needed
        if not isinstance(states, np.ndarray):
            if isinstance(states, list):
                states = np.array(states)
            else:
                # Single state replicated for all agents
                states = np.array([states] * self.n_agents)
        
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        if not isinstance(rewards, (list, np.ndarray)):
            rewards = np.array([rewards] * self.n_agents)
        elif not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        
        if not isinstance(next_states, np.ndarray):
            if isinstance(next_states, list):
                next_states = np.array(next_states)
            else:
                next_states = np.array([next_states] * self.n_agents)
        
        # Store as tuple
        self.buffer.append((states, actions, rewards, next_states, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of numpy arrays:
                - states: (batch_size, n_agents, state_dim)
                - actions: (batch_size, n_agents, action_dim)
                - rewards: (batch_size, n_agents)
                - next_states: (batch_size, n_agents, state_dim)
                - dones: (batch_size, n_agents)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([[x[4]] * self.n_agents for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def get_state(self):
        """
        Get buffer state for checkpointing
        
        Returns:
            Dictionary with buffer state
        """
        return {
            'buffer': list(self.buffer),
            'capacity': self.capacity,
            'n_agents': self.n_agents
        }
    
    def load_state(self, state_dict):
        """
        Load buffer state from checkpoint
        
        Args:
            state_dict: Dictionary containing buffer state
        """
        self.capacity = state_dict['capacity']
        self.n_agents = state_dict['n_agents']
        self.buffer = deque(state_dict['buffer'], maxlen=self.capacity)
