"""
Model Evaluator
Evaluates trained models on the environment
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional
import sys
import os

# Add project root to path
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))

from environments.uav_mec_env import UAVMECEnvironment  # ✅ اصلاح شده
from models.actor_critic.maddpg_agent import MADDPGAgent


class ModelEvaluator:
    """Evaluates trained models"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to model checkpoint (.pth)
            config_path: Path to config.json (if None, looks in model directory)
        """
        self.model_path = Path(model_path)
        
        # Load config
        if config_path is None:
            config_path = self.model_path.parent / 'config.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract dimensions with fallback
        if 'dimensions' in self.config:
            dims = self.config['dimensions']
            self.state_dim = dims['state']
            self.action_dim = dims['action']
        else:
            self.state_dim = self.config.get('state_dim', 537)
            self.action_dim = self.config.get('action_dim', 11)
        
        # Initialize environment
        env_config = self.config.get('env_config', {})
        self.env = UAVMECEnvironment(**env_config)
        
        # Initialize agent
        self.agent = MADDPGAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=512,
            lr_actor=1e-4,
            lr_critic=1e-3
        )
        
        # Load model weights
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.actor.eval()
    
    def _adapt_state(self, state: np.ndarray) -> np.ndarray:
        """
        Adapt state dimension if needed (padding or truncation)
        
        Args:
            state: Input state from environment
            
        Returns:
            Adapted state matching model's expected dimension
        """
        current_dim = state.shape[0] if len(state.shape) == 1 else state.shape[-1]
        
        if current_dim == self.state_dim:
            return state
        
        if current_dim < self.state_dim:
            # Pad with zeros
            padding = self.state_dim - current_dim
            return np.concatenate([state, np.zeros(padding)])
        else:
            # Truncate
            return state[:self.state_dim]
    
    def evaluate(self, num_episodes: int = 10, max_steps: int = 50, 
                 noise: float = 0.0) -> Dict:
        """
        Evaluate model over multiple episodes
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            noise: Exploration noise (0.0 for deterministic)
            
        Returns:
            Dictionary with evaluation results
        """
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            state = self.env.reset()
            
            # Handle tuple return from reset
            if isinstance(state, tuple):
                state = state[0]
            
            # Adapt state dimension
            state = self._adapt_state(state)
            
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                with torch.no_grad():
                    action = self.agent.select_action(state, noise=noise)
                
                result = self.env.step(action)
                
                # Handle different return formats
                if len(result) == 4:
                    next_state, reward, done, info = result
                elif len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    break
                
                # Handle tuple state
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                
                # Adapt next state
                next_state = self._adapt_state(next_state)
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        # Calculate statistics
        rewards_array = np.array(episode_rewards)
        
        results = {
            'num_episodes': num_episodes,
            'mean_reward': float(rewards_array.mean()),
            'std_reward': float(rewards_array.std()),
            'min_reward': float(rewards_array.min()),
            'max_reward': float(rewards_array.max()),
            'median_reward': float(np.median(rewards_array)),
            'mean_length': float(np.mean(episode_lengths)),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'model_path': str(self.model_path),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
