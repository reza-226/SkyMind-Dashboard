"""
Action Analyzer
Analyzes action distributions and patterns from trained model
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import sys

# Add project root to path
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))

from environments.uav_mec_env import UAVMECEnvironment  # ✅ اصلاح شده
from models.actor_critic.maddpg_agent import MADDPGAgent


class ActionAnalyzer:
    """Analyzes action distributions and decision patterns"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """Initialize analyzer"""
        self.model_path = Path(model_path)
        
        # Load config
        if config_path is None:
            config_path = self.model_path.parent / 'config.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract dimensions
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
        
        # Load model
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.actor.eval()
    
    def _adapt_state(self, state: np.ndarray) -> np.ndarray:
        """Adapt state dimension"""
        current_dim = state.shape[0] if len(state.shape) == 1 else state.shape[-1]
        
        if current_dim == self.state_dim:
            return state
        elif current_dim < self.state_dim:
            padding = self.state_dim - current_dim
            return np.concatenate([state, np.zeros(padding)])
        else:
            return state[:self.state_dim]
    
    def analyze(self, num_episodes: int = 20, max_steps: int = 50) -> Dict:
        """
        Analyze actions over multiple episodes
        
        Returns:
            Dictionary with action analysis results
        """
        all_actions = []
        all_offload_decisions = []
        all_cpu_allocations = []
        all_bandwidth_allocations = []
        all_movements = []
        episode_data = []
        
        for ep in range(num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state = self._adapt_state(state)
            
            episode_actions = []
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                with torch.no_grad():
                    action = self.agent.select_action(state, noise=0.0)
                
                episode_actions.append(action)
                
                # Parse action components (assuming standard format)
                # action = [offload_one_hot(5), cpu(1), bandwidth(3), movement(2)]
                if len(action) >= 11:
                    offload_idx = np.argmax(action[:5])
                    cpu = action[5]
                    bandwidth = action[6:9]
                    movement = action[9:11]
                    
                    all_offload_decisions.append(offload_idx)
                    all_cpu_allocations.append(cpu)
                    all_bandwidth_allocations.append(bandwidth)
                    all_movements.append(movement)
                
                result = self.env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                elif len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    break
                
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                next_state = self._adapt_state(next_state)
                
                state = next_state
                steps += 1
            
            all_actions.extend(episode_actions)
            episode_data.append({
                'episode': ep,
                'num_actions': len(episode_actions)
            })
        
        # Convert to arrays
        all_actions = np.array(all_actions)
        all_offload_decisions = np.array(all_offload_decisions)
        all_cpu_allocations = np.array(all_cpu_allocations)
        all_bandwidth_allocations = np.array(all_bandwidth_allocations)
        all_movements = np.array(all_movements)
        
        # Analysis results
        results = {
            'num_episodes': num_episodes,
            'total_actions': len(all_actions),
            
            # Offload distribution
            'offload_distribution': {
                f'server_{i}': int(np.sum(all_offload_decisions == i))
                for i in range(5)
            },
            
            # CPU allocation stats
            'cpu_allocation': {
                'mean': float(all_cpu_allocations.mean()),
                'std': float(all_cpu_allocations.std()),
                'min': float(all_cpu_allocations.min()),
                'max': float(all_cpu_allocations.max())
            },
            
            # Bandwidth allocation stats (per dimension)
            'bandwidth_allocation': {
                f'dim_{i}': {
                    'mean': float(all_bandwidth_allocations[:, i].mean()),
                    'std': float(all_bandwidth_allocations[:, i].std())
                }
                for i in range(min(3, all_bandwidth_allocations.shape[1]))
            },
            
            # Movement stats
            'movement_patterns': {
                'x_movement': {
                    'mean': float(all_movements[:, 0].mean()),
                    'std': float(all_movements[:, 0].std())
                },
                'y_movement': {
                    'mean': float(all_movements[:, 1].mean()),
                    'std': float(all_movements[:, 1].std())
                }
            } if all_movements.shape[1] >= 2 else {},
            
            'model_path': str(self.model_path)
        }
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save analysis results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
