import os
import json

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    print(f"✅ Created: {path}")

def create_file(path, content=""):
    """Create file with optional content"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def setup_maddpg_structure():
    """Setup complete MADDPG training structure"""
    
    print("\n🚀 Setting up MADDPG Training Structure...\n")
    
    # ==================== DIRECTORIES ====================
    dirs = [
        "training",
        "environment",
        "checkpoints/maddpg",
        "logs/maddpg_training/tensorboard",
        "results/maddpg_results/charts"
    ]
    
    for d in dirs:
        create_directory(d)
    
    print("\n📝 Creating module files...\n")
    
    # ==================== INIT FILES ====================
    init_files = [
        "training/__init__.py",
        "environment/__init__.py"
    ]
    
    for init_file in init_files:
        create_file(init_file, '"""MADDPG Module"""\n')
    
    # ==================== TRAINING MODULE ====================
    
    # 1. Replay Buffer
    replay_buffer_code = '''import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """Experience Replay Buffer for MADDPG"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
'''
    create_file("training/replay_buffer.py", replay_buffer_code)
    
    # 2. Logger
    logger_code = '''import csv
import json
import os
from datetime import datetime

class TrainingLogger:
    """Log training metrics"""
    
    def __init__(self, log_dir="logs/maddpg_training"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.rewards_file = os.path.join(log_dir, "rewards.csv")
        self.losses_file = os.path.join(log_dir, "losses.csv")
        self.metrics_file = os.path.join(log_dir, "metrics.csv")
        
        # Initialize CSV files
        self._init_csv(self.rewards_file, ["episode", "reward", "avg_reward"])
        self._init_csv(self.losses_file, ["episode", "actor_loss", "critic_loss"])
        self._init_csv(self.metrics_file, ["episode", "success_rate", "avg_latency", "avg_energy"])
    
    def _init_csv(self, filepath, headers):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_reward(self, episode, reward, avg_reward):
        with open(self.rewards_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, avg_reward])
    
    def log_loss(self, episode, actor_loss, critic_loss):
        with open(self.losses_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, actor_loss, critic_loss])
    
    def log_metrics(self, episode, success_rate, avg_latency, avg_energy):
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, success_rate, avg_latency, avg_energy])
    
    def save_config(self, config):
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
'''
    create_file("training/logger.py", logger_code)
    
    # 3. Evaluator
    evaluator_code = '''import numpy as np
import torch

class Evaluator:
    """Evaluate agent performance"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def evaluate(self, num_episodes=10):
        """Run evaluation episodes"""
        rewards = []
        success_rates = []
        latencies = []
        energies = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_success = 0
            episode_latency = []
            episode_energy = []
            done = False
            
            while not done:
                with torch.no_grad():
                    action = self.agent.select_action(state, explore=False)
                
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if info.get('success', False):
                    episode_success += 1
                    episode_latency.append(info.get('latency', 0))
                    episode_energy.append(info.get('energy', 0))
                
                state = next_state
            
            rewards.append(episode_reward)
            success_rates.append(episode_success)
            if episode_latency:
                latencies.append(np.mean(episode_latency))
            if episode_energy:
                energies.append(np.mean(episode_energy))
        
        return {
            'avg_reward': np.mean(rewards),
            'avg_success_rate': np.mean(success_rates),
            'avg_latency': np.mean(latencies) if latencies else 0,
            'avg_energy': np.mean(energies) if energies else 0
        }
'''
    create_file("training/evaluator.py", evaluator_code)
    
    # 4. Training Script Placeholder
    train_script = '''"""
MADDPG Training Script

Usage:
    python training/train_maddpg.py --episodes 1000 --batch_size 64
"""

import argparse
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_actor', type=float, default=0.001)
    parser.add_argument('--lr_critic', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    
    args = parser.parse_args()
    
    print("🚀 MADDPG Training will be implemented here!")
    print(f"Episodes: {args.episodes}")
    print(f"Batch Size: {args.batch_size}")

if __name__ == "__main__":
    main()
'''
    create_file("training/train_maddpg.py", train_script)
    
    # ==================== ENVIRONMENT MODULE ====================
    
    # 1. State Builder
    state_builder_code = '''import numpy as np

class StateBuilder:
    """Build 537-dim state vector"""
    
    def __init__(self):
        self.graph_dim = 256
        self.node_dim = 256
        self.flat_dim = 25
    
    def build_state(self, graph_features, node_features, flat_features):
        """
        Combine features into single state vector
        
        Args:
            graph_features: (256,) graph embedding
            node_features: (256,) node embedding
            flat_features: (25,) flat state [battery, cpu, position, etc.]
        
        Returns:
            state: (537,) concatenated state
        """
        state = np.concatenate([
            graph_features,
            node_features,
            flat_features
        ])
        return state
'''
    create_file("environment/state_builder.py", state_builder_code)
    
    # 2. Reward Calculator
    reward_calculator_code = '''class RewardCalculator:
    """Calculate reward based on latency, energy, and success"""
    
    def __init__(self, w_latency=0.4, w_energy=0.3, w_success=0.3):
        self.w_latency = w_latency
        self.w_energy = w_energy
        self.w_success = w_success
    
    def calculate(self, latency, energy, success):
        """
        Calculate total reward
        
        Args:
            latency: Task latency (ms)
            energy: Energy consumed (J)
            success: Task success (bool)
        
        Returns:
            reward: float
        """
        # Normalize latency (lower is better)
        latency_reward = -latency / 1000.0  # scale to [-1, 0]
        
        # Normalize energy (lower is better)
        energy_reward = -energy / 10.0  # scale to [-1, 0]
        
        # Success bonus
        success_reward = 10.0 if success else -5.0
        
        reward = (
            self.w_latency * latency_reward +
            self.w_energy * energy_reward +
            self.w_success * success_reward
        )
        
        return reward
'''
    create_file("environment/reward_calculator.py", reward_calculator_code)
    
    # 3. UAV Environment Placeholder
    uav_env_code = '''"""
UAV Offloading Environment (OpenAI Gym Interface)

TODO: Implement complete environment with:
- State space: (537,) vector
- Action space: offload + continuous actions
- Reward: latency + energy + success
"""

import gym
import numpy as np

class UAVEnvironment(gym.Env):
    """Custom UAV Environment"""
    
    def __init__(self):
        super().__init__()
        self.state_dim = 537
        self.action_dim = 11
    
    def reset(self):
        """Reset environment"""
        state = np.zeros(self.state_dim)
        return state
    
    def step(self, action):
        """Execute action"""
        next_state = np.zeros(self.state_dim)
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info
'''
    create_file("environment/uav_env.py", uav_env_code)
    
    # ==================== CONFIG FILES ====================
    
    # Requirements
    requirements = '''torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.13.0
gym>=0.26.0
'''
    create_file("requirements_maddpg.txt", requirements)
    
    # Training Config
    config = {
        "training": {
            "episodes": 1000,
            "max_steps": 500,
            "batch_size": 64,
            "replay_buffer_size": 100000,
            "warmup_episodes": 50
        },
        "hyperparameters": {
            "lr_actor": 0.001,
            "lr_critic": 0.001,
            "gamma": 0.99,
            "tau": 0.005
        },
        "monitoring": {
            "log_interval": 10,
            "eval_interval": 50,
            "save_interval": 100
        }
    }
    
    config_path = "checkpoints/maddpg/training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Created: {config_path}")
    
