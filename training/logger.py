import csv
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
