"""
Training Logger for MADDPG
"""

import json
import os
import csv
from datetime import datetime
from collections import defaultdict
import numpy as np


class TrainingLogger:
    """
    Logger for tracking training progress
    """
    
    def __init__(self, log_dir):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Log files
        self.episode_log_file = os.path.join(log_dir, 'episodes.csv')
        self.eval_log_file = os.path.join(log_dir, 'evaluations.csv')
        self.summary_file = os.path.join(log_dir, 'training_summary.json')
        
        # Episode data
        self.episodes = []
        self.evaluations = []
        
        # Running statistics
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # Initialize CSV files
        self._init_episode_log()
        self._init_eval_log()
        
    def _init_episode_log(self):
        """Initialize episode log CSV"""
        with open(self.episode_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode',
                'reward',
                'steps',
                'actor_loss',
                'critic_loss',
                'total_steps',
                'timestamp'
            ])
    
    def _init_eval_log(self):
        """Initialize evaluation log CSV"""
        with open(self.eval_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode',
                'avg_reward',
                'std_reward',
                'min_reward',
                'max_reward',
                'timestamp'
            ])
    
    def log_episode(self, episode_data):
        """
        Log episode results
        
        Args:
            episode_data: Dictionary containing:
                - episode: Episode number
                - reward: Total episode reward
                - steps: Number of steps
                - losses: Dictionary with actor_loss and critic_loss
        """
        episode = episode_data['episode']
        reward = episode_data['reward']
        steps = episode_data['steps']
        losses = episode_data.get('losses', {})
        
        self.total_steps += steps
        
        # Update best reward
        if reward > self.best_reward:
            self.best_reward = reward
            print(f"🎉 New best reward: {reward:.2f}")
        
        # Save to episodes list
        self.episodes.append({
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'actor_loss': losses.get('actor_loss', 0) if losses else 0,
            'critic_loss': losses.get('critic_loss', 0) if losses else 0,
            'total_steps': self.total_steps
        })
        
        # Write to CSV
        with open(self.episode_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                reward,
                steps,
                losses.get('actor_loss', 0) if losses else 0,
                losses.get('critic_loss', 0) if losses else 0,
                self.total_steps,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
    
    def log_evaluation(self, eval_data):
        """
        Log evaluation results
        
        Args:
            eval_data: Dictionary containing evaluation metrics
        """
        self.evaluations.append(eval_data)
        
        # Write to CSV
        with open(self.eval_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                eval_data.get('episode', len(self.episodes)),
                eval_data['avg_reward'],
                eval_data['std_reward'],
                eval_data['min_reward'],
                eval_data['max_reward'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
    
    def get_recent_avg_reward(self, n=10):
        """
        Get average reward over last n episodes
        
        Args:
            n: Number of recent episodes
            
        Returns:
            Average reward
        """
        if len(self.episodes) == 0:
            return 0.0
        
        recent = self.episodes[-n:]
        return np.mean([ep['reward'] for ep in recent])
    
    def save_summary(self):
        """Save training summary"""
        if len(self.episodes) == 0:
            print("⚠️ No episodes to summarize")
            return
        
        rewards = [ep['reward'] for ep in self.episodes]
        
        summary = {
            'total_episodes': len(self.episodes),
            'total_steps': self.total_steps,
            'best_reward': float(self.best_reward),
            'final_avg_reward_10': float(self.get_recent_avg_reward(10)),
            'final_avg_reward_50': float(self.get_recent_avg_reward(50)),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:10.2f}")
            else:
                print(f"{key:25s}: {value}")
        print("="*60)
        
        return summary
    
    def plot_training_curves(self):
        """
        Plot training curves (requires matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(self.episodes) == 0:
                print("⚠️ No data to plot")
                return
            
            episodes_nums = [ep['episode'] for ep in self.episodes]
            rewards = [ep['reward'] for ep in self.episodes]
            
            # Calculate moving average
            window = min(50, len(rewards) // 10)
            if window > 0:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            else:
                moving_avg = rewards
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(episodes_nums, rewards, alpha=0.3, label='Episode Reward')
            if len(moving_avg) > 0:
                plt.plot(episodes_nums[window-1:], moving_avg, label=f'Moving Avg ({window})')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            if len(self.episodes) > 0 and self.episodes[0].get('actor_loss', 0) != 0:
                actor_losses = [ep['actor_loss'] for ep in self.episodes if ep['actor_loss'] != 0]
                critic_losses = [ep['critic_loss'] for ep in self.episodes if ep['critic_loss'] != 0]
                
                if len(actor_losses) > 0:
                    plt.plot(actor_losses, label='Actor Loss', alpha=0.7)
                if len(critic_losses) > 0:
                    plt.plot(critic_losses, label='Critic Loss', alpha=0.7)
                    
                plt.xlabel('Update Step')
                plt.ylabel('Loss')
                plt.title('Training Losses')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.log_dir, 'training_curves.png')
            plt.savefig(plot_file, dpi=150)
            print(f"📈 Plot saved to {plot_file}")
            
        except ImportError:
            print("⚠️ matplotlib not installed. Skipping plot generation.")
