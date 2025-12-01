"""
Random Policy Baseline
Selects random actions at each step.
"""

import numpy as np
import torch

class RandomAgent:
    """
    Random policy agent that selects actions uniformly at random.
    """
    
    def __init__(self, offload_dim=5, continuous_dim=6):
        self.offload_dim = offload_dim
        self.continuous_dim = continuous_dim
        
    def select_action(self, state, explore=True):
        """
        Select random action.
        
        Returns:
            dict with keys: 'offload', 'cpu', 'bandwidth', 'move'
        """
        # Random offload decision (0-4)
        offload = np.random.randint(0, self.offload_dim)
        
        # Random CPU allocation [0, 1]
        cpu = np.random.uniform(0, 1)
        
        # Random bandwidth (normalized to sum=1)
        bw_raw = np.random.uniform(0, 1, size=3)
        bandwidth = bw_raw / bw_raw.sum()
        
        # Random movement [-5, 5]
        move = np.random.uniform(-5, 5, size=2)
        
        return {
            'offload': int(offload),
            'cpu': float(cpu),
            'bandwidth': bandwidth,
            'move': move
        }
    
    def update(self, *args, **kwargs):
        """No learning for random policy"""
        pass
    
    def save(self, path):
        """Nothing to save"""
        pass
    
    def load(self, path):
        """Nothing to load"""
        pass


def run_random_baseline(env, num_episodes=500, save_dir='results/baselines/random'):
    """
    Run random policy baseline and save results.
    """
    import os
    import json
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    
    agent = RandomAgent()
    episode_rewards = []
    
    print(f"\n{'='*60}")
    print("🎲 Running Random Policy Baseline")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward (last 50): {avg_reward:.2f}")
    
    # Save results
    results = {
        'algorithm': 'Random Policy',
        'episode_rewards': episode_rewards,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {save_dir}/results.json")
    print(f"📊 Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return results
