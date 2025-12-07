# compare_results.py
"""
Compare old vs new training results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(path):
    with open(path) as f:
        return json.load(f)

def plot_comparison():
    # Load both results
    old = load_results('results/4layer_3level/level_1/training_results.json')
    new = load_results('results/improved_training/training_results.json')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Raw rewards
    ax = axes[0, 0]
    ax.plot(old['history']['episode_rewards'], alpha=0.3, label='Old', color='red')
    ax.plot(new['history'], alpha=0.3, label='New', color='green')
    ax.set_title('Raw Rewards Comparison')
    ax.legend()
    
    # Plot 2: Moving average
    ax = axes[0, 1]
    old_ma = moving_average(old['history']['episode_rewards'], 50)
    new_rewards = [h['reward'] for h in new['history']]
    new_ma = moving_average(new_rewards, 50)
    ax.plot(old_ma, label='Old (MA50)', color='red', linewidth=2)
    ax.plot(new_ma, label='New (MA50)', color='green', linewidth=2)
    ax.set_title('Moving Average (50)')
    ax.legend()
    
    # Plot 3: Distribution
    ax = axes[1, 0]
    ax.hist(old['history']['episode_rewards'], bins=50, alpha=0.5, label='Old', color='red')
    ax.hist(new_rewards, bins=50, alpha=0.5, label='New', color='green')
    ax.set_title('Reward Distribution')
    ax.legend()
    
    # Plot 4: Statistics comparison
    ax = axes[1, 1]
    stats = {
        'Old': {
            'Mean': np.mean(old['history']['episode_rewards']),
            'Best': np.max(old['history']['episode_rewards']),
            'Final 100': np.mean(old['history']['episode_rewards'][-100:])
        },
        'New': {
            'Mean': np.mean(new_rewards),
            'Best': np.max(new_rewards),
            'Final 100': np.mean(new_rewards[-100:])
        }
    }
    
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, list(stats['Old'].values()), width, label='Old', color='red', alpha=0.7)
    ax.bar(x + width/2, list(stats['New'].values()), width, label='New', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(stats['Old'].keys())
    ax.set_title('Statistics Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Comparison saved to: results/comparison.png")

def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == '__main__':
    plot_comparison()
