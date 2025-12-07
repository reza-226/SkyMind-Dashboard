# visualize_training.py

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(summary_path='./training_results/logs/training_summary.json'):
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    metrics = data['metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    
    # Moving average
    window = 50
    if len(metrics['episode_rewards']) >= window:
        moving_avg = np.convolve(metrics['episode_rewards'], 
                                  np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Reward (window={window})')
        axes[0, 1].set_xlabel('Episode')
    
    # Losses
    axes[1, 0].plot(metrics['actor_losses'], label='Actor')
    axes[1, 0].plot(metrics['critic_losses'], label='Critic')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].legend()
    
    # Q-values
    axes[1, 1].plot(metrics['avg_q_values'])
    axes[1, 1].set_title('Average Q-values')
    axes[1, 1].set_xlabel('Episode')
    
    plt.tight_layout()
    plt.savefig('./training_results/training_plots.png', dpi=300)
    print("✅ Plots saved to: ./training_results/training_plots.png")
    plt.show()

if __name__ == "__main__":
    plot_training_results()
