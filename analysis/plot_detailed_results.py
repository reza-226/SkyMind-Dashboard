# فایل: analysis/plot_detailed_results.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# Load metrics
data = np.load('results/training_metrics.npz')
rewards = data['episode_rewards']
actor_losses = data['actor_losses']
critic_losses = data['critic_losses']

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Reward with moving average
ax1 = axes[0, 0]
window = 50
rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
episodes_smooth = np.arange(window-1, len(rewards))

ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
ax1.plot(episodes_smooth, rewards_smooth, color='red', linewidth=2, 
         label=f'Moving Avg (window={window})')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Total Reward', fontsize=12)
ax1.set_title('📊 Reward Progression', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Critic Loss (log scale)
ax2 = axes[0, 1]
ax2.plot(critic_losses, color='green', linewidth=1.5)
ax2.set_yscale('log')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Critic Loss (log scale)', fontsize=12)
ax2.set_title('📉 Critic Loss Convergence', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Actor Loss
ax3 = axes[1, 0]
ax3.plot(actor_losses, color='purple', linewidth=1.5)
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Actor Loss', fontsize=12)
ax3.set_title('🎭 Actor Loss Evolution', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Reward Distribution (last 200 episodes)
ax4 = axes[1, 1]
last_rewards = rewards[-200:]
ax4.hist(last_rewards, bins=30, color='orange', alpha=0.7, edgecolor='black')
ax4.axvline(x=np.mean(last_rewards), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(last_rewards):.2f}')
ax4.set_xlabel('Reward Value', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('📊 Reward Distribution (Last 200 Episodes)', 
              fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✅ نمودار ذخیره شد: results/detailed_analysis.png")
plt.show()
