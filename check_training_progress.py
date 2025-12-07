# check_training_progress.py
import json
import matplotlib.pyplot as plt
import numpy as np

# بارگذاری تاریخچه آموزش
with open('results/4layer_3level/level_1/training_history.json', 'r') as f:
    history = json.load(f)

episodes = history['episodes']
rewards = history['rewards']

# رسم نمودار
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards, alpha=0.3, label='Episode Reward')

# میانگین متحرک
window = 50
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress - Level 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_analysis.png', dpi=150)
plt.show()

print(f"Best Reward: {max(rewards):.2f}")
print(f"Final 100-ep Avg: {np.mean(rewards[-100:]):.2f}")
