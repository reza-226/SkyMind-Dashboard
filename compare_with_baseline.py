# فایل: compare_with_baseline.py

import numpy as np
import matplotlib.pyplot as plt

def random_policy_baseline(env, num_episodes=100):
    """
    پایه‌سنجی با policy تصادفی
    """
    baseline_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(200):
            # اکشن تصادفی
            actions = np.random.uniform(-1, 1, size=(3, 4))
            next_state, reward, done, _ = env.step(actions)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        baseline_rewards.append(episode_reward)
    
    return np.mean(baseline_rewards), np.std(baseline_rewards)

# Load trained model results
data = np.load('results/training_metrics.npz')
trained_rewards = data['episode_rewards'][-100:]

# Calculate baseline
from core.env_multi import MADDPGMultiEnv
env = MADDPGMultiEnv(n_agents=3, n_users=5)
baseline_mean, baseline_std = random_policy_baseline(env)

# مقایسه
trained_mean = np.mean(trained_rewards)
trained_std = np.std(trained_rewards)

print("\n" + "="*60)
print("📊 مقایسه با Baseline:")
print("="*60)
print(f"Random Policy:   {baseline_mean:.2f} ± {baseline_std:.2f}")
print(f"Trained MADDPG:  {trained_mean:.2f} ± {trained_std:.2f}")
print(f"Improvement:     {((trained_mean - baseline_mean) / abs(baseline_mean) * 100):.1f}%")
print("="*60)

# رسم نمودار
fig, ax = plt.subplots(figsize=(10, 6))
policies = ['Random\nPolicy', 'Trained\nMADDPG']
means = [baseline_mean, trained_mean]
stds = [baseline_std, trained_std]

bars = ax.bar(policies, means, yerr=stds, capsize=10, 
              color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
ax.set_title('🏆 Performance Comparison', fontsize=16, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, mean) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width()/2, mean + stds[i] + 0.5,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/baseline_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ نمودار مقایسه ذخیره شد!")
plt.show()
