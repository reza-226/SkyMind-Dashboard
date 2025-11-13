"""
compare_all_policies.py
مقایسه MADDPG با Random, Greedy و سایر روش‌ها + ذخیره‌سازی نتایج
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from run_maddpg_experiment_FINAL import MultiAgentMADDPG, env_config, state_to_vector
from core.env_multi import MultiUAVEnv
import torch

def evaluate_policy(policy_name, env, n_episodes=50):
    """ارزیابی یک policy"""
    results = {'delays': [], 'energies': [], 'rewards': []}
    
    print(f"\n🔍 Evaluating: {policy_name.upper()}")
    
    for ep in range(n_episodes):
        state = env.reset()
        state_vec = state_to_vector(state)
        
        episode_reward = 0.0
        episode_delay = 0.0
        episode_energy = 0.0
        done = False
        step = 0
        
        while not done and step < 100:
            # انتخاب action بر اساس policy
            if policy_name == 'random':
                actions = np.random.randn(3, 4)
            elif policy_name == 'maddpg':
                actions = multi_agent.act(state_vec, noise_scale=0.0)
            elif policy_name == 'greedy_local':
                actions = np.zeros((3, 4))  # پردازش محلی
            elif policy_name == 'greedy_offload':
                actions = np.ones((3, 4))   # تخلیه کامل
            elif policy_name == 'balanced':
                actions = np.ones((3, 4)) * 0.5  # استراتژی متعادل
            
            step_output = env.step(actions)
            
            if len(step_output) == 4:
                next_state, reward, done, info = step_output
            else:
                break
            
            reward_scalar = np.sum(reward) if isinstance(reward, np.ndarray) else reward
            episode_reward += reward_scalar
            
            if isinstance(info, dict):
                if 'delay_total' in info:
                    episode_delay += float(np.sum(info['delay_total']))
                if 'energy_total' in info:
                    episode_energy += float(np.sum(info['energy_total']))
            
            state = next_state
            state_vec = state_to_vector(state)
            step += 1
        
        results['rewards'].append(episode_reward)
        results['delays'].append(episode_delay)
        results['energies'].append(episode_energy)
        
        # نمایش پیشرفت
        if (ep + 1) % 10 == 0:
            print(f"   Progress: {ep + 1}/{n_episodes} episodes completed")
    
    # محاسبه آمار
    stats = {}
    for k, v in results.items():
        stats[k] = {
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
            'min': float(np.min(v)),
            'max': float(np.max(v)),
            'all_values': [float(x) for x in v]
        }
    
    return stats

# ============================================================================
# اجرا
# ============================================================================
print("=" * 70)
print("📊 Multi-Policy Comparison Experiment")
print("=" * 70)

env = MultiUAVEnv(**env_config)

# لود MADDPG (Random policy برای این مرحله)
multi_agent = MultiAgentMADDPG(state_dim=38, action_dim=4, n_agents=3)
print("\n⚠️  MADDPG using random policy (no trained model)")

# لیست policy ها
policies = ['random', 'greedy_local', 'greedy_offload', 'balanced']

print(f"\n🎯 Policies to compare: {', '.join(policies)}")
print(f"📋 Episodes per policy: 50")

# اجرای ارزیابی
all_results = {}
for policy in policies:
    all_results[policy] = evaluate_policy(policy, env, n_episodes=50)
    
    print(f"\n   ✅ {policy.upper()} completed:")
    print(f"      Delay:  {all_results[policy]['delays']['mean']:.4f} ± {all_results[policy]['delays']['std']:.4f}s")
    print(f"      Energy: {all_results[policy]['energies']['mean']:.2e} ± {all_results[policy]['energies']['std']:.2e}J")
    print(f"      Reward: {all_results[policy]['rewards']['mean']:.2e} ± {all_results[policy]['rewards']['std']:.2e}")

# ============================================================================
# خلاصه نتایج
# ============================================================================
print("\n" + "=" * 70)
print("📊 Comparison Summary")
print("=" * 70)

for metric in ['delays', 'energies', 'rewards']:
    print(f"\n{metric.upper()}:")
    for policy in policies:
        mean_val = all_results[policy][metric]['mean']
        std_val = all_results[policy][metric]['std']
        print(f"   {policy:20s}: {mean_val:12.4e} ± {std_val:12.4e}")

# ============================================================================
# ذخیره‌سازی نتایج
# ============================================================================
print("\n" + "=" * 70)
print("💾 Saving Results")
print("=" * 70)

# ساخت مسیر
results_dir = Path("results/comparison")
results_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. ذخیره JSON
json_path = results_dir / f"comparison_{timestamp}.json"
output_data = {
    'experiment_info': {
        'n_episodes_per_policy': 50,
        'policies': policies,
        'state_dim': 38,
        'action_dim': 4,
        'n_agents': 3,
        'timestamp': timestamp
    },
    'env_config': env_config,
    'results': all_results
}

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON saved: {json_path}")

# 2. رسم نمودارهای مقایسه‌ای
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Policy Comparison Results', fontsize=16, fontweight='bold')

metrics = ['delays', 'energies', 'rewards']
titles = ['Average Delay (s)', 'Average Energy (J)', 'Average Reward']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]
    
    means = [all_results[p][metric]['mean'] for p in policies]
    stds = [all_results[p][metric]['std'] for p in policies]
    
    x_pos = np.arange(len(policies))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Policy', fontsize=12, fontweight='bold')
    ax.set_ylabel(title, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policies, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # اضافه کردن مقادیر روی میله‌ها
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2e}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()

png_path = results_dir / f"comparison_{timestamp}.png"
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"✅ Comparison plot saved: {png_path}")

plt.show()

# 3. ذخیره گزارش متنی
txt_path = results_dir / f"comparison_report_{timestamp}.txt"
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("Multi-Policy Comparison Report\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Policies Compared: {', '.join(policies)}\n")
    f.write(f"Episodes per Policy: 50\n")
    f.write(f"State Dimension: 38\n")
    f.write(f"Action Dimension: 4\n")
    f.write(f"Number of Agents: 3\n\n")
    
    f.write("="*70 + "\n")
    f.write("Detailed Results\n")
    f.write("="*70 + "\n\n")
    
    for policy in policies:
        f.write(f"{policy.upper()}:\n")
        f.write("-" * 40 + "\n")
        
        for metric in ['delays', 'energies', 'rewards']:
            stats = all_results[policy][metric]
            f.write(f"  {metric.capitalize()}:\n")
            f.write(f"    Mean: {stats['mean']:.4e}\n")
            f.write(f"    Std:  {stats['std']:.4e}\n")
            f.write(f"    Min:  {stats['min']:.4e}\n")
            f.write(f"    Max:  {stats['max']:.4e}\n")
        f.write("\n")
    
    f.write("="*70 + "\n")
    f.write("Summary Table\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"{'Policy':<20} {'Delay (s)':<20} {'Energy (J)':<20} {'Reward':<20}\n")
    f.write("-" * 80 + "\n")
    
    for policy in policies:
        delay_mean = all_results[policy]['delays']['mean']
        energy_mean = all_results[policy]['energies']['mean']
        reward_mean = all_results[policy]['rewards']['mean']
        f.write(f"{policy:<20} {delay_mean:<20.4e} {energy_mean:<20.4e} {reward_mean:<20.4e}\n")

print(f"✅ Text report saved: {txt_path}")

print("\n" + "=" * 70)
print("✅ Comparison experiment completed!")
print(f"📁 All results saved in: {results_dir}")
print("=" * 70)
