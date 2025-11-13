"""
run_maddpg_experiment_FINAL.py
اجرای کامل MADDPG با استخراج صحیح Delay و Energy + قابلیت ذخیره‌سازی
"""

import numpy as np
import sys
from pathlib import Path
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent

print("=" * 70)
print("🚀 MADDPG Multi-Agent Experiment (Random Policy)")
print("=" * 70)

# ============================================================================
# Multi-Agent Wrapper
# ============================================================================
class MultiAgentMADDPG:
    """Wrapper برای مدیریت چند agent مستقل"""
    
    def __init__(self, state_dim, action_dim, n_agents):
        self.n_agents = n_agents
        self.action_dim = action_dim
        
        # ساخت یک agent برای هر UAV
        self.agents = [
            MADDPG_Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                lr=1e-4,
                gamma=0.95
            )
            for _ in range(n_agents)
        ]
        
        print(f"✅ Created {n_agents} independent agents")
    
    def act(self, state, noise_scale=0.0):
        """هر agent action مستقل انتخاب می‌کنه"""
        actions = []
        for agent in self.agents:
            action = agent.act(state, noise_scale)
            actions.append(action)
        
        return np.array(actions)  # (n_agents, action_dim)

# ============================================================================
# محیط
# ============================================================================
env_config = {
    'n_agents': 3,
    'n_users': 10,
    'dt': 1.0,
    'area_size': 1000.0,
    'c1': 9.26e-4,
    'c2': 2250.0,
    'bandwidth': 1e6,
    'noise_power': 1e-10,
    'alpha_delay': 1.0,
    'beta_energy': 1e-6,
    'gamma_eff': 1e3
}

env = MultiUAVEnv(**env_config)
print(f"\n📋 Environment created successfully!")
print(f"   n_agents: {env_config['n_agents']}")
print(f"   n_users : {env_config['n_users']}")
print(f"   area_size: {env_config['area_size']}m")

# ============================================================================
# بررسی State
# ============================================================================
print("\n" + "=" * 70)
print("🔍 State Structure Analysis")
print("=" * 70)

state = env.reset()

def state_to_vector(state):
    """تبدیل state dictionary به vector"""
    if isinstance(state, dict):
        state_vector = []
        for key in sorted(state.keys()):
            val = state[key]
            if isinstance(val, np.ndarray):
                state_vector.append(val.flatten())
            else:
                state_vector.append(np.array([val]).flatten())
        return np.concatenate(state_vector)
    elif isinstance(state, np.ndarray):
        return state.flatten()
    else:
        return state

state_flat = state_to_vector(state)
state_dim = len(state_flat)

print(f"\n✅ State dimension: {state_dim}")

# ============================================================================
# Agent
# ============================================================================
multi_agent = MultiAgentMADDPG(
    state_dim=state_dim,
    action_dim=4,
    n_agents=3
)

print(f"\n📋 Agent Configuration:")
print(f"   State dim: {state_dim}")
print(f"   Action dim: 4")
print(f"   N agents: 3")
print(f"\n⚠️  Using RANDOM policy (no pre-trained models)")

# ============================================================================
# Helper برای تبدیل به scalar
# ============================================================================
def to_scalar(value):
    """تبدیل value به scalar"""
    if isinstance(value, np.ndarray):
        return float(np.sum(value))
    elif isinstance(value, (list, tuple)):
        return float(np.sum(value))
    else:
        return float(value)

# ============================================================================
# اجرای Episodes
# ============================================================================
print("\n" + "=" * 70)
print("🎮 Running Episodes")
print("=" * 70)

n_episodes = 10
results = {
    'rewards': [],
    'delays': [],
    'energies': []
}

for ep in range(n_episodes):
    state = env.reset()
    state_vec = state_to_vector(state)
    
    episode_reward = 0.0
    episode_delay = 0.0
    episode_energy = 0.0
    done = False
    step = 0
    
    print(f"\n📍 Episode {ep + 1}/{n_episodes}")
    
    while not done and step < 100:
        # گرفتن action از multi-agent
        actions = multi_agent.act(state_vec, noise_scale=0.0)
        
        # اجرا در محیط
        step_output = env.step(actions)
        
        if len(step_output) == 4:
            next_state, reward, done, info = step_output
        else:
            print(f"⚠️  Unexpected step output length: {len(step_output)}")
            break
        
        # محاسبه reward
        reward_scalar = to_scalar(reward)
        episode_reward += reward_scalar
        
        # استخراج delay و energy از info
        if isinstance(info, dict):
            # استخراج delay_total
            if 'delay_total' in info:
                episode_delay += to_scalar(info['delay_total'])
            elif 'mean_delay' in info:
                episode_delay += to_scalar(info['mean_delay'])
            
            # استخراج energy_total
            if 'energy_total' in info:
                episode_energy += to_scalar(info['energy_total'])
        
        state = next_state
        state_vec = state_to_vector(state)
        step += 1
        
        if done:
            break
    
    results['rewards'].append(episode_reward)
    results['delays'].append(episode_delay)
    results['energies'].append(episode_energy)
    
    print(f"   Steps: {step}")
    print(f"   Total Reward: {episode_reward:.2e}")
    print(f"   Total Delay: {episode_delay:.4f}s")
    print(f"   Total Energy: {episode_energy:.2e}J")

# ============================================================================
# خلاصه نتایج
# ============================================================================
print("\n" + "=" * 70)
print("📊 Results Summary")
print("=" * 70)

summary_stats = {}
for metric_name, values in results.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    summary_stats[metric_name] = {
        'mean': float(mean_val),
        'std': float(std_val),
        'min': float(min_val),
        'max': float(max_val),
        'all_values': [float(v) for v in values]
    }
    
    print(f"\n{metric_name.upper()}:")
    print(f"   Mean: {mean_val:.4e}")
    print(f"   Std:  {std_val:.4e}")
    print(f"   Min:  {min_val:.4e}")
    print(f"   Max:  {max_val:.4e}")

# ============================================================================
# ذخیره‌سازی نتایج
# ============================================================================
print("\n" + "=" * 70)
print("💾 Saving Results")
print("=" * 70)

# ساخت مسیر results
results_dir = Path("results/maddpg_random")
results_dir.mkdir(parents=True, exist_ok=True)

# تاریخ و زمان
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. ذخیره JSON
json_path = results_dir / f"results_{timestamp}.json"
output_data = {
    'experiment_info': {
        'policy': 'Random',
        'n_episodes': n_episodes,
        'state_dim': state_dim,
        'action_dim': 4,
        'n_agents': 3,
        'timestamp': timestamp
    },
    'env_config': env_config,
    'summary_statistics': summary_stats
}

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON saved: {json_path}")

# 2. رسم نمودارها
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('MADDPG Random Policy Results', fontsize=14, fontweight='bold')

metrics_titles = ['Reward', 'Delay (s)', 'Energy (J)']
for idx, (metric_name, values) in enumerate(results.items()):
    ax = axes[idx]
    episodes = np.arange(1, len(values) + 1)
    
    ax.plot(episodes, values, marker='o', linestyle='-', linewidth=2, markersize=6)
    ax.axhline(y=np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.2e}')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel(metrics_titles[idx], fontsize=11)
    ax.set_title(f'{metrics_titles[idx]} per Episode', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# ذخیره نمودار
png_path = results_dir / f"results_{timestamp}.png"
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f"✅ Plot saved: {png_path}")

plt.show()

# 3. ذخیره گزارش متنی
txt_path = results_dir / f"report_{timestamp}.txt"
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("MADDPG Random Policy Experiment Report\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Policy: Random\n")
    f.write(f"Episodes: {n_episodes}\n")
    f.write(f"State Dimension: {state_dim}\n")
    f.write(f"Action Dimension: 4\n")
    f.write(f"Number of Agents: 3\n\n")
    
    f.write("="*70 + "\n")
    f.write("Results Summary\n")
    f.write("="*70 + "\n\n")
    
    for metric_name, stats in summary_stats.items():
        f.write(f"{metric_name.upper()}:\n")
        f.write(f"  Mean: {stats['mean']:.4e}\n")
        f.write(f"  Std:  {stats['std']:.4e}\n")
        f.write(f"  Min:  {stats['min']:.4e}\n")
        f.write(f"  Max:  {stats['max']:.4e}\n\n")

print(f"✅ Text report saved: {txt_path}")

print("\n" + "=" * 70)
print("✅ Experiment completed!")
print(f"📁 All results saved in: {results_dir}")
print("=" * 70)
print("\n💡 Note: Results are based on RANDOM policy.")
print("   Train the model with correct state_dim=38 for better results.")
