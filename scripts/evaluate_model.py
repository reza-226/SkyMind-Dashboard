import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from core.env.environment import UAVMECEnvironment
from agents.maddpg_agent import MADDPGAgent
from utils.dag_generator import generate_random_dag

# تنظیمات
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPISODES = 100
MAX_STEPS = 100
SAVE_DIR = Path("evaluation_results")
SAVE_DIR.mkdir(exist_ok=True)

print("="*80)
print("🎯 COMPREHENSIVE MODEL EVALUATION")
print("="*80)
print(f"\n📊 Configuration:")
print(f"   Device: {DEVICE}")
print(f"   Episodes: {NUM_EPISODES}")
print(f"   Max steps per episode: {MAX_STEPS}")
print(f"   Save directory: {SAVE_DIR}")

# ساخت محیط
print("\n[1] Initializing environment...")
env = UAVMECEnvironment(device=DEVICE, max_steps=MAX_STEPS)

# بارگذاری Agent
print("\n[2] Loading trained MADDPG agent...")
agent = MADDPGAgent(
    state_dim=537,
    device=DEVICE
)

# تلاش برای بارگذاری مدل آموزش دیده
model_path = Path("models/maddpg_final.pth")
if model_path.exists():
    print(f"   ✅ Loading model from: {model_path}")
    agent.load(str(model_path))
else:
    print(f"   ⚠️  No trained model found at {model_path}")
    print(f"   ℹ️  Using randomly initialized weights")

# ساختار برای ذخیره نتایج
results = {
    'episodes': [],
    'metrics': {
        'delay': [],
        'energy_consumption': [],
        'distance': [],
        'qos_satisfaction': []
    },
    'rewards': [],
    'episode_lengths': []
}

# اجرای Evaluation
print("\n[3] Running evaluation...")
print("-"*80)

for episode in tqdm(range(NUM_EPISODES), desc="Evaluating"):
    # تولید DAG جدید
    dag = generate_random_dag(
        num_nodes=np.random.randint(5, 10),
        edge_probability=0.3,
        device=DEVICE
    )
    
    state = env.reset(dag)
    episode_reward = 0.0
    episode_metrics = {
        'delay': [],
        'energy_consumption': [],
        'distance': [],
        'qos_satisfaction': []
    }
    
    for step in range(MAX_STEPS):
        # انتخاب action بدون noise
        action = agent.select_action(state, explore=False)
        
        # اجرای action
        next_state, reward, done, info = env.step(action)
        
        # ذخیره metrics
        episode_metrics['delay'].append(info['delay'])
        episode_metrics['energy_consumption'].append(info['energy_consumption'])
        episode_metrics['distance'].append(info['distance'])
        episode_metrics['qos_satisfaction'].append(info['qos_satisfaction'])
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    # ذخیره نتایج episode
    results['episodes'].append(episode)
    results['rewards'].append(episode_reward)
    results['episode_lengths'].append(step + 1)
    
    # میانگین metrics در این episode
    for metric_name in episode_metrics:
        avg_value = np.mean(episode_metrics[metric_name])
        results['metrics'][metric_name].append(avg_value)

print("-"*80)
print("\n[4] Computing statistics...")

# محاسبه آمار
stats = {
    'avg_reward': float(np.mean(results['rewards'])),
    'std_reward': float(np.std(results['rewards'])),
    'min_reward': float(np.min(results['rewards'])),
    'max_reward': float(np.max(results['rewards'])),
    'avg_episode_length': float(np.mean(results['episode_lengths'])),
    'metrics': {}
}

for metric_name in results['metrics']:
    values = results['metrics'][metric_name]
    stats['metrics'][metric_name] = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }

# نمایش نتایج
print("\n" + "="*80)
print("📊 EVALUATION RESULTS")
print("="*80)

print(f"\n🎯 Reward Statistics:")
print(f"   Average Reward: {stats['avg_reward']:.4f} ± {stats['std_reward']:.4f}")
print(f"   Min/Max: [{stats['min_reward']:.4f}, {stats['max_reward']:.4f}]")
print(f"   Avg Episode Length: {stats['avg_episode_length']:.2f} steps")

print(f"\n📈 Metrics Statistics:")
for metric_name, metric_stats in stats['metrics'].items():
    print(f"\n   {metric_name.upper()}:")
    print(f"      Mean: {metric_stats['mean']:.6f} ± {metric_stats['std']:.6f}")
    print(f"      Range: [{metric_stats['min']:.6f}, {metric_stats['max']:.6f}]")

# ذخیره نتایج JSON
print("\n[5] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = SAVE_DIR / f"evaluation_{timestamp}.json"

with open(json_path, 'w') as f:
    json.dump({
        'config': {
            'num_episodes': NUM_EPISODES,
            'max_steps': MAX_STEPS,
            'device': DEVICE,
            'model_path': str(model_path) if model_path.exists() else 'random_init'
        },
        'statistics': stats,
        'raw_results': {
            'episodes': results['episodes'],
            'rewards': results['rewards'],
            'episode_lengths': results['episode_lengths'],
            'metrics': results['metrics']
        }
    }, f, indent=2)

print(f"   ✅ Results saved to: {json_path}")

# ساخت نمودارها
print("\n[6] Generating visualizations...")

# تنظیمات نمودار
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle(f'MADDPG Evaluation Results (N={NUM_EPISODES} episodes)', 
             fontsize=16, fontweight='bold')

# 1. Reward over episodes
ax1 = axes[0, 0]
ax1.plot(results['episodes'], results['rewards'], alpha=0.6, color='blue')
ax1.axhline(stats['avg_reward'], color='red', linestyle='--', 
            label=f"Mean: {stats['avg_reward']:.2f}")
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.set_title('Episode Rewards')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Reward distribution
ax2 = axes[0, 1]
ax2.hist(results['rewards'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(stats['avg_reward'], color='red', linestyle='--', linewidth=2,
            label=f"Mean: {stats['avg_reward']:.2f}")
ax2.set_xlabel('Reward')
ax2.set_ylabel('Frequency')
ax2.set_title('Reward Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3-6. Metrics over time
metric_colors = {
    'delay': 'orange',
    'energy_consumption': 'green',
    'distance': 'purple',
    'qos_satisfaction': 'red'
}

metric_positions = {
    'delay': (1, 0),
    'energy_consumption': (1, 1),
    'distance': (2, 0),
    'qos_satisfaction': (2, 1)
}

for metric_name, (row, col) in metric_positions.items():
    ax = axes[row, col]
    values = results['metrics'][metric_name]
    color = metric_colors[metric_name]
    
    ax.plot(results['episodes'], values, alpha=0.6, color=color)
    ax.axhline(stats['metrics'][metric_name]['mean'], 
               color='red', linestyle='--',
               label=f"Mean: {stats['metrics'][metric_name]['mean']:.4f}")
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("_", " ").title()} Over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = SAVE_DIR / f"evaluation_plots_{timestamp}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Plots saved to: {plot_path}")

# نمودار مقایسه metrics
fig2, ax = plt.subplots(figsize=(10, 6))
metric_names = list(stats['metrics'].keys())
metric_means = [stats['metrics'][m]['mean'] for m in metric_names]
metric_stds = [stats['metrics'][m]['std'] for m in metric_names]

x_pos = np.arange(len(metric_names))
bars = ax.bar(x_pos, metric_means, yerr=metric_stds, 
              color=['orange', 'green', 'purple', 'red'],
              alpha=0.7, capsize=5)

ax.set_xlabel('Metrics')
ax.set_ylabel('Average Value')
ax.set_title('Average Metrics with Standard Deviation')
ax.set_xticks(x_pos)
ax.set_xticklabels([m.replace('_', '\n') for m in metric_names])
ax.grid(True, alpha=0.3, axis='y')

# اضافه کردن مقادیر روی میله‌ها
for i, (bar, mean, std) in enumerate(zip(bars, metric_means, metric_stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.4f}\n±{std:.4f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
comparison_path = SAVE_DIR / f"metrics_comparison_{timestamp}.png"
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Comparison plot saved to: {comparison_path}")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
print(f"\n📁 All results saved in: {SAVE_DIR}/")
print(f"   - JSON results: evaluation_{timestamp}.json")
print(f"   - Main plots: evaluation_plots_{timestamp}.png")
print(f"   - Comparison: metrics_comparison_{timestamp}.png")
