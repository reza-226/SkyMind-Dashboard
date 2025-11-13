"""
plot_experiment_results.py
==========================
رسم نمودارهای مقایسه‌ای برای نتایج آزمایش
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# تنظیمات فارسی
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

def load_results(filepath='results/obstacle_experiments_fixed.json'):
    """بارگذاری نتایج"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_comparison_bars(results, output_dir='results/plots'):
    """نمودارهای میله‌ای مقایسه"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    policies = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # 1️⃣ مقایسه Reward
    fig, ax = plt.subplots(figsize=(10, 6))
    rewards = [results[p]['mean_reward'] for p in policies]
    errors = [results[p]['std_reward'] for p in policies]
    
    bars = ax.bar(policies, rewards, yerr=errors, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Average Rewards Across Policies', 
                 fontsize=14, fontweight='bold')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax.grid(axis='y', alpha=0.3)
    
    # برچسب‌های روی میله‌ها
    for bar, val in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/reward_comparison.png', dpi=300)
    plt.close()
    print(f"✅ رسم شد: {output_dir}/reward_comparison.png")
    
    # 2️⃣ مقایسه Delay
    fig, ax = plt.subplots(figsize=(10, 6))
    delays = [results[p]['mean_delay'] for p in policies]
    errors = [results[p]['std_delay'] for p in policies]
    
    bars = ax.bar(policies, delays, yerr=errors, capsize=5,
                   color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average Delay (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Average Delay Across Policies',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, delays):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/delay_comparison.png', dpi=300)
    plt.close()
    print(f"✅ رسم شد: {output_dir}/delay_comparison.png")
    
    # 3️⃣ مقایسه Energy
    fig, ax = plt.subplots(figsize=(10, 6))
    energies = [results[p]['mean_energy'] for p in policies]
    errors = [results[p]['std_energy'] for p in policies]
    
    bars = ax.bar(policies, energies, yerr=errors, capsize=5,
                   color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average Energy Consumption (Joules)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Average Energy Consumption Across Policies',
                 fontsize=14, fontweight='bold')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_comparison.png', dpi=300)
    plt.close()
    print(f"✅ رسم شد: {output_dir}/energy_comparison.png")

def plot_convergence(results, output_dir='results/plots'):
    """نمودار همگرایی برای هر سیاست"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    policies = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, (policy, color) in enumerate(zip(policies, colors)):
        ax = axes[idx // 2, idx % 2]
        
        rewards = results[policy]['all_rewards']
        episodes = range(1, len(rewards) + 1)
        
        ax.plot(episodes, rewards, color=color, linewidth=2, alpha=0.7)
        
        # خط میانگین متحرک
        window = 5
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(rewards)+1), moving_avg, 
                   color='black', linewidth=2, label='Moving Avg (5)')
        
        ax.set_title(f"Reward Convergence - {policy}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Reward", fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(1, len(rewards))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_convergence.png", dpi=300)
    plt.close()
    print(f"✅ رسم شد: {output_dir}/reward_convergence.png")

def plot_pareto_front(results, output_dir='results/plots'):
    """رسم Pareto Front: Energy vs Delay"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    policies = list(results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    markers = ['o', 's', '^', 'D']
    
    for policy, color, marker in zip(policies, colors, markers):
        energy = results[policy]['mean_energy']
        delay = results[policy]['mean_delay']
        
        ax.scatter(energy, delay, s=200, color=color, marker=marker,
                  edgecolor='black', linewidth=2, alpha=0.8, label=policy)
        
        # برچسب نقطه
        ax.annotate(policy, (energy, delay), 
                   textcoords="offset points", xytext=(10,10),
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Energy Consumption (Joules)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delay (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Front: Energy vs Delay Trade-off', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_front.png', dpi=300)
    plt.close()
    print(f"✅ رسم شد: {output_dir}/pareto_front.png")

if __name__ == "__main__":
    results_path = "results/obstacle_experiments_fixed.json"
    
    print("🔄 در حال بارگذاری نتایج...")
    results = load_results(results_path)
    
    print("\n📊 رسم نمودارهای مقایسه‌ای...")
    plot_comparison_bars(results)
    
    print("\n📈 رسم نمودارهای همگرایی...")
    plot_convergence(results)
    
    print("\n🎯 رسم Pareto Front...")
    plot_pareto_front(results)
    
    print("\n✅ تمام نمودارها با موفقیت رسم شدند!")
    print("📂 محل ذخیره: results/plots/")
