import matplotlib.pyplot as plt
import numpy as np
import json
import os

def create_comparison_plots():
    """Create comprehensive comparison plots for baseline algorithms"""
    
    # ✅ داده‌های واقعی از اجراها
    algorithms = ['Random', 'I-DDPG', 'MADDPG\n(Decentralized)']
    best_rewards = [-150.00, 3284.18, 79.05]
    final_avgs = [-150.00, 2529.07, 31.41]
    training_times = [0, 45.3, 90.1]  # تخمینی از log
    
    # رنگ‌بندی
    colors = ['#e74c3c', '#3498db', '#27ae60']
    
    # ایجاد figure با 3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # نمودار 1: Best Reward
    bars1 = axes[0].bar(algorithms, best_rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_title('Best Reward Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Best Reward', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # اضافه کردن مقادیر روی ستون‌ها
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}' if abs(height) < 1000 else f'{height:.0f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    # نمودار 2: Final Average
    bars2 = axes[1].bar(algorithms, final_avgs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_title('Final Average Reward (100 episodes)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Average Reward', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}' if abs(height) < 1000 else f'{height:.0f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    # نمودار 3: Training Time (فقط I-DDPG و MADDPG)
    bars3 = axes[2].bar(algorithms[1:], training_times[1:], 
                       color=colors[1:], alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[2].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Time (minutes)', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f} min',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # ذخیره نمودار
    os.makedirs('results', exist_ok=True)
    output_path = 'results/baseline_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    # ذخیره نتایج در JSON
    results = {
        'algorithms': ['Random', 'I-DDPG', 'MADDPG (Decentralized)'],
        'best_rewards': best_rewards,
        'final_averages': final_avgs,
        'training_times': training_times,
        'analysis': {
            'iddpg_vs_random_best': f"{((3284.18 - (-150)) / abs(-150) * 100):.1f}%",
            'iddpg_vs_random_avg': f"{((2529.07 - (-150)) / abs(-150) * 100):.1f}%",
            'note': 'I-DDPG shows significantly better performance than Random baseline',
            'maddpg_note': 'MADDPG (Decentralized) used different reward scaling'
        }
    }
    
    json_path = 'results/baseline_comparison.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {json_path}")
    
    # نمایش نمودار
    plt.show()
    
    # چاپ خلاصه
    print("\n" + "="*70)
    print("📊 BASELINE COMPARISON SUMMARY")
    print("="*70)
    for i, alg in enumerate(['Random', 'I-DDPG', 'MADDPG (Decentralized)']):
        print(f"\n{alg}:")
        print(f"  Best Reward:    {best_rewards[i]:>10.2f}")
        print(f"  Final Avg(100): {final_avgs[i]:>10.2f}")
        if training_times[i] > 0:
            print(f"  Training Time:  {training_times[i]:>10.1f} min")
    
    print("\n" + "="*70)
    print("🏆 KEY FINDINGS")
    print("="*70)
    print(f"✅ I-DDPG vs Random (Best):  +{results['analysis']['iddpg_vs_random_best']} improvement")
    print(f"✅ I-DDPG vs Random (Avg):   +{results['analysis']['iddpg_vs_random_avg']} improvement")
    print("\n⚠️  Note: MADDPG uses different reward scaling/environment")
    print("   Direct numerical comparison may not be meaningful.")
    print("="*70)

if __name__ == "__main__":
    create_comparison_plots()
