"""
تحلیل سریع نتایج آموزش - نسخه سفارشی برای SkyMind
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_training_data(json_path):
    """بارگذاری داده با پشتیبانی از ساختار SkyMind"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"🔍 Top-level keys: {list(data.keys())}")
    
    # ساختار SkyMind: {'history': {'episode_rewards': [...]}}
    if 'history' in data and 'episode_rewards' in data['history']:
        rewards = np.array(data['history']['episode_rewards'])
        episodes = np.arange(len(rewards))
        level_name = data.get('level', 'Unknown Level')
        
        # اطلاعات اضافی
        config = data.get('config', {})
        final_stats = data.get('final_stats', {})
        
        print(f"✅ Format: SkyMind structure (history.episode_rewards)")
        print(f"📊 Config: {config.get('num_uavs', '?')} UAVs, {config.get('num_devices', '?')} devices")
        
        return episodes, rewards, level_name, config, final_stats
    
    # ساختار قدیمی: {'results': {'training_history': {...}}}
    elif 'results' in data and 'training_history' in data['results']:
        history = data['results']['training_history']
        episodes = np.array(history['episodes'])
        rewards = np.array(history['rewards'])
        level_name = data.get('level_display_name', data.get('level_name', 'Unknown'))
        print(f"✅ Format: Standard structure (results.training_history)")
        return episodes, rewards, level_name, {}, {}
    
    else:
        raise KeyError(
            f"❌ Unknown format! Available keys: {list(data.keys())}\n"
            f"Expected 'history.episode_rewards' or 'results.training_history'"
        )

def analyze_training_quick(json_path):
    """تحلیل سریع training history"""
    
    try:
        episodes, rewards, level_name, config, final_stats = load_training_data(json_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    n = len(episodes)
    
    # Header
    print(f"\n{'='*70}")
    print(f"📈 Training Analysis - {level_name.upper()}")
    print(f"{'='*70}")
    
    # Configuration
    if config:
        print(f"\n🔧 Environment Config:")
        print(f"  UAVs: {config.get('num_uavs', 'N/A')}, "
              f"Devices: {config.get('num_devices', 'N/A')}, "
              f"Servers: {config.get('num_edge_servers', 'N/A')}")
        print(f"  Grid: {config.get('grid_size', 'N/A')}m, "
              f"Max Steps: {config.get('max_steps', 'N/A')}")
    
    # Overall stats
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Episodes:   {n}")
    print(f"  Best Reward:      {rewards.max():>10.2f}  (Episode {episodes[rewards.argmax()]})")
    print(f"  Worst Reward:     {rewards.min():>10.2f}  (Episode {episodes[rewards.argmin()]})")
    print(f"  Mean Reward:      {rewards.mean():>10.2f}  ± {rewards.std():.2f}")
    print(f"  Median Reward:    {np.median(rewards):>10.2f}")
    
    if final_stats:
        print(f"\n📈 Final Stats (from file):")
        print(f"  Avg Reward:       {final_stats.get('avg_reward', 'N/A')}")
        print(f"  Best Reward:      {final_stats.get('best_reward', 'N/A')}")
        print(f"  Final 100 Avg:    {final_stats.get('final_100_avg', 'N/A')}")
    
    # Phase analysis
    early = rewards[:n//3]
    mid = rewards[n//3:2*n//3]
    late = rewards[2*n//3:]
    
    print(f"\n📊 Learning Phases:")
    print(f"  Early (Ep 0-{len(early)-1:>3}):      {early.mean():>10.2f}  ± {early.std():>8.2f}")
    print(f"  Mid   (Ep {len(early)}-{len(early)+len(mid)-1:>3}):    {mid.mean():>10.2f}  ± {mid.std():>8.2f}")
    print(f"  Late  (Ep {len(early)+len(mid)}-{n-1:>3}):   {late.mean():>10.2f}  ± {late.std():>8.2f}")
    
    improvement_early_to_late = ((late.mean() - early.mean()) / abs(early.mean()) * 100) if early.mean() != 0 else 0
    print(f"\n  Early→Late Change: {improvement_early_to_late:>+6.1f}%")
    
    # Convergence analysis
    window = min(50, n//4)
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    improvement = ((moving_avg[-1] - moving_avg[0]) / abs(moving_avg[0]) * 100) if moving_avg[0] != 0 else 0
    
    print(f"\n🎯 Convergence Analysis (Window={window}):")
    print(f"  Initial Avg:      {moving_avg[0]:>10.2f}")
    print(f"  Final Avg:        {moving_avg[-1]:>10.2f}")
    print(f"  Improvement:      {improvement:>9.1f}%")
    
    if improvement > 20:
        status = "✅ Strong Learning"
    elif improvement > 5:
        status = "⚠️  Moderate Learning"
    elif improvement > -5:
        status = "⚠️  Stagnant"
    else:
        status = "❌ Degrading"
    print(f"  Status:           {status}")
    
    # Stability
    last_50 = rewards[-50:] if n >= 50 else rewards[-n//2:]
    print(f"\n🎲 Stability (Last {len(last_50)} episodes):")
    print(f"  Mean:             {last_50.mean():>10.2f}")
    print(f"  Std Dev:          {last_50.std():>10.2f}")
    print(f"  CV (σ/μ):         {abs(last_50.std()/last_50.mean()):.3f}")
    
    # Quartiles
    q1, q2, q3 = np.percentile(rewards, [25, 50, 75])
    print(f"\n📐 Percentiles:")
    print(f"  25th (Q1):        {q1:>10.2f}")
    print(f"  50th (Median):    {q2:>10.2f}")
    print(f"  75th (Q3):        {q3:>10.2f}")
    print(f"  IQR:              {q3-q1:>10.2f}")
    
    # Plotting
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main training curve (2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=0.8, label='Episode Reward')
    ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2.5, label=f'{window}-Episode MA')
    ax1.axhline(0, color='green', linestyle='--', alpha=0.4, linewidth=1)
    ax1.axhline(rewards.mean(), color='orange', linestyle=':', alpha=0.6, linewidth=1.5, label='Mean')
    ax1.fill_between(episodes, rewards.mean()-rewards.std(), rewards.mean()+rewards.std(), 
                      alpha=0.1, color='orange')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title(f'Training Progress - {level_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(rewards, bins=50, alpha=0.7, edgecolor='black', color='skyblue', orientation='horizontal')
    ax2.axhline(rewards.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean')
    ax2.axhline(np.median(rewards), color='orange', linestyle='--', linewidth=2, label=f'Median')
    ax2.set_ylabel('Reward')
    ax2.set_xlabel('Frequency')
    ax2.set_title('Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Phase comparison
    ax3 = fig.add_subplot(gs[1, 2])
    phases = ['Early', 'Mid', 'Late']
    means = [early.mean(), mid.mean(), late.mean()]
    stds = [early.std(), mid.std(), late.std()]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    bars = ax3.bar(phases, means, yerr=stds, capsize=8, alpha=0.8, 
                   color=colors, edgecolor='black', linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('Learning Phases', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.0f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    # 4. Box plot
    ax4 = fig.add_subplot(gs[2, 0])
    bp = ax4.boxplot([early, mid, late], labels=phases, patch_artist=True,
                     notch=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax4.set_ylabel('Reward')
    ax4.set_title('Phase Distribution (Boxplot)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Rolling statistics
    ax5 = fig.add_subplot(gs[2, 1])
    rolling_std = np.array([rewards[max(0,i-window):i+1].std() for i in range(len(rewards))])
    ax5.plot(episodes, rolling_std, color='purple', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Rolling Std Dev')
    ax5.set_title(f'Stability Over Time (Window={window})', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative best
    ax6 = fig.add_subplot(gs[2, 2])
    cumulative_best = np.maximum.accumulate(rewards)
    ax6.plot(episodes, cumulative_best, color='darkgreen', linewidth=2)
    ax6.fill_between(episodes, rewards.min(), cumulative_best, alpha=0.3, color='green')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Best Reward So Far')
    ax6.set_title('Cumulative Best Performance', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Complete Training Analysis - {level_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = Path(json_path).parent / 'training_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Plot saved: {output_path}")
    
    plt.show()
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # پیش‌فرض
        json_path = 'results/4layer_3level/level_1/training_results.json'
    
    try:
        analyze_training_quick(json_path)
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        print("\n📁 Available training_results.json files:")
        for f in Path('.').rglob('training_results.json'):
            print(f"  - {f}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
