# -*- coding: utf-8 -*-
"""
Training Status Analyzer - تحلیل جامع نتایج آموزش MADDPG
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from datetime import datetime

# تنظیمات نمودار
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

class TrainingAnalyzer:
    def __init__(self, base_dir="models"):
        self.base_dir = Path(base_dir)
        self.results = {}
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
    def find_training_files(self):
        """جستجوی فایل‌های training_results.json"""
        pattern = str(self.base_dir / "**" / "training_results.json")
        files = glob.glob(pattern, recursive=True)
        
        print(f"\n{'='*80}")
        print("TRAINING STATUS ANALYSIS")
        print(f"{'='*80}\n")
        print(f"🔍 جستجوی فایل‌های Training Results...\n")
        
        if not files:
            print("❌ هیچ فایل training_results.json پیدا نشد!")
            return []
        
        print(f"✅ {len(files)} فایل پیدا شد:")
        for f in files:
            size = os.path.getsize(f) / (1024 * 1024)  # MB
            print(f"   📄 {f} ({size:.2f} MB)")
        
        return files
    
    def load_results(self, file_path):
        """بارگذاری نتایج از فایل JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            level_name = data.get('level_name', 'unknown')
            episodes = data['results']['training_history']['episodes']
            rewards = data['results']['training_history']['rewards']
            noise_values = data['results']['training_history'].get('noise_values', [])
            
            print(f"✅ بارگذاری {level_name}: {len(episodes)} اپیزود")
            
            return {
                'level_name': level_name,
                'display_name': data.get('level_display_name', level_name),
                'episodes': episodes,
                'rewards': rewards,
                'noise_values': noise_values,
                'config': data.get('config', {}),
                'best_reward': data['results'].get('best_reward', max(rewards)),
                'total_episodes': data['results'].get('total_episodes', len(episodes)),
                'timestamp': data.get('timestamp', 'N/A')
            }
        except Exception as e:
            print(f"❌ خطا در بارگذاری {file_path}: {e}")
            return None
    
    def analyze_convergence(self, rewards, window=100):
        """تحلیل همگرایی"""
        if len(rewards) < window:
            return {
                'converged': False,
                'convergence_episode': None,
                'final_performance': np.mean(rewards[-min(50, len(rewards)):]),
                'stability': 0
            }
        
        # محاسبه میانگین متحرک
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # محاسبه واریانس در پنجره‌های مختلف
        variances = [np.var(rewards[i:i+window]) for i in range(0, len(rewards)-window, window//2)]
        
        # تشخیص همگرایی: واریانس کم و عملکرد بالا
        recent_variance = np.var(rewards[-window:])
        recent_mean = np.mean(rewards[-window:])
        
        converged = recent_variance < 10 and recent_mean > -5  # معیارهای همگرایی
        
        # پیدا کردن نقطه همگرایی (اولین بار که شرایط برقرار شد)
        convergence_ep = None
        for i in range(window, len(rewards)):
            if np.var(rewards[i-window:i]) < 10 and np.mean(rewards[i-window:i]) > -5:
                convergence_ep = i
                break
        
        return {
            'converged': converged,
            'convergence_episode': convergence_ep,
            'final_performance': recent_mean,
            'stability': 1 / (1 + recent_variance),  # معیار پایداری
            'variance_trend': variances
        }
    
    def plot_analysis(self, data, analysis):
        """ترسیم نمودارهای تحلیلی"""
        level_name = data['level_name']
        rewards = data['rewards']
        episodes = data['episodes']
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(f"{data['display_name']} - Training Analysis", fontsize=16, fontweight='bold')
        
        # 1. Raw Rewards
        axes[0, 0].plot(episodes, rewards, alpha=0.6, linewidth=0.8)
        axes[0, 0].set_title('Episode Rewards (Raw)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Moving Average (100 episodes)
        if len(rewards) >= 100:
            window = 100
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(episodes[window-1:], moving_avg, linewidth=2, color='orange')
            axes[0, 1].set_title(f'Moving Average (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Best Reward Progress
        best_rewards = np.maximum.accumulate(rewards)
        axes[1, 0].plot(episodes, best_rewards, linewidth=2, color='green')
        axes[1, 0].set_title('Best Reward Progress')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Best Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Reward Distribution
        axes[1, 1].hist(rewards, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        axes[1, 1].axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Noise Decay (if available)
        if data['noise_values']:
            axes[2, 0].plot(episodes, data['noise_values'], linewidth=2, color='blue')
            axes[2, 0].set_title('Exploration Noise Decay')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Noise Std')
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'No Noise Data', ha='center', va='center', transform=axes[2, 0].transAxes)
        
        # 6. Recent Performance (Last 100 episodes)
        recent_episodes = min(100, len(rewards))
        axes[2, 1].plot(episodes[-recent_episodes:], rewards[-recent_episodes:], linewidth=2, color='red', marker='o', markersize=3)
        axes[2, 1].axhline(np.mean(rewards[-recent_episodes:]), color='green', linestyle='--', linewidth=2, 
                          label=f'Recent Avg: {np.mean(rewards[-recent_episodes:]):.2f}')
        axes[2, 1].set_title(f'Recent Performance (Last {recent_episodes} Episodes)')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Reward')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ذخیره نمودار
        output_path = self.output_dir / f"training_analysis_{level_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ نمودار ذخیره شد: {output_path}")
        
        return output_path
    
    def generate_report(self):
        """تولید گزارش متنی"""
        report_path = self.output_dir / "convergence_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING CONVERGENCE ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for level_name, data in self.results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"{data['display_name']}\n")
                f.write(f"{'='*80}\n\n")
                
                analysis = data['analysis']
                
                f.write(f"📊 Training Statistics:\n")
                f.write(f"   • Total Episodes: {data['total_episodes']}\n")
                f.write(f"   • Average Reward: {np.mean(data['rewards']):.4f}\n")
                f.write(f"   • Best Reward: {data['best_reward']:.4f}\n")
                f.write(f"   • Worst Reward: {min(data['rewards']):.4f}\n")
                f.write(f"   • Std Deviation: {np.std(data['rewards']):.4f}\n\n")
                
                f.write(f"🎯 Convergence Analysis:\n")
                if analysis['converged']:
                    f.write(f"   ✅ Status: CONVERGED\n")
                    if analysis['convergence_episode']:
                        f.write(f"   • Convergence Episode: {analysis['convergence_episode']}\n")
                    f.write(f"   • Final Performance: {analysis['final_performance']:.4f}\n")
                    f.write(f"   • Stability Score: {analysis['stability']:.4f}\n")
                else:
                    f.write(f"   ❌ Status: NOT CONVERGED\n")
                    f.write(f"   • Recommendation: Continue training\n")
                
                f.write("\n")
        
        print(f"\n✅ گزارش ذخیره شد: {report_path}")
        return report_path
    
    def run_analysis(self):
        """اجرای تحلیل کامل"""
        # یافتن فایل‌ها
        files = self.find_training_files()
        if not files:
            return
        
        print(f"\n{'-'*80}\n")
        
        # بارگذاری و تحلیل
        for file_path in files:
            data = self.load_results(file_path)
            if data:
                # تحلیل همگرایی
                analysis = self.analyze_convergence(data['rewards'])
                data['analysis'] = analysis
                
                # ذخیره نتایج
                self.results[data['level_name']] = data
                
                # نمایش خلاصه
                print(f"\n📊 تحلیل {data['level_name']}...")
                print(f"   📈 تعداد اپیزودها: {data['total_episodes']}")
                print(f"   📊 میانگین Reward: {np.mean(data['rewards']):.4f}")
                print(f"   🏆 بهترین Reward: {data['best_reward']:.4f}")
                print(f"   📉 بدترین Reward: {min(data['rewards']):.4f}")
                
                if analysis['converged']:
                    print(f"   ✅ مدل به همگرایی رسیده")
                    if analysis['convergence_episode']:
                        print(f"   📍 اپیزود همگرایی: {analysis['convergence_episode']}")
                else:
                    print(f"   ⚠️  مدل هنوز به همگرایی نرسیده")
                
                # ترسیم نمودارها
                self.plot_analysis(data, analysis)
        
        # تولید گزارش
        print(f"\n{'-'*80}\n")
        self.generate_report()
        
        print(f"\n{'='*80}")
        print("✅ تحلیل کامل شد!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    analyzer = TrainingAnalyzer()
    analyzer.run_analysis()
